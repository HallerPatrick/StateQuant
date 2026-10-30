"""End to end on a real model: statequant's patch against the PACE harness's own patch (carry_patch.stream_carry),
prompt plus greedy decoding through the cache, for PACE-4 (no calibration) and PACE-4/8 (calibrated head allocation).
GPU script, run with the PACE venv (its transformers binds the ops per layer, ours patches them module-level too):

  PACE_SRC=<pace>/src PYTHONPATH=src <pace venv>/bin/python tests/e2e_vs_pace.py [--repo Qwen/Qwen3.5-0.8B]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.environ['PACE_SRC'])
import fla.models  # noqa: F401,E402
from carry_patch import stream_carry  # noqa: E402
from damp import build_mix_op  # noqa: E402
from eval_matrix import calibrate_carrypatch  # noqa: E402
from state_quant import fake_quant_state  # noqa: E402

import statequant as sq  # noqa: E402


def run(model, prompt, n_gen):
    """Logits of the prompt and of n_gen greedy decode steps through the cache."""
    with torch.no_grad():
        out = model(input_ids=prompt, use_cache=True)
        logits, pkv = [out.logits[:, -1]], out.past_key_values
        full = out.logits
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(n_gen):
            out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            logits.append(out.logits[:, -1])
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    return full, torch.stack(logits, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='Qwen/Qwen3.5-0.8B')
    p.add_argument('--prompt', type=int, default=600)
    p.add_argument('--gen', type=int, default=20)
    a = p.parse_args()
    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    mt = AutoConfig.from_pretrained(a.repo).model_type
    model = AutoModelForCausalLM.from_pretrained(a.repo, dtype=torch.float32).cuda().eval()
    tok = AutoTokenizer.from_pretrained(a.repo)
    ids = tok('\n'.join(load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='test')['text'][:4000]),
              return_tensors='pt').input_ids
    prompt = ids[:, :a.prompt].cuda()
    calib = ids[:, 10000:10000 + 2 * 2048].view(2, 2048)
    ref_full, ref_dec = run(model, prompt, a.gen)

    def compare(name, pace, ours):
        for (x, y), what in zip(zip(pace, ours), ('prompt', 'decode')):
            d = (x - y).abs().max().item()
            agree = (x.argmax(-1) == y.argmax(-1)).float().mean().item()
            print(f'{name:8s} {what:6s} max|logit diff| {d:.3e}  argmax agreement {agree:.4f}  '
                  f'(vs fp32: {(x - (ref_full if what == "prompt" else ref_dec)).abs().max().item():.3e})', flush=True)

    # PACE-4: uniform:4:256:kg64:dP8:buf8
    q4 = lambda S, i, kw: fake_quant_state(S, 4, 'rtn', ste=False, group='kg64')  # noqa: E731
    with stream_carry(model, mt, 256, q4, decode_period=8, buf_bits=8):
        pace = run(model, prompt, a.gen)
    with sq.apply(model, sq.pace4(paper=True)):
        ours = run(model, prompt, a.gen)
    compare('PACE-4', pace, ours)

    # PACE-4/8: mix:4:8:256:g32:int:product:0.25:full:bufk4:dP8:axk
    cal, _ = calibrate_carrypatch(model, mt, 256, 4, [calib[i:i + 1] for i in range(len(calib))], 'cuda')
    c = dict(sel='product', unit='head', period=256, decode_period=8, frac=0.25, bits=4, hi_bits=8, group='g32',
             zp='int', had='full', mode='rtn', axis='k', label='pace48')
    op, info = build_mix_op(c, cal, 'cuda', verbose=False)
    with stream_carry(model, mt, 256, op, decode_period=8, buf_bits=4, buf_only='k'):
        pace = run(model, prompt, a.gen)
    q48 = sq.calibrate(model, sq.pace48(paper=True), calib, period=256)
    same = all(set(info['units'][i]) == set(q48.masks[i].nonzero().flatten().tolist()) for i in info['units'])
    print(f'PACE-4/8 head allocation identical to the PACE harness: {same}', flush=True)
    with sq.apply(model, q48):
        ours = run(model, prompt, a.gen)
    compare('PACE-4/8', pace, ours)
    print(sq.report(model, q48))


if __name__ == '__main__':
    main()
