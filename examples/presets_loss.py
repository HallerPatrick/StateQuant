"""Every preset on a transformers hybrid: WikiText-103 loss of N windows (prompt path, one forward), a teacher-forced
decode loss (a prompt of P tokens through the cache, then D tokens fed one at a time through the decode path, NLL of
each next token), deltas against the unpatched model ('none'), and the memory report.

  python examples/presets_loss.py --repo Qwen/Qwen3.5-0.8B --windows 4
  python examples/presets_loss.py --presets none,pace4,lrb_bdr,leapquant8 --decode-windows 2 --prompt 512 --decode 512
  python examples/presets_loss.py --repo inclusionAI/Ling-3.0-tiny --windows 4 --seq 1024 --calib-seq 1024 \
      --decode-windows 2 --prompt 512 --decode 256
"""
import argparse
import contextlib
import json
import time

import torch

import statequant as sq


@torch.no_grad()
def prompt_loss(model, x):
    return sum(model(input_ids=x[i:i + 1], labels=x[i:i + 1], use_cache=False).loss.item()
               for i in range(len(x))) / len(x)


@torch.no_grad()
def decode_loss(model, x, P, D):
    """Mean NLL of tokens P .. P+D-1: the first predicted by the prompt pass, the rest by single-token decode steps."""
    tot, n = 0., 0
    for i in range(len(x)):
        ids = x[i:i + 1]
        out = model(input_ids=ids[:, :P], use_cache=True)
        pkv, logits = out.past_key_values, [out.logits[:, -1:]]
        for t in range(P, P + D - 1):
            out = model(input_ids=ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            logits.append(out.logits)
        lg = torch.cat(logits, 1).float()
        tot += torch.nn.functional.cross_entropy(lg[0], ids[0, P:P + D], reduction='sum').item()
        n += D
    return tot / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='Qwen/Qwen3.5-0.8B')
    p.add_argument('--windows', type=int, default=4, help='windows for the prompt loss (0: skip)')
    p.add_argument('--seq', type=int, default=2048)
    p.add_argument('--calib-seq', type=int, default=None, help='calibration sequence length (default --seq)')
    p.add_argument('--decode-windows', type=int, default=0, help='windows for the decode loss (0: skip)')
    p.add_argument('--prompt', type=int, default=512)
    p.add_argument('--decode', type=int, default=512)
    p.add_argument('--presets', default='none,fp32,rtn,nemotron,qmamba_dsq,damp_int8,damp_int4,pace4,pace48')
    p.add_argument('--json', default=None, help='append one JSON line per preset to this file')
    a = p.parse_args()
    from datasets import load_dataset
    from statequant.compat import from_pretrained, load_tokenizer       # (+ the Ling-3.0 loading fixes)
    model = from_pretrained(a.repo, dtype=torch.bfloat16).cuda().eval()
    tok = load_tokenizer(a.repo)
    ids = tok('\n'.join(load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='test')['text']),
              return_tensors='pt').input_ids[0]
    cs = a.calib_seq or a.seq
    calib = ids[:8 * cs].view(8, cs)
    n = max(a.windows, a.decode_windows)
    x = ids[8 * cs:8 * cs + n * a.seq].view(n, a.seq).cuda()
    xd = x[:a.decode_windows, :a.prompt + a.decode]
    ref = {}
    for name in a.presets.split(','):
        t = time.time()
        q = None if name == 'none' else sq.PRESETS[name]()
        if q is not None and q.needs_calibration:
            sq.calibrate(model, q, calib, period=getattr(q, 'calibration_period', 256))
        with (sq.apply(model, q) if q is not None else contextlib.nullcontext()):
            lp = prompt_loss(model, x[:a.windows]) if a.windows else float('nan')
            ld = decode_loss(model, xd, a.prompt, a.decode) if a.decode_windows else float('nan')
        if name == 'none':
            ref = {'prompt': lp, 'decode': ld}
        r = sq.report(model, q) if q is not None else sq.report(model, sq.Identity(32), sq.per_token())
        row = {'preset': name, 'method': r['method'], 'prompt_loss': lp, 'decode_loss': ld,
               'd_prompt': lp - ref.get('prompt', float('nan')), 'd_decode': ld - ref.get('decode', float('nan')),
               'payload_bits': r['payload_bits_per_value'], 'resident_bits': r['resident_bits_per_value'],
               'peak_bits': r['peak_bits_per_value'], 'seconds': time.time() - t}
        print(f'{name:18s} prompt {lp:.4f} ({row["d_prompt"]:+.4f})  decode {ld:.4f} ({row["d_decode"]:+.4f})  '
              f'payload {row["payload_bits"]:.3f} resident {row["resident_bits"]:.3f} peak {row["peak_bits"]:.3f} '
              f'b/val  {row["seconds"]:.0f}s', flush=True)
        if a.json:
            with open(a.json, 'a') as f:
                f.write(json.dumps({**row, 'args': vars(a)}) + '\n')


if __name__ == '__main__':
    main()
