"""End-to-end checks of the statequant vLLM plugin (packed GDN state) on Qwen3.5 (default) or OLMo-Hybrid
(``--model allenai/Olmo-Hybrid-7B --out runs_vllm_olmo``). Needs a GPU, vLLM with statequant installed (the plugin
registers via the ``vllm.general_plugins`` entry point) and the model in the HF cache.

One engine configuration per process (the plugin is chosen by STATEQUANT_VLLM before vLLM starts):

    python tests/vllm_packed_check.py run   --engine stock|stockbf16|stockfp32|bf16|pace4|int4v|int8k [--tasks ...]
                                            (stock: vLLM as shipped: fp32 GDN state from Qwen3.5's mamba_ssm_dtype,
                                             the model dtype (bf16) for OLMo-Hybrid; stockbf16 / stockfp32: stock
                                             with mamba_ssm_cache_dtype='bfloat16' / 'float32')
    python tests/vllm_packed_check.py mem   --engine ... --max_model_len 4232     (KV-cache size, max concurrency)
    python tests/vllm_packed_check.py hf    --engine stock|pace4                    (statequant's HF packed backend,
                                                                                    teacher-forced decode loss; run
                                                                                    in a venv with transformers + fla)
    python tests/vllm_packed_check.py report                                        (compare the saved results)

Tasks
  gen    greedy continuations (64 tokens) of a few prompts                     -> token agreement with stock
  loss   WikiText-103 test windows: prompt_logprobs over 640 tokens (prefill path) and teacher-forced decoding of the
         last 128 tokens after a 512-token prompt (decode path; tokens forced with vLLM's trace replay,
         SamplingParams.trace_decode_token_ids, logprobs from the unmodified logits)
  mixed  many concurrent requests of different prompt / output lengths with a small token budget (chunked prefill,
         mixed prefill + decode steps, sequences at different buffer positions); agreement with stock and samples
Results: <out>/<engine>.json (+ mem_<engine>_<len>.json); <out> = runs_vllm by default.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'runs_vllm')
MODEL = 'Qwen/Qwen3.5-4B'
STOCK = ('stock', 'stockbf16', 'stockfp32')
SEED_PROMPTS = [
    'The history of the Roman Empire can be divided into',
    'def quicksort(arr):\n    """Sort a list in place."""\n',
    'Photosynthesis is the process by which plants',
    'Q: What is the capital of Australia, and why was it chosen?\nA:',
    'Once upon a time in a small village by the sea, there lived',
    'The key differences between TCP and UDP are',
    'In linear algebra, an eigenvector of a matrix is',
    'Write a short poem about autumn leaves:\n',
]
PROMPT_LEN, DECODE_LEN = 512, 128


# ---------------------------------------------------------------------------------------------------------------
def wikitext_ids(tok):
    from datasets import load_dataset
    wt = '\n'.join(load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='test')['text'])
    return tok(wt).input_ids


def windows(ids, n, seq):
    return [ids[(8 + i) * seq:(9 + i) * seq] for i in range(n)]


def set_engine_env(engine):
    if engine in STOCK:
        os.environ.pop('STATEQUANT_VLLM', None)
    else:
        os.environ['STATEQUANT_VLLM'] = engine


def make_llm(a, **kw):
    from vllm import LLM
    args = dict(model=MODEL, dtype='bfloat16', max_model_len=a.max_model_len, enable_prefix_caching=False, seed=0)
    if 'qwen3.5' in MODEL.lower():               # a vision-language checkpoint: text only
        args['limit_mm_per_prompt'] = {'image': 0, 'video': 0}
    if a.kv_gib:                                 # a fixed KV-cache budget (comparable concurrency); gpu_mem is
        args['kv_cache_memory_bytes'] = int(a.kv_gib * 2 ** 30)            # then only vLLM's startup check
    args['gpu_memory_utilization'] = a.gpu_mem
    if a.engine == 'stockbf16':                  # stock vLLM with a bf16 recurrent state (Qwen3.5's config asks
        args['mamba_ssm_cache_dtype'] = 'bfloat16'  # for float32, which stock vLLM follows)
    if a.engine == 'stockfp32':                  # ... with an fp32 one (OLMo-Hybrid's default is the model dtype)
        args['mamba_ssm_cache_dtype'] = 'float32'
    args.update(kw)
    return LLM(**args)


def task_gen(llm, res):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=64, temperature=0, ignore_eos=True)
    t0 = time.time()
    outs = llm.generate(SEED_PROMPTS, sp, use_tqdm=False)
    res['gen'] = [list(o.outputs[0].token_ids) for o in outs]
    res['gen_text'] = [o.outputs[0].text for o in outs]
    print(f'[gen] {len(outs)} prompts in {time.time() - t0:.1f}s; first: {res["gen_text"][0][:200]!r}', flush=True)


def task_loss(llm, tok, res, n_win):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    ids = wikitext_ids(tok)
    wins = windows(ids, n_win, PROMPT_LEN + DECODE_LEN)
    # prefill path
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=w) for w in wins], sp, use_tqdm=False)
    nll, n = 0., 0
    for w, o in zip(wins, outs):
        for tid, lp in zip(w[1:], o.prompt_logprobs[1:]):
            nll -= lp[tid].logprob
            n += 1
    res['loss_prefill'] = nll / n
    # decode path: prompt PROMPT_LEN, forced continuation DECODE_LEN
    sps = [SamplingParams(max_tokens=DECODE_LEN, temperature=0, logprobs=0, ignore_eos=True,
                          trace_decode_token_ids=w[PROMPT_LEN:]) for w in wins]
    outs = llm.generate([TokensPrompt(prompt_token_ids=w[:PROMPT_LEN]) for w in wins], sps, use_tqdm=False)
    nll, n, bad, per_pos, per_win = 0., 0, 0, [0.] * DECODE_LEN, []
    for w, o in zip(wins, outs):
        per_win.append(-sum(lp[tid].logprob for tid, lp in zip(w[PROMPT_LEN:], o.outputs[0].logprobs)) / DECODE_LEN)
        got = list(o.outputs[0].token_ids)
        bad += sum(x != y for x, y in zip(got, w[PROMPT_LEN:]))
        for j, (tid, lp) in enumerate(zip(w[PROMPT_LEN:], o.outputs[0].logprobs)):
            nll -= lp[tid].logprob
            per_pos[j] -= lp[tid].logprob
            n += 1
    res['loss_decode'] = nll / n
    res['loss_decode_per_pos'] = [x / len(wins) for x in per_pos]
    res['forced_mismatch'] = bad
    res['loss_decode_per_win'] = per_win
    print(f'[loss] {len(wins)} windows: prefill loss {res["loss_prefill"]:.4f} ({PROMPT_LEN + DECODE_LEN - 1} tok/win)'
          f', teacher-forced decode loss {res["loss_decode"]:.4f} ({DECODE_LEN} tok/win), forced-token mismatches '
          f'{bad}', flush=True)


def mixed_requests(tok, n=48):
    import random
    ids = wikitext_ids(tok)
    rng = random.Random(0)
    reqs = []
    for i in range(n):
        L = rng.choice([7, 33, 120, 300, 517, 900, 1400])
        start = 20000 + 3000 * i
        reqs.append((ids[start:start + L], rng.choice([5, 17, 64, 129, 200])))
    return reqs


def task_mixed(llm, tok, res):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    reqs = mixed_requests(tok)
    sps = [SamplingParams(max_tokens=m, temperature=0, ignore_eos=True) for _, m in reqs]
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p, _ in reqs], sps, use_tqdm=False)
    res['mixed'] = [list(o.outputs[0].token_ids) for o in outs]
    res['mixed_text'] = [o.outputs[0].text for o in outs]
    res['mixed_lens'] = [(len(p), m) for p, m in reqs]
    print(f'[mixed] {len(reqs)} requests in {time.time() - t0:.1f}s', flush=True)


def cmd_run(a):
    set_engine_env(a.engine)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tasks = a.tasks.split(',')
    res = {'engine': a.engine, 'env': {k: v for k, v in os.environ.items() if k.startswith('STATEQUANT')}}
    path = os.path.join(OUT, f'{a.engine}{a.suffix}.json')
    if os.path.exists(path):
        res = {**json.load(open(path)), **res}
    os.makedirs(OUT, exist_ok=True)
    if 'gen' in tasks or 'loss' in tasks:
        llm = make_llm(a, enable_trace_replay=True)
        if 'gen' in tasks:
            task_gen(llm, res)
        if 'loss' in tasks:
            task_loss(llm, tok, res, a.windows)
        del llm
    if 'mixed' in tasks:
        if 'gen' in tasks or 'loss' in tasks:
            print('[mixed] run it in its own process (--tasks mixed): it needs a small token budget', flush=True)
        else:
            llm = make_llm(a, max_num_batched_tokens=a.budget, max_num_seqs=32)
            task_mixed(llm, tok, res)
    json.dump(res, open(path, 'w'), indent=1)


def cmd_mem(a):
    set_engine_env(a.engine)
    import gc
    llm = make_llm(a, enforce_eager=True)
    core = getattr(llm.llm_engine, 'engine_core', None)
    cfg = llm.llm_engine.vllm_config.cache_config
    info = {'engine': a.engine, 'max_model_len': a.max_model_len, 'kv_gib': a.kv_gib,
            'num_gpu_blocks': cfg.num_gpu_blocks, 'block_size': cfg.block_size,
            'mamba_page_size_padded': cfg.mamba_page_size_padded}
    print('[mem]', json.dumps(info), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(info, open(os.path.join(OUT, f'mem_{a.engine}_{a.max_model_len}.json'), 'w'), indent=1)
    del llm, core
    gc.collect()


def cmd_hf(a):
    """statequant's HF transformers path on the same windows: teacher-forced decode loss of the last DECODE_LEN
    tokens after a PROMPT_LEN prompt (stock bf16 state, or the packed backend with the plugin's format)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
    import statequant as sq
    from statequant.integrations.vllm_plugin import _PRESETS
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    wins = torch.tensor(windows(wikitext_ids(tok), a.windows, PROMPT_LEN + DECODE_LEN))

    @torch.no_grad()
    def run(ids):
        out = model(input_ids=ids[:, :PROMPT_LEN], use_cache=True)
        pkv, lps = out.past_key_values, [out.logits[:, -1].float().log_softmax(-1)]
        for j in range(DECODE_LEN - 1):
            out = model(input_ids=ids[:, PROMPT_LEN + j:PROMPT_LEN + j + 1], past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            lps.append(out.logits[:, -1].float().log_softmax(-1))
        lp = torch.stack(lps, 1)                                              # [B, DECODE_LEN, V]
        tgt = ids[:, PROMPT_LEN:].to(lp.device)
        return -lp.gather(-1, tgt[..., None]).squeeze(-1)                     # [B, DECODE_LEN]

    nlls = []
    for i in range(0, len(wins), a.hf_batch):
        ids = wins[i:i + a.hf_batch].cuda()
        if a.engine == 'stock':
            nlls.append(run(ids))
        else:
            fmt = 'bf16' if a.engine == 'bf16' else sq.Quantizer(*_PRESETS[a.engine])
            m = int(os.environ.get('STATEQUANT_VLLM_M', '8'))
            bb = int(os.environ.get('STATEQUANT_VLLM_BUFFER_BITS', '8'))
            # one rounding at the end of the prompt (vLLM's prefill writes the slot once), then the packed decode
            sch = sq.Schedule(prefill=1 << 20, decode=m, buffer_bits=bb)
            with sq.apply(model, fmt if fmt != 'bf16' else sq.FloatCast('bf16'), sch, backend='packed'):
                nlls.append(run(ids))
    nll = torch.cat(nlls)
    res = {'engine': a.engine, 'hf_loss_decode': nll.mean().item(),
           'hf_loss_decode_per_pos': nll.mean(0).tolist(), 'hf_loss_decode_per_win': nll.mean(1).tolist()}
    print(f'[hf {a.engine}] teacher-forced decode loss {res["hf_loss_decode"]:.4f}', flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f'hf_{a.engine}{a.suffix}.json'), 'w'), indent=1)


def _agree(a, b):
    """(fraction of equal tokens, tokens before the first difference) of two greedy continuations."""
    eq = [x == y for x, y in zip(a, b)]
    first = next((i for i, e in enumerate(eq) if not e), len(eq))
    return sum(eq) / max(len(eq), 1), first


def cmd_report(a):
    import glob
    R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(OUT, '*.json'))}
    for ref in STOCK:
        stock = R.get(ref, {})
        print(f'== greedy agreement with {ref} (8 prompts x 64 tokens) ==')
        for name, r in sorted(R.items()):
            if name.startswith(('mem_', 'hf_')) or 'gen' not in r or name == ref or 'gen' not in stock:
                continue
            ag = [_agree(x, y) for x, y in zip(r['gen'], stock['gen'])]
            print(f'{name:12s} token agreement {sum(x for x, _ in ag) / len(ag):.3f}; identical sequences '
                  f'{sum(f == 64 for _, f in ag)}/{len(ag)}; first divergence {[f for _, f in ag]}')
    stock = R.get('stockbf16', R.get('stock', {}))
    print('== losses (WikiText-103 test windows) ==')
    for name, r in sorted(R.items()):
        if 'loss_decode' in r:
            hf = R.get(f'hf_{name}', {}).get('hf_loss_decode')
            print(f'{name:12s} prefill {r["loss_prefill"]:.4f}  teacher-forced decode {r["loss_decode"]:.4f}'
                  + (f'  (HF statequant path: {hf:.4f})' if hf is not None else '')
                  + f'  forced mismatches {r["forced_mismatch"]}')
    for ref in STOCK:
        print(f'== mixed batch (48 requests, chunked prefill, budget 256 tokens/step) vs {ref} ==')
        st = R.get(ref, {})
        for name, r in sorted(R.items()):
            if 'mixed' in r and name != ref and 'mixed' in st:
                ag = [_agree(x, y) for x, y in zip(r['mixed'], st['mixed'])]
                fd = sorted(f for _, f in ag)
                print(f'{name:12s} token agreement {sum(x for x, _ in ag) / len(ag):.3f}; identical '
                      f'{sum(f == len(x) for (_, f), x in zip(ag, r["mixed"]))}/{len(ag)}; median first '
                      f'divergence {fd[len(fd) // 2]}')
    print('== memory ==')
    for name, r in sorted(R.items()):
        if name.startswith('mem_'):
            print(name, r)


def main():
    global MODEL, OUT
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['run', 'mem', 'hf', 'report'])
    p.add_argument('--engine', default='stock')
    p.add_argument('--tasks', default='gen,loss')
    p.add_argument('--max_model_len', type=int, default=2048)
    p.add_argument('--gpu_mem', type=float, default=0.45)
    p.add_argument('--kv_gib', type=float, default=0.)
    p.add_argument('--windows', type=int, default=32)
    p.add_argument('--budget', type=int, default=256, help='max_num_batched_tokens of the mixed task')
    p.add_argument('--hf_batch', type=int, default=8)
    p.add_argument('--suffix', default='')
    p.add_argument('--model', default=MODEL, help='HF repo (Qwen/Qwen3.5-4B, allenai/Olmo-Hybrid-7B, ...)')
    p.add_argument('--out', default=None, help='results directory (default runs_vllm)')
    a = p.parse_args()
    MODEL = a.model
    if a.out:
        OUT = a.out if os.path.isabs(a.out) else os.path.join(os.path.dirname(HERE), a.out)
    {'run': cmd_run, 'mem': cmd_mem, 'hf': cmd_hf, 'report': cmd_report}[a.cmd](a)


if __name__ == '__main__':
    main()
