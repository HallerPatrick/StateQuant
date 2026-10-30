"""End-to-end checks of the statequant vLLM plugin on a Mamba2-family model (packed ``MambaMixer2`` ssm state).
Same tasks and report as ``vllm_packed_check.py`` (greedy agreement, prefill / teacher-forced decode loss on
WikiText-103, a mixed batch with chunked prefill), on another model; needs a GPU, vLLM, the model in the HF cache, and
this checkout's ``src`` first on PYTHONPATH (the plugin entry point then loads this copy; the log line
``statequant (<path>): Mamba2 ssm state stored as ...`` says which).

    python tests/vllm_mamba2_check.py run  --engine stock|stockbf16|bf16|pace4|int4v|int8k [--model ...] [--tasks ...]
    python tests/vllm_mamba2_check.py mem  --engine ... --max_model_len 4232
    python tests/vllm_mamba2_check.py report [--model ...]

stock: vLLM as shipped (NemotronH: fp32 ssm state, the model's ``mamba_ssm_cache_dtype`` default); stockbf16: stock
with mamba_ssm_cache_dtype='bfloat16'; bf16: the plugin's bf16 slot layout (the control: same storage width as
stockbf16, plus the decode buffer); pace4 / int4v / int8k: int4 key-axis g64, int4 value-axis g32 (halved to divide
head_dim if needed, e.g. g16 for head_dim 80), int8 key-axis g32.
Results: runs_vllm_mamba2/<model>/<engine>.json.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vllm_packed_check as base  # noqa: E402

DEFAULT_MODEL = 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16'


def make_llm(a, **kw):
    from vllm import LLM
    args = dict(model=base.MODEL, dtype='bfloat16', max_model_len=a.max_model_len, enable_prefix_caching=False,
                seed=0, trust_remote_code=True)
    args['gpu_memory_utilization'] = a.gpu_mem                 # (also the startup check with kv_gib)
    if a.kv_gib:                                               # a fixed KV-cache budget: comparable concurrency
        args['kv_cache_memory_bytes'] = int(a.kv_gib * 2 ** 30)
    if a.engine == 'stockbf16':
        args['mamba_ssm_cache_dtype'] = 'bfloat16'
    args.update(kw)
    return LLM(**args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['run', 'mem', 'report'])
    p.add_argument('--engine', default='stock')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--tasks', default='gen,loss')
    p.add_argument('--max_model_len', type=int, default=2048)
    p.add_argument('--gpu_mem', type=float, default=0.45)
    p.add_argument('--kv_gib', type=float, default=0.)
    p.add_argument('--windows', type=int, default=32)
    p.add_argument('--budget', type=int, default=256)
    p.add_argument('--suffix', default='')
    a = p.parse_args()
    base.MODEL = a.model
    base.OUT = os.path.join(os.path.dirname(HERE), 'runs_vllm_mamba2', a.model.split('/')[-1])
    base.make_llm = make_llm
    {'run': base.cmd_run, 'mem': base.cmd_mem, 'report': base.cmd_report}[a.cmd](a)


if __name__ == '__main__':
    main()
