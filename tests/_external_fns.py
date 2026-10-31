"""External state quantizers for the vLLM plugin's ``STATEQUANT_VLLM=external:<this file>:<fn>`` mode (checks)."""
import statequant as sq

_INT4K64 = sq.Quantizer(4, 'k', 64, scale_dtype='fp16')     # PACE-4's packed format (fp16 scales)


def identity(S, layer):
    return S


def int4_k64(S, layer):
    return _INT4K64.fake(S)
