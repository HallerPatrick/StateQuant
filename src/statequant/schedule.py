"""When the recurrent state is rounded.

Prior work rounds the state at every write (every token). PACE rounds once every ``prefill`` tokens of the prompt (one
kernel chunk) and once every ``decode`` generated tokens; the tokens in between are held in a decode buffer of
``buffer_bits``-wide keys and values and applied to the full-precision working state, so rounding happens only at the
flush. Positions count along the whole sequence, so a 4000-token prompt followed by decoding rounds at the same places
however the caller splits it into calls.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Schedule:
    """prefill        rounding period (tokens) in multi-token calls (prompt processing)
    decode            rounding period (tokens) in single-token decode calls
    buffer_bits       width of the buffered keys/values between roundings: 4 or 8 (int, absmax per vector), 16 (bf16),
                      0 (full precision)
    buffer            which buffered tensors are rounded: 'kv' | 'k' | 'v'
    buffer_in_prefill also round k/v of prompt tokens to ``buffer_bits`` (the PACE evaluation harness does this;
                      a deployment's prefill kernel reads full-precision k/v, so the default is False)
    prompt_end        also round at the end of every multi-token call (a deployment stores the state packed after
                      prefill even when the prompt length is not a multiple of ``prefill``; DAMP, LeapQuant)
    """
    prefill: int = 256
    decode: int = 8
    buffer_bits: int = 8
    buffer: str = 'kv'
    buffer_in_prefill: bool = False
    prompt_end: bool = False

    def __post_init__(self):
        assert self.prefill >= 1 and self.decode >= 1
        assert self.buffer in ('kv', 'k', 'v') and self.buffer_bits in (0, 4, 8, 16)

    def buffer_bits_per_value(self, d_k: int, d_v: int, heads_per_key: int = 1) -> float:
        """Resident cost of a full decode buffer per state value (PACE Eq. 6): per buffered token one key (shared by
        ``heads_per_key`` value heads), one value, a 16-bit scale per rounded vector and 32 bits of gate/beta, for
        ``decode`` tokens. PACE-4 on Qwen3.5 (d_k = d_v = 128, 2 value heads per key): 0.781, resident 5.03."""
        if self.decode <= 1:
            return 0.
        qk = bool(self.buffer_bits) and 'k' in self.buffer
        qv = bool(self.buffer_bits) and 'v' in self.buffer
        bk = self.buffer_bits if qk else 16
        bv = self.buffer_bits if qv else 16
        per_tok = bk * d_k / heads_per_key + bv * d_v + 32 + 16 * (qk + qv)
        return per_tok * self.decode / (d_k * d_v)


def per_token() -> Schedule:
    """Round at every write, prefill and decode (the convention of DAMP's Eq. 4 and of Q-Mamba)."""
    return Schedule(prefill=1, decode=1, buffer_bits=0)


def periodic(prefill: int = 256, decode: int = 8, buffer_bits: int = 8, **kw) -> Schedule:
    return Schedule(prefill=prefill, decode=decode, buffer_bits=buffer_bits, **kw)
