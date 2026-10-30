"""State formats: what a stored recurrent state looks like.

A recurrent state is a float tensor S of shape [B, H, d_k, d_v] (fla / transformers layout). A format maps S to the
dequantized state it would be after a store/load round trip (``fake``) and prices that store in bits per value.

``Quantizer``   integer grid with per-group scales, symmetric (absmax) or affine (min/max + zero point), groups along
                the key axis (``axis='k'``: one scale per ``group`` key rows of a value column, PACE's choice), the value
                axis (``axis='v'``, DAMP's), or the whole head (``axis='head'``). Optional Hadamard rotation along the
                grouping axis, stochastic rounding.
``FloatCast``   a float storage dtype (bf16, fp16, fp8), optionally with stochastic rounding (Nemotron's fp16 + SR).
``Identity``    full precision.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

AXES = ('k', 'v', 'head')


def hadamard(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Normalised Sylvester Hadamard matrix (n a power of two)."""
    assert n > 0 and n & (n - 1) == 0, f'hadamard size must be a power of two, got {n}'
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / n ** 0.5


class Format:
    """A storage format for one state tensor. Subclasses implement ``fake`` and ``bits_per_value``."""

    def fake(self, S: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def bits_per_value(self, d_k: int, d_v: int) -> float:
        raise NotImplementedError

    def __call__(self, S: torch.Tensor) -> torch.Tensor:
        return self.fake(S)


@dataclass(frozen=True)
class Identity(Format):
    """No quantization (bits_per_value reports ``bits``, the dtype the state is kept in)."""
    bits: int = 32

    def fake(self, S):
        return S

    def bits_per_value(self, d_k, d_v):
        return float(self.bits)


@dataclass(frozen=True)
class FloatCast(Format):
    """Round trip through a float dtype: 'bf16' | 'fp16' | 'fp8' (e4m3). ``stochastic`` rounds unbiasedly."""
    dtype: str = 'fp16'
    stochastic: bool = False

    _DT = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp8': torch.float8_e4m3fn}

    def fake(self, S):
        dt = self._DT[self.dtype]
        x = S.float()
        if not self.stochastic:
            return x.to(dt).to(S.dtype)
        # unbiased: round to one of the two neighbouring representable values with probability given by the distance
        lo = x.to(dt).float()
        nxt = torch.nextafter(lo.to(dt), torch.where(x >= lo, torch.full_like(lo, float('inf')),
                                                     torch.full_like(lo, float('-inf'))).to(dt)).float()
        gap = (nxt - lo)
        p = torch.where(gap != 0, (x - lo) / gap, torch.zeros_like(x))
        return torch.where(torch.rand_like(x) < p, nxt, lo).to(S.dtype)

    def bits_per_value(self, d_k, d_v):
        return {'bf16': 16., 'fp16': 16., 'fp8': 8.}[self.dtype]


@dataclass(frozen=True)
class Quantizer(Format):
    """Integer state grid with per-group scales.

    bits        payload width, 2..8 (any width up to 15 works in ``fake``)
    axis        'k': groups of ``group`` consecutive key rows within one value column (PACE);
                'v': groups of ``group`` consecutive value columns within one key row (DAMP);
                'head': one scale for the whole [d_k, d_v] head state
    group       group length (ignored for axis='head'). If it does not divide its axis, the last group is shorter
                (ceil(n / group) groups; e.g. group 64 on OLMo-Hybrid's d_k 96: 64 + 32)
    affine      False: symmetric absmax, levels -qmax..qmax with qmax = 2^(b-1)-1;
                True: min/max affine, 2^b levels with a zero point
    zero_point  affine only. 'int': zero point snapped to the payload grid, stored as a b-bit code; 'fp': stored as float
    rotation    None | 'hadamard': orthogonal rotation along the grouping axis before rounding (inverted after, so the
                returned state is in the original basis; in deployment it folds into the read-out and costs nothing).
                An axis whose length is not a power of two (OLMo-Hybrid: d_k 96, d_v 192) is rotated block-diagonally,
                in blocks of the largest power of two dividing it (32 resp. 64)
    rounding    'rtn' | 'sr' (stochastic)
    scale_bits  bits per stored scale (and per float zero point)
    scale_dtype 'fp32' (default): with scale_bits=16 the scale is used as computed, in fp32, which is what the PACE
                evaluation harness does; 'fp16': the scale (and a float zero point) is rounded to fp16 before the
                payload is rounded against it, i.e. exactly what a packed store with fp16 scales holds
                (``statequant.kernels``). An fp16 scale is clamped below at 2^-24, the smallest fp16 subnormal, so an
                all-zero group never gets a zero scale. Requires scale_bits=16.
    """
    bits: int = 4
    axis: str = 'k'
    group: int = 32
    affine: bool = False
    zero_point: str = 'int'
    rotation: str | None = None
    rounding: str = 'rtn'
    scale_bits: int = 16
    scale_dtype: str = 'fp32'

    def __post_init__(self):
        assert self.axis in AXES, f'axis must be one of {AXES}, got {self.axis!r}'
        assert 2 <= self.bits <= 15, f'bits must be in 2..15, got {self.bits}'
        assert self.zero_point in ('int', 'fp') and self.rounding in ('rtn', 'sr') and self.rotation in (None, 'hadamard')
        assert self.scale_bits in (8, 16)
        assert self.scale_dtype in ('fp32', 'fp16') and (self.scale_dtype == 'fp32' or self.scale_bits == 16)

    # -- layout: move the grouping axis last, [..., n_groups, group] ---------------------------------------------
    def _groups(self, S):
        if self.axis == 'k':
            S = S.transpose(-1, -2)                          # [..., d_v, d_k]: groups run along k
        if self.axis == 'head':
            return S.reshape(*S.shape[:-2], 1, S.shape[-2] * S.shape[-1])
        n, g = S.shape[-1], self.group
        pad = (-n) % g
        if pad:                                          # ragged last group (OLMo-Hybrid d_k 96, group 64): padded
            S = torch.cat([S, S[..., -1:].expand(*S.shape[:-1], pad)], -1)   # by repeating its last value, which
        return S.reshape(*S.shape[:-1], (n + pad) // g, g)                    # changes neither absmax nor min / max

    def _ungroup(self, X, shape):
        if self.axis == 'k':
            return X.reshape(*shape[:-2], shape[-1], -1)[..., :shape[-2]].transpose(-1, -2)
        if self.axis == 'head':
            return X.reshape(shape)
        return X.reshape(*shape[:-1], -1)[..., :shape[-1]]

    def _rot(self, S, inverse=False):
        # always as a right-multiplication of the grouping axis, transposed to last for axis 'k': the same float
        # association as the PACE reference, so rounding thresholds land identically
        def right(X):
            n = X.shape[-1]
            if n & (n - 1):                              # not a power of two (OLMo-Hybrid d_k 96, d_v 192): block-
                b = n & -n                               # diagonal Hadamard, blocks of the largest power of two
                assert b >= 2, f'hadamard rotation needs an even axis, got {n}'   # dividing n
                Hm = hadamard(b, X.device, X.dtype)
                Xb = X.reshape(*X.shape[:-1], n // b, b)
                return (Xb @ Hm.t() if inverse else Xb @ Hm).reshape(X.shape)
            Hm = hadamard(n, X.device, X.dtype)
            return X @ Hm.t() if inverse else X @ Hm
        if self.axis == 'k':
            return right(S.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
        return right(S)

    def _cast_scale(self, scale):
        if self.scale_bits == 8:
            return scale.to(torch.float8_e4m3fn).float().clamp_min(1e-12)
        if self.scale_dtype == 'fp16':
            return scale.to(torch.float16).float().clamp_min(2. ** -24)
        return scale

    def _round(self, u):
        return torch.floor(u + torch.rand_like(u)) if self.rounding == 'sr' else torch.round(u)

    def fake(self, S):
        x = S.float()
        if self.rotation:
            assert self.axis != 'head', 'rotation needs a grouping axis'
            x = self._rot(x)
        X = self._groups(x)
        if not self.affine:
            qmax = 2 ** (self.bits - 1) - 1
            scale = self._cast_scale(X.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax)
            Xq = self._round(X / scale).clamp(-qmax, qmax) * scale
        else:
            qmax = 2 ** self.bits - 1
            lo, hi = X.amin(-1, keepdim=True), X.amax(-1, keepdim=True)
            scale = self._cast_scale(((hi - lo) / qmax).clamp_min(1e-12))
            if self.zero_point == 'int':
                lo = torch.round(lo / scale) * scale
            elif self.scale_bits == 8:
                lo = lo.to(torch.float8_e4m3fn).float()
            elif self.scale_dtype == 'fp16':
                lo = lo.to(torch.float16).float()
            Xq = self._round((X - lo) / scale).clamp(0, qmax) * scale + lo
        out = self._ungroup(Xq, x.shape)
        if self.rotation:
            out = self._rot(out, inverse=True)
        return out.to(S.dtype)

    def bits_per_value(self, d_k, d_v):
        side = self.scale_bits
        if self.affine:
            side += self.bits if self.zero_point == 'int' else self.scale_bits
        if self.axis == 'head':
            return self.bits + side / (d_k * d_v)
        n = d_k if self.axis == 'k' else d_v
        return self.bits + side * -(-n // self.group) / n          # ragged last group: ceil(n / group) per n
