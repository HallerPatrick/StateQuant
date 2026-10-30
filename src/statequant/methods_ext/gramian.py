"""Calibration-free decay-weighted mixed-precision state allocation: "Low-Bit Recurrent States in Hybrid LMs"
(arXiv 2609.30950, Sec. 2-3).

Bit allocation (Eq. 3-4): per allocation unit i, c_i = Gamma^_i R~_i^2 with
    Gamma^_i = 1 / expm1(2 (s^_i + r_i)),   s^_i = -1/2 log EMA_64(exp(2 g_t,i)),   r_i = 1/2 EMA_64[beta_t (2 - beta_t) k_t,i^2]
(EMA: chunk-aware, target span 64 tokens, recomputed at every write-back from the sequence so far; g = log decay),
R~_i the unit's range normalised by the RMS range of its output-normalisation group; widths
b_i = b_mean + 1/2 log2(c_i / GM(c)), rounded, clipped to [b_min, b_max] (b_max 8, 16 at C = 1) and adjusted to the
layer budget sum_i b_i = b_mean * n_units.

Units per family (``unit='auto'``, Sec. 3 "Allocation units"):
    gdn         per-head decay and per-head output norm: a unit is a head (one width per head, symmetric levels with
                an fp16 scale per block of 32 key channels of a value column), R~ = 1, b_min = 2
    kda         per-channel decay: a unit is a key row (one fp16 scale + width per row), R~ within the head, b_min = 1
    mamba2      normalisation spanning heads: rows as units, R~ within the gated RMSNorm group (H / n_groups heads;
                n_groups read from B's group axis), b_min = 1; no erasure (r = 0)
    lightning   (not in the paper) MiniMax normalises over all heads: rows, R~ over the whole layer, b_min = 1, r = 0
    rwkv6       (not in the paper) per-row decay, per-head GroupNorm: rows, R~ within the head, r = 0
    rwkv7       NotImplementedError (diag + rank-one transition)

Levels (Sec. 3): b >= 2: {-n..n} Delta, n = 2^(b-1) - 1, Delta = f R / n with R the group's absmax and f chosen per
scale group among 16 values in [1/4, 1] by squared error, stored as an fp16 scale. b = 1: sign(x) * mean|x| per group.
Ternary (b = 2) is counted as two bits. Storage: payload + 16/32 bits (block scales) + width bits per head.

Write-back: every token (C = 1, ``per_token()``) or once per 64-token prefill chunk and every decode token (C = 64).
"""
from __future__ import annotations

import math

import torch

from ..schedule import Schedule, per_token
from .hooks import StatefulMethod, terms


def clip_quantize(X, bits, n_clip=16, f_min=0.25):
    """X [..., G] scale groups; bits broadcastable to X[..., :1] (integer widths >= 1). Symmetric levels with a clip
    factor f searched over ``n_clip`` values in [f_min, 1] per group, fp16 Delta; 1 bit = sign * fp16 mean|x|."""
    X = X.float()
    b = torch.as_tensor(bits, device=X.device).float()
    n = (2 ** (b - 1) - 1).clamp_min(1)                               # [..., 1]
    R = X.abs().amax(-1, keepdim=True)                                 # [..., 1]
    fs = torch.linspace(f_min, 1., n_clip, device=X.device)
    best, best_err = None, None
    for f in fs:
        step = (f * R / n).half().float().clamp_min(2. ** -24)
        Xq = torch.round(X / step).clamp(-n, n) * step
        err = (Xq - X).square().sum(-1, keepdim=True)
        if best is None:
            best, best_err = Xq, err
        else:
            better = err < best_err
            best = torch.where(better, Xq, best)
            best_err = torch.where(better, err, best_err)
    one = torch.sign(X) * X.abs().mean(-1, keepdim=True).half().float()
    return torch.where(b == 1, one, best)


def ema_update(num, den, x, alpha):
    """Token-sequential debiased EMA over a piece: x [B, T, ...]; num, den carry the running sums (decay 1 - alpha per
    token), so the result does not depend on how the tokens are chunked."""
    T = x.shape[1]
    w = alpha * (1 - alpha) ** torch.arange(T - 1, -1, -1, device=x.device, dtype=torch.float64)
    w = w.view(1, T, *([1] * (x.dim() - 2)))
    keep = (1 - alpha) ** T
    num = keep * num + (w * x.double()).sum(1)
    den = keep * den + w.sum(1)
    return num, den


def allocate(c, mean_bits, b_min, b_max):
    """Eq. 4, then round, clip and adjust to sum = mean_bits * n per row of c [..., n]: greedily, one bit at a time,
    take a bit where the high-rate proxy c_i 2^(-2 b_i) grows least / give one where it falls most. A unit's marginal
    costs form an increasing sequence, so the greedy equals taking the |excess| smallest of all units' marginal costs
    at once (ties to the lower unit index, as a sequential argmin would)."""
    c = c.double().clamp_min(1e-300)
    n = c.shape[-1]
    lc = c.log2()
    b = (mean_bits + 0.5 * (lc - lc.mean(-1, keepdim=True))).round().clamp(b_min, b_max)
    budget = int(round(mean_bits * n))
    assert n * b_min <= budget <= n * b_max, 'infeasible bit budget'
    R = b_max - b_min
    j = torch.arange(R, device=c.device, dtype=c.dtype)
    excess = (b.sum(-1) - budget).long()                                       # [...]
    # removing the (j+1)-th bit from unit i costs c_i * 3 * 4^-(b_i - j); adding the (j+1)-th gains c_i * .75 * 4^-(b_i+j)
    cost = torch.where(j < (b - b_min)[..., None], c[..., None] * 3 * 4. ** (-(b[..., None] - j)), math.inf)
    gain = torch.where(j < (b_max - b)[..., None], c[..., None] * 0.75 * 4. ** (-(b[..., None] + j)), -math.inf)
    flat_c, flat_g = cost.flatten(-2), gain.flatten(-2)
    oc = torch.sort(flat_c, dim=-1, stable=True).indices
    og = torch.sort(flat_g, dim=-1, descending=True, stable=True).indices
    rank = torch.arange(n * R, device=c.device)
    take = torch.zeros_like(flat_c).scatter(-1, oc, (rank < excess.clamp_min(0)[..., None]).to(c.dtype))
    give = torch.zeros_like(flat_g).scatter(-1, og, (rank < (-excess).clamp_min(0)[..., None]).to(c.dtype))
    b = b - take.view(*b.shape, R).sum(-1) + give.view(*b.shape, R).sum(-1)
    return b.long()


class GramianAlloc(StatefulMethod):
    """bits        layer-wise mean payload b_mean (4 in the paper's per-token comparison)
    C            write-back period: 1 (every token) or 64 (per 64-token prefill chunk + every decode token)
    allocation   'gramian' (the paper), 'uniform' (all units b_mean: the "uniform bit widths" ablation / fitted INT
                 block-32 baseline with the same levels)
    unit         'auto' (per family, above) | 'head' | 'row'
    norm_heads   heads per output-normalisation group for row units (None: per family, above)
    erase        include the erasure rate r_i (the paper keeps it)
    span         EMA target span in tokens (64); alpha = 2 / (span + 1)
    """

    UNITS = {'gdn': 'head', 'kda': 'row', 'mamba2': 'row', 'lightning': 'row', 'rwkv6': 'row'}

    def __init__(self, bits: int = 4, C: int = 1, allocation: str = 'gramian', unit: str = 'auto', block: int = 32,
                 b_min: int | None = None, b_max: int | None = None, erase: bool = True, span: int = 64,
                 n_clip: int = 16, norm_heads: int | None = None, name: str | None = None):
        assert allocation in ('gramian', 'uniform') and unit in ('auto', 'head', 'row')
        self.bits, self.C, self.allocation, self.unit, self.block = bits, C, allocation, unit, block
        self._b_min, self.norm_heads = b_min, norm_heads
        self.b_max = b_max if b_max is not None else (16 if C == 1 else 8)
        self.erase, self.span, self.n_clip = erase, span, n_clip
        self.alpha = 2. / (span + 1)
        self.layers, self.widths, self.units_seen = {}, {}, set()
        self.name = name or f'Gramian-{allocation} b={bits} C={C} ({unit} units)'
        self.schedule = per_token() if C == 1 else Schedule(prefill=C, decode=1, buffer_bits=0)

    def reset(self):
        self.layers, self.widths = {}, {}

    def unit_of(self, family):
        if self.unit != 'auto':
            return self.unit
        if family not in self.UNITS:
            raise NotImplementedError(f'GramianAlloc: no allocation units for the {family} family')
        return self.UNITS[family]

    def b_min(self, unit):
        return self._b_min if self._b_min is not None else (2 if unit == 'head' else 1)

    def group_heads(self, ctx, H):
        """Heads per output-normalisation group (row units)."""
        if self.norm_heads is not None:
            return self.norm_heads
        name = ctx.family.name
        if name == 'mamba2':                                  # gated RMSNorm groups = n_groups (B's group axis)
            Bg = ctx.pc['B']
            G = Bg.shape[2] if Bg.dim() == 4 else Bg.shape[1]
            return H // G
        if name == 'lightning':
            return H
        return 1

    def _init(self, ctx):
        return {'n_s': 0., 'd_s': 0., 'n_r': 0., 'd_r': 0.}

    def unit_weights(self, st, S, unit='head', group=1):
        """c_i per unit [B, n_units] from the EMAs (and the state's ranges for row units, normalised by the RMS row
        range of the unit's output-normalisation group of ``group`` heads)."""
        a2 = (st['n_s'] / st['d_s']).clamp_min(1e-300)                       # EMA(exp(2 g)) [B, H] or [B, H, d_k]
        s_hat = -0.5 * a2.log()
        r = 0.5 * st['n_r'] / st['d_r'] if (self.erase and st['d_r'] is not None) else 0.  # [B, H, d_k]
        if s_hat.dim() == 2:
            s_hat = s_hat[..., None]
        gam = 1. / torch.expm1(2 * (s_hat + r)).clamp_min(1e-30)              # [B, H, d_k]
        B, H, dk = S.shape[:3]
        gam = gam.expand(B, H, dk)
        if unit == 'head':
            return gam.mean(-1)                                               # R~ = 1 (per-head output norm)
        R = S.abs().amax(-1).double().reshape(B, H // group, group * dk)      # row ranges by norm group
        Rt = (R / R.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-30)).reshape(B, H, dk)
        return (gam * Rt.square()).flatten(1)

    def step(self, ctx):
        st = self._layer_state(ctx, self._init)
        lg, key, beta, _, rule = terms(ctx.pc)                                # [B,T,H(,d_k)], [B,T,H,d_k], [B,T,H]
        unit = self.unit_of(ctx.family.name)
        self.units_seen.add(unit)
        st['n_s'], st['d_s'] = ema_update(st['n_s'], st['d_s'], (2 * lg).exp(), self.alpha)
        if rule == 'delta':                                                   # erasure rate (delta rule only)
            er = (beta.float() * (2 - beta.float()))[..., None] * key.square()
            st['n_r'], st['d_r'] = ema_update(st['n_r'], st['d_r'], er, self.alpha)
        else:
            st['d_r'] = None
        if not bool(ctx.hit.any()):
            return ctx.S_out
        S = ctx.S_out
        B, H, dk, dv = S.shape
        n_units = H if unit == 'head' else H * dk
        if self.allocation == 'uniform':
            b = torch.full((B, n_units), self.bits, dtype=torch.long, device=S.device)
        else:
            c = self.unit_weights(st, S, unit, self.group_heads(ctx, H) if unit == 'row' else 1)
            b = allocate(c, self.bits, self.b_min(unit), self.b_max)
        self.widths[ctx.layer] = b
        Sq = self.quantize_widths(S, b, unit)
        if not bool(ctx.hit.all()):
            Sq = torch.where(ctx.hit.to(S.device).view(-1, 1, 1, 1), Sq, S)
        return Sq

    def quantize_widths(self, S, b, unit='head'):
        """S [B, H, d_k, d_v]; b [B, H] (head units: scales per ``block`` key channels of a value column) or
        [B, H*d_k] (row units: one scale per row)."""
        B, H, dk, dv = S.shape
        if unit == 'head':
            blk = min(self.block, dk)
            X = S.transpose(-1, -2).reshape(B, H, dv, dk // blk, blk)
            Xq = clip_quantize(X, b.view(B, H, 1, 1, 1), self.n_clip)
            return Xq.reshape(B, H, dv, dk).transpose(-1, -2)
        return clip_quantize(S, b.view(B, H, dk, 1), self.n_clip)

    def width_bits(self):
        return 4 if self.b_max > 8 else 3

    def bits_per_value(self, d_k, d_v):
        """Mean payload (= b_mean by the budget) + scales + widths, incl. metadata as the paper counts it."""
        unit = self.unit if self.unit != 'auto' else (sorted(self.units_seen)[0] if self.units_seen else 'head')
        if unit == 'head':
            return self.bits + 16 / self.block + self.width_bits() / (d_k * d_v)
        return self.bits + (16 + self.width_bits()) / d_v
