"""DAMP as the paper specifies it (Zhang et al., arXiv 2608.27513 v2, Sec. 5, App. A.1-A.3, Algorithm 1).

Storage (App. A.1, A.4): per head, K_hi = 16 key channels in FP16, the rest INT8 symmetric (Eq. 5-6) in groups of
32 consecutive values along the value dimension with one FP32 scale and no zero point (9.0 bits; total
16/128 * 16 + 112/128 * 9 = 9.875 = "9.9 bits"); INT4: asymmetric, same groups, FP32 scale and FP32 zero point
(6.0 bits). Group parameters are recomputed at every write; every decode step is requantized; the prefill runs in
FP32 within each chunk and writes its final state packed (``Schedule(prefill=chunk, decode=1, prompt_end=True)``).

Value scaling and reordering (SR, App. A.1/A.3): T = P_K S D_c^-1 P_V^T, with per-(layer, head) value scales c and
value order P_V (dimensions sorted by RMS after scaling) fixed by calibration. In deployment D_c and P_V are folded
into the V / O projections; here the fake-quant applies scale + permute before quantization and the inverse after,
which is the same arithmetic. The key permutation P_K only moves the protected rows to the front, so it is
represented by the row mask.

Calibration (App. A.2, Algorithm 1): FP32-state trajectories; scoring anchors at fixed positions, each followed by
H = 256 per-token decays. For each layer/head and SR candidate c: e_{t,u} = ||[Q(T) - T]_u restored||^2 (Eq. 14) on
fitting anchors, C_u = E[e_{t,u} W_{t,u}] with W = 1 + sum_{j<=H} prod_{r<=j} a_{t+r,u}^2 (Eq. 15), H_c = TopK(C, K_hi);
J(c) (Eq. 16) on the parameter-selection anchors; keep argmin_c J.
"""
from __future__ import annotations

import math

import torch

from ..methods import StateQuantizer
from ..quantizer import FloatCast, Quantizer

# anchors of App. A.2: decode updates 256/512/768 of two windows starting at decode positions 0 and 2048, after a
# 1,024-token prompt (as absolute sequence positions); only anchors with H further tokens in a sequence are used
PAPER_ANCHORS = tuple(1024 + w + a for w in (0, 2048) for a in (256, 512, 768))
SR_ALPHAS = (None, 0.0, 0.25, 0.5, 0.75, 1.0)   # None = identity (no SR); else c = rms^alpha, see sr_candidates


def sr_candidates(rms, alphas=SR_ALPHAS):
    """Candidate (scale c [H, d_v], order perm [H, d_v]) pairs from per-value-dimension state RMS [H, d_v]:
    c = rms^alpha / GM(rms^alpha), dimensions sorted by rms / c descending (ties kept in index order)."""
    out = []
    H, dv = rms.shape
    ar = torch.arange(dv).expand(H, dv)
    for a in alphas:
        if a is None:
            out.append((torch.ones_like(rms), ar.clone()))
            continue
        lr = rms.clamp_min(1e-30).log() * a
        c = (lr - lr.mean(-1, keepdim=True)).exp()
        out.append((c, torch.sort(rms / c, dim=-1, descending=True, stable=True).indices))
    return out


def sr_forward(S, c, perm):
    """T = S D_c^-1 P_V^T: scale value dims, then reorder. S [..., H, d_k, d_v]; c, perm [H, d_v]."""
    X = S / c[:, None, :]
    return torch.gather(X, -1, perm[:, None, :].expand_as(X))


def sr_inverse(T, c, perm):
    X = torch.empty_like(T).scatter_(-1, perm[:, None, :].expand_as(T), T)
    return X * c[:, None, :]


def retention_weight(log_decay):
    """W = 1 + sum_{j=1..H} prod_{r<=j} a_r^2 from the H decays after an anchor: [n, H_steps, heads(, d_k)] ->
    [n, heads(, d_k)]."""
    return 1. + (2 * log_decay.double().cumsum(1)).exp().sum(1)


class DampPaper(StateQuantizer):
    """lo          low-precision format: INT8 symmetric g32 fp32 scale (default) or INT4 asymmetric g32 fp32 scale + zp
    k_hi         protected key channels per head (FP16)
    horizon      H (256)
    anchors      absolute sequence positions of the scoring anchors (multiples of the calibration period)
    alphas       SR candidates (``sr_candidates``)
    """
    needs_calibration = True
    calibration_period = 256

    def __init__(self, bits: int = 8, k_hi: int = 16, horizon: int = 256, anchors=PAPER_ANCHORS, alphas=SR_ALPHAS,
                 chunk: int = 8192, lo=None, name: str | None = None):
        assert bits in (8, 4)
        self.bits, self.k_hi, self.horizon, self.anchors, self.alphas = bits, k_hi, horizon, tuple(anchors), alphas
        self.calibration_period = horizon                    # one calibration piece = the H decays after an anchor
        self.lo = lo or (Quantizer(8, 'v', 32) if bits == 8 else Quantizer(4, 'v', 32, affine=True, zero_point='fp'))
        self.hi = FloatCast('fp16')
        self.meta_bits = 32 if bits == 8 else 64                           # fp32 scale (+ fp32 zero point) per group
        self.name = name or f'DAMP paper INT{bits} (K_hi {k_hi} fp16, SR)'
        self.masks, self.sr, self.choice = {}, {}, {}
        self._anchor_S, self._anchor_lg, self._seq = {}, {}, {}
        from ..schedule import Schedule
        self.schedule = Schedule(prefill=chunk, decode=1, buffer_bits=0, prompt_end=True)

    # -- calibration -----------------------------------------------------------------------------------------------
    def observe(self, layer, S, log_decay, kw=None):
        assert getattr(kw, 'position', None) is not None, \
            'DampPaper needs statequant.calibrate (a families.Piece with positions)'
        from .hooks import check_family
        check_family(kw.family, 'DampPaper')
        pos = int(kw.position[0])
        assert bool((kw.position == pos).all())
        period = log_decay.shape[1]
        if pos == period:                                    # first boundary of a new batch of sequences
            self._seq[layer] = self._seq.get(layer, -1) + 1
        seq = self._seq.get(layer, 0)
        if pos in self.anchors:
            self._anchor_S.setdefault(layer, {})[(seq, pos)] = S.float().cpu()
        a0 = pos - log_decay.shape[1]
        if a0 in self.anchors:
            assert log_decay.shape[1] == self.horizon, 'calibration period must equal the horizon H'
            self._anchor_lg.setdefault(layer, {})[(seq, a0)] = log_decay.float().cpu()

    def _anchors(self, layer):
        """Complete anchors: (S [n,H,dk,dv], log decays [n,horizon,H(,dk)], split [n] 0 = fit / 1 = tune)."""
        Ss, ls, split = [], [], []
        for key, S in sorted(self._anchor_S.get(layer, {}).items()):
            lg = self._anchor_lg.get(layer, {}).get(key)
            if lg is None:
                continue
            for b in range(S.shape[0]):
                Ss.append(S[b])
                ls.append(lg[b])
                split.append((key[0] * S.shape[0] + b) % 2)
        assert Ss, f'no complete DAMP anchors for layer {layer}: calibrate on sequences longer than ' \
                   f'min(anchors) + H = {min(self.anchors) + self.horizon}'
        return torch.stack(Ss), torch.stack(ls), torch.tensor(split)

    def _errors(self, S, c, perm, fmt):
        """Squared error per key channel in original coordinates: [n, H, d_k]."""
        T = sr_forward(S, c, perm)
        return (sr_inverse(self._fake(fmt, T), c, perm) - S).square().sum(-1)

    @staticmethod
    def _fake(fmt, T, g=32):
        """fmt on value-axis groups of 32; if d_v is not a multiple of 32 (Nemotron-3-Nano's Mamba2 head_dim 80), the
        last group is shorter: padded by repeating its last value (changes neither absmax nor min / max) and cut."""
        pad = (-T.shape[-1]) % g if isinstance(fmt, Quantizer) and fmt.axis == 'v' else 0
        if not pad:
            return fmt.fake(T)
        Tp = torch.cat([T, T[..., -1:].expand(*T.shape[:-1], pad)], -1)
        return fmt.fake(Tp)[..., :T.shape[-1]]

    def finalize(self, device=None):
        assert self._anchor_S, f'DampPaper saw no anchor states: calibrate on sequences longer than ' \
                               f'min(anchors) + H = {min(self.anchors) + self.horizon} tokens'
        for layer in sorted(self._anchor_S):
            S, lg, split = self._anchors(layer)
            dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            S, lg = S.to(dev), lg.to(dev)
            W = retention_weight(lg).float()                                  # [n, H(, d_k)]
            if W.dim() == 2:
                W = W[..., None]                                              # GDN: shared by a head's channels
            fit = split == 0
            tune = split == 1 if bool((split == 1).any()) else fit
            fit, tune = fit.to(dev), tune.to(dev)
            rms = S[fit].square().mean((0, 2)).sqrt()                         # [H, d_v]
            best = None
            for i, (c, perm) in enumerate(sr_candidates(rms, self.alphas)):
                c, perm = c.to(dev), perm.to(dev)
                e_lo = self._errors(S, c, perm, self.lo)                      # [n, H, d_k]
                e_hi = self._errors(S, c, perm, self.hi)
                Cu = (e_lo[fit] * W[fit]).mean(0)                             # Eq. 15 [H, d_k]
                mask = torch.zeros_like(Cu, dtype=torch.bool).scatter(-1, Cu.topk(self.k_hi, -1).indices, True)
                J = (W[tune] * torch.where(mask, e_hi[tune], e_lo[tune])).sum(-1).mean(0)   # Eq. 16 [H]
                if best is None:
                    best = [J, mask, c, perm, torch.zeros_like(J, dtype=torch.long)]
                else:
                    better = J < best[0]
                    best[0] = torch.where(better, J, best[0])
                    best[1] = torch.where(better[:, None], mask, best[1])
                    best[2] = torch.where(better[:, None], c, best[2])
                    best[3] = torch.where(better[:, None], perm, best[3])
                    best[4] = torch.where(better, torch.full_like(best[4], i), best[4])
            self.masks[layer] = best[1].cpu()
            self.sr[layer] = (best[2].cpu(), best[3].cpu())
            self.choice[layer] = best[4].cpu()
        self._anchor_S, self._anchor_lg = {}, {}

    # -- inference ------------------------------------------------------------------------------------------------
    def quantize(self, S, layer):
        m = self.masks.get(layer)
        if m is None:
            raise RuntimeError('DampPaper is not calibrated: run statequant.calibrate(...) first')
        c, perm = (x.to(S.device) for x in self.sr[layer])
        T = sr_forward(S.float(), c, perm)
        Tq = torch.where(m.to(S.device)[None, :, :, None], self.hi.fake(T), self._fake(self.lo, T))
        return sr_inverse(Tq, c, perm).to(S.dtype)

    def bits_per_value(self, d_k, d_v):
        f = self.k_hi / d_k
        lo_bits = self.bits + self.meta_bits * math.ceil(d_v / 32) / d_v
        return f * 16 + (1 - f) * lo_bits
