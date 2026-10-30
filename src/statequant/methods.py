"""State quantizers: a storage format applied per layer, possibly with a per-layer allocation fitted on calibration data.

Every method implements
    quantize(S, layer)          S: [B, H, d_k, d_v] (the family's canonical layout, ``statequant.families``) ->
                                dequantized state of the same shape
    bits_per_value(d_k, d_v)    resident payload incl. scales / zero points / protected tiers
    needs_calibration           True if ``statequant.calibrate`` must run first; then ``observe(layer, S, log_decay)``
                                is called at every calibration boundary and ``finalize()`` once at the end;
                                log_decay is [B, T, H] (scalar per head) or [B, T, H, d_k] (per key channel: KDA,
                                RWKV); HeadMix's 'retained' / 'slow' scores use the head's mean decay over channels

Methods
    Uniform(fmt)                one format for every head (RTN, PACE-4, Nemotron casts, ...)
    HeadMix(lo, hi, frac)       a fraction of heads per layer stored with ``hi``, chosen by a calibrated score (PACE-4/8)
    ChannelProtect(lo, hi, k)   ``k`` key channels per head stored with ``hi``, chosen by DAMP's score E_u * P_u
    DSQ(bits)                   Q-Mamba's decoupled scales: one scale vector per value coordinate and per key coordinate
"""
from __future__ import annotations

import torch

from .quantizer import Format, Identity


class StateQuantizer:
    needs_calibration = False
    name = 'state-quantizer'

    def quantize(self, S: torch.Tensor, layer: int) -> torch.Tensor:
        raise NotImplementedError

    def bits_per_value(self, d_k: int, d_v: int) -> float:
        raise NotImplementedError

    # calibration protocol (only used when needs_calibration)
    def observe(self, layer: int, S: torch.Tensor, log_decay: torch.Tensor | None, kw: dict | None = None):
        pass

    def finalize(self):
        pass

    def __repr__(self):
        return f'{type(self).__name__}({self.name})'


class Uniform(StateQuantizer):
    """The same format for every layer and head."""

    def __init__(self, fmt: Format, name: str | None = None):
        self.fmt = fmt
        self.name = name or repr(fmt)

    def quantize(self, S, layer):
        return self.fmt.fake(S)

    def bits_per_value(self, d_k, d_v):
        return self.fmt.bits_per_value(d_k, d_v)


class _Stats:
    """Per-layer calibration statistics: error energy of ``fmt`` per (head, key channel) and mean log decay per head."""

    def __init__(self, fmt: Format):
        self.fmt = fmt
        self.err, self.loga, self.n, self.nd = {}, {}, {}, {}

    def observe(self, layer, S, log_decay):
        S = S.float()
        e = (self.fmt.fake(S) - S).pow(2).sum(-1).sum(0).double()           # [H, d_k]
        self.err[layer] = self.err.get(layer, 0) + e
        self.n[layer] = self.n.get(layer, 0) + S.shape[0]
        if log_decay is not None:                                             # [B, T, H] (or [B, T, H, d_k])
            lg = log_decay.float()
            assert (lg < 0).float().mean() > 0.5, 'log decay should be negative; got the wrong gate?'
            lg = lg.clamp_min(-18.42)                                         # log(1e-8), as a.clamp_min(1e-8).log()
            la = lg.mean(1).sum(0).double()
            if la.dim() == 1:
                la = la[:, None]                                              # [H, 1]
            self.loga[layer] = self.loga.get(layer, 0) + la
            self.nd[layer] = self.nd.get(layer, 0) + lg.shape[0]

    def energy(self, layer):                                                  # [H, d_k]
        return (self.err[layer] / self.n[layer]).float()

    def decay(self, layer):                                                   # a_eff per [H, d_k or 1]
        assert layer in self.loga, 'calibration saw no decay for this layer'
        return (self.loga[layer] / self.nd[layer]).float().exp()


HEAD_SCORES = ('retained', 'energy', 'slow', 'random')


class HeadMix(StateQuantizer):
    """Mixed precision by head: in every layer the ``frac`` highest-scoring heads use ``hi``, the rest ``lo``.

    score       'retained': error energy e_h / (1 - a_h^(2 * score_period)), the error a rounding leaves after the
                            decay of one period (PACE); 'energy': e_h; 'slow': mean decay a_h; 'random' (control)
    cal_format  the format whose rounding error e_h is measured (default ``lo``)
    """
    needs_calibration = True

    def __init__(self, lo: Format, hi: Format, frac: float = 0.25, score: str = 'retained', score_period: int = 8,
                 cal_format: Format | None = None, name: str | None = None):
        assert score in HEAD_SCORES, score
        self.lo, self.hi, self.frac, self.score, self.score_period = lo, hi, frac, score, score_period
        self.stats = _Stats(cal_format or lo)
        self.masks, self.scores = {}, {}
        self.name = name or f'{frac:g} of heads {hi!r}, rest {lo!r}, by {score}'

    def observe(self, layer, S, log_decay, kw=None):
        self.stats.observe(layer, S, log_decay)

    def head_scores(self, layer):
        st = self.stats
        e = st.energy(layer).sum(-1)                                          # [H]
        if self.score == 'energy':
            return e
        if self.score == 'random':
            return torch.rand(e.numel(), generator=torch.Generator().manual_seed(1000 + layer))
        a = st.decay(layer).mean(-1)                                          # [H]
        if self.score == 'slow':
            return a
        return e / (1. - a.pow(2 * max(1, self.score_period))).clamp_min(1e-4)

    def finalize(self):
        for layer in self.stats.err:
            sc = self.head_scores(layer).cpu()
            k = max(1, round(self.frac * sc.numel()))
            m = torch.zeros(sc.numel(), dtype=torch.bool)
            m[torch.argsort(sc, descending=True)[:k]] = True
            self.masks[layer], self.scores[layer] = m, sc

    def quantize(self, S, layer):
        m = self.masks.get(layer)
        lo = self.lo.fake(S)
        if m is None:
            assert not self.masks, f'layer {layer} was not seen in calibration'
            raise RuntimeError('HeadMix is not calibrated: run statequant.calibrate(...) first')
        return torch.where(m.to(S.device)[None, :, None, None], self.hi.fake(S), lo)

    def bits_per_value(self, d_k, d_v):
        f = self.frac
        if self.masks:
            m = next(iter(self.masks.values()))
            f = m.float().mean().item()
        return f * self.hi.bits_per_value(d_k, d_v) + (1 - f) * self.lo.bits_per_value(d_k, d_v)


class ChannelProtect(StateQuantizer):
    """DAMP (arXiv 2608.27513): per head, the ``k`` key channels with the largest C_u = E_u * P_u are stored with ``hi``
    (fp16 in the paper), the rest with ``lo``. E_u is the error energy of ``cal_format`` (default ``lo``) on channel u,
    P_u = 1 / max(1 - a_u^(2 * persistence_period), tau) the persistence of an error under the decay.

    Channels are rows of the canonical state's key axis d_k. For Gated DeltaNet, Mamba2 and lightning the decay is one
    scalar per head, so P_u is constant within a head and only E_u ranks channels; for KDA and RWKV the decay is per key
    channel and P_u differs between channels.
    """
    needs_calibration = True

    def __init__(self, lo: Format, hi: Format, k: int = 16, persistence_period: int = 1, tau: float = 1e-4,
                 cal_format: Format | None = None, name: str | None = None):
        self.lo, self.hi, self.k, self.pp, self.tau = lo, hi, k, persistence_period, tau
        self.stats = _Stats(cal_format or lo)
        self.masks = {}
        self.name = name or f'{k} key channels/head {hi!r}, rest {lo!r}'

    def observe(self, layer, S, log_decay, kw=None):
        self.stats.observe(layer, S, log_decay)

    def finalize(self):
        for layer in self.stats.err:
            e = self.stats.energy(layer)                                      # [H, d_k]
            a = self.stats.decay(layer).expand_as(e)
            score = e / (1. - a.pow(2 * self.pp)).clamp_min(self.tau)
            idx = score.topk(self.k, dim=-1).indices
            self.masks[layer] = torch.zeros_like(score, dtype=torch.bool).scatter(-1, idx, True).cpu()

    def quantize(self, S, layer):
        m = self.masks.get(layer)
        if m is None:
            raise RuntimeError('ChannelProtect is not calibrated: run statequant.calibrate(...) first')
        return torch.where(m.to(S.device)[None, :, :, None], self.hi.fake(S), self.lo.fake(S))

    def bits_per_value(self, d_k, d_v):
        f = self.k / d_k
        return f * self.hi.bits_per_value(d_k, d_v) + (1 - f) * self.lo.bits_per_value(d_k, d_v)


class DSQ(StateQuantizer):
    """Q-Mamba's decoupled scale quantization (Chen et al., Findings of ACL 2025), ported to a gated-delta state:
    scale = s_state (x) s_channel with s_channel = sqrt(mean_k |S|) per value coordinate and s_state = max_v |S| / s_channel
    per key coordinate, recomputed at every store, symmetric with qmax = 2^(b-1) - 1. Without Q-Mamba's fine-tuning."""

    def __init__(self, bits: int = 4, scale_bits: int = 16):
        self.bits, self.scale_bits = bits, scale_bits
        self.name = f'DSQ int{bits}'

    def quantize(self, S, layer):
        qmax = 2 ** (self.bits - 1) - 1
        x = S.float()
        a = x.abs()
        s_channel = a.mean(dim=-2).sqrt().clamp_min(1e-12)                    # [.., d_v]
        s_state = (a / s_channel.unsqueeze(-2)).amax(dim=-1).clamp_min(1e-12)  # [.., d_k]
        scale = s_state.unsqueeze(-1) * s_channel.unsqueeze(-2)
        q = torch.round(x / scale.clamp_min(1e-12) * qmax).clamp(-qmax, qmax)
        return (q / qmax * scale).to(S.dtype)

    def bits_per_value(self, d_k, d_v):
        return self.bits + self.scale_bits * (d_k + d_v) / (d_k * d_v)


def as_method(q) -> StateQuantizer:
    if isinstance(q, StateQuantizer):
        return q
    if isinstance(q, Format):
        return Uniform(q)
    if q is None:
        return Uniform(Identity())
    raise TypeError(f'not a state quantizer or format: {q!r}')
