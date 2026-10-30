"""LeapQuant (Pan, Xi, Zhu, Li, Wu, Mishra, Zhang, Zheng, Kasikci, Han, Keutzer, Iyer, Stoica; arXiv 2609.38166).

Per-window quantization (Sec. 3.2, Eq. 3-4): the boundary state S^_0 is held fixed for a window of p tokens; the p
updates (decay a_i, key k_i, correction u_i) are buffered in fp16 and every output inside the window reads
Gamma S^_0 + sum_j Gamma_{j+1:l} k_j u_j^T; only at the window end is S_p materialised and quantized again.

Compensator tokens (Sec. 3.3, Eq. 5-6): before quantizing S_p, r fp16 rank-one terms K~ U~^T (fitted by power
iteration, the best rank-r approximation of S_p) are subtracted; they start the next window as records with identity
decay placed before the real tokens (so they are decayed with the window, Eq. 7).

Residual smoothing (Sec. 3.4, Eq. 7, App. B.3): c_i = sqrt(mean_j |R_ij|) per key row (fp32, small floor), the
residual C^-1 R is quantized symmetric per value channel (``bits`` = 8, also 6 and 4; Z / (2^(b-1)-1) * B with an
fp32 scale per value column), dequantized and multiplied by C.

Storage per boundary state: b + 32/d_k (value-channel scales) + 32/d_v (smoothing scales) + 16 r (d_k + d_v)/(d_k d_v)
bits per element; at 8 bits, r = 4, d = 128: 9.5 bits = 1.1875 B ("1.19 bytes per element", Sec. 4.3). The window
buffer (p records of 1 + d_k + d_v fp16 values) is reported separately (``buffer_bits_per_value``).
"""
from __future__ import annotations

import torch

from ..quantizer import Quantizer
from ..schedule import Schedule
from .hooks import StatefulMethod, decay_shape, terms, written_value


def compensators(S, r, iters=4, init=None, generator=None):
    """Block power (subspace) iteration for the top-r right singular subspace of S [..., d_k, d_v]:
    V <- orth(S^T S V), ``iters`` times from ``init`` (previous window's U~, warm start) or a seeded Gaussian.
    Returns fp16 (K~ = S V [.., d_k, r], U~ = V [.., d_v, r])."""
    if r == 0:
        return S.new_zeros(*S.shape[:-1], 0).half(), S.new_zeros(*S.shape[:-2], S.shape[-1], 0).half()
    if init is None or init.shape[-1] != r:
        init = torch.randn(*S.shape[:-2], S.shape[-1], r, device=S.device, generator=generator)
    V = torch.linalg.qr(init.float()).Q
    for _ in range(iters):
        V = torch.linalg.qr(S.transpose(-1, -2) @ (S @ V)).Q
    return (S @ V).half(), V.half()


class LeapQuant(StatefulMethod):
    """bits        residual width: 8 (the paper's main setting), 6 or 4
    r            compensator tokens (default 4; 8 at 4 bits)
    p            window length (16)
    records      'fp16': the window's records (log decay, key, correction) are fp16 as in the paper's buffer;
                 needs one-token decode pieces (schedule decode=1; the method closes windows itself every p tokens).
                 'fp32': the window runs on p-token pieces and S_p is the op's exact fp32 state (faster).
    prefill      prompt chunk: the prompt runs in full precision and its state is quantized (with compensators) at
                 the end of every chunk and of the prompt, as a serving engine stores it (the paper's windows are
                 decode windows; it does not describe prefill)
    power_iters  power-iteration steps per boundary (warm-started from the previous window's compensators)
    floor        smoothing-scale floor
    """

    def __init__(self, bits: int = 8, r: int | None = None, p: int = 16, records: str = 'fp16', prefill: int = 8192,
                 power_iters: int = 4, floor: float = 1e-6, seed: int = 0, name: str | None = None):
        assert (bits is None or bits in range(2, 16)) and records in ('fp16', 'fp32')   # None: residual unquantized
        self.bits, self.r, self.p = bits, (8 if bits == 4 else 4) if r is None else r, p
        self._identity = bits is None
        self.records, self.power_iters, self.floor, self.seed = records, power_iters, floor, seed
        self.layers, self._gen, self.channel_decay = {}, {}, False
        self.name = name or f'LeapQuant int{bits} r={self.r} p={p} records {records}'
        self.schedule = Schedule(prefill=prefill, decode=1 if records == 'fp16' else p, buffer_bits=0,
                                 prompt_end=True)

    def reset(self):
        self.layers, self._gen = {}, {}

    def _init(self, ctx):
        return {'S0': torch.zeros_like(ctx.S_out), 'acc': None, 'W': torch.zeros_like(ctx.S_out), 'U': None}

    def step(self, ctx):
        st = self._layer_state(ctx, self._init)
        lg, key, _, v, rule = terms(ctx.pc)
        self.channel_decay = self.channel_decay or ctx.family.decay == 'channel'
        S_out = ctx.S_out
        if self.records == 'fp16' and lg.shape[1] == 1:
            g16 = lg[:, 0].half().float()                              # fp16 log-decay record
            a16 = decay_shape(g16, S_out).exp()
            k16 = key[:, 0].half()
            u16 = written_value(S_out, ctx.S_in, lg, key, v, rule).half()   # the op's own update k u^T
            st['W'] = a16 * st['W'] + k16.float()[..., :, None] * u16.float()[..., None, :]
            st['acc'] = a16 if st['acc'] is None else st['acc'] * a16
        else:                                                           # exact window (or a prompt piece)
            a = decay_shape(lg.sum(1), S_out).exp()
            st['acc'] = a if st['acc'] is None else st['acc'] * a
            st['W'] = S_out - st['acc'] * st['S0']
        boundary = ((ctx.pos % self.p == 0) & (ctx.pos > 0)) | torch.full_like(ctx.hit, ctx.prompt_end)
        if lg.shape[1] > 1:
            boundary = boundary & ctx.hit                              # prompt pieces: chunk / prompt ends only
        Sp = st['acc'] * st['S0'] + st['W']
        if not bool(boundary.any()):
            return Sp
        assert bool(boundary.all()), 'LeapQuant: all sequences of a batch must be at the same position'
        return self._boundary(st, Sp)

    def residual_format(self, d_k):
        if self._identity:
            from ..quantizer import Identity
            return Identity()
        return Quantizer(self.bits, 'k', d_k)                          # symmetric, one fp32 scale per value column

    def quantize_state(self, Sp, init=None):
        """One window boundary: compensators, smoothing, residual quantization. Returns (S~_p, U~)."""
        g = self._gen.get(Sp.device)
        if g is None:
            g = self._gen[Sp.device] = torch.Generator(device=Sp.device).manual_seed(self.seed)
        K, U = compensators(Sp, self.r, self.power_iters, init, g)
        comp = K.float() @ U.float().transpose(-1, -2)
        R = Sp - comp
        c = R.abs().mean(-1, keepdim=True).sqrt().clamp_min(self.floor)   # [.., d_k, 1]
        Rq = self.residual_format(Sp.shape[-2]).fake(R / c) * c
        return Rq + comp, U

    def _boundary(self, st, Sp):
        S0, U = self.quantize_state(Sp, st['U'])
        st.update(S0=S0, U=U, acc=torch.ones_like(st['acc']), W=torch.zeros_like(Sp))
        return S0

    def bits_per_value(self, d_k, d_v):
        return (self.bits or 32) + 32 / d_k + 32 / d_v + 16 * self.r * (d_k + d_v) / (d_k * d_v)

    def buffer_bits_per_value(self, d_k, d_v, heads_per_key=1):
        """A full window buffer: p records of (decay, key, correction) in fp16 (Table 4 of the paper: 1 + d_k + d_v
        values for a per-head decay, 2 d_k + d_v for a per-row decay; set once a per-row-decay family was seen)."""
        return 16 * self.p * ((d_k if self.channel_decay else 1) + d_k + d_v) / (d_k * d_v)
