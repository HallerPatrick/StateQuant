"""Packed recurrent state for KDA (Kimi Delta Attention) layers: torch reference, lockstep Triton state, slot layout.

KDA is gated delta with a decay per KEY channel: per head, one token, with a_t = exp(g_t) in R^{d_k},

    S_t = (I - beta_t k_t k_t^T) diag(a_t) S_{t-1} + beta_t k_t v_t^T          S: [d_k, d_v] per head
    o_t = S_t^T q_t                       (q, k l2-normalised as fla's kernels do, q scaled by d_k^-1/2)

i.e. with S' = diag(a_t) S_{t-1}:  u_t = beta_t (v_t - S'^T k_t),  S_t = S' + k_t u_t^T,  o_t = S'^T q_t + (k_t.q_t) u_t.

Packed decode (``fold.py``'s delta-value buffer, with the scalar decays replaced by vectors). The stored state S_0 is
rounded and only rewritten every ``m`` tokens; in between the tokens sit in a buffer of (cumulative log decay
G_i in R^{d_k}, normalised key k_i, delta value u_i), and with alpha_i = exp(G_i) (per key channel)

    S_{n-1}^T (a_n * x) = S_0^T (alpha_n * x) + sum_{i<n} [(alpha_n / alpha_i) * k_i] . x  u_i     (x = q or k)
    fold    S = diag(alpha_m) S_0 + sum_i diag(alpha_m / alpha_i) k_i u_i^T,  re-rounded          (every m tokens)

(``*`` elementwise), so a step reads the stored state once (against alpha_n * q and alpha_n * k) and the buffer once;
the decay ratio of each buffered key is formed per channel in log space, exp(G_n - G_i) <= 1. Unlike GDN the buffer
keeps the cumulative decay per channel: G is [m, d_k] per head (fp32), not [m].

Layouts: states are value-major, [B, H, d_v, d_k] (vLLM's KDA recurrent state; fla's chunk_kda takes the transpose
[B, H, d_k, d_v]). Storage formats are ``fold.pack_vk``'s: int4 / int8 with fp16 scales along the key axis ('k':
``group`` consecutive d_k entries of one value row) or the value axis ('v'), or bf16. d_k, d_v powers of two >= 16;
key heads = value heads (no KDA model has GQA).

``ReferenceKDAState``  the packed arithmetic in torch (and, with ``fmt=Identity()`` and ``buffer_bits=32``, the exact
                       fp32 recurrence through the same buffer algebra); for tests
``kda_recurrence``     the plain fp32 KDA step (no packing at all)
``PackedKDAState``     one launch per step (``_kda_triton._kda_step``), sequences in lockstep; CUDA-graph safe
``KDASlotLayout``      the serving layout (``slots.SlotLayout``'s per-slot blob with G per channel, read / write
                       around prefill) with the KDA step on a batch of slots (slot <= 0 = padding)
"""
from __future__ import annotations

import math

import torch

from ..quantizer import Identity
from .fold import _bits, _nrm, _round_buf, pack_vk, packed_format, unpack_vk
from .slots import _ALIGN, SlotLayout


def kda_recurrence(S, q, k, v, g, beta, scale=None):
    """One plain fp32 KDA step on S [BH, d_v, d_k] (value-major, updated in place); q, k raw [BH, d_k] (normalised
    here), v [BH, d_v], g [BH, d_k] log decay, beta [BH]. Returns o [BH, d_v] fp32."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    qn, kn = _nrm(q) * scale, _nrm(k)
    S.mul_(torch.exp(g.float())[:, None, :])
    u = beta.float()[:, None] * (v.float() - torch.einsum('bvk,bk->bv', S, kn))
    S.add_(u[:, :, None] * kn[:, None, :])
    return torch.einsum('bvk,bk->bv', S, qn)


def _check(fmt, dk, dv, hv, hk, m, buffer_bits):
    assert dk >= 16 and dk & (dk - 1) == 0 and dv >= 16 and dv & (dv - 1) == 0, (dk, dv)
    assert hk == hv, 'KDA: key heads must equal value heads'
    assert m >= 1 and m & (m - 1) == 0 and m <= 32, 'm must be a power of two <= 32'
    assert buffer_bits in (8, 16), buffer_bits
    if fmt is not None:
        g = fmt.group
        assert g >= 2 and g & (g - 1) == 0, f'group {g} must be a power of two'
        assert (dk if fmt.axis == 'k' else dv) % g == 0, f'group {g} must divide d_{fmt.axis} ({dk}, {dv})'
        assert fmt.axis == 'v' or dk // g <= 64, 'at most 64 key groups'


def _kda_config(st, sbv=None, fbv=None, warps=None):
    """(state block rows, fold block rows, warps) of the KDA step: the state pass reads 64 value rows per block (two
    [64, d_k] tiles live); the fold runs 16 rows at a time (bf16: 32), value-axis formats one scale group."""
    DV = st.DV
    sbv = sbv or min(DV, 64)
    if fbv is None:
        if st.fmt is not None and not st.axk:
            fbv = st.group
        else:
            fbv = min(DV, 32 if st.fmt is None else 16)
    assert DV % sbv == 0 and DV // sbv in (1, 2, 4, 8) and DV % fbv == 0, (DV, sbv, fbv)
    return sbv, fbv, warps or 4


def _common(st, fmt, hv, hk, dk, dv, m, buffer_bits, sbv, fbv, warps):
    st.fmt = packed_format(fmt)
    st.bits = _bits(st.fmt)
    st.HV, st.HK, st.DK, st.DV, st.m = hv, hk, dk, dv, m
    _check(st.fmt, dk, dv, hv, hk, m, buffer_bits)
    st.buffer_bits, st.qbuf = buffer_bits, buffer_bits == 8
    st.axk = st.fmt is not None and st.fmt.axis == 'k'
    st.group = st.fmt.group if st.fmt is not None else 16
    st.sbv, st.fbv, st.warps = _kda_config(st, sbv, fbv, warps)


# ---------------------------------------------------------------------------------------------------------------
# torch reference
# ---------------------------------------------------------------------------------------------------------------
class ReferenceKDAState:
    """The packed KDA decode arithmetic in torch (dequantize -> step -> buffer -> fold every m tokens).

    S            [B, H, d_v, d_k] or [B*H, d_v, d_k] float state (value-major)
    fmt          as ``fold.packed_format`` (Quantizer int4 / int8, axis 'k' / 'v'; 'bf16'), or ``Identity()``: the
                 state is kept in fp32 and never rounded
    m, hv        buffer length; heads per sequence
    buffer_bits  8 (int8 + fp16 absmax scale per vector), 16 (bf16) or 32 (fp32, no rounding)
    auto_fold    False: a full buffer stays until ``fold()`` (tests inspect / adopt the kernel's entries first)

    ``step(q, k, v, g, beta, scale)``: q, k raw [BH, d_k], v [BH, d_v], g [BH, d_k] (log decay), beta [BH] (after
    sigmoid) -> o [BH, d_v] fp32.
    """

    def __init__(self, S, fmt, m=8, hv=None, buffer_bits=8, auto_fold=True):
        self.ident = isinstance(fmt, Identity)
        self.fmt = None if self.ident else packed_format(fmt)
        S = S.float().reshape(-1, S.shape[-2], S.shape[-1])
        self.BH, self.DV, self.DK = S.shape
        self.m, self.buffer_bits, self.auto_fold = m, buffer_bits, auto_fold
        self.HV = hv or self.BH
        self.B = self.BH // self.HV
        self.load(S)

    def _store(self, S):
        if self.ident:
            self.codes, self.scales = S.clone(), None
            return S.clone()
        self.codes, self.scales = pack_vk(S, self.fmt)
        return unpack_vk(self.codes, self.scales, self.fmt, self.DK)

    def load(self, S):
        S = S.float().reshape(self.BH, self.DV, self.DK)
        self.S0 = self._store(S)
        dev = S.device
        self.K = torch.zeros(self.BH, self.m, self.DK, device=dev)
        self.U = torch.zeros(self.BH, self.m, self.DV, device=dev)
        self.G = torch.zeros(self.BH, self.m, self.DK, device=dev)
        self.gcur = torch.zeros(self.BH, self.DK, device=dev)
        self.n = 0

    def dequantize(self):
        """The stored state S_0 [BH, d_v, d_k] (without the buffered tokens)."""
        return self.S0.clone()

    def state(self):
        """The current state (S_0 advanced across the pending buffer), [BH, d_v, d_k]."""
        S = torch.exp(self.gcur)[:, None, :] * self.S0
        n = self.n
        if n:
            dec = torch.exp(self.gcur[:, None, :] - self.G[:, :n])
            S = S + torch.einsum('biv,bik->bvk', self.U[:, :n], self.K[:, :n] * dec)
        return S

    def _round(self, x):
        return x.float() if self.buffer_bits == 32 else _round_buf(x.float(), self.buffer_bits)

    @torch.no_grad()
    def step(self, q, k, v, g, beta, scale=None):
        BH, n = self.BH, self.n
        scale = self.DK ** -0.5 if scale is None else scale
        qn = _nrm(q.reshape(BH, self.DK)) * scale
        kn = _nrm(k.reshape(BH, self.DK))
        gnew = self.gcur + g.float().reshape(BH, self.DK)
        A = torch.exp(gnew)
        sq = torch.einsum('bvk,bk->bv', self.S0, A * qn)
        sk = torch.einsum('bvk,bk->bv', self.S0, A * kn)
        if n:
            Kd = self.K[:, :n] * torch.exp(gnew[:, None, :] - self.G[:, :n])
            sq = sq + torch.einsum('bi,biv->bv', torch.einsum('bik,bk->bi', Kd, qn), self.U[:, :n])
            sk = sk + torch.einsum('bi,biv->bv', torch.einsum('bik,bk->bi', Kd, kn), self.U[:, :n])
        u = beta.float().reshape(BH, 1) * (v.float().reshape(BH, self.DV) - sk)
        o = sq + (kn * qn).sum(-1, keepdim=True) * u
        self.U[:, n] = self._round(u)
        self.K[:, n] = self._round(kn)
        self.gcur = gnew
        self.G[:, n] = gnew
        self.n += 1
        if self.n == self.m and self.auto_fold:
            self.fold()
        return o

    def fold(self):
        assert self.n == self.m
        S = torch.exp(self.gcur)[:, None, :] * self.S0
        Kd = self.K * torch.exp(self.gcur[:, None, :] - self.G)
        for i in range(self.m):                               # the kernel's accumulation order
            S = S + self.U[:, i, :, None] * Kd[:, i, None, :]
        self.S_raw = S
        self.S0 = self._store(S)
        self.gcur = torch.zeros_like(self.gcur)
        self.n = 0


# ---------------------------------------------------------------------------------------------------------------
# the kernel call
# ---------------------------------------------------------------------------------------------------------------
def _launch(st, rows, q, k, v, g, beta, out, scale, idx, T, strides, nptr_stride):
    from . import _kda_triton as K
    for t in (q, k, v, g):
        assert t.stride(-1) == 1, 'unit inner stride'
    assert q.stride(-2) == st.DK and k.stride(-2) == st.DK and v.stride(-2) == st.DV and g.stride(-2) == st.DK
    assert beta.stride(-1) == 1 and out.stride(-1) == 1 and out.stride(-2) == st.DV
    sS, sSC, sKB, sKBS, sUB, sUBS, sGB, sGC = strides
    K._kda_step[(rows * st.HV,)](
        q, q.stride(0), k, k.stride(0), v, v.stride(0), g, g.stride(0), beta, beta.stride(0), idx,
        T['S'], sS, T['SC'], sSC, T['KB'], sKB, T['KBS'], sKBS, T['UB'], sUB, T['UBS'], sUBS,
        T['GB'], sGB, T['GCUR'], sGC, T['NPTR'], nptr_stride, out, out.stride(0), float(scale),
        DK=st.DK, DV=st.DV, SBV=st.sbv, FBV=st.fbv, M=st.m, BITS=st.bits, AXK=st.axk, GROUP=st.group,
        QBUF=st.qbuf, HV=st.HV, HAS_IDX=idx is not None, num_warps=st.warps)
    return out


class PackedKDAState:
    """One KDA layer's packed state and decode buffer for B sequences in lockstep, stepped by one Triton launch.

    S [B, H, d_v, d_k] (or [B*H, d_v, d_k]); fmt, m, hv (= H), buffer_bits as ``ReferenceKDAState`` (no Identity /
    fp32 buffer). ``step(q, k, v, g, beta, scale)`` (q, k raw [B, H, d_k] or [BH, d_k]; v [.., d_v]; g [.., d_k]
    log decay; beta [B, H]) returns the output buffer [B, H, d_v] (the same tensor every step). All tensors are
    allocated once, so ``step`` can be captured in a CUDA graph."""

    def __init__(self, S, fmt, m=8, hv=None, buffer_bits=8, out_dtype=torch.bfloat16, sbv=None, fbv=None,
                 warps=None):
        S = S.float().reshape(-1, S.shape[-2], S.shape[-1])
        BH, DV, DK = S.shape
        hv = hv or BH
        assert BH % hv == 0
        _common(self, fmt, hv, hv, DK, DV, m, buffer_bits, sbv, fbv, warps)
        self.BH, self.B = BH, BH // hv
        dev = S.device
        c, s = pack_vk(S, self.fmt)
        bdt = torch.int8 if self.qbuf else torch.bfloat16
        self.T = dict(S=c.clone(), SC=s.clone(),
                      KB=torch.zeros(BH, m, DK, device=dev, dtype=bdt),
                      UB=torch.zeros(BH, m, DV, device=dev, dtype=bdt),
                      KBS=torch.zeros(BH, m, device=dev, dtype=torch.float16),
                      UBS=torch.zeros(BH, m, device=dev, dtype=torch.float16),
                      GB=torch.zeros(BH, m, DK, device=dev), GCUR=torch.zeros(BH, DK, device=dev),
                      NPTR=torch.zeros(self.B, hv, device=dev, dtype=torch.int32))
        self.O = torch.zeros(self.B, hv, DV, device=dev, dtype=out_dtype)

    def __getattr__(self, name):
        T = self.__dict__.get('T')
        if T is not None and name in T:
            return T[name]
        raise AttributeError(name)

    @property
    def codes(self):
        return self.T['S']

    @property
    def scales(self):
        return self.T['SC']

    @property
    def n(self):
        return int(self.T['NPTR'].reshape(-1)[0])

    @torch.no_grad()
    def load(self, S):
        c, s = pack_vk(S.float().reshape(self.BH, self.DV, self.DK), self.fmt)
        self.T['S'].copy_(c)
        self.T['SC'].copy_(s)
        for k in ('KB', 'UB', 'KBS', 'UBS', 'GB', 'GCUR', 'NPTR'):
            self.T[k].zero_()

    def dequantize(self):
        return unpack_vk(self.T['S'], self.T['SC'], self.fmt, self.DK)

    def stored_bytes(self):
        nb = lambda t: t.numel() * t.element_size()          # noqa: E731
        T = self.T
        per = dict(state=nb(T['S']) + (nb(T['SC']) if self.fmt is not None else 0),
                   buffer=nb(T['KB']) + nb(T['UB']) + ((nb(T['KBS']) + nb(T['UBS'])) if self.qbuf else 0),
                   decay=nb(T['GB']) + nb(T['GCUR']) + nb(T['NPTR']))
        per['total'] = sum(per.values())
        return per

    def step(self, q, k, v, g, beta, scale=None):
        Bn, HV, m, DK, DV = self.B, self.HV, self.m, self.DK, self.DV
        q, k = q.reshape(Bn, HV, DK), k.reshape(Bn, HV, DK)
        v, g, beta = v.reshape(Bn, HV, DV), g.reshape(Bn, HV, DK), beta.reshape(Bn, HV)
        T = self.T
        strides = (HV * T['S'][0].numel(), HV * T['SC'][0].numel() if self.fmt is not None else 0,
                   HV * m * DK, HV * m, HV * m * DV, HV * m, HV * m * DK, HV * DK)
        return _launch(self, Bn, q, k, v, g, beta, self.O, DK ** -0.5 if scale is None else scale, None, T,
                       strides, HV)


# ---------------------------------------------------------------------------------------------------------------
# slots
# ---------------------------------------------------------------------------------------------------------------
class KDASlotLayout(SlotLayout):
    """The per-slot blob of one KDA layer and the KDA decode step on a batch of slots.

    Fields (16-byte aligned, in this order): GB fp32 [H, m, d_k] (cumulative log decay per channel of each buffered
    token), GCUR fp32 [H, d_k] (running log decay since the last fold), NPTR int32 [max(H, 4)] (fill counter, one
    copy per head), SC fp16 scales (as ``SlotLayout``), KBS / UBS fp16 [H, m] (int8 buffer only), S codes
    [H, d_v, d_k(/2)], KB [H, m, d_k] (normalised keys), UB [H, m, d_v] (delta values). ``views`` / ``alloc`` as
    ``SlotLayout``.

    fmt, hv (heads on this rank), dk, dv, m, buffer_bits as ``PackedKDAState``."""

    def __init__(self, fmt, hv, dk, dv, m=8, buffer_bits=8, sbv=None, fbv=None, warps=None):
        _common(self, fmt, hv, hv, dk, dv, m, buffer_bits, sbv, fbv, warps)
        self.gqa = 1
        self.fused = True
        bdt = torch.int8 if self.qbuf else torch.bfloat16
        if self.fmt is None:
            sdt, sshape, scshape = torch.bfloat16, (hv, dv, dk), None
        else:
            sdt = torch.uint8 if self.bits == 4 else torch.int8
            sshape = (hv, dv, dk // 2) if self.bits == 4 else (hv, dv, dk)
            g = self.group
            scshape = (hv, dv, dk // g) if self.axk else (hv, dv // g, dk)
        fields = [('GB', torch.float32, (hv, m, dk)), ('GCUR', torch.float32, (hv, dk)),
                  ('NPTR', torch.int32, (max(hv, 4),))]
        if scshape is not None:
            fields.append(('SC', torch.float16, scshape))
        if self.qbuf:
            fields += [('KBS', torch.float16, (hv, m)), ('UBS', torch.float16, (hv, m))]
        fields += [('S', sdt, sshape), ('KB', bdt, (hv, m, dk)), ('UB', bdt, (hv, m, dv))]
        self.fields, off = {}, 0
        for name, dt, shape in fields:
            nb = math.prod(shape) * torch.empty((), dtype=dt).element_size()
            self.fields[name] = (off, dt, shape, nb)
            off += -(-nb // _ALIGN) * _ALIGN
        self.nbytes = off

    def __repr__(self):
        f = 'bf16' if self.fmt is None else f'int{self.bits}-{self.fmt.axis}-g{self.group}'
        return (f'KDASlotLayout({f}, heads={self.HV}, d_k={self.DK}, d_v={self.DV}, m={self.m}, '
                f'buffer_bits={self.buffer_bits}, {self.nbytes} B/slot)')

    def step(self, views, idx, q, k, v, g, beta, out, scale=None):
        """One decode token for every batch row b on slot ``idx[b]`` (int32 [rows]; <= 0 = padding: output zeroed,
        nothing read or written).

        q, k [rows, H, d_k] raw (l2-normalised in the kernel), v [rows, H, d_v], g [rows, H, d_k] log decay (vLLM's
        ``fused_kda_gate`` output), beta [rows, H] (after sigmoid): any row stride, contiguous rows; out
        [rows, H, d_v] (row stride free, rows contiguous). Capture-safe (one launch, no host syncs)."""
        rows = idx.shape[0]
        if rows == 0:
            return out
        assert idx.dtype == torch.int32 and idx.is_contiguous()
        V = views
        T = dict(S=V['S'], SC=V.get('SC', V['GB']), KB=V['KB'], KBS=V.get('KBS', V['GB']), UB=V['UB'],
                 UBS=V.get('UBS', V['GB']), GB=V['GB'], GCUR=V['GCUR'], NPTR=V['NPTR'])
        st = lambda n: V[n].stride(0) if n in V else 0     # noqa: E731
        strides = (st('S'), st('SC'), st('KB'), st('KBS'), st('UB'), st('UBS'), st('GB'), st('GCUR'))
        return _launch(self, rows, q, k, v, g, beta, out, self.DK ** -0.5 if scale is None else scale, idx, T,
                       strides, st('NPTR'))

    # -- prefill side (torch) ---------------------------------------------------------------------------------
    @torch.no_grad()
    def read(self, views, idx):
        """The current state of slots ``idx`` as fp32 [P, H, d_v, d_k]: S_0 advanced across its pending buffer,
        diag(alpha_n) S_0 + sum_{i<n} diag(alpha_n / alpha_i) k_i u_i^T (value-major). Garbage slots give garbage
        (mask the result with ``torch.where``)."""
        idx = idx.long()
        P, HV, DK, DV, m = idx.shape[0], self.HV, self.DK, self.DV, self.m
        V = views
        codes = V['S'][idx].reshape(P * HV, *V['S'].shape[2:])
        sc = V['SC'][idx].reshape(P * HV, *V['SC'].shape[2:]) if 'SC' in V else None
        S0 = unpack_vk(codes, sc, self.fmt, DK).view(P, HV, DV, DK)
        Kb = V['KB'][idx].float()                                             # [P, H, m, DK]
        Ub = V['UB'][idx].float()                                             # [P, H, m, DV]
        if self.qbuf:
            Kb = Kb * V['KBS'][idx].float()[..., None]
            Ub = Ub * V['UBS'][idx].float()[..., None]
        n = V['NPTR'][idx, 0] & (m - 1)                                       # [P]
        gcur = V['GCUR'][idx]                                                 # [P, H, DK]
        live = (torch.arange(m, device=idx.device)[None, :] < n[:, None])[:, None, :, None]   # [P, 1, m, 1]
        dec = torch.where(live, torch.exp(gcur[:, :, None, :] - torch.where(live, V['GB'][idx], 0.)), 0.)
        Ub = torch.where(live, Ub, 0.)
        Kd = torch.where(live, Kb, 0.) * dec
        return torch.exp(gcur)[:, :, None, :] * S0 + torch.einsum('phmv,phmk->phvk', Ub, Kd)

    @torch.no_grad()
    def write(self, views, idx, S):
        """Store fp32/bf16 states S [P, H, d_v, d_k] (value-major) packed into slots ``idx`` and empty their buffers
        (n = 0, running decay 0)."""
        idx = idx.long()
        P = idx.shape[0]
        codes, sc = pack_vk(S.reshape(P * self.HV, self.DV, self.DK), self.fmt)
        V = views
        V['S'][idx] = codes.view(P, *V['S'].shape[1:])
        if 'SC' in V:
            V['SC'][idx] = sc.view(P, *V['SC'].shape[1:])
        V['NPTR'][idx] = 0
        V['GCUR'][idx] = 0.


__all__ = ['ReferenceKDAState', 'PackedKDAState', 'KDASlotLayout', 'kda_recurrence']
