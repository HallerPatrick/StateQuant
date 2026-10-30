"""Packed recurrent state for Mamba2 / SSD layers: torch reference, lockstep Triton state, slot-addressed layout.

Mamba2 (``mamba_ssm`` / vLLM ``MambaMixer2``), per head h of H, group g = h // (H / G), one token:

    dt  = softplus(dt_raw + dt_bias), clamped to dt_limit;   A_h = -exp(A_log_h) (vLLM keeps A itself)
    S   = exp(A_h dt) S + (dt x) B^T                          S: [head_dim, d_state] per head (vLLM's ssm state)
    y   = S C + D_h x

This is gated-delta's packed decode (``fold.py``) with key B (per group: the GQA of ``fold``), query C, buffered value
u = dt x (no delta term, no normalisation) and a scalar decay per head. The stored state S_0 is rounded and only
rewritten every ``m`` tokens; in between the tokens sit in a buffer (cumulative log decay G_i, B_i, u_i), rounded to
``buffer_bits`` (int8 with one fp16 absmax scale per vector, or bf16), and with alpha_i = exp(G_i)

    y_n  = alpha_n S_0 C + sum_{i<=n} (alpha_n / alpha_i) (B_i . C) u_i + D x        (token n itself unrounded)
    fold   S = alpha_m S_0 + sum_i (alpha_m / alpha_i) u_i B_i^T, re-rounded         (every m tokens)

Layouts: states are [B, H, head_dim, d_state] = [B, H, d_v, d_k] (value-major, as mamba_ssm / vLLM; note that
``PackedGDNState`` takes [BH, d_k, d_v]). Storage formats are ``fold.pack_vk``'s: int4 / int8 with fp16 scales along
the key axis ('k': ``group`` consecutive d_state entries of one head_dim row) or the value axis ('v'), or bf16.
Any head dims (d_k >= 16, even; powers of two are not required); a ``group`` must be a power of two dividing its axis.

``ReferenceMamba2State``  the packed arithmetic in torch (and, with ``fmt=Identity()`` and ``buffer_bits=32``, the
                          exact fp32 recurrence through the same buffer algebra); for tests
``mamba2_recurrence``     the plain fp32 Mamba2 step, S <- exp(A dt) S + dt x B^T (no packing at all)
``PackedMamba2State``     one launch per step (``_mamba2_triton._m2_step``), sequences in lockstep; CUDA-graph safe
``Mamba2SlotLayout``      the serving layout (``slots.SlotLayout``'s per-slot blob, read / write around prefill) with
                          the Mamba2 step on a batch of slots (slot <= 0 = padding)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..quantizer import Identity
from .fold import _bits, _round_buf, pack_vk, packed_format, unpack_vk
from .slots import _ALIGN, SlotLayout


def _pow2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


def _gate(dt, A, dt_bias=None, dt_softplus=False, dt_limit=None):
    """dt (raw [B, H]) -> (dt, log decay g = A dt), fp32, as the kernel."""
    dt = dt.float()
    if dt_bias is not None:
        dt = dt + dt_bias.float()
    if dt_softplus:
        dt = torch.where(dt <= 20., torch.log(1. + torch.exp(dt)), dt)
    if dt_limit is not None and tuple(dt_limit) != (0., float('inf')):
        dt = dt.clamp(dt_limit[0], dt_limit[1])
    return dt, A.float() * dt


def mamba2_recurrence(S, x, dt, B, C, A, D=None, dt_bias=None, dt_softplus=False, dt_limit=None):
    """One plain fp32 Mamba2 step on S [B, H, d_v, d_k] (updated in place); x [B, H, d_v], dt [B, H], B / C
    [B, G, d_k], A / D / dt_bias [H]. Returns y [B, H, d_v] fp32."""
    Bn, H = dt.shape
    G = B.shape[1]
    dt, g = _gate(dt, A, dt_bias, dt_softplus, dt_limit)
    Bh = B.float().repeat_interleave(H // G, 1)
    Ch = C.float().repeat_interleave(H // G, 1)
    u = dt[..., None] * x.float()
    S.mul_(torch.exp(g)[..., None, None]).add_(u[..., :, None] * Bh[..., None, :])
    y = torch.einsum('bhvk,bhk->bhv', S, Ch)
    if D is not None:
        y = y + D.float()[:, None] * x.float()
    return y


def _check_dims(fmt, dk, dv):
    assert dk >= 16 and dk % 2 == 0, dk
    if fmt is not None:
        g = fmt.group
        assert g & (g - 1) == 0, f'group {g} must be a power of two'
        assert (dk if fmt.axis == 'k' else dv) % g == 0, f'group {g} must divide d_{fmt.axis} ({dk}, {dv})'


def _block(dv, cap):
    """Largest power of two <= cap dividing dv (at least 16; rows beyond dv are masked)."""
    b = cap
    while b > 16 and dv % b:
        b //= 2
    return b


# Fewer warps per program pay off once there are enough programs to fill the GPU: below this many (rows x heads),
# 4 warps, above it 2 (benchmarks/decode_state.py --mamba2 on an A100: Nemotron-H-4B, Nemotron-3-Nano-4B,
# Granite-4.0-H-1B shapes; 2 warps are 15-45% faster at 128 rows, 4 warps a few percent faster at 1 row).
WARPS_SMALL_BH = 256


def _m2_config(st, sbv=None, fbv=None, warps=None):
    """(state block rows, fold block rows, warps or None = by batch) for a layout with DK, DV, bits, axk, group.
    From benchmarks/decode_state.py on an A100: the state pass reads 64 value rows per block, or the whole padded
    head for a head_dim that is not a power of two (80 -> one 128-row block beats 5 x 16); the fold runs 16 rows at
    a time (int4 key-axis: 64), value-axis formats one scale group."""
    DV = st.DV
    if sbv is None:
        sbv = _pow2(DV) if (DV & (DV - 1) and _pow2(DV) <= 128) else min(_pow2(DV), 64)
    if fbv is None:
        if st.fmt is not None and not st.axk:
            fbv = st.group
        elif st.bits == 4 and DV % 64 == 0:
            fbv = 64
        else:
            fbv = _block(DV, 16)
    return sbv, fbv, warps


# ---------------------------------------------------------------------------------------------------------------
# torch reference
# ---------------------------------------------------------------------------------------------------------------
class ReferenceMamba2State:
    """The packed Mamba2 decode arithmetic in torch (dequantize -> step -> buffer -> fold every m tokens).

    S            [B, H, d_v, d_k] or [B*H, d_v, d_k] float state (value-major)
    fmt          as ``fold.packed_format`` (Quantizer int4 / int8, axis 'k' / 'v'; 'bf16'), or ``Identity()``: the
                 state is kept in fp32 and never rounded
    m, hv, hk    buffer length; heads H and groups G per sequence (hv % hk == 0)
    buffer_bits  8 (int8 + fp16 absmax scale per vector), 16 (bf16) or 32 (fp32, no rounding)
    auto_fold    False: a full buffer stays until ``fold()`` (tests inspect / adopt the kernel's entries first)
    """

    def __init__(self, S, fmt, m=8, hv=None, hk=None, buffer_bits=8, auto_fold=True):
        self.ident = isinstance(fmt, Identity)
        self.fmt = None if self.ident else packed_format(fmt)
        S = S.float().reshape(-1, S.shape[-2], S.shape[-1])
        self.BH, self.DV, self.DK = S.shape
        self.m, self.buffer_bits, self.auto_fold = m, buffer_bits, auto_fold
        self.HV = hv or self.BH
        self.HK = hk or self.HV
        self.gqa, self.B = self.HV // self.HK, self.BH // self.HV
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
        self.K = torch.zeros(self.B * self.HK, self.m, self.DK, device=dev)
        self.U = torch.zeros(self.BH, self.m, self.DV, device=dev)
        self.G = torch.zeros(self.BH, self.m, device=dev)
        self.gcur = torch.zeros(self.BH, device=dev)
        self.n = 0

    def dequantize(self):
        """The stored state S_0 [BH, d_v, d_k] (without the buffered tokens)."""
        return self.S0.clone()

    def state(self):
        """The current state (S_0 advanced across the pending buffer), [BH, d_v, d_k]."""
        S = torch.exp(self.gcur)[:, None, None] * self.S0
        if self.n:
            Kb, Ub = self._kv()
            dec = torch.exp(self.gcur[:, None] - self.G[:, :self.n])
            S = S + torch.einsum('bi,biv,bik->bvk', dec, Ub, Kb)
        return S

    def _round(self, x):
        return x.float() if self.buffer_bits == 32 else _round_buf(x.float(), self.buffer_bits)

    def _kv(self):
        n = self.n
        K = self.K[:, :n].reshape(self.B, self.HK, 1, n, self.DK).expand(-1, -1, self.gqa, -1, -1)
        return K.reshape(self.BH, n, self.DK), self.U[:, :n]

    @torch.no_grad()
    def step(self, x, dt, B, C, A, D=None, dt_bias=None, dt_softplus=False, dt_limit=None):
        """x [B, H, d_v] (or [BH, d_v]), dt [B, H], B / C [B, G, d_k], A / D / dt_bias [H] -> y [BH, d_v] fp32."""
        BH, n = self.BH, self.n
        dt = dt.reshape(self.B, self.HV)
        dt, g = _gate(dt, A, dt_bias, dt_softplus, dt_limit)
        dt, g = dt.reshape(BH), g.reshape(BH)
        x = x.float().reshape(BH, self.DV)
        Bg = B.float().reshape(self.B * self.HK, self.DK)
        Bh = Bg.reshape(self.B, self.HK, 1, -1).expand(-1, -1, self.gqa, -1).reshape(BH, -1)
        Ch = C.float().reshape(self.B, self.HK, 1, -1).expand(-1, -1, self.gqa, -1).reshape(BH, -1)
        u = dt[:, None] * x
        alpha = torch.exp(self.gcur)
        sq = alpha[:, None] * torch.einsum('bvk,bk->bv', self.S0, Ch)
        if n:
            Kb, Ub = self._kv()
            dec = torch.exp(self.gcur[:, None] - self.G[:, :n])
            sq = sq + torch.einsum('bi,biv->bv', dec * torch.einsum('bik,bk->bi', Kb, Ch), Ub)
        a = torch.exp(g)[:, None]
        y = a * sq + (Bh * Ch).sum(-1, keepdim=True) * u
        if D is not None:
            y = y + D.float().repeat(self.B)[:, None] * x
        self.U[:, n] = self._round(u)
        self.K[:, n] = self._round(Bg)
        self.gcur = self.gcur + g
        self.G[:, n] = self.gcur
        self.n += 1
        if self.n == self.m and self.auto_fold:
            self.fold()
        return y

    def fold(self):
        assert self.n == self.m
        Kb, Ub = self._kv()
        S = torch.exp(self.gcur)[:, None, None] * self.S0
        dec = torch.exp(self.gcur[:, None] - self.G)
        for i in range(self.m):                               # the kernel's accumulation order
            S = S + (dec[:, i, None] * Ub[:, i])[:, :, None] * Kb[:, i, None, :]
        self.S_raw = S
        self.S0 = self._store(S)
        self.gcur = torch.zeros_like(self.gcur)
        self.n = 0


# ---------------------------------------------------------------------------------------------------------------
# the kernel call
# ---------------------------------------------------------------------------------------------------------------
def _launch(st, rows, x, dt, B, C, out, A, D, dt_bias, dt_softplus, dt_limit, idx, T, strides, nptr_stride):
    from . import _mamba2_triton as K
    assert x.stride(-1) == 1 and B.stride(-1) == 1 and C.stride(-1) == 1 and dt.stride(-1) == 1
    assert x.stride(-2) == st.DV and out.stride(-1) == 1 and out.stride(-2) == st.DV
    assert B.stride(-2) == st.DK and C.stride(-2) == st.DK
    assert A.is_contiguous() and (D is None or D.is_contiguous()) and (dt_bias is None or dt_bias.is_contiguous())
    clamp = dt_limit is not None and tuple(dt_limit) != (0., float('inf'))
    lo, hi = (float(dt_limit[0]), float(dt_limit[1])) if clamp else (0., 0.)
    sS, sSC, sKB, sKBS, sUB, sUBS, sGB, sGC = strides
    K._m2_step[(rows * st.HV,)](
        x, x.stride(0), dt, dt.stride(0), B, B.stride(0), C, C.stride(0), A,
        dt_bias if dt_bias is not None else A, D if D is not None else A, idx,
        T['S'], sS, T['SC'], sSC, T['KB'], sKB, T['KBS'], sKBS, T['UB'], sUB, T['UBS'], sUBS,
        T['GB'], sGB, T['GCUR'], sGC, T['NPTR'], nptr_stride, out, out.stride(0), lo, hi,
        DK=st.DK, DV=st.DV, DKP=_pow2(st.DK), DVP=_pow2(st.DV), SBV=st.sbv, FBV=st.fbv, M=st.m, BITS=st.bits,
        AXK=st.axk, GROUP=st.group, QBUF=st.qbuf, HV=st.HV, HK=st.HK, GQA=st.gqa, HAS_IDX=idx is not None,
        HAS_DT_BIAS=dt_bias is not None, DT_SOFTPLUS=bool(dt_softplus), DT_CLAMP=clamp, HAS_D=D is not None,
        num_warps=st.warps or (4 if rows * st.HV <= WARPS_SMALL_BH else 2))
    return out


def _common(st, fmt, hv, hk, dk, dv, m, buffer_bits, sbv, fbv, warps):
    st.fmt = packed_format(fmt)
    st.bits = _bits(st.fmt)
    st.HV, st.HK, st.DK, st.DV, st.m = hv, hk, dk, dv, m
    assert hv % hk == 0, (hv, hk)
    st.gqa = hv // hk
    assert m >= 1 and m & (m - 1) == 0 and m <= 32, 'm must be a power of two <= 32'
    _check_dims(st.fmt, dk, dv)
    assert buffer_bits in (8, 16)
    st.buffer_bits, st.qbuf = buffer_bits, buffer_bits == 8
    st.axk = st.fmt is not None and st.fmt.axis == 'k'
    st.group = st.fmt.group if st.fmt is not None else 16
    st.sbv, st.fbv, st.warps = _m2_config(st, sbv, fbv, warps)


class PackedMamba2State:
    """One Mamba2 layer's packed state and decode buffer for B sequences in lockstep, stepped by one Triton launch.

    S [B, H, d_v, d_k] (or [B*H, d_v, d_k]); fmt, m, hv (= H), hk (= G groups), buffer_bits as
    ``ReferenceMamba2State`` (no Identity / fp32 buffer). ``step(x, dt, B, C, A, D, dt_bias, dt_softplus,
    dt_limit)`` returns the output buffer [B, H, d_v] (the same tensor every step). All tensors are allocated once,
    so ``step`` can be captured in a CUDA graph."""

    def __init__(self, S, fmt, m=8, hv=None, hk=None, buffer_bits=8, out_dtype=torch.bfloat16, sbv=None, fbv=None,
                 warps=None):
        S = S.float().reshape(-1, S.shape[-2], S.shape[-1])
        BH, DV, DK = S.shape
        hv = hv or BH
        hk = hk or hv
        assert BH % hv == 0
        _common(self, fmt, hv, hk, DK, DV, m, buffer_bits, sbv, fbv, warps)
        self.BH, self.B = BH, BH // hv
        dev = S.device
        c, s = pack_vk(S, self.fmt)
        bdt = torch.int8 if self.qbuf else torch.bfloat16
        self.T = dict(S=c.clone(), SC=s.clone(),
                      KB=torch.zeros(self.B * hk, m, DK, device=dev, dtype=bdt),
                      UB=torch.zeros(BH, m, DV, device=dev, dtype=bdt),
                      KBS=torch.zeros(self.B * hk, m, device=dev, dtype=torch.float16),
                      UBS=torch.zeros(BH, m, device=dev, dtype=torch.float16),
                      GB=torch.zeros(BH, m, device=dev), GCUR=torch.zeros(BH, device=dev),
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

    def step(self, x, dt, B, C, A, D=None, dt_bias=None, dt_softplus=False, dt_limit=None):
        Bn, HV, HK, m = self.B, self.HV, self.HK, self.m
        x = x.reshape(Bn, HV, self.DV)
        dt = dt.reshape(Bn, HV)
        B = B.reshape(Bn, HK, self.DK)
        C = C.reshape(Bn, HK, self.DK)
        T = self.T
        strides = (HV * T['S'][0].numel(), HV * T['SC'][0].numel() if self.fmt is not None else 0,
                   HK * m * self.DK, HK * m, HV * m * self.DV, HV * m, HV * m, HV)
        return _launch(self, Bn, x, dt, B, C, self.O, A, D, dt_bias, dt_softplus, dt_limit, None, T, strides, HV)


# ---------------------------------------------------------------------------------------------------------------
# slots
# ---------------------------------------------------------------------------------------------------------------
class Mamba2SlotLayout(SlotLayout):
    """The per-slot blob of one Mamba2 layer (``slots.SlotLayout``'s fields: KB / KBS per group, UB / UBS, GB, GCUR
    per head; ``views``, ``alloc``, ``read``, ``write`` as there) and the Mamba2 decode step on a batch of slots.

    fmt, hv (heads on this rank), hk (groups on this rank), dk (d_state), dv (head_dim), m, buffer_bits as
    ``PackedMamba2State``; any even d_k >= 16 and any d_v."""

    def __init__(self, fmt, hv, hk, dk, dv, m=8, buffer_bits=8, sbv=None, fbv=None, warps=None):
        _common(self, fmt, hv, hk, dk, dv, m, buffer_bits, sbv, fbv, warps)
        self.fused = True
        bdt = torch.int8 if self.qbuf else torch.bfloat16
        if self.fmt is None:
            sdt, sshape, scshape = torch.bfloat16, (hv, dv, dk), None
        else:
            sdt = torch.uint8 if self.bits == 4 else torch.int8
            sshape = (hv, dv, dk // 2) if self.bits == 4 else (hv, dv, dk)
            g = self.group
            scshape = (hv, dv, dk // g) if self.axk else (hv, dv // g, dk)
        fields = [('GB', torch.float32, (hv, m)), ('GCUR', torch.float32, (hv,)), ('NPTR', torch.int32, (max(hv, 4),))]
        if scshape is not None:
            fields.append(('SC', torch.float16, scshape))
        if self.qbuf:
            fields += [('KBS', torch.float16, (hk, m)), ('UBS', torch.float16, (hv, m))]
        fields += [('S', sdt, sshape), ('KB', bdt, (hk, m, dk)), ('UB', bdt, (hv, m, dv))]
        self.fields, off = {}, 0
        for name, dt, shape in fields:
            nb = math.prod(shape) * torch.empty((), dtype=dt).element_size()
            self.fields[name] = (off, dt, shape, nb)
            off += -(-nb // _ALIGN) * _ALIGN
        self.nbytes = off

    def __repr__(self):
        f = 'bf16' if self.fmt is None else f'int{self.bits}-{self.fmt.axis}-g{self.group}'
        return (f'Mamba2SlotLayout({f}, heads={self.HV}, groups={self.HK}, d_state={self.DK}, head_dim={self.DV}, '
                f'm={self.m}, buffer_bits={self.buffer_bits}, {self.nbytes} B/slot)')

    def step(self, views, idx, x, dt, B, C, out, A, D=None, dt_bias=None, dt_softplus=True,
             dt_limit=(0., float('inf'))):
        """One decode token for every batch row b on slot ``idx[b]`` (int32 [rows]; <= 0 = padding: output zeroed,
        nothing read or written).

        x [rows, H, d_v], dt [rows, H] (raw: + dt_bias, softplus, clamp in the kernel, as vLLM's
        ``selective_state_update(..., dt_softplus=True)``), B / C [rows, G, d_k]: any row stride, contiguous rows;
        out [rows, H, d_v] (row stride free, rows contiguous); A [H] (= -exp(A_log)), D, dt_bias [H] or None.
        Capture-safe (one launch, no host syncs)."""
        rows = idx.shape[0]
        if rows == 0:
            return out
        assert idx.dtype == torch.int32 and idx.is_contiguous()
        V = views
        T = dict(S=V['S'], SC=V.get('SC', V['GB']), KB=V['KB'], KBS=V.get('KBS', V['GB']), UB=V['UB'],
                 UBS=V.get('UBS', V['GB']), GB=V['GB'], GCUR=V['GCUR'], NPTR=V['NPTR'])
        st = lambda n: V[n].stride(0) if n in V else 0     # noqa: E731
        strides = (st('S'), st('SC'), st('KB'), st('KBS'), st('UB'), st('UBS'), st('GB'), st('GCUR'))
        return _launch(self, rows, x, dt, B, C, out, A, D, dt_bias, dt_softplus, dt_limit, idx, T, strides,
                       st('NPTR'))
