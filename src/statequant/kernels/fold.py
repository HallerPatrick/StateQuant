"""Packed recurrent-state storage and the decode step with a delta-value buffer (PACE's fold path).

Gated-delta decode, S_t = a_t S_{t-1} + k_t u_t^T with u_t = beta_t (v_t - a_t S_{t-1}^T k_t), is run against a state
S_0 that is stored rounded and only rewritten every ``m`` tokens. In between, the tokens sit in a buffer of
(cumulative log decay G_i, normalised key k_i, delta value u_i), and with alpha_i = exp(G_i)

    S_{n}^T x = alpha_n (S_0^T x) + sum_i (alpha_n / alpha_i) (k_i . x) u_i          (any read-out x: q or k)

so a step reads the stored state once and the buffer once. When the buffer is full the state is folded,
S = alpha_m S_0 + sum_i (alpha_m / alpha_i) k_i u_i^T, and rounded again, on the device (PACE's ``foldk``).

``pack`` / ``unpack``      the storage format, torch only. ``unpack(pack(S, q)) == packed_format(q).fake(S)`` bit for bit.
``ReferenceGDNState``      the same decode arithmetic in torch (buffer rounding, fold every m), for tests.
``PackedGDNState``         the Triton path. All tensors are allocated once and written in place, so ``step`` can be
                           captured in a CUDA graph (the graph holds addresses; nothing here is ever rebound).

Layouts: states come in and go out as [BH, d_k, d_v] (statequant's [B, H, d_k, d_v] with B and H merged; a 4-d
state is accepted). Inside, the state is held transposed, [BH, d_v, d_k]; see ``_fold_triton``.
"""
from __future__ import annotations

import dataclasses

import torch

from ..quantizer import FloatCast, Quantizer

_FP16_MIN = 2. ** -24

# ``fused=None`` (auto): the single-launch fused step, except for value-axis formats above FUSED_MAX_BH_AXIS_V
# (batch rows x value heads), where the five-kernel step's finer-grained fold wins. From benchmarks/decode_state.py
# on an A100 (Qwen3.5-4B shapes, HV = 32; see the README): fused is faster at every batch 1 .. 512 for bf16 and
# key-axis int4 / int8; int4 axis 'v' g32 is faster fused up to B = 4 (BH = 128) and slower from B = 8.
FUSED_MAX_BH = None                 # key-axis / bf16 formats: no limit
FUSED_MAX_BH_AXIS_V = 128


def _fused_config(st, warps=None):
    """(state value block, fold value block, warps) of the fused step for a layout ``st`` (PackedGDNState /
    SlotLayout attributes), from benchmarks/decode_state.py on an A100 (d_k = d_v = 128): the state pass reads the
    whole head in one block (int8: two; fewer registers); the fold runs in blocks of 64 rows (bf16), 16 / 32 rows
    (key-axis int4 / int8: the re-rounding needs registers) or one scale group (value axis). 4 warps (8 are slower
    and can change Triton's reduction trees, i.e. lose bit-exactness with the five-kernel step)."""
    DV = st.DV
    p2 = DV & -DV                                     # largest power of two dividing d_v (192 -> 64)
    sbv = min(p2, 64) if st.bits == 8 else (DV if p2 == DV else min(p2, 64))
    if st.fmt is None:
        fbv = min(p2, 64)
    elif st.axk:
        fbv = min(p2, 16 if st.bits == 4 else 32)
    else:
        fbv = st.group
        sbv = max(sbv, st.group)
    assert DV % sbv == 0 and DV // sbv in (1, 2, 3, 4) and DV % fbv == 0, (DV, sbv, fbv)
    return sbv, fbv, warps or 4


def pow2(n):
    """The smallest power of two >= n (the padded tile width of a head dim)."""
    return 1 << max(0, (int(n) - 1).bit_length())


def check_dims(fmt, dk, dv):
    """Head dims and group a packed GDN store supports: d_k even and >= 16 (any, e.g. OLMo-Hybrid's 96: the kernels
    pad the key tile to the next power of two and mask it); d_v a multiple of 32; axis 'k': a power-of-two group
    with d_k padded holding at most 8 groups, the last group may be ragged (64 on d_k 96: 64 + 32); axis 'v': a
    power-of-two group dividing d_v."""
    assert dk >= 16 and dk % 2 == 0, f'd_k must be even and >= 16, got {dk}'
    assert dv % 32 == 0, f'd_v must be a multiple of 32, got {dv}'
    if fmt is not None:
        g = fmt.group
        assert g >= 2 and g & (g - 1) == 0, f'group {g} must be a power of two'
        if fmt.axis == 'k':
            assert g <= pow2(dk) and pow2(dk) // g <= 8, f'key group {g} on d_k {dk}: 1 .. 8 groups'
        else:
            assert dv % g == 0, f'value group {g} must divide d_v {dv}'


def use_fused(fused, bh, axis_v=False):
    """Resolve a ``fused`` option (None = auto, True, False) for ``bh`` = batch rows x value heads and a format
    with (``axis_v``) or without value-axis scales."""
    if fused is None:
        lim = FUSED_MAX_BH_AXIS_V if axis_v else FUSED_MAX_BH
        return lim is None or bh <= lim
    return bool(fused)


# ---------------------------------------------------------------------------------------------------------------
# format
# ---------------------------------------------------------------------------------------------------------------
def packed_format(fmt):
    """The format a packed store actually holds: ``fmt`` with fp16 scales, or None for the bf16 control.

    Accepts a symmetric RTN ``Quantizer`` with axis 'k' or 'v', 4 or 8 bits, no rotation; or 'bf16' /
    ``FloatCast('bf16')`` / None for the unquantized bf16 state.
    """
    if fmt is None or fmt == 'bf16' or (isinstance(fmt, FloatCast) and fmt.dtype == 'bf16' and not fmt.stochastic):
        return None
    assert isinstance(fmt, Quantizer), f'packed storage needs a Quantizer or bf16, got {fmt!r}'
    assert fmt.bits in (4, 8) and fmt.axis in ('k', 'v'), (fmt.bits, fmt.axis)
    assert not fmt.affine and fmt.rotation is None and fmt.rounding == 'rtn' and fmt.scale_bits == 16, \
        'packed storage implements symmetric RTN with 16-bit scales and no rotation'
    assert fmt.group % 2 == 0
    return dataclasses.replace(fmt, scale_dtype='fp16')


def _bits(fmt):
    return 16 if fmt is None else fmt.bits


def pack(S, fmt):
    """[BH, d_k, d_v] float -> (codes, scales).

    codes   4 bit: uint8 [BH, d_v, d_k/2], byte j = (c[2j] + 8) | (c[2j+1] + 8) << 4;  8 bit: int8 [BH, d_v, d_k];
            bf16: bf16 [BH, d_v, d_k]
    scales  fp16; axis 'v': [BH, d_v/group, d_k]; axis 'k': [BH, d_v, d_k/group]; bf16: a [1] tensor of ones
    The arithmetic is ``Quantizer.fake``'s on the same groups, so codes * scales == packed_format(fmt).fake(S).
    """
    x = S.float().reshape(-1, S.shape[-2], S.shape[-1]).transpose(1, 2).contiguous()      # [BH, DV, DK]
    return pack_vk(x, fmt)


def pack_vk(x, fmt):
    """``pack`` of a state given value-major, [BH, d_v, d_k] (the stored orientation; vLLM's ssm-state layout)."""
    fmt = packed_format(fmt)
    x = x.float().reshape(-1, x.shape[-2], x.shape[-1])
    BH, DV, DK = x.shape
    if fmt is None:
        return x.to(torch.bfloat16).contiguous(), torch.ones(1, device=x.device, dtype=torch.float16)
    g, qmax = fmt.group, 2 ** (fmt.bits - 1) - 1
    if fmt.axis == 'v':
        assert DV % g == 0, f'value group {g} must divide d_v {DV}'
        X = x.reshape(BH, DV // g, g, DK)
        scale = fmt._cast_scale(X.abs().amax(2, keepdim=True).clamp_min(1e-8) / qmax)
        codes = torch.round(X / scale).clamp(-qmax, qmax).to(torch.int8).reshape(BH, DV, DK)
        scales = scale.reshape(BH, DV // g, DK)
    else:                                             # ragged last key group: zero padded (symmetric: same absmax)
        ng = -(-DK // g)
        X = torch.nn.functional.pad(x, (0, ng * g - DK)).reshape(BH, DV, ng, g)
        scale = fmt._cast_scale(X.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax)
        codes = torch.round(X / scale).clamp(-qmax, qmax).to(torch.int8).reshape(BH, DV, ng * g)[..., :DK]
        scales = scale.reshape(BH, DV, ng)
    if fmt.bits == 4:
        lo = (codes[..., 0::2] + 8).to(torch.uint8)
        hi = (codes[..., 1::2] + 8).to(torch.uint8)
        codes = lo | (hi << 4)
    return codes.contiguous(), scales.to(torch.float16).contiguous()


def unpack(codes, scales, fmt, d_k):
    """Inverse of ``pack``: -> [BH, d_k, d_v] fp32."""
    return unpack_vk(codes, scales, fmt, d_k).transpose(1, 2).contiguous()


def unpack_vk(codes, scales, fmt, d_k):
    """Inverse of ``pack_vk``: -> [BH, d_v, d_k] fp32."""
    fmt = packed_format(fmt)
    if fmt is None:
        return codes.float()
    BH, DV = codes.shape[:2]
    if fmt.bits == 4:
        c = codes.to(torch.int16)
        x = torch.empty(BH, DV, d_k, device=codes.device, dtype=torch.float32)
        x[..., 0::2] = (c & 0xF).float() - 8
        x[..., 1::2] = ((c >> 4) & 0xF).float() - 8
    else:
        x = codes.float()
    g, sc = fmt.group, scales.float()
    if fmt.axis == 'v':
        x = (x.reshape(BH, DV // g, g, d_k) * sc[:, :, None, :]).reshape(BH, DV, d_k)
    else:
        x = sc.repeat_interleave(g, -1)[..., :d_k] * x            # ragged last group: scale columns cut at d_k
    return x


def _flat(S):
    S = S.float() if S.dim() == 3 else S.float().reshape(-1, S.shape[-2], S.shape[-1])
    return S.contiguous()


def _nrm(x):
    x = x.float()
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _round_buf(x, bits):
    """Buffer rounding exactly as the append kernel: int8 with one fp16 absmax scale per vector, or bf16."""
    if bits == 16:
        return x.to(torch.bfloat16).float()
    s = (x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127).to(torch.float16).float().clamp_min(_FP16_MIN)
    return torch.round(x / s).clamp(-127, 127) * s


# ---------------------------------------------------------------------------------------------------------------
# torch reference
# ---------------------------------------------------------------------------------------------------------------
class ReferenceGDNState:
    """The packed decode arithmetic in torch: rounded S_0, (G_i, k_i, u_i) buffer rounded to ``buffer_bits``, fold and
    re-round every ``m`` tokens. Same constructor and ``step`` as ``PackedGDNState``.

    ``auto_fold=False`` leaves a full buffer in place after ``step`` until ``fold()`` is called, so a test can inspect
    (or adopt) the buffered entries before they are folded."""

    def __init__(self, S, fmt, m=8, hv=None, hk=None, buffer_bits=8, auto_fold=True):
        self.auto_fold = auto_fold
        S = _flat(S)
        self.BH, self.DK, self.DV = S.shape
        self.fmt, self.m, self.buffer_bits = packed_format(fmt), m, buffer_bits
        self.HV = hv or self.BH
        self.HK = hk or self.HV
        self.gqa = self.HV // self.HK
        self.B = self.BH // self.HV
        self.load(S)

    def load(self, S):
        self.codes, self.scales = pack(_flat(S), self.fmt)
        self.S0 = unpack(self.codes, self.scales, self.fmt, self.DK)
        dev = self.S0.device
        self.K = torch.zeros(self.B * self.HK, self.m, self.DK, device=dev)
        self.U = torch.zeros(self.BH, self.m, self.DV, device=dev)
        self.G = torch.zeros(self.BH, self.m, device=dev)
        self.gcur = torch.zeros(self.BH, device=dev)
        self.n = 0

    def dequantize(self):
        return self.S0.clone()

    def _kv(self):
        K = self.K[:, :self.n]
        K = K.reshape(self.B, self.HK, 1, self.n, self.DK).expand(-1, -1, self.gqa, -1, -1)
        return K.reshape(self.BH, self.n, self.DK), self.U[:, :self.n]

    @torch.no_grad()
    def step(self, q, k, v, g, beta, scale):
        BH, n = self.BH, self.n
        if q.shape[0] != BH:                                  # q, k given per key head
            q = q.reshape(self.B, self.HK, 1, -1).expand(-1, -1, self.gqa, -1).reshape(BH, -1)
            k = k.reshape(self.B, self.HK, 1, -1).expand(-1, -1, self.gqa, -1).reshape(BH, -1)
        qn = _nrm(q) * scale
        kn = _nrm(k)
        alpha = torch.exp(self.gcur)
        sq = alpha[:, None] * torch.einsum('bkv,bk->bv', self.S0, qn)
        sk = alpha[:, None] * torch.einsum('bkv,bk->bv', self.S0, kn)
        if n:
            Kb, Ub = self._kv()
            dec = torch.exp(self.gcur[:, None] - self.G[:, :n])
            sq = sq + torch.einsum('bi,biv->bv', dec * torch.einsum('bik,bk->bi', Kb, qn), Ub)
            sk = sk + torch.einsum('bi,biv->bv', dec * torch.einsum('bik,bk->bi', Kb, kn), Ub)
        a = torch.exp(g.float())[:, None]
        u = beta.float()[:, None] * (v.float() - a * sk)
        o = a * sq + (kn * qn).sum(-1, keepdim=True) * u
        # append
        self.U[:, n] = _round_buf(u, self.buffer_bits)
        kh = kn.reshape(self.B, self.HK, self.gqa, self.DK)[:, :, 0].reshape(-1, self.DK)
        self.K[:, n] = _round_buf(kh, self.buffer_bits)
        self.gcur = self.gcur + g.float()
        self.G[:, n] = self.gcur
        self.n += 1
        if self.n == self.m and self.auto_fold:
            self.fold()
        return o

    def fold(self):
        assert self.n == self.m
        Kb, Ub = self._kv()
        S = torch.exp(self.gcur)[:, None, None] * self.S0
        dec = torch.exp(self.gcur[:, None] - self.G)
        for i in range(self.m):                               # the kernel's accumulation order
            S = S + Kb[:, i, :, None] * (dec[:, i, None] * Ub[:, i])[:, None, :]
        self.S_raw = S                                        # the unrounded folded state, for tests
        self.codes, self.scales = pack(S, self.fmt)
        self.S0 = unpack(self.codes, self.scales, self.fmt, self.DK)
        self.gcur = torch.zeros_like(self.gcur)
        self.n = 0


# ---------------------------------------------------------------------------------------------------------------
# Triton path
# ---------------------------------------------------------------------------------------------------------------
class PackedGDNState:
    """One layer's packed recurrent state and its decode buffer, stepped by Triton kernels.

    S            [BH, d_k, d_v] (or [B, H, d_k, d_v]) float state to start from
    fmt          ``Quantizer(bits in {4, 8}, axis in {'k', 'v'}, group)``, symmetric RTN; or 'bf16' (16-bit control).
                 Scales are stored fp16, so the stored state is ``packed_format(fmt).fake(S)`` exactly.
    m            buffer length (power of two); the state is folded and re-rounded every m steps
    hv, hk       value and key heads per sequence (GQA: hv % hk == 0); BH = B * hv
    buffer_bits  8 (int8 + fp16 absmax scale per vector) or 16 (bf16) for buffered keys and delta values
    out_dtype    dtype of the output o
    fused        True: one fused launch per step (``_fused_triton``); False: the five-kernel step; None (default):
                 ``use_fused`` (fused, except value-axis formats above ``FUSED_MAX_BH_AXIS_V`` batch rows x heads).
                 Fixed at construction (the counter layout differs).

    ``step(q, k, v, g, beta, scale)``: q, k raw [BH, d_k] or per key head [B * hk, d_k]; v [BH, d_v]; g (log decay)
    and beta (after sigmoid) [BH]. Returns the output buffer [BH, d_v] (the same tensor every step).
    """

    def __init__(self, S, fmt, m=8, hv=None, hk=None, buffer_bits=8, out_dtype=torch.bfloat16,
                 block_v=None, warps=None, fused=None, fused_warps=None):
        from . import _fold_triton, _fused_triton
        self._K, self._F = _fold_triton, _fused_triton
        S = _flat(S)
        BH, DK, DV = S.shape
        dev = S.device
        self.BH, self.DK, self.DV, self.m = BH, DK, DV, m
        self.fmt = packed_format(fmt)
        self.bits = _bits(self.fmt)
        self.HV = hv or BH
        self.HK = hk or self.HV
        assert self.HV % self.HK == 0 and BH % self.HV == 0, (BH, self.HV, self.HK)
        self.gqa, self.B = self.HV // self.HK, BH // self.HV
        assert m >= 1 and m & (m - 1) == 0, 'm must be a power of two'
        check_dims(self.fmt, DK, DV)
        self.DKP, self.DVP = pow2(DK), pow2(DV)
        assert buffer_bits in (8, 16)
        self.buffer_bits, self.qbuf = buffer_bits, buffer_bits == 8
        self.axk = self.fmt is not None and self.fmt.axis == 'k'
        self.group = self.fmt.group if self.fmt is not None else 32
        # axis 'v' needs one scale group per program (BV == group); axis 'k' and bf16 take any BV
        self.BV = block_v or (self.group if (self.fmt is not None and not self.axk) else 32)
        if self.fmt is not None and not self.axk:
            assert self.BV == self.group, 'axis v: block_v must equal the group'
        assert DV % self.BV == 0
        self.warps = warps or (2 if self.BV <= 32 else 4)
        self.fold_warps = 4 if self.BV <= 32 else 8
        self.fused = use_fused(fused, BH, self.fmt is not None and self.fmt.axis == 'v')
        self.fused_sbv, self.fused_bv, self.fused_warps = _fused_config(self, fused_warps)

        codes, scales = pack(S, self.fmt)
        self.codes, self.scales = codes.clone(), scales.clone()
        bdt = torch.int8 if self.qbuf else torch.bfloat16
        self.KB = torch.zeros(self.B * self.HK, m, DK, device=dev, dtype=bdt)
        self.UB = torch.zeros(BH, m, DV, device=dev, dtype=bdt)
        self.KBS = torch.zeros(self.B * self.HK, m, device=dev, dtype=torch.float16)
        self.UBS = torch.zeros(BH, m, device=dev, dtype=torch.float16)
        self.GB = torch.zeros(BH, m, device=dev, dtype=torch.float32)
        self.GCUR = torch.zeros(BH, device=dev, dtype=torch.float32)
        # the shared lockstep counter; the fused step keeps one copy per (sequence, value head), all equal
        self.NPTR = (torch.zeros(self.B, self.HV, device=dev, dtype=torch.int32) if self.fused
                     else torch.zeros((), device=dev, dtype=torch.int32))
        # per-step scratch
        self.QN = torch.zeros(BH, DK, device=dev)
        self.KN = torch.zeros(BH, DK, device=dev)
        self.CQ = torch.zeros(BH, m, device=dev)
        self.CK = torch.zeros(BH, m, device=dev)
        self.ALPHA = torch.zeros(BH, device=dev)
        self.KQ = torch.zeros(BH, device=dev)
        self.UCUR = torch.zeros(BH, DV, device=dev)
        self.GS = torch.zeros(BH, device=dev)
        self.BS = torch.zeros(BH, device=dev)
        self.O = torch.zeros(BH, DV, device=dev, dtype=out_dtype)

    # -- state ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def load(self, S):
        """Replace the state (in place) and empty the buffer."""
        c, sc = pack(_flat(S), self.fmt)
        self.codes.copy_(c)
        self.scales.copy_(sc)
        for t in (self.KB, self.UB, self.KBS, self.UBS, self.GB, self.GCUR, self.NPTR):
            t.zero_()

    def dequantize(self):
        """The stored state S_0 (without the buffered tokens) as [BH, d_k, d_v] fp32."""
        return unpack(self.codes, self.scales, self.fmt, self.DK)

    @property
    def n(self):
        return int(self.NPTR.reshape(-1)[0])

    def stored_bytes(self):
        """Resident bytes: the state payload + scales, the buffer, the decay. Scratch is excluded."""
        nb = lambda t: t.numel() * t.element_size()
        per = dict(state=nb(self.codes) + (nb(self.scales) if self.fmt is not None else 0),
                   buffer=nb(self.KB) + nb(self.UB) + ((nb(self.KBS) + nb(self.UBS)) if self.qbuf else 0),
                   decay=nb(self.GB) + nb(self.GCUR) + nb(self.NPTR))
        per['total'] = sum(per.values())
        return per

    # -- the step ---------------------------------------------------------------------------------------------
    def step(self, q, k, v, g, beta, scale):
        K = self._K
        BH, DK, DV, m = self.BH, self.DK, self.DV, self.m
        HV, HK = self.HV, self.HK
        per_key = q.shape[0] != BH
        if per_key:
            assert q.shape[0] == self.B * HK and k.shape[0] == self.B * HK
        for t in (q, k, v, g, beta):
            assert t.is_contiguous()
        sqk = (HK if per_key else HV) * DK
        DKP, DVP = self.DKP, self.DVP
        cst = dict(DK=DK, M=m, HV=HV, HK=HK, GQA=self.gqa, QBUF=self.qbuf, HAS_IDX=False)
        st = dict(DK=DK, DKP=DKP, DV=DV, BV=self.BV, M=m, BITS=self.bits, AXK=self.axk, GROUP=self.group,
                  QBUF=self.qbuf, HV=HV, HAS_IDX=False)
        # per-sequence strides of the lockstep layout: contiguous sequences, one shared counter (stride 0)
        sS = HV * self.codes[0].numel()
        sSC = HV * self.scales[0].numel() if self.fmt is not None else 0
        sKB, sKBS, sUB, sUBS, sGB = HK * m * DK, HK * m, HV * m * DV, HV * m, HV * m
        idx = self.NPTR                                       # unused (HAS_IDX=False)
        if self.fused:
            self._F._fused_step[(BH,)](
                q, k, sqk, sqk, v, HV * DV, g, beta, HV, HV, g, beta, 0, 0, g, g, idx,
                self.codes, sS, self.scales, sSC, self.KB, sKB, self.KBS, sKBS, self.UB, sUB, self.UBS, sUBS,
                self.GB, sGB, self.GCUR, HV, self.NPTR, HV, self.O, float(scale),
                DK=DK, DV=DV, DKP=DKP, DVP=DVP, SBV=self.fused_sbv, FBV=self.fused_bv, M=m, BITS=self.bits, AXK=self.axk,
                GROUP=self.group, QBUF=self.qbuf, HV=HV, HK=HK, GQA=self.gqa, QK_PER_KEY_HEAD=per_key, HAS_IDX=False,
                GATE=False, num_warps=self.fused_warps)
            return self.O
        K._prep[(BH,)](q, k, sqk, sqk, g, beta, HV, HV, g, beta, 0, 0, g, g, idx,
                       self.GB, sGB, self.GCUR, HV, self.KB, sKB, self.KBS, sKBS, self.NPTR, 0,
                       self.QN, self.KN, self.CQ, self.CK, self.KQ, self.ALPHA, self.GS, self.BS, float(scale),
                       BM=min(m, 16), QK_PER_KEY_HEAD=per_key, GATE=False, DKP=DKP, num_warps=4, **cst)
        K._state[(BH, DV // self.BV)](
            self.codes, sS, self.scales, sSC, self.QN, self.KN, self.UB, sUB, self.UBS, sUBS, self.CQ, self.CK,
            self.ALPHA, v, HV * DV, self.GS, self.BS, self.KQ, self.O, self.UCUR, self.NPTR, 0, idx,
            num_warps=self.warps, **st)
        K._append[(BH,)](self.UCUR, self.KN, self.GS, idx, self.GB, sGB, self.GCUR, HV, self.KB, sKB,
                         self.KBS, sKBS, self.UB, sUB, self.UBS, sUBS, self.NPTR, 0, DV=DV, DKP=DKP, DVP=DVP,
                         num_warps=4, **cst)
        K._fold[(BH, DV // self.BV)](
            self.codes, sS, self.scales, sSC, self.UB, sUB, self.UBS, sUBS, self.KB, sKB, self.KBS, sKBS,
            self.GB, sGB, self.GCUR, HV, self.NPTR, 0, idx, HK=HK, GQA=self.gqa, num_warps=self.fold_warps, **st)
        K._advance[(1,)](self.GCUR, self.NPTR, BH, M=m, BLK=1024, num_warps=4)
        return self.O
