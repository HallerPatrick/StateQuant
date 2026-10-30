"""Triton kernels of the packed decode step (ported from PACE's ``runs/20260921-e2e/fold_kv_kernel.py``).

One decode step is five launches with a fixed sequence, so a CUDA graph can hold it:

    _prep     normalise q, k; (optionally) the gates g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b); the
              buffer's two coefficient vectors c_q[i] = (a_n/a_i) k_i.q, c_k[i] = (a_n/a_i) k_i.k
    _state    the only read of the stored state: S_{t-1}^T q and S_{t-1}^T k from codes + buffer, then
              u_t = beta (v_t - a_t S_{t-1}^T k_t) and o_t = a_t S_{t-1}^T q + (k_t.q) u_t
    _append   push (cumulative log decay, key, delta value) into the buffer, rounded to the buffer width
    _fold     exits at once unless the buffer is now full; otherwise S = a_m S_0 + sum_i (a_m/a_i) k_i u_i^T and
              re-round in place (PACE's ``foldk`` path)
    _advance  one program: zero the running log decay after a fold and advance the shared slot pointer
              n -> (n+1) mod m (all sequences in lockstep);  ``_advance_seq``: one program per sequence, each with
              its own counter (the serving layout)

The state is held TRANSPOSED, [HV, d_v, d_k] per sequence, so contractions with q and k reduce along the contiguous
axis and four-bit codes pack two per byte along d_k (byte j = code[2j] + 8 | (code[2j+1] + 8) << 4). Scales are fp16:

    axis 'v'  one scale per ``group`` consecutive value columns at a fixed key row: scales [HV, d_v/group, d_k]
    axis 'k'  one scale per ``group`` consecutive key rows within a value column:   scales [HV, d_v, d_k/group]

Head dims that are not powers of two (OLMo-Hybrid: d_k 96, d_v 192): key-axis tiles are padded to DKP, the next
power of two, and masked (loads read 0 past d_k, stores stop at d_k; ``_pad0``), so memory holds exactly d_k x d_v
and a padded run is bit-identical to the same problem zero-padded to DKP (tests/test_kernels_nonpow2.py). Key-axis
scale groups may be ragged: ceil(d_k / group) groups per value row, the last one shorter (64 on 96: 64 + 32); a group
of the padded tile that lies entirely past d_k is neither stored nor read. Value rows run in blocks that divide d_v
(the five-kernel step: 32 or the value group; the fused step: the largest power of two dividing d_v, up to 64), so
d_v needs no padding except in ``_append``'s whole-row tile (DVP, masked).

Addressing. Every per-sequence state tensor (codes S, scales SC, buffers KB/KBS/UB/UBS, decays GB/GCUR, counter
NPTR) is passed with its own per-sequence stride ``s*`` (in elements). Batch row b reads sequence ``seq`` =
IDX[b] if HAS_IDX (slot indirection, e.g. a serving engine's state-slot table; rows with seq <= 0 are padding and
are skipped, their output zeroed) else b. The lockstep path (``PackedGDNState``) is the special case of contiguous
sequences, identity index and one shared counter (stride 0). Inputs q, k, v, g, beta / a, b are read at batch row
b with their own row strides; scratch (QN, KN, CQ, CK, KQ, ALPHA, GS, BS, UCUR) and the output O are contiguous over
(b, value head).

Rounding is made to agree with ``statequant.Quantizer(..., scale_dtype='fp16').fake`` element for element: the
divisions are IEEE (``div_rn``), rounding is half-to-even (``rint``, as torch.round), and the payload is rounded
against the fp16-ROUNDED scale. PACE's kernel used libdevice ``round`` (half away from zero) and the fp32 scale
for the payload, so its codes after a fold could differ from its own host quantiser on ties.
"""
from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

FP16_MIN = 5.960464477539063e-08            # 2^-24, smallest fp16 subnormal: the floor of a stored scale


@triton.jit
def _fp16_scale(amax, QMAX: tl.constexpr):
    """Quantizer: (amax.clamp_min(1e-8) / qmax) -> fp16 -> fp32, clamped at 2^-24."""
    s = tl.div_rn(tl.maximum(amax, 1e-8), QMAX * 1.0)
    s = s.to(tl.float16).to(tl.float32)
    return tl.maximum(s, 5.960464477539063e-08)


@triton.jit
def _round_buf(x, scale):
    return tl.minimum(tl.maximum(libdevice.rint(tl.div_rn(x, scale)), -127.0), 127.0)


@triton.jit
def _seq_of(IDX, b, HAS_IDX: tl.constexpr):
    if HAS_IDX:
        s = tl.load(IDX + b).to(tl.int64)
    else:
        s = b.to(tl.int64)
    return s


@triton.jit
def _nib(x):
    """A 4-bit code field (0..15, int32) -> code - 8 as fp32, via the exponent trick (2^23 + x as float bits)
    instead of an integer->float conversion (a quarter-rate instruction): exact."""
    return (x | 0x4B000000).to(tl.float32, bitcast=True) - 8388616.0


@triton.jit
def _i2f(x):
    """Small integers (|x| < 2^22: int8 codes, int8 buffers) -> fp32, exact, by the 1.5 * 2^23 trick (integer add +
    float subtract instead of a quarter-rate conversion)."""
    return (x.to(tl.int32) + 0x4B400000).to(tl.float32, bitcast=True) - 12582912.0


@triton.jit
def _tof(x):
    """Stored codes / buffer entries -> fp32 (int8: ``_i2f``; bf16 / fp16: the cast)."""
    if x.dtype == tl.int8:
        return _i2f(x)
    else:
        return x.to(tl.float32)


@triton.jit
def _pad0(x, mask, PAD: tl.constexpr):
    """x with the padded lanes (mask False) set to 0, for tiles padded past d_k (PAD); a no-op for power-of-two d_k.
    Needed for four-bit codes: Triton 3.8 fills the masked lanes of a vectorised uint8 load with ``other=136``
    sign-extended across the vector word (lanes read 0x88, 0xFF, 0xFF, 0xFF, i.e. codes 0 and 7), which a ragged
    key group's absmax in the fold would see (``.dbg``-style repro: a [16, 64] uint8 tile, row stride 48). Other
    loads are zeroed too where a padded lane could reach a scale or a stored value."""
    if PAD:
        return tl.where(mask, x, 0.0)
    else:
        return x


@triton.jit
def _load_codes(Sb, ov, oh, DK: tl.constexpr, BITS: tl.constexpr):
    """Codes (or bf16 values) of rows ``ov`` of one head (base ``Sb``) at even and odd key columns 2*oh, 2*oh+1,
    as fp32 [BV, len(oh)]; ``oh`` may run past d_k / 2 (padded tiles): those columns read as 0."""
    hm = (oh < DK // 2)[None, :]
    if BITS == 4:
        by = tl.load(Sb + ov[:, None] * (DK // 2) + oh[None, :], mask=hm, other=136).to(tl.int32)
        ce = _nib(by & 0xF)
        co = _nib((by >> 4) & 0xF)
    else:
        base = Sb + ov[:, None] * DK + 2 * oh[None, :]
        ce = _tof(tl.load(base, mask=hm, other=0))
        co = _tof(tl.load(base + 1, mask=hm, other=0))
    PAD: tl.constexpr = oh.shape[0] != DK // 2
    return _pad0(ce, hm, PAD), _pad0(co, hm, PAD)


@triton.jit
def _half1(x):
    """[L] -> its halves [L/2], [L/2]."""
    L: tl.constexpr = x.shape[0]
    return tl.split(tl.permute(tl.reshape(x, (2, L // 2)), (1, 0)))


@triton.jit
def _grp2(x, gi: tl.constexpr, NG: tl.constexpr):
    a, b = _half1(x)
    if NG == 4:
        if gi < 2:
            y = a
        else:
            y = b
        a, b = _half1(y)
    if gi % 2 == 0:
        return a
    else:
        return b


@triton.jit
def _grp(x, gi: tl.constexpr, NG: tl.constexpr):
    """Piece gi of NG equal consecutive pieces of the vector x (NG <= 8), by halving."""
    if NG == 1:
        return x
    elif NG <= 4:
        return _grp2(x, gi, NG)
    else:
        tl.static_assert(NG == 8, 'fused step: at most 8 key groups')
        a, b = _half1(x)
        if gi < 4:
            return _grp2(a, gi, 4)
        else:
            return _grp2(b, gi - 4, 4)


@triton.jit
def _state_dots_wide(Sb, SCh, q, k, ov, DK: tl.constexpr, DKP: tl.constexpr, BV: tl.constexpr, BITS: tl.constexpr,
                     AXK: tl.constexpr, GROUP: tl.constexpr):
    """``_state_dots`` for 8- and 16-bit codes: rows read whole ([BV, DKP] or [BV, GROUP] per key group, unit
    stride, vectorised) against the full q, k vectors ([DKP], zero past d_k)."""
    ok = tl.arange(0, DKP)
    km = (ok < DK)[None, :]
    if BITS == 16 or not AXK:
        c = _tof(tl.load(Sb + ov[:, None] * DK + ok[None, :], mask=km, other=0))
        if BITS == 16:
            sq = tl.sum(c * q[None, :], 1)
            sk = tl.sum(c * k[None, :], 1)
        elif BV == GROUP:
            sc = tl.load(SCh + ok, mask=ok < DK, other=0.0).to(tl.float32)
            sq = tl.sum(c * (q * sc)[None, :], 1)
            sk = tl.sum(c * (k * sc)[None, :], 1)
        else:
            sc = tl.load(SCh + (ov[:, None] // GROUP) * DK + ok[None, :], mask=km, other=0.0).to(tl.float32)
            sq = tl.sum(c * (q[None, :] * sc), 1)
            sk = tl.sum(c * (k[None, :] * sc), 1)
    else:
        NG: tl.constexpr = (DK + GROUP - 1) // GROUP              # stored key groups (the last may be ragged)
        NGP: tl.constexpr = DKP // GROUP                          # groups of the padded tile
        og = tl.arange(0, GROUP)
        sq = tl.zeros((BV,), tl.float32)
        sk = tl.zeros((BV,), tl.float32)
        for gi in tl.static_range(NG):
            oc = gi * GROUP + og
            c = _tof(tl.load(Sb + ov[:, None] * DK + oc[None, :], mask=(oc < DK)[None, :], other=0))
            sc = tl.load(SCh + ov * NG + gi).to(tl.float32)
            sq = tl.fma(tl.sum(c * _grp(q, gi, NGP)[None, :], 1), sc, sq)
            sk = tl.fma(tl.sum(c * _grp(k, gi, NGP)[None, :], 1), sc, sk)
    return sq, sk


@triton.jit
def _state_dots(Sb, SCh, QNb, KNb, ov, DK: tl.constexpr, DKP: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
                BITS: tl.constexpr, AXK: tl.constexpr, GROUP: tl.constexpr):
    """S_0^T q and S_0^T k on value rows ``ov`` (one block of BV) of one head: codes at ``Sb``, scales at ``SCh``,
    normalised q, k at ``QNb``, ``KNb``. The fused step (``_fused_triton._state_dots_v``) is the same arithmetic on
    the same tiles, with q, k from registers.

    4-bit codes (8 / 16 bit: ``_state_dots_wide``) are read [BV, DK/2] at unit stride and the two nibbles reduce
    separately against q, k at even and odd d_k (PACE's VEC form). Axis 'v' with BV == GROUP: every row shares one
    scale group and the scale is a [DK] vector folded into q and k once (BV a multiple of GROUP: the same products
    with the scales gathered per row). Axis 'k' takes one key group at a time ([BV, GROUP/2] codes), sums it and
    accumulates (group sum) * (its [BV] scales) group by group in a fixed order."""
    if BITS == 16 or not AXK:
        oh = tl.arange(0, DKP // 2)
        hm = oh < DK // 2
        qe = tl.load(QNb + 2 * oh, mask=hm, other=0.0)
        qo = tl.load(QNb + 2 * oh + 1, mask=hm, other=0.0)
        ke = tl.load(KNb + 2 * oh, mask=hm, other=0.0)
        ko = tl.load(KNb + 2 * oh + 1, mask=hm, other=0.0)
        ce, co = _load_codes(Sb, ov, oh, DK, BITS)
        if BITS == 16:
            sq = tl.sum(ce * qe[None, :], 1) + tl.sum(co * qo[None, :], 1)
            sk = tl.sum(ce * ke[None, :], 1) + tl.sum(co * ko[None, :], 1)
        elif BV == GROUP:                                     # SCh: this value block's scale row
            sce = tl.load(SCh + 2 * oh, mask=hm, other=0.0).to(tl.float32)
            sco = tl.load(SCh + 2 * oh + 1, mask=hm, other=0.0).to(tl.float32)
            sq = tl.sum(ce * (qe * sce)[None, :], 1) + tl.sum(co * (qo * sco)[None, :], 1)
            sk = tl.sum(ce * (ke * sce)[None, :], 1) + tl.sum(co * (ko * sco)[None, :], 1)
        else:                                                 # several groups per block; SCh: the head's scales
            sp = SCh + (ov[:, None] // GROUP) * DK + 2 * oh[None, :]
            sce = tl.load(sp, mask=hm[None, :], other=0.0).to(tl.float32)
            sco = tl.load(sp + 1, mask=hm[None, :], other=0.0).to(tl.float32)
            sq = tl.sum(ce * (qe[None, :] * sce), 1) + tl.sum(co * (qo[None, :] * sco), 1)
            sk = tl.sum(ce * (ke[None, :] * sce), 1) + tl.sum(co * (ko[None, :] * sco), 1)
    else:
        NG: tl.constexpr = (DK + GROUP - 1) // GROUP
        og = tl.arange(0, GROUP // 2)
        sq = tl.zeros((BV,), tl.float32)
        sk = tl.zeros((BV,), tl.float32)
        for gi in tl.static_range(NG):
            oj = gi * (GROUP // 2) + og
            jm = oj < DK // 2
            qe = tl.load(QNb + 2 * oj, mask=jm, other=0.0)
            qo = tl.load(QNb + 2 * oj + 1, mask=jm, other=0.0)
            ke = tl.load(KNb + 2 * oj, mask=jm, other=0.0)
            ko = tl.load(KNb + 2 * oj + 1, mask=jm, other=0.0)
            if BITS == 4:
                by = tl.load(Sb + ov[:, None] * (DK // 2) + oj[None, :], mask=jm[None, :], other=136).to(tl.int32)
                ce = _nib(by & 0xF)
                co = _nib((by >> 4) & 0xF)
            else:
                base = Sb + ov[:, None] * DK + 2 * oj[None, :]
                ce = _tof(tl.load(base, mask=jm[None, :], other=0))
                co = _tof(tl.load(base + 1, mask=jm[None, :], other=0))
            sc = tl.load(SCh + ov * NG + gi).to(tl.float32)
            sq = tl.fma(tl.sum(tl.fma(co, qo[None, :], ce * qe[None, :]), 1), sc, sq)
            sk = tl.fma(tl.sum(tl.fma(co, ko[None, :], ce * ke[None, :]), 1), sc, sk)
    return sq, sk


@triton.jit
def _coefs(kbase, ksc, gb, gcur, q, k, om, n, ok, DK: tl.constexpr, QBUF: tl.constexpr):
    """c_q[i] = (a_n/a_i) k_i.q and c_k[i] = (a_n/a_i) k_i.k for buffer slots ``om`` (0 unless i < n), one [len(om), DK]
    tile. The loads are not masked by n (so they need not wait for the counter); dead slots' values, possibly
    garbage, are discarded."""
    kb = _tof(tl.load(kbase + om[:, None] * DK + ok[None, :], mask=(ok < DK)[None, :], other=0))
    if QBUF:
        kb = kb * tl.load(ksc + om).to(tl.float32)[:, None]
    # alpha_n / alpha_i = exp(G_n - G_i) <= 1, formed in log space
    dec = libdevice.exp(gcur - tl.load(gb + om))
    live = om < n
    return (tl.where(live, dec * tl.sum(kb * q[None, :], 1), 0.0),
            tl.where(live, dec * tl.sum(kb * k[None, :], 1), 0.0))


@triton.jit
def _half(X):
    """[R, W] -> its column halves [R, W/2], [R, W/2]."""
    R: tl.constexpr = X.shape[0]
    W: tl.constexpr = X.shape[1]
    return tl.split(tl.permute(tl.reshape(X, (R, 2, W // 2)), (0, 2, 1)))


@triton.jit
def _acc1(sq, sk, U, C, D):
    R: tl.constexpr = U.shape[0]
    u = tl.reshape(U, (R,))
    return tl.fma(u, tl.reshape(C, (R,)), sq), tl.fma(u, tl.reshape(D, (R,)), sk)


@triton.jit
def _acc2(sq, sk, U, C, D):
    U0, U1 = _half(U)
    C0, C1 = _half(C)
    D0, D1 = _half(D)
    sq, sk = _acc1(sq, sk, U0, C0, D0)
    return _acc1(sq, sk, U1, C1, D1)


@triton.jit
def _acc4(sq, sk, U, C, D):
    U0, U1 = _half(U)
    C0, C1 = _half(C)
    D0, D1 = _half(D)
    sq, sk = _acc2(sq, sk, U0, C0, D0)
    return _acc2(sq, sk, U1, C1, D1)


@triton.jit
def _acc8(sq, sk, U, C, D):
    U0, U1 = _half(U)
    C0, C1 = _half(C)
    D0, D1 = _half(D)
    sq, sk = _acc4(sq, sk, U0, C0, D0)
    return _acc4(sq, sk, U1, C1, D1)


@triton.jit
def _acc16(sq, sk, U, C, D):
    U0, U1 = _half(U)
    C0, C1 = _half(C)
    D0, D1 = _half(D)
    sq, sk = _acc8(sq, sk, U0, C0, D0)
    return _acc8(sq, sk, U1, C1, D1)


@triton.jit
def _acc32(sq, sk, U, C, D):
    U0, U1 = _half(U)
    C0, C1 = _half(C)
    D0, D1 = _half(D)
    sq, sk = _acc16(sq, sk, U0, C0, D0)
    return _acc16(sq, sk, U1, C1, D1)


@triton.jit
def _acc(sq, sk, U, C, D, M: tl.constexpr):
    """sq = fma(u_i, c_i, sq), sk = fma(u_i, d_i, sk) for i = 0 .. M-1 in order, from [R, M] tiles (column i =
    slot i): ``_buffer_dots``' sequence, independent of the tiles' layout. m <= 32."""
    if M == 1:
        return _acc1(sq, sk, U, C, D)
    elif M == 2:
        return _acc2(sq, sk, U, C, D)
    elif M == 4:
        return _acc4(sq, sk, U, C, D)
    elif M == 8:
        return _acc8(sq, sk, U, C, D)
    elif M == 16:
        return _acc16(sq, sk, U, C, D)
    else:
        tl.static_assert(M == 32, 'fused step: m <= 32')
        return _acc32(sq, sk, U, C, D)


@triton.jit
def _buffer_dots(sq, sk, ubh, ubsh, CQb, CKb, ov, n, M: tl.constexpr, DV: tl.constexpr, QBUF: tl.constexpr):
    """sq += sum_{i<n} c_q[i] u_i, sk += sum_{i<n} c_k[i] u_i on rows ``ov``: the delta values in one [m, BV] tile
    (unmasked, dead slots discarded), accumulated slot by slot in a fixed order by ``_acc`` (not a tree over slots,
    whose order would depend on the layout Triton picks). ``CQb`` / ``CKb``: the coefficients (zero at dead slots)."""
    om = tl.arange(0, M)
    BV: tl.constexpr = ov.shape[0]
    ubt = _tof(tl.load(ubh + om[:, None] * DV + ov[None, :]))
    if QBUF:
        ubt = ubt * tl.load(ubsh + om).to(tl.float32)[:, None]
    ubt = tl.where((om < n)[:, None], ubt, 0.0)
    cq = tl.load(CQb + om)
    ck = tl.load(CKb + om)
    return _acc(sq, sk, tl.trans(ubt), tl.trans(tl.broadcast_to(cq[:, None], (M, BV))),
                tl.trans(tl.broadcast_to(ck[:, None], (M, BV))), M)


@triton.jit
def _prep(Q, K, sQ, sK, G, BETA, sG, sBETA, A, BT, sA, sBT, ALOG, DTB, IDX,
          GB, sGB, GCUR, sGC, KB, sKB, KBS, sKBS, NPTR, sN,
          QN, KN, CQ, CK, KQ, ALPHA, GS, BS, SCALE,
          DK: tl.constexpr, DKP: tl.constexpr, M: tl.constexpr, BM: tl.constexpr, HV: tl.constexpr,
          HK: tl.constexpr, GQA: tl.constexpr, QBUF: tl.constexpr, QK_PER_KEY_HEAD: tl.constexpr,
          HAS_IDX: tl.constexpr, GATE: tl.constexpr):
    """Buffer side: the two coefficient vectors (``_coefs``, BM slots at a time). One program per (batch row, value
    head). Slots >= n are zeroed."""
    bh = tl.program_id(0)
    ok = tl.arange(0, DKP)
    km = ok < DK
    b = bh // HV
    h = bh % HV
    hk = h // GQA
    seq = _seq_of(IDX, b, HAS_IDX)
    if HAS_IDX:
        if seq <= 0:
            return
    n = tl.load(NPTR + seq * sN) & (M - 1)
    if QK_PER_KEY_HEAD:
        hq = hk
    else:
        hq = h

    q = tl.load(Q + b * sQ + hq * DK + ok, mask=km, other=0.0).to(tl.float32)
    k = tl.load(K + b * sK + hq * DK + ok, mask=km, other=0.0).to(tl.float32)
    # fla's normalisation, epsilon inside the sqrt
    q = tl.div_rn(q, tl.sqrt_rn(tl.sum(q * q) + 1e-6)) * SCALE
    k = tl.div_rn(k, tl.sqrt_rn(tl.sum(k * k) + 1e-6))
    tl.store(QN + bh * DK + ok, q, mask=km)
    tl.store(KN + bh * DK + ok, k, mask=km)
    tl.store(KQ + bh, tl.sum(k * q))
    if GATE:
        # vLLM / fla's gating: g = -exp(A_log) * softplus(a + dt_bias) (threshold 20), beta = sigmoid(b)
        x = tl.load(A + b * sA + h).to(tl.float32) + tl.load(DTB + h).to(tl.float32)
        sp = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        g = -tl.exp(tl.load(ALOG + h).to(tl.float32)) * sp
        beta = tl.sigmoid(tl.load(BT + b * sBT + h).to(tl.float32))
    else:
        g = tl.load(G + b * sG + h).to(tl.float32)
        beta = tl.load(BETA + b * sBETA + h).to(tl.float32)
    tl.store(GS + bh, g)
    tl.store(BS + bh, beta)

    kbase = KB + seq * sKB + hk * M * DK
    ksc = KBS + seq * sKBS + hk * M
    gb = GB + seq * sGB + h * M
    gcur = tl.load(GCUR + seq * sGC + h)
    tl.store(ALPHA + bh, libdevice.exp(gcur))
    for i in range(0, M, BM):
        om = i + tl.arange(0, BM)
        cq, ck = _coefs(kbase, ksc, gb, gcur, q, k, om, n, ok, DK, QBUF)
        tl.store(CQ + bh * M + om, cq)
        tl.store(CK + bh * M + om, ck)


@triton.jit
def _state(S, sS, SC, sSC, QN, KN, UB, sUB, UBS, sUBS, CQ, CK, ALPHA, V, sV, GS, BS, KQ, O, UCUR, NPTR, sN, IDX,
           DK: tl.constexpr, DKP: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr, M: tl.constexpr,
           BITS: tl.constexpr, AXK: tl.constexpr, GROUP: tl.constexpr, QBUF: tl.constexpr, HV: tl.constexpr,
           HAS_IDX: tl.constexpr):
    """The only pass over the stored state, read once in its stored width; never written here (``_state_dots``),
    then the buffer (``_buffer_dots``)."""
    bh = tl.program_id(0)
    iv = tl.program_id(1)
    b = bh // HV
    h = bh % HV
    ov = iv * BV + tl.arange(0, BV)
    seq = _seq_of(IDX, b, HAS_IDX)
    if HAS_IDX:
        if seq <= 0:
            tl.store(O + bh * DV + ov, tl.zeros((BV,), tl.float32).to(O.dtype.element_ty))
            return
    NG: tl.constexpr = (DK + GROUP - 1) // GROUP
    if BITS == 4:
        Sb = S + seq * sS + h * (DV * (DK // 2))
    else:
        Sb = S + seq * sS + h * (DV * DK)
    if AXK:
        SCh = SC + seq * sSC + h * (DV * NG)
    else:
        SCh = SC + seq * sSC + h * ((DV // GROUP) * DK) + (iv * BV // GROUP) * DK
    alpha = tl.load(ALPHA + bh)
    if BITS == 4:
        sq, sk = _state_dots(Sb, SCh, QN + bh * DK, KN + bh * DK, ov, DK, DKP, DV, BV, BITS, AXK, GROUP)
    else:
        ok = tl.arange(0, DKP)
        km = ok < DK
        sq, sk = _state_dots_wide(Sb, SCh, tl.load(QN + bh * DK + ok, mask=km, other=0.0),
                                  tl.load(KN + bh * DK + ok, mask=km, other=0.0), ov, DK, DKP, BV, BITS, AXK, GROUP)
    sq = alpha * sq
    sk = alpha * sk
    n = tl.load(NPTR + seq * sN) & (M - 1)
    sq, sk = _buffer_dots(sq, sk, UB + seq * sUB + h * (M * DV), UBS + seq * sUBS + h * M, CQ + bh * M,
                          CK + bh * M, ov, n, M, DV, QBUF)

    # the current token, by the recurrence itself: u_t = beta (v_t - a_t S_{t-1}^T k_t)
    a = libdevice.exp(tl.load(GS + bh))
    beta = tl.load(BS + bh)
    vt = tl.load(V + b * sV + h * DV + ov).to(tl.float32)
    u = beta * tl.fma(-a, sk, vt)
    o = tl.fma(tl.load(KQ + bh), u, a * sq)
    tl.store(O + bh * DV + ov, o.to(O.dtype.element_ty))
    tl.store(UCUR + bh * DV + ov, u)


@triton.jit
def _append(UCUR, KN, GS, IDX, GB, sGB, GCUR, sGC, KB, sKB, KBS, sKBS, UB, sUB, UBS, sUBS, NPTR, sN,
            DK: tl.constexpr, DV: tl.constexpr, DKP: tl.constexpr, DVP: tl.constexpr, M: tl.constexpr,
            HV: tl.constexpr, HK: tl.constexpr, GQA: tl.constexpr, QBUF: tl.constexpr, HAS_IDX: tl.constexpr):
    """Append (decay, key, delta value) at slot n. The key is written only by its group's first value head."""
    bh = tl.program_id(0)
    ok = tl.arange(0, DKP)
    km = ok < DK
    ov = tl.arange(0, DVP)
    vm = ov < DV
    b = bh // HV
    h = bh % HV
    hk = h // GQA
    seq = _seq_of(IDX, b, HAS_IDX)
    if HAS_IDX:
        if seq <= 0:
            return
    n = tl.load(NPTR + seq * sN) & (M - 1)
    u = tl.load(UCUR + bh * DV + ov, mask=vm, other=0.0)
    ubase = UB + seq * sUB + h * (M * DV) + n * DV
    if QBUF:
        us = _fp16_scale(tl.max(tl.abs(u)), 127)
        tl.store(UBS + seq * sUBS + h * M + n, us.to(tl.float16))
        tl.store(ubase + ov, _round_buf(u, us).to(tl.int8), mask=vm)
    else:
        tl.store(ubase + ov, u.to(UB.dtype.element_ty), mask=vm)
    if h % GQA == 0:
        k = tl.load(KN + bh * DK + ok, mask=km, other=0.0)
        kb = KB + seq * sKB + (hk * M + n) * DK
        if QBUF:
            ks = _fp16_scale(tl.max(tl.abs(k)), 127)
            tl.store(KBS + seq * sKBS + hk * M + n, ks.to(tl.float16))
            tl.store(kb + ok, _round_buf(k, ks).to(tl.int8), mask=km)
        else:
            tl.store(kb + ok, k.to(KB.dtype.element_ty), mask=km)
    g = tl.load(GS + bh)
    gcur = tl.load(GCUR + seq * sGC + h)
    tl.store(GB + seq * sGB + h * M + n, gcur + g)
    tl.store(GCUR + seq * sGC + h, gcur + g)


@triton.jit
def _fold(S, sS, SC, sSC, UB, sUB, UBS, sUBS, KB, sKB, KBS, sKBS, GB, sGB, GCUR, sGC, NPTR, sN, IDX,
          DK: tl.constexpr, DKP: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr, M: tl.constexpr,
          BITS: tl.constexpr, AXK: tl.constexpr, GROUP: tl.constexpr, QBUF: tl.constexpr,
          HV: tl.constexpr, HK: tl.constexpr, GQA: tl.constexpr, HAS_IDX: tl.constexpr):
    """Advance the stored state across the full buffer and re-round it in place. No-op unless n == M - 1 (checked
    per sequence, so in a batch only the sequences whose buffer just filled are folded).

    Runs after the append, so all M slots are live and S = alpha_M S_0 + sum_i (alpha_M / alpha_i) k_i u_i^T is the
    whole fold, current token included. The buffer is not cleared: n wraps to 0 and the masked reads ignore it.
    A program owns BV consecutive value rows and all of d_k, so every scale group (BV == GROUP rows for axis 'v',
    GROUP key columns of one row for axis 'k') lies inside it and re-rounding needs no cross-program reduction.
    """
    bh = tl.program_id(0)
    iv = tl.program_id(1)
    b = bh // HV
    h = bh % HV
    seq = _seq_of(IDX, b, HAS_IDX)
    if HAS_IDX:
        if seq <= 0:
            return
    if (tl.load(NPTR + seq * sN) & (M - 1)) != M - 1:
        return
    ov = iv * BV + tl.arange(0, BV)
    oh = tl.arange(0, DKP // 2)                               # padded key tile: columns >= d_k read 0, never stored
    hm = oh < DK // 2
    hk = h // GQA
    kbase = KB + seq * sKB + hk * (M * DK)
    ksc = KBS + seq * sKBS + hk * M
    QMAX: tl.constexpr = 2 ** (BITS - 1) - 1
    NG: tl.constexpr = (DK + GROUP - 1) // GROUP              # stored key groups (the last may be ragged)
    NGP: tl.constexpr = DKP // GROUP
    if BITS == 4:
        Sb = S + seq * sS + h * (DV * (DK // 2))
    else:
        Sb = S + seq * sS + h * (DV * DK)
    SCk = SC + seq * sSC + h * (DV * NG)
    scb = SC + seq * sSC + h * ((DV // GROUP) * DK) + (iv * BV // GROUP) * DK

    gcur = tl.load(GCUR + seq * sGC + h)
    alpha = libdevice.exp(gcur)
    acce, acco = _load_codes(Sb, ov, oh, DK, BITS)
    if BITS != 16:
        if AXK:
            sce = tl.load(SCk + ov[:, None] * NG + (2 * oh[None, :]) // GROUP, mask=hm[None, :],
                          other=0.0).to(tl.float32)
            sco = sce
        else:
            sce = tl.load(scb + 2 * oh, mask=hm, other=0.0).to(tl.float32)[None, :]
            sco = tl.load(scb + 2 * oh + 1, mask=hm, other=0.0).to(tl.float32)[None, :]
        acce = acce * sce
        acco = acco * sco
    acce = alpha * acce
    acco = alpha * acco

    ub = UB + seq * sUB + h * (M * DV)
    ubs = UBS + seq * sUBS + h * M
    gb = GB + seq * sGB + h * M
    for i in range(0, M):
        dec = libdevice.exp(gcur - tl.load(gb + i))
        u = tl.load(ub + i * DV + ov).to(tl.float32)
        ke = _pad0(tl.load(kbase + i * DK + 2 * oh, mask=hm, other=0).to(tl.float32), hm, DKP != DK)
        ko = _pad0(tl.load(kbase + i * DK + 2 * oh + 1, mask=hm, other=0).to(tl.float32), hm, DKP != DK)
        if QBUF:
            u = u * tl.load(ubs + i).to(tl.float32)
            ks = tl.load(ksc + i).to(tl.float32)
            ke = ke * ks
            ko = ko * ks
        du = dec * u
        acce = tl.fma(du[:, None], ke[None, :], acce)
        acco = tl.fma(du[:, None], ko[None, :], acco)

    if BITS == 16:
        base = Sb + ov[:, None] * DK + 2 * oh[None, :]
        tl.store(base, acce.to(tl.bfloat16), mask=hm[None, :])
        tl.store(base + 1, acco.to(tl.bfloat16), mask=hm[None, :])
        return

    if AXK:
        ae = tl.reshape(acce, (BV, NGP, GROUP // 2))
        ao = tl.reshape(acco, (BV, NGP, GROUP // 2))
        s = _fp16_scale(tl.maximum(tl.max(tl.abs(ae), 2), tl.max(tl.abs(ao), 2)), QMAX)        # [BV, NGP]
        og = tl.arange(0, NGP)
        tl.store(SCk + ov[:, None] * NG + og[None, :], s.to(tl.float16), mask=(og < NG)[None, :])
        ce = tl.reshape(libdevice.rint(tl.div_rn(ae, s[:, :, None])), (BV, DKP // 2))
        co = tl.reshape(libdevice.rint(tl.div_rn(ao, s[:, :, None])), (BV, DKP // 2))
    else:
        se = _fp16_scale(tl.max(tl.abs(acce), 0), QMAX)                                       # [DKP/2]
        so = _fp16_scale(tl.max(tl.abs(acco), 0), QMAX)
        tl.store(scb + 2 * oh, se.to(tl.float16), mask=hm)
        tl.store(scb + 2 * oh + 1, so.to(tl.float16), mask=hm)
        ce = libdevice.rint(tl.div_rn(acce, se[None, :]))
        co = libdevice.rint(tl.div_rn(acco, so[None, :]))
    ce = tl.minimum(tl.maximum(ce, -QMAX * 1.0), QMAX * 1.0)
    co = tl.minimum(tl.maximum(co, -QMAX * 1.0), QMAX * 1.0)
    if BITS == 4:
        by = (ce.to(tl.int32) + 8) | ((co.to(tl.int32) + 8) << 4)
        tl.store(Sb + ov[:, None] * (DK // 2) + oh[None, :], by.to(tl.uint8), mask=hm[None, :])
    else:
        base = Sb + ov[:, None] * DK + 2 * oh[None, :]
        tl.store(base, ce.to(tl.int8), mask=hm[None, :])
        tl.store(base + 1, co.to(tl.int8), mask=hm[None, :])


@triton.jit
def _advance(GCUR, NPTR, BH, M: tl.constexpr, BLK: tl.constexpr):
    """One program, shared counter: after a fold zero the running log decay; then n -> (n + 1) mod M. Last launch
    of the lockstep step."""
    n = tl.load(NPTR)
    if n == M - 1:
        for i in range(0, BH, BLK):
            o = i + tl.arange(0, BLK)
            tl.store(GCUR + o, tl.zeros((BLK,), tl.float32), mask=o < BH)
        tl.store(NPTR, 0)
    else:
        tl.store(NPTR, n + 1)


@triton.jit
def _advance_seq(IDX, GCUR, sGC, NPTR, sN, M: tl.constexpr, HV: tl.constexpr, BHV: tl.constexpr):
    """One program per batch row, per-sequence counter: after this sequence's fold zero its running log decays;
    then its n -> (n + 1) mod M, written to all HV copies of the counter (element h is the fused step's counter
    for value head h; element 0 is the one these kernels read). Padding rows (IDX <= 0) are skipped. Last launch
    of the slot step."""
    b = tl.program_id(0)
    seq = tl.load(IDX + b).to(tl.int64)
    if seq <= 0:
        return
    n = tl.load(NPTR + seq * sN) & (M - 1)
    o = tl.arange(0, BHV)
    if n == M - 1:
        tl.store(GCUR + seq * sGC + o, tl.zeros((BHV,), tl.float32), mask=o < HV)
        tl.store(NPTR + seq * sN + o, tl.zeros((BHV,), tl.int32), mask=o < HV)
    else:
        tl.store(NPTR + seq * sN + o, tl.zeros((BHV,), tl.int32) + (n + 1), mask=o < HV)
