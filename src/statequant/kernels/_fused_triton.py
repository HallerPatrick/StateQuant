"""The packed decode step as ONE Triton launch (the five kernels of ``_fold_triton`` fused).

One program per (batch row, value head) does that head's whole step:

    prep     load and normalise q, k (fla's normalisation), the gates; the buffer coefficients
             c_q[i] = (a_n/a_i) k_i.q, c_k[i] = (a_n/a_i) k_i.k for all slots in one tile           (as ``_prep``)
    state    value blocks of SBV rows: the stored codes read once, S_{t-1}^T q and S_{t-1}^T k from codes + buffer,
             u_t = beta (v_t - a_t S_{t-1}^T k_t), o_t = a_t S_{t-1}^T q + (k_t.q) u_t                  (as ``_state``)
    append   the delta value at slot n, int8 with one absmax scale over the whole d_v row (the program owns the
             row), or bf16; the key (per KEY head) only by the group's first value head; the decay  (as ``_append``)
    fold     if this sequence's buffer is now full: S = a_m S_0 + sum_i (a_m/a_i) k_i u_i^T, re-rounded in place,
             FBV value rows at a time                                                              (as ``_fold``)
    advance  this head's running log decay -> 0 after a fold, and this head's copy of the counter
             n -> (n+1) mod m                                                               (as ``_advance_seq``)

The step is latency bound at small batch (one program per head, a chain of dependent memory round trips), so no load
waits for the counter n unless it must: buffer slots are read unmasked and dead slots (i >= n, possibly garbage)
discarded with ``where``; nothing goes through scratch memory.

Cross-program hazards and why there are none:
  * the key buffer is shared by the GQA value heads of a key head. Only the group's first value head writes slot n.
    The others read slot n only to discard it (c_q, c_k: dead slot) and the fold, which needs slot n = m-1, takes it
    from registers: every program of the group computes the group's buffer key itself (the same inputs and the same
    instructions, so the same bits as the stored one) and rounds it identically.
  * the counter: one copy per value head (``NPTR[seq, h]``), read and written only by that head's program, so no
    program can see it move before reading it, without atomics. The five-kernel step reads copy 0 and writes all
    copies (``_advance_seq``), so both steps keep the copies equal and can be mixed on the same state.
  * everything else (state codes and scales, delta-value buffer, decays) is owned by one value head.

Non-power-of-two head dims: as ``_fold_triton`` (key tiles padded to DKP and masked, ragged key groups); the
state pass runs d_v / SBV in {1, 2, 3, 4} value blocks (d_v 192: three blocks of 64).

Bit-exactness with the five-kernel step: the shared helpers (``_coefs``, ``_state_dots_wide``, ``_grp``), the
fixed slot-by-slot fma order of the buffer term (``_acc`` here, ``_buffer_dots`` there), explicit fma, the fold's
accumulation order, and tiles whose row reductions Triton lowers identically (one row's reduction never depends on how
many rows a tile has); checked in ``tests/test_fused.py`` (outputs, codes, scales, buffers, decays, counters).
"""
from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ._fold_triton import (_acc, _coefs, _fp16_scale, _grp, _half, _load_codes, _nib, _pad0, _round_buf, _seq_of,
                           _state_dots_wide, _tof)


@triton.jit
def _fused_step(Q, K, sQ, sK, V, sV, G, BETA, sG, sBETA, A, BT, sA, sBT, ALOG, DTB, IDX,
                S, sS, SC, sSC, KB, sKB, KBS, sKBS, UB, sUB, UBS, sUBS, GB, sGB, GCUR, sGC, NPTR, sN,
                O, SCALE,
                DK: tl.constexpr, DV: tl.constexpr, DKP: tl.constexpr, DVP: tl.constexpr, SBV: tl.constexpr,
                FBV: tl.constexpr, M: tl.constexpr, BITS: tl.constexpr, AXK: tl.constexpr, GROUP: tl.constexpr,
                QBUF: tl.constexpr, HV: tl.constexpr, HK: tl.constexpr, GQA: tl.constexpr,
                QK_PER_KEY_HEAD: tl.constexpr, HAS_IDX: tl.constexpr, GATE: tl.constexpr):
    bh = tl.program_id(0)
    b = bh // HV
    h = bh % HV
    hk = h // GQA
    ov = tl.arange(0, DVP)
    seq = _seq_of(IDX, b, HAS_IDX)
    if HAS_IDX:
        if seq <= 0:
            tl.store(O + bh * DV + ov, tl.zeros((DVP,), tl.float32).to(O.dtype.element_ty), mask=ov < DV)
            return
    ok = tl.arange(0, DKP)                                          # padded key tile (d_k not a power of two):
    km = ok < DK                                                    # columns >= d_k load 0 and are never stored
    NG: tl.constexpr = (DK + GROUP - 1) // GROUP
    QMAX: tl.constexpr = 2 ** (BITS - 1) - 1
    n = tl.load(NPTR + seq * sN + h) & (M - 1)                      # this head's copy of the counter
    if QK_PER_KEY_HEAD:
        hq = hk
    else:
        hq = h

    # ---- prep: q, k (fla's normalisation), the gates ----
    qb = Q + b * sQ + hq * DK
    kp = K + b * sK + hq * DK
    q = tl.load(qb + ok, mask=km, other=0.0).to(tl.float32)
    k = tl.load(kp + ok, mask=km, other=0.0).to(tl.float32)
    nq = tl.sqrt_rn(tl.sum(q * q) + 1e-6)
    nk = tl.sqrt_rn(tl.sum(k * k) + 1e-6)
    q = tl.div_rn(q, nq) * SCALE
    k = tl.div_rn(k, nk)
    kq = tl.sum(k * q)
    # the key that goes into the (shared, per key head) buffer: the group's first value head's
    if QK_PER_KEY_HEAD or GQA == 1:
        kxp = kp
        kx = k
        nkx = nk
    else:
        kxp = K + b * sK + (hk * GQA) * DK
        kx = tl.load(kxp + ok, mask=km, other=0.0).to(tl.float32)
        nkx = tl.sqrt_rn(tl.sum(kx * kx) + 1e-6)
        kx = tl.div_rn(kx, nkx)
    if GATE:
        x = tl.load(A + b * sA + h).to(tl.float32) + tl.load(DTB + h).to(tl.float32)
        sp = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        g = -tl.exp(tl.load(ALOG + h).to(tl.float32)) * sp
        beta = tl.sigmoid(tl.load(BT + b * sBT + h).to(tl.float32))
    else:
        g = tl.load(G + b * sG + h).to(tl.float32)
        beta = tl.load(BETA + b * sBETA + h).to(tl.float32)

    kbase = KB + seq * sKB + hk * (M * DK)
    ksc = KBS + seq * sKBS + hk * M
    gb = GB + seq * sGB + h * M
    ub = UB + seq * sUB + h * (M * DV)
    ubs = UBS + seq * sUBS + h * M
    gcur = tl.load(GCUR + seq * sGC + h)
    alpha = libdevice.exp(gcur)
    if BITS == 4:
        Sb = S + seq * sS + h * (DV * (DK // 2))
    else:
        Sb = S + seq * sS + h * (DV * DK)
    SCk = SC + seq * sSC + h * (DV * NG)
    SCv = SC + seq * sSC + h * ((DV // GROUP) * DK)
    a = libdevice.exp(g)

    # ---- state: value blocks of SBV rows (codes read once), then the buffer slot by slot ----
    if AXK:
        SCh = SCk
    else:
        SCh = SCv
    # even / odd columns of the normalised q, k by a register split (the same values; recomputing them in the
    # tile's layout would replicate the divisions across its rows)
    qe, qo = tl.split(tl.reshape(q, (DKP // 2, 2)))
    ke, ko = tl.split(tl.reshape(k, (DKP // 2, 2)))
    # c_q, c_k for all slots in one [m, d_k] tile, as ``_prep``
    om = tl.arange(0, M)
    cq, ck = _coefs(kbase, ksc, gb, gcur, q, k, om, n, ok, DK, QBUF)
    NB: tl.constexpr = DV // SBV
    u0 = _block(0, Sb, SCh, q, k, qe, qo, ke, ko, cq, ck, ub, ubs, om, n, alpha, a, beta, kq, V + b * sV + h * DV,
                O + bh * DV, DK, DKP, DV, SBV, M, BITS, AXK, GROUP, QBUF)
    if NB > 1:
        u1 = _block(1, Sb, SCh, q, k, qe, qo, ke, ko, cq, ck, ub, ubs, om, n, alpha, a, beta, kq, V + b * sV + h * DV,
                    O + bh * DV, DK, DKP, DV, SBV, M, BITS, AXK, GROUP, QBUF)
    if NB > 2:
        u2 = _block(2, Sb, SCh, q, k, qe, qo, ke, ko, cq, ck, ub, ubs, om, n, alpha, a, beta, kq, V + b * sV + h * DV,
                    O + bh * DV, DK, DKP, DV, SBV, M, BITS, AXK, GROUP, QBUF)
    if NB > 3:
        tl.static_assert(NB == 4, 'fused step: d_v / block in {1, 2, 3, 4}')
        u3 = _block(3, Sb, SCh, q, k, qe, qo, ke, ko, cq, ck, ub, ubs, om, n, alpha, a, beta, kq, V + b * sV + h * DV,
                    O + bh * DV, DK, DKP, DV, SBV, M, BITS, AXK, GROUP, QBUF)

    # ---- append: decay, delta value (int8: one scale over the whole row), key (group's first head only) ----
    gnew = gcur + g
    tl.store(gb + n, gnew)
    if QBUF:
        amax = tl.max(tl.abs(u0))
        if NB > 1:
            amax = tl.maximum(amax, tl.max(tl.abs(u1)))
        if NB == 3:
            amax = tl.maximum(amax, tl.max(tl.abs(u2)))
        if NB == 4:
            amax = tl.maximum(tl.maximum(amax, tl.max(tl.abs(u2))), tl.max(tl.abs(u3)))
        us = _fp16_scale(amax, 127)
        tl.store(ubs + n, us.to(tl.float16))
        ks = _fp16_scale(tl.max(tl.abs(kx)), 127)
    else:
        us = 1.0
    ubn = ub + n * DV + tl.arange(0, SBV)
    _put_u(ubn, u0, us, QBUF)
    if NB > 1:
        _put_u(ubn + SBV, u1, us, QBUF)
    if NB > 2:
        _put_u(ubn + 2 * SBV, u2, us, QBUF)
    if NB > 3:
        _put_u(ubn + 3 * SBV, u3, us, QBUF)
    if h % GQA == 0:
        kb = kbase + n * DK
        if QBUF:
            tl.store(ksc + n, ks.to(tl.float16))
            tl.store(kb + ok, _round_buf(kx, ks).to(tl.int8), mask=km)
        else:
            tl.store(kb + ok, kx.to(KB.dtype.element_ty), mask=km)

    # ---- fold (buffer full): slots 0..m-2 from memory, slot m-1: own delta value just stored, key from registers ----
    full = n == M - 1
    if full:
        # the state pass's reads of codes / scales and the delta-value store complete before the fold
        tl.debug_barrier()
        if not QBUF:
            ks = 1.0
        _fold_head(Sb, SCk, SCv, kbase, ksc, gb, ub, ubs, kx, ks, us, gnew, DK, DKP, DV, FBV, M, BITS, AXK, GROUP,
                   QBUF)

    # ---- advance: this head's running decay and its copy of the counter ----
    tl.store(GCUR + seq * sGC + h, tl.where(full, 0.0, gnew))
    tl.store(NPTR + seq * sN + h, tl.where(full, 0, n + 1))


@triton.jit
def _state_dots_v(Sb, SCh, qe, qo, ke, ko, ov, DK: tl.constexpr, DKP: tl.constexpr, BV: tl.constexpr,
                  BITS: tl.constexpr, AXK: tl.constexpr, GROUP: tl.constexpr):
    """``_fold_triton._state_dots`` on value rows ``ov`` (BV of them) with q, k at even / odd columns given as
    vectors ([DKP / 2], zero past d_k): the same tiles and the same arithmetic. Axis 'v': scales gathered per row
    (BV / GROUP groups)."""
    oh = tl.arange(0, DKP // 2)
    hm = (oh < DK // 2)[None, :]
    if BITS == 16 or not AXK:
        ce, co = _load_codes(Sb, ov, oh, DK, BITS)
        if BITS == 16:
            sq = tl.sum(ce * qe[None, :], 1) + tl.sum(co * qo[None, :], 1)
            sk = tl.sum(ce * ke[None, :], 1) + tl.sum(co * ko[None, :], 1)
        else:
            sp = SCh + (ov[:, None] // GROUP) * DK + 2 * oh[None, :]
            sce = tl.load(sp, mask=hm, other=0.0).to(tl.float32)
            sco = tl.load(sp + 1, mask=hm, other=0.0).to(tl.float32)
            sq = tl.sum(ce * (qe[None, :] * sce), 1) + tl.sum(co * (qo[None, :] * sco), 1)
            sk = tl.sum(ce * (ke[None, :] * sce), 1) + tl.sum(co * (ko[None, :] * sco), 1)
    else:
        NG: tl.constexpr = (DK + GROUP - 1) // GROUP
        NGP: tl.constexpr = DKP // GROUP
        og = tl.arange(0, GROUP // 2)
        sq = tl.zeros((BV,), tl.float32)
        sk = tl.zeros((BV,), tl.float32)
        for gi in tl.static_range(NG):
            oj = gi * (GROUP // 2) + og
            jm = (oj < DK // 2)[None, :]
            qeg = _grp(qe, gi, NGP)
            qog = _grp(qo, gi, NGP)
            keg = _grp(ke, gi, NGP)
            kog = _grp(ko, gi, NGP)
            if BITS == 4:
                by = tl.load(Sb + ov[:, None] * (DK // 2) + oj[None, :], mask=jm, other=136).to(tl.int32)
                ce = _nib(by & 0xF)
                co = _nib((by >> 4) & 0xF)
            else:
                base = Sb + ov[:, None] * DK + 2 * oj[None, :]
                ce = _tof(tl.load(base, mask=jm, other=0))
                co = _tof(tl.load(base + 1, mask=jm, other=0))
            sc = tl.load(SCh + ov * NG + gi).to(tl.float32)
            sq = tl.fma(tl.sum(tl.fma(co, qog[None, :], ce * qeg[None, :]), 1), sc, sq)
            sk = tl.fma(tl.sum(tl.fma(co, kog[None, :], ce * keg[None, :]), 1), sc, sk)
    return sq, sk


@triton.jit
def _block(iv: tl.constexpr, Sb, SCh, q, k, qe, qo, ke, ko, cq, ck, ub, ubs, om, n, alpha, a, beta, kq, Vh, Oh,
           DK: tl.constexpr, DKP: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr, M: tl.constexpr,
           BITS: tl.constexpr, AXK: tl.constexpr, GROUP: tl.constexpr, QBUF: tl.constexpr):
    """Value rows iv*BV .. +BV of one head: S^T q, S^T k from the codes and the buffer, u and o (o stored);
    returns u. The buffer tile [m, BV] and the coefficients broadcast to it are split into per-slot vectors, so the
    accumulation is ``_buffer_dots``' fixed slot-by-slot fma sequence."""
    ov = iv * BV + tl.arange(0, BV)
    if BITS == 4:
        sq, sk = _state_dots_v(Sb, SCh, qe, qo, ke, ko, ov, DK, DKP, BV, BITS, AXK, GROUP)
    else:
        if BITS == 8 and not AXK and BV == GROUP:
            SCb = SCh + (iv * BV // GROUP) * DK                         # this block's scale row
        else:
            SCb = SCh
        sq, sk = _state_dots_wide(Sb, SCb, q, k, ov, DK, DKP, BV, BITS, AXK, GROUP)
    sq = alpha * sq
    sk = alpha * sk
    ubt = _tof(tl.load(ub + om[:, None] * DV + ov[None, :]))
    if QBUF:
        ubt = ubt * tl.load(ubs + om).to(tl.float32)[:, None]
    ubt = tl.where((om < n)[:, None], ubt, 0.0)
    sq, sk = _acc(sq, sk, tl.trans(ubt), tl.trans(tl.broadcast_to(cq[:, None], (M, BV))),
                  tl.trans(tl.broadcast_to(ck[:, None], (M, BV))), M)
    vt = tl.load(Vh + ov).to(tl.float32)
    u = beta * tl.fma(-a, sk, vt)
    o = tl.fma(kq, u, a * sq)
    tl.store(Oh + ov, o.to(Oh.dtype.element_ty))
    return u


@triton.jit
def _put_u(p, u, us, QBUF: tl.constexpr):
    if QBUF:
        tl.store(p, _round_buf(u, us).to(tl.int8))
    else:
        tl.store(p, u.to(p.dtype.element_ty))


@triton.jit
def _fold_head(Sb, SCk, SCv, kbase, ksc, gb, ub, ubs, kx, ks, us, gnew, DK: tl.constexpr, DKP: tl.constexpr,
               DV: tl.constexpr, FBV: tl.constexpr, M: tl.constexpr, BITS: tl.constexpr, AXK: tl.constexpr,
               GROUP: tl.constexpr, QBUF: tl.constexpr):
    """S = a_m S_0 + sum_i (a_m/a_i) k_i u_i^T over the full buffer, re-rounded in place, FBV value rows at a time
    (``_fold``'s arithmetic and accumulation order). Slots 0..m-2 from memory, slot m-1 (this token): the delta
    value as just stored, the key from registers (the group's first value head may still be writing it).
    4-bit codes: even / odd key columns (the nibbles); 8 / 16 bit: whole rows."""
    NG: tl.constexpr = (DK + GROUP - 1) // GROUP                   # stored key groups (the last may be ragged)
    NGP: tl.constexpr = DKP // GROUP                               # groups of the padded tile
    og = tl.arange(0, NGP)
    gm = (og < NG)[None, :]
    QMAX: tl.constexpr = 2 ** (BITS - 1) - 1
    alphan = libdevice.exp(gnew)
    # per-slot factors, slot m-1 (this token) from registers: decay a_m/a_i, delta-value scale, key scale
    om = tl.arange(0, M)
    last = om == M - 1
    dec = libdevice.exp(gnew - tl.where(last, gnew, tl.load(gb + om, mask=om < M - 1, other=0.0)))
    if QBUF:
        usv = tl.where(last, us, tl.load(ubs + om, mask=om < M - 1, other=0.0).to(tl.float32))
        ksv = tl.load(ksc + om, mask=om < M - 1, other=0.0).to(tl.float32)
    if BITS == 4:
        oh = tl.arange(0, DKP // 2)
        hm = (oh < DK // 2)[None, :]
        kxe, kxo = tl.split(tl.reshape(kx, (DKP // 2, 2)))
        if QBUF:
            kre = _round_buf(kxe, ks) * ks                            # = stored int8 * fp16 scale, exactly
            kro = _round_buf(kxo, ks) * ks
        else:
            kre = kxe.to(tl.bfloat16).to(tl.float32)
            kro = kxo.to(tl.bfloat16).to(tl.float32)
        Ke = tl.load(kbase + om[:, None] * DK + 2 * oh[None, :], mask=(om < M - 1)[:, None] & hm, other=0.0)
        Ko = tl.load(kbase + om[:, None] * DK + 2 * oh[None, :] + 1, mask=(om < M - 1)[:, None] & hm, other=0.0)
        Ke = _pad0(_tof(Ke), hm, DKP // 2 != DK // 2)
        Ko = _pad0(_tof(Ko), hm, DKP // 2 != DK // 2)
        if QBUF:
            Ke = Ke * ksv[:, None]
            Ko = Ko * ksv[:, None]
        KeT = tl.trans(tl.where(last[:, None], kre[None, :], Ke))
        KoT = tl.trans(tl.where(last[:, None], kro[None, :], Ko))
        for iv in range(0, DV // FBV):                                 # loops, not unrolled: registers
            fv = iv * FBV + tl.arange(0, FBV)
            acce, acco = _load_codes(Sb, fv, oh, DK, BITS)
            if AXK:                                                      # [FBV, NGP] scales, broadcast per group
                fsc = tl.load(SCk + fv[:, None] * NG + og[None, :], mask=gm, other=0.0).to(tl.float32)[:, :, None]
                acce = tl.reshape(tl.reshape(acce, (FBV, NGP, GROUP // 2)) * fsc, (FBV, DKP // 2))
                acco = tl.reshape(tl.reshape(acco, (FBV, NGP, GROUP // 2)) * fsc, (FBV, DKP // 2))
            else:
                fsb = SCv + (iv * FBV // GROUP) * DK                      # FBV == GROUP
                acce = acce * tl.load(fsb + 2 * oh, mask=oh < DK // 2, other=0.0).to(tl.float32)[None, :]
                acco = acco * tl.load(fsb + 2 * oh + 1, mask=oh < DK // 2, other=0.0).to(tl.float32)[None, :]
            acce = alphan * acce
            acco = alphan * acco
            U = _tof(tl.load(ub + om[:, None] * DV + fv[None, :]))           # slot m-1: as just stored
            if QBUF:
                U = U * usv[:, None]
            acce, acco = _fold_acc(acce, acco, tl.trans(dec[:, None] * U), KeT, KoT, M, True)
            if AXK:
                ae = tl.reshape(acce, (FBV, NGP, GROUP // 2))
                ao = tl.reshape(acco, (FBV, NGP, GROUP // 2))
                sc = _fp16_scale(tl.maximum(tl.max(tl.abs(ae), 2), tl.max(tl.abs(ao), 2)), QMAX)
                tl.store(SCk + fv[:, None] * NG + og[None, :], sc.to(tl.float16), mask=gm)
                fce = tl.reshape(_rint_quot(ae, sc[:, :, None], tl.div_rn(1.0, sc)[:, :, None]), (FBV, DKP // 2))
                fco = tl.reshape(_rint_quot(ao, sc[:, :, None], tl.div_rn(1.0, sc)[:, :, None]), (FBV, DKP // 2))
            else:
                se = _fp16_scale(tl.max(tl.abs(acce), 0), QMAX)
                so = _fp16_scale(tl.max(tl.abs(acco), 0), QMAX)
                tl.store(fsb + 2 * oh, se.to(tl.float16), mask=oh < DK // 2)
                tl.store(fsb + 2 * oh + 1, so.to(tl.float16), mask=oh < DK // 2)
                fce = _rint_quot(acce, se[None, :], tl.div_rn(1.0, se)[None, :])
                fco = _rint_quot(acco, so[None, :], tl.div_rn(1.0, so)[None, :])
            fce = tl.minimum(tl.maximum(fce, -QMAX * 1.0), QMAX * 1.0)
            fco = tl.minimum(tl.maximum(fco, -QMAX * 1.0), QMAX * 1.0)
            by = _pack4(fce, fco)
            tl.store(Sb + fv[:, None] * (DK // 2) + oh[None, :], by.to(tl.uint8), mask=hm)
    else:
        ok = tl.arange(0, DKP)
        km = (ok < DK)[None, :]
        if QBUF:
            kr = _round_buf(kx, ks) * ks
        else:
            kr = kx.to(tl.bfloat16).to(tl.float32)
        Kf = _pad0(_tof(tl.load(kbase + om[:, None] * DK + ok[None, :], mask=(om < M - 1)[:, None] & km, other=0.0)),
                   km, DKP != DK)
        if QBUF:
            Kf = Kf * ksv[:, None]
        KfT = tl.trans(tl.where(last[:, None], kr[None, :], Kf))
        for iv in range(0, DV // FBV):
            fv = iv * FBV + tl.arange(0, FBV)
            base = Sb + fv[:, None] * DK + ok[None, :]
            acc = _pad0(_tof(tl.load(base, mask=km, other=0)), km, DKP != DK)
            if BITS == 8:
                if AXK:
                    fsc = tl.load(SCk + fv[:, None] * NG + og[None, :], mask=gm, other=0.0).to(tl.float32)[:, :, None]
                    acc = tl.reshape(tl.reshape(acc, (FBV, NGP, GROUP)) * fsc, (FBV, DKP))
                else:
                    fsb = SCv + (iv * FBV // GROUP) * DK                  # FBV == GROUP
                    acc = acc * tl.load(fsb + ok, mask=ok < DK, other=0.0).to(tl.float32)[None, :]
            acc = alphan * acc
            U = _tof(tl.load(ub + om[:, None] * DV + fv[None, :]))           # slot m-1: as just stored
            if QBUF:
                U = U * usv[:, None]
            acc, unused = _fold_acc(acc, acc, tl.trans(dec[:, None] * U), KfT, KfT, M, False)
            if BITS == 16:
                tl.store(base, acc.to(tl.bfloat16), mask=km)
            else:
                if AXK:
                    a3 = tl.reshape(acc, (FBV, NGP, GROUP))
                    sc = _fp16_scale(tl.max(tl.abs(a3), 2), QMAX)
                    tl.store(SCk + fv[:, None] * NG + og[None, :], sc.to(tl.float16), mask=gm)
                    fc = tl.reshape(_rint_quot(a3, sc[:, :, None], tl.div_rn(1.0, sc)[:, :, None]), (FBV, DKP))
                else:
                    sc = _fp16_scale(tl.max(tl.abs(acc), 0), QMAX)
                    tl.store(fsb + ok, sc.to(tl.float16), mask=ok < DK)
                    fc = _rint_quot(acc, sc[None, :], tl.div_rn(1.0, sc)[None, :])
                fc = tl.minimum(tl.maximum(fc, -QMAX * 1.0), QMAX * 1.0)
                tl.store(base, _f2i8(fc), mask=km)

@triton.jit
def _fa1(ae, ao, D, E, F, TWO: tl.constexpr):
    R: tl.constexpr = D.shape[0]
    C: tl.constexpr = E.shape[0]
    d = tl.reshape(D, (R,))
    ae = tl.fma(d[:, None], tl.reshape(E, (C,))[None, :], ae)
    if TWO:
        ao = tl.fma(d[:, None], tl.reshape(F, (C,))[None, :], ao)
    return ae, ao


@triton.jit
def _fa2(ae, ao, D, E, F, TWO: tl.constexpr):
    D0, D1 = _half(D)
    E0, E1 = _half(E)
    F0, F1 = _half(F)
    ae, ao = _fa1(ae, ao, D0, E0, F0, TWO)
    return _fa1(ae, ao, D1, E1, F1, TWO)


@triton.jit
def _fa4(ae, ao, D, E, F, TWO: tl.constexpr):
    D0, D1 = _half(D)
    E0, E1 = _half(E)
    F0, F1 = _half(F)
    ae, ao = _fa2(ae, ao, D0, E0, F0, TWO)
    return _fa2(ae, ao, D1, E1, F1, TWO)


@triton.jit
def _fa8(ae, ao, D, E, F, TWO: tl.constexpr):
    D0, D1 = _half(D)
    E0, E1 = _half(E)
    F0, F1 = _half(F)
    ae, ao = _fa4(ae, ao, D0, E0, F0, TWO)
    return _fa4(ae, ao, D1, E1, F1, TWO)


@triton.jit
def _fa16(ae, ao, D, E, F, TWO: tl.constexpr):
    D0, D1 = _half(D)
    E0, E1 = _half(E)
    F0, F1 = _half(F)
    ae, ao = _fa8(ae, ao, D0, E0, F0, TWO)
    return _fa8(ae, ao, D1, E1, F1, TWO)


@triton.jit
def _fa32(ae, ao, D, E, F, TWO: tl.constexpr):
    D0, D1 = _half(D)
    E0, E1 = _half(E)
    F0, F1 = _half(F)
    ae, ao = _fa16(ae, ao, D0, E0, F0, TWO)
    return _fa16(ae, ao, D1, E1, F1, TWO)


@triton.jit
def _fold_acc(ae, ao, D, E, F, M: tl.constexpr, TWO: tl.constexpr):
    """ae = fma(d_i[:, None], e_i[None, :], ae) (and ao with f_i) for slots i = 0 .. M-1 in order, from [R, M] /
    [C, M] tiles (column i = slot i): ``_fold``'s accumulation, with each tile loaded once instead of per slot."""
    if M == 1:
        return _fa1(ae, ao, D, E, F, TWO)
    elif M == 2:
        return _fa2(ae, ao, D, E, F, TWO)
    elif M == 4:
        return _fa4(ae, ao, D, E, F, TWO)
    elif M == 8:
        return _fa8(ae, ao, D, E, F, TWO)
    elif M == 16:
        return _fa16(ae, ao, D, E, F, TWO)
    else:
        tl.static_assert(M == 32, 'fused step: m <= 32')
        return _fa32(ae, ao, D, E, F, TWO)


@triton.jit
def _rint_quot(x, s, r):
    """rint(div_rn(x, s)) bit for bit, for s > 0 an fp16 value and r = div_rn(1, s), mostly without the division
    (whose IEEE slow-path branch makes it the fold's most expensive operation).

    q = x r is within |q| 2^-23 of x / s, and RN(x / s) within |q| 2^-24 of it, so the two round to the same integer
    unless q lies within ~|q| 2^-22 of a half-integer. Lanes within max(|q|, 1) 2^-20 of one are flagged; if a tile
    has any (a few percent of the fold's tiles), it computes the exact rint(div_rn(x, s)) for them. Checked against
    rint(div_rn(x, s)) on 1e10 samples concentrated at ties (no mismatch)."""
    q = x * r
    c = (q + 12582912.0) - 12582912.0                     # rint (ties to even) for |q| < 2^22, at full rate
    # q - c is exact; flag |q - c| >= 1/2 - max(|q|, 1) 2^-20 (within the margin of a half-integer)
    near = tl.abs(q - c) >= tl.fma(tl.maximum(tl.abs(q), 1.0), -9.5367431640625e-07, 0.5)
    if tl.max(tl.reshape(near.to(tl.int32), (near.numel,))) > 0:
        c = tl.where(near, libdevice.rint(tl.div_rn(x, s)), c)
    return c


@triton.jit
def _pack4(ce, co):
    """Integral fp32 codes in [-7, 7] at even / odd key columns -> bytes (ce + 8) | (co + 8) << 4, via the float
    bits of 2^23 + 8 + c (no float->int conversion)."""
    be = (ce + 8388616.0).to(tl.int32, bitcast=True) & 0xF
    bo = (co + 8388616.0).to(tl.int32, bitcast=True) & 0xF
    return (be | (bo << 4)).to(tl.uint8)


@triton.jit
def _f2i8(c):
    """Integral fp32 codes in [-127, 127] -> int8, via the float bits of 1.5 * 2^23 + c."""
    return ((c + 12582912.0).to(tl.int32, bitcast=True) - 0x4B400000).to(tl.int8)
