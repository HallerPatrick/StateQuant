"""The packed decode step of a KDA layer as ONE Triton launch (one program per (batch row, head)).

KDA in the delta-buffer picture (``kda.py``): per head, with the decay a_t = exp(g_t) per key channel and the
buffer of (cumulative log decay G_i [d_k], normalised key k_i, delta value u_i) since the last fold,

    A     = exp(G_n),  G_n = G_{n-1} + g_n                     (running decay incl. this token, per key channel)
    c_x[i] = sum_j k_i[j] x[j] exp(G_n[j] - G_i[j])           (x = q, k; slots i < n; one [m, d_k] tile)
    S'^T x = S_0^T (A * x) + sum_i c_x[i] u_i                 (S' = diag(a_n) S_{n-1})
    u_n   = beta (v - S'^T k),   o_n = S'^T q + (k.q) u_n
    fold (every m tokens):  S = diag(A) S_0 + sum_i u_i (exp(G_n - G_i) * k_i)^T,  re-rounded in place

One program owns one head's whole state ([d_v, d_k], value-major like vLLM's KDA recurrent state) and does

    prep     load and normalise q, k (fla's normalisation), the gate g [d_k] and beta, A, the buffer coefficients
    state    value blocks of SBV rows (up to 4 blocks, kept in registers): the stored codes read once against A * q
             and A * k, the buffer term, u and o (o stored)
    append   u at slot n (int8 with one absmax scale over the whole d_v row, or bf16), the key, G_n
    fold     if this sequence's buffer is now full: FBV value rows at a time, slot m-1's key and decay from registers,
             re-rounding as ``fold.pack_vk``
    advance  this head's running decay (a [d_k] vector) and its copy of the counter

No cross-program hazards: KDA has no GQA, so every buffer entry, decay and counter copy is owned by one program.
Dead buffer slots (i >= n, possibly garbage) are loaded unmasked and discarded with ``where``.

The arithmetic matches ``kda.ReferenceKDAState`` (rounding exactly as ``fold.pack_vk``: IEEE divisions, half-to-even,
against the fp16-rounded scale); fp32 sums may associate differently (tests: ``tests/test_kda_kernels.py``).
"""
from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ._fold_triton import _fp16_scale, _nib, _round_buf, _seq_of, _tof
from ._fused_triton import _f2i8, _fold_acc, _pack4, _rint_quot


@triton.jit
def _kda_step(Q, sQ, K, sK, V, sV, G, sG, BETA, sBETA, IDX,
              S, sS, SC, sSC, KB, sKB, KBS, sKBS, UB, sUB, UBS, sUBS, GB, sGB, GCUR, sGC, NPTR, sN,
              O, sO, SCALE,
              DK: tl.constexpr, DV: tl.constexpr, SBV: tl.constexpr, FBV: tl.constexpr, M: tl.constexpr,
              BITS: tl.constexpr, AXK: tl.constexpr, GROUP: tl.constexpr, QBUF: tl.constexpr, HV: tl.constexpr,
              HAS_IDX: tl.constexpr):
    bh = tl.program_id(0)
    b = bh // HV
    h = bh % HV
    seq = _seq_of(IDX, b, HAS_IDX)
    if HAS_IDX:
        if seq <= 0:
            tl.store(O + b * sO + h * DV + tl.arange(0, DV), tl.zeros((DV,), tl.float32).to(O.dtype.element_ty))
            return
    ok = tl.arange(0, DK)
    n = tl.load(NPTR + seq * sN + h) & (M - 1)                      # this head's copy of the counter

    # ---- prep: q, k (fla's normalisation), gate, beta, running decay ----
    q = tl.load(Q + b * sQ + h * DK + ok).to(tl.float32)
    k = tl.load(K + b * sK + h * DK + ok).to(tl.float32)
    nq = tl.sqrt_rn(tl.sum(q * q) + 1e-6)
    nk = tl.sqrt_rn(tl.sum(k * k) + 1e-6)
    q = tl.div_rn(q, nq) * SCALE
    k = tl.div_rn(k, nk)
    kq = tl.sum(k * q)
    g = tl.load(G + b * sG + h * DK + ok).to(tl.float32)
    beta = tl.load(BETA + b * sBETA + h).to(tl.float32)
    gc = GCUR + seq * sGC + h * DK
    gnew = tl.load(gc + ok) + g
    A = libdevice.exp(gnew)
    qa = A * q
    ka = A * k

    kbase = KB + seq * sKB + h * (M * DK)
    ksc = KBS + seq * sKBS + h * M
    gb = GB + seq * sGB + h * (M * DK)
    ub = UB + seq * sUB + h * (M * DV)
    ubs = UBS + seq * sUBS + h * M
    NG: tl.constexpr = DK // GROUP
    if BITS == 4:
        Sb = S + seq * sS + h * (DV * (DK // 2))
    else:
        Sb = S + seq * sS + h * (DV * DK)
    if AXK:
        SCh = SC + seq * sSC + h * (DV * NG)
    else:
        SCh = SC + seq * sSC + h * ((DV // GROUP) * DK)

    # c_x[i] = sum_j k_i[j] x[j] exp(G_n[j] - G_i[j]), slots i < n (dead slots, possibly garbage, discarded)
    om = tl.arange(0, M)
    live = om < n
    kb = _tof(tl.load(kbase + om[:, None] * DK + ok[None, :]))
    if QBUF:
        kb = kb * tl.load(ksc + om).to(tl.float32)[:, None]
    kd = tl.where(live[:, None], kb * libdevice.exp(gnew[None, :] - tl.load(gb + om[:, None] * DK + ok[None, :])),
                  0.0)
    cq = tl.sum(kd * q[None, :], 1)
    ck = tl.sum(kd * k[None, :], 1)

    # ---- state: value blocks of SBV rows (codes read once), u kept in registers for the append ----
    qae, qao = tl.split(tl.reshape(qa, (DK // 2, 2)))
    kae, kao = tl.split(tl.reshape(ka, (DK // 2, 2)))
    NB: tl.constexpr = DV // SBV
    Vh = V + b * sV + h * DV
    Oh = O + b * sO + h * DV
    u0 = _kblock(0, Sb, SCh, qa, ka, qae, qao, kae, kao, cq, ck, ub, ubs, om, live, beta, kq, Vh, Oh,
                 DK, DV, SBV, M, BITS, AXK, GROUP, QBUF)
    if NB > 1:
        u1 = _kblock(1, Sb, SCh, qa, ka, qae, qao, kae, kao, cq, ck, ub, ubs, om, live, beta, kq, Vh, Oh,
                     DK, DV, SBV, M, BITS, AXK, GROUP, QBUF)
    if NB > 2:
        tl.static_assert(NB == 4, 'KDA step: d_v / block in {1, 2, 4}')
        u2 = _kblock(2, Sb, SCh, qa, ka, qae, qao, kae, kao, cq, ck, ub, ubs, om, live, beta, kq, Vh, Oh,
                     DK, DV, SBV, M, BITS, AXK, GROUP, QBUF)
        u3 = _kblock(3, Sb, SCh, qa, ka, qae, qao, kae, kao, cq, ck, ub, ubs, om, live, beta, kq, Vh, Oh,
                     DK, DV, SBV, M, BITS, AXK, GROUP, QBUF)

    # ---- append: decay vector, delta value (int8: one scale over the whole row), key ----
    tl.store(gb + n * DK + ok, gnew)
    if QBUF:
        amax = tl.max(tl.abs(u0))
        if NB > 1:
            amax = tl.maximum(amax, tl.max(tl.abs(u1)))
        if NB > 2:
            amax = tl.maximum(tl.maximum(amax, tl.max(tl.abs(u2))), tl.max(tl.abs(u3)))
        us = _fp16_scale(amax, 127)
        tl.store(ubs + n, us.to(tl.float16))
        ks = _fp16_scale(tl.max(tl.abs(k)), 127)
        tl.store(ksc + n, ks.to(tl.float16))
        tl.store(kbase + n * DK + ok, _round_buf(k, ks).to(tl.int8))
    else:
        us = 1.0
        ks = 1.0
        tl.store(kbase + n * DK + ok, k.to(KB.dtype.element_ty))
    ubn = ub + n * DV + tl.arange(0, SBV)
    _put(ubn, u0, us, QBUF)
    if NB > 1:
        _put(ubn + SBV, u1, us, QBUF)
    if NB > 2:
        _put(ubn + 2 * SBV, u2, us, QBUF)
        _put(ubn + 3 * SBV, u3, us, QBUF)

    # ---- fold ----
    full = n == M - 1
    if full:
        tl.debug_barrier()                  # the state pass's reads and the buffer stores complete before the fold
        _kda_fold(Sb, SCh, kbase, ksc, gb, ub, ubs, k, ks, gnew, A, DK, DV, FBV, M, BITS, AXK, GROUP, QBUF)

    # ---- advance ----
    tl.store(gc + ok, tl.where(full, 0.0, gnew))
    tl.store(NPTR + seq * sN + h, tl.where(full, 0, n + 1))


@triton.jit
def _put(p, u, us, QBUF: tl.constexpr):
    if QBUF:
        tl.store(p, _round_buf(u, us).to(tl.int8))
    else:
        tl.store(p, u.to(p.dtype.element_ty))


@triton.jit
def _dots(Sb, SCh, qa, ka, qae, qao, kae, kao, fv, DK: tl.constexpr, SBV: tl.constexpr, BITS: tl.constexpr,
          AXK: tl.constexpr, GROUP: tl.constexpr):
    """S_0^T (A q), S_0^T (A k) on value rows fv: codes (and scales) read once, both vectors against them."""
    NG: tl.constexpr = DK // GROUP
    og = tl.arange(0, NG)
    if BITS == 4:
        oh = tl.arange(0, DK // 2)
        by = tl.load(Sb + fv[:, None] * (DK // 2) + oh[None, :]).to(tl.int32)
        ce = _nib(by & 0xF)
        co = _nib((by >> 4) & 0xF)
        if AXK:                              # per key group: (codes . x) summed, times the group's scale
            sc = tl.load(SCh + fv[:, None] * NG + og[None, :]).to(tl.float32)
            pq = tl.reshape(ce * qae[None, :] + co * qao[None, :], (SBV, NG, GROUP // 2))
            pk = tl.reshape(ce * kae[None, :] + co * kao[None, :], (SBV, NG, GROUP // 2))
            sq = tl.sum(tl.sum(pq, 2) * sc, 1)
            sk = tl.sum(tl.sum(pk, 2) * sc, 1)
        else:
            sp = SCh + (fv[:, None] // GROUP) * DK + 2 * oh[None, :]
            ce = ce * tl.load(sp).to(tl.float32)
            co = co * tl.load(sp + 1).to(tl.float32)
            sq = tl.sum(ce * qae[None, :] + co * qao[None, :], 1)
            sk = tl.sum(ce * kae[None, :] + co * kao[None, :], 1)
    else:
        ok = tl.arange(0, DK)
        cs = _tof(tl.load(Sb + fv[:, None] * DK + ok[None, :]))
        if BITS == 8 and AXK:
            sc = tl.load(SCh + fv[:, None] * NG + og[None, :]).to(tl.float32)
            sq = tl.sum(tl.sum(tl.reshape(cs * qa[None, :], (SBV, NG, GROUP)), 2) * sc, 1)
            sk = tl.sum(tl.sum(tl.reshape(cs * ka[None, :], (SBV, NG, GROUP)), 2) * sc, 1)
        else:
            if BITS == 8:
                cs = cs * tl.load(SCh + (fv[:, None] // GROUP) * DK + ok[None, :]).to(tl.float32)
            sq = tl.sum(cs * qa[None, :], 1)
            sk = tl.sum(cs * ka[None, :], 1)
    return sq, sk


@triton.jit
def _kblock(iv: tl.constexpr, Sb, SCh, qa, ka, qae, qao, kae, kao, cq, ck, ub, ubs, om, live, beta, kq, Vh, Oh,
            DK: tl.constexpr, DV: tl.constexpr, SBV: tl.constexpr, M: tl.constexpr, BITS: tl.constexpr,
            AXK: tl.constexpr, GROUP: tl.constexpr, QBUF: tl.constexpr):
    """Value rows iv*SBV .. +SBV of one head: S'^T q, S'^T k from the codes and the buffer, u and o (o stored);
    returns u."""
    fv = iv * SBV + tl.arange(0, SBV)
    sq, sk = _dots(Sb, SCh, qa, ka, qae, qao, kae, kao, fv, DK, SBV, BITS, AXK, GROUP)
    ubt = _tof(tl.load(ub + om[:, None] * DV + fv[None, :]))
    if QBUF:
        ubt = ubt * tl.load(ubs + om).to(tl.float32)[:, None]
    ubt = tl.where(live[:, None], ubt, 0.0)
    sq = sq + tl.sum(ubt * cq[:, None], 0)
    sk = sk + tl.sum(ubt * ck[:, None], 0)
    u = beta * (tl.load(Vh + fv).to(tl.float32) - sk)
    tl.store(Oh + fv, tl.fma(kq, u, sq).to(Oh.dtype.element_ty))
    return u


@triton.jit
def _kda_fold(Sb, SCh, kbase, ksc, gb, ub, ubs, kx, ks, gnew, A, DK: tl.constexpr, DV: tl.constexpr,
              FBV: tl.constexpr, M: tl.constexpr, BITS: tl.constexpr, AXK: tl.constexpr, GROUP: tl.constexpr,
              QBUF: tl.constexpr):
    """S = diag(A) S_0 + sum_i u_i (exp(G_n - G_i) * k_i)^T over the full buffer, re-rounded in place, FBV value rows
    at a time. Slots 0..m-2 from memory; slot m-1: the value as just stored by this program, the key and its decay
    (ratio 1) from registers. 4-bit codes: even / odd key columns (the nibbles); 8 / 16 bit: whole rows."""
    NG: tl.constexpr = DK // GROUP
    QMAX: tl.constexpr = 2 ** (BITS - 1) - 1
    ok = tl.arange(0, DK)
    om = tl.arange(0, M)
    last = om == M - 1
    old = om < M - 1
    og = tl.arange(0, NG)
    gt = tl.load(gb + om[:, None] * DK + ok[None, :], mask=old[:, None], other=0.0)
    dec = libdevice.exp(gnew[None, :] - tl.where(last[:, None], gnew[None, :], gt))          # [M, DK]
    if QBUF:
        usv = tl.load(ubs + om).to(tl.float32)
        ksv = tl.load(ksc + om, mask=old, other=0.0).to(tl.float32)
        kr = _round_buf(kx, ks) * ks                                  # = stored int8 * fp16 scale, exactly
    else:
        kr = kx.to(tl.bfloat16).to(tl.float32)
    Kf = _tof(tl.load(kbase + om[:, None] * DK + ok[None, :], mask=old[:, None], other=0))
    if QBUF:
        Kf = Kf * ksv[:, None]
    Kd = tl.where(last[:, None], kr[None, :], Kf) * dec                                       # [M, DK]
    if BITS == 4:
        oh = tl.arange(0, DK // 2)
        Kde, Kdo = tl.split(tl.reshape(Kd, (M, DK // 2, 2)))
        KeT = tl.trans(Kde)
        KoT = tl.trans(Kdo)
        Ae, Ao = tl.split(tl.reshape(A, (DK // 2, 2)))
        for iv in range(0, DV // FBV):
            fv = iv * FBV + tl.arange(0, FBV)
            by = tl.load(Sb + fv[:, None] * (DK // 2) + oh[None, :]).to(tl.int32)
            acce = _nib(by & 0xF)
            acco = _nib((by >> 4) & 0xF)
            if AXK:
                fsc = tl.load(SCh + fv[:, None] * NG + og[None, :]).to(tl.float32)[:, :, None]
                acce = tl.reshape(tl.reshape(acce, (FBV, NG, GROUP // 2)) * fsc, (FBV, DK // 2))
                acco = tl.reshape(tl.reshape(acco, (FBV, NG, GROUP // 2)) * fsc, (FBV, DK // 2))
            else:
                fsb = SCh + (iv * FBV // GROUP) * DK                      # FBV == GROUP
                acce = acce * tl.load(fsb + 2 * oh).to(tl.float32)[None, :]
                acco = acco * tl.load(fsb + 2 * oh + 1).to(tl.float32)[None, :]
            acce = acce * Ae[None, :]
            acco = acco * Ao[None, :]
            U = _tof(tl.load(ub + om[:, None] * DV + fv[None, :]))
            if QBUF:
                U = U * usv[:, None]
            acce, acco = _fold_acc(acce, acco, tl.trans(U), KeT, KoT, M, True)
            if AXK:
                ae = tl.reshape(acce, (FBV, NG, GROUP // 2))
                ao = tl.reshape(acco, (FBV, NG, GROUP // 2))
                sc = _fp16_scale(tl.maximum(tl.max(tl.abs(ae), 2), tl.max(tl.abs(ao), 2)), QMAX)
                tl.store(SCh + fv[:, None] * NG + og[None, :], sc.to(tl.float16))
                fce = tl.reshape(_rint_quot(ae, sc[:, :, None], tl.div_rn(1.0, sc)[:, :, None]), (FBV, DK // 2))
                fco = tl.reshape(_rint_quot(ao, sc[:, :, None], tl.div_rn(1.0, sc)[:, :, None]), (FBV, DK // 2))
            else:
                se = _fp16_scale(tl.max(tl.abs(acce), 0), QMAX)
                so = _fp16_scale(tl.max(tl.abs(acco), 0), QMAX)
                tl.store(fsb + 2 * oh, se.to(tl.float16))
                tl.store(fsb + 2 * oh + 1, so.to(tl.float16))
                fce = _rint_quot(acce, se[None, :], tl.div_rn(1.0, se)[None, :])
                fco = _rint_quot(acco, so[None, :], tl.div_rn(1.0, so)[None, :])
            fce = tl.minimum(tl.maximum(fce, -QMAX * 1.0), QMAX * 1.0)
            fco = tl.minimum(tl.maximum(fco, -QMAX * 1.0), QMAX * 1.0)
            tl.store(Sb + fv[:, None] * (DK // 2) + oh[None, :], _pack4(fce, fco))
    else:
        KdT = tl.trans(Kd)
        for iv in range(0, DV // FBV):
            fv = iv * FBV + tl.arange(0, FBV)
            base = Sb + fv[:, None] * DK + ok[None, :]
            acc = _tof(tl.load(base))
            if BITS == 8:
                if AXK:
                    fsc = tl.load(SCh + fv[:, None] * NG + og[None, :]).to(tl.float32)[:, :, None]
                    acc = tl.reshape(tl.reshape(acc, (FBV, NG, GROUP)) * fsc, (FBV, DK))
                else:
                    fsb = SCh + (iv * FBV // GROUP) * DK                  # FBV == GROUP
                    acc = acc * tl.load(fsb + ok).to(tl.float32)[None, :]
            acc = acc * A[None, :]
            U = _tof(tl.load(ub + om[:, None] * DV + fv[None, :]))
            if QBUF:
                U = U * usv[:, None]
            acc, unused = _fold_acc(acc, acc, tl.trans(U), KdT, KdT, M, False)
            if BITS == 16:
                tl.store(base, acc.to(tl.bfloat16))
            else:
                if AXK:
                    a3 = tl.reshape(acc, (FBV, NG, GROUP))
                    sc = _fp16_scale(tl.max(tl.abs(a3), 2), QMAX)
                    tl.store(SCh + fv[:, None] * NG + og[None, :], sc.to(tl.float16))
                    fc = tl.reshape(_rint_quot(a3, sc[:, :, None], tl.div_rn(1.0, sc)[:, :, None]), (FBV, DK))
                else:
                    sc = _fp16_scale(tl.max(tl.abs(acc), 0), QMAX)
                    tl.store(fsb + ok, sc.to(tl.float16))
                    fc = _rint_quot(acc, sc[None, :], tl.div_rn(1.0, sc)[None, :])
                fc = tl.minimum(tl.maximum(fc, -QMAX * 1.0), QMAX * 1.0)
                tl.store(base, _f2i8(fc))
