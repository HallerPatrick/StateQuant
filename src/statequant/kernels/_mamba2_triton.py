"""The packed decode step of a Mamba2 / SSD layer as ONE Triton launch (one program per (batch row, head)).

Mamba2 in the delta-buffer picture (``fold.py``): per head h with group g = h // (H / G),

    dt  = softplus(dt_raw + dt_bias), clamped to dt_limit            (or given)
    S_t = exp(A_h dt) S_{t-1} + B_t u_t^T,   u_t = dt x_t              (S: [d_state, head_dim] = [d_k, d_v])
    y_t = S_t^T C_t + D_h x_t

so B is the key (one per group, shared by the group's H / G heads, as GQA), C the query, u = dt x the buffered value
and exp(A dt) a scalar decay. Unlike gated delta there is no delta term (u does not depend on S, so the state is
read once, against C only) and q, k are not normalised. With the buffer of (cumulative log decay G_i, B_i, u_i) since
the last fold and alpha_i = exp(G_i):

    y_n = a_n [alpha_{n-1} S_0^T C + sum_{i<n} (alpha_{n-1} / alpha_i) (B_i . C) u_i] + (B_n . C) u_n + D x_n
    fold (every m tokens):  S = alpha_m S_0 + sum_i (alpha_m / alpha_i) B_i u_i^T,  re-rounded in place

One program owns one head's whole state ([d_v, d_k], stored value-major like vLLM's ssm state) and does:

    prep     B, C of the head's group, dt (bias, softplus, clamp) and the decay, u = dt x over the whole row, the
             buffer coefficients c[i] = (alpha_{n-1}/alpha_i) B_i . C for all slots in one [m, d_k] tile
    state    value blocks of SBV rows: the stored codes read once, S_0^T C, the buffer term, y (stored)
    append   u at slot n (int8 with one absmax scale over the row, or bf16), B by the group's first head only, the
             cumulative decay
    fold     if this sequence's buffer is now full: FBV value rows at a time, slot m-1's key from registers (the
             group's first head may still be writing it), re-rounding as ``fold.pack_vk``
    advance  this head's running decay and its copy of the counter

Cross-program hazards: as ``_fused_triton`` (the key buffer is shared by a group; only its first head writes slot n,
the others never read slot n before the fold, which takes it from registers; the counter has one copy per head).

Non-power-of-two head dims (d_v = 80, d_k = 96, ...): the tiles are padded to the next power of two (DKP, DVP) and
masked; memory holds exactly d_k, d_v. Groups (``GROUP``) must be powers of two that divide their axis, so a padded
key row is a whole number of (padded) groups.

The arithmetic matches ``mamba2.ReferenceMamba2State`` (rounding exactly as ``fold.pack_vk``: IEEE divisions,
half-to-even, against the fp16-rounded scale); fp32 sums may associate differently (tests: ``tests/test_mamba2.py``).
"""
from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ._fold_triton import _fp16_scale, _nib, _round_buf, _seq_of, _tof
from ._fused_triton import _f2i8, _fold_acc, _pack4, _rint_quot


@triton.jit
def _m2_step(X, sX, DT, sDT, BM, sB, CM, sC, A, DTB, DP, IDX,
             S, sS, SC, sSC, KB, sKB, KBS, sKBS, UB, sUB, UBS, sUBS, GB, sGB, GCUR, sGC, NPTR, sN,
             O, sO, DT_MIN, DT_MAX,
             DK: tl.constexpr, DV: tl.constexpr, DKP: tl.constexpr, DVP: tl.constexpr, SBV: tl.constexpr,
             FBV: tl.constexpr, M: tl.constexpr, BITS: tl.constexpr, AXK: tl.constexpr, GROUP: tl.constexpr,
             QBUF: tl.constexpr, HV: tl.constexpr, HK: tl.constexpr, GQA: tl.constexpr, HAS_IDX: tl.constexpr,
             HAS_DT_BIAS: tl.constexpr, DT_SOFTPLUS: tl.constexpr, DT_CLAMP: tl.constexpr, HAS_D: tl.constexpr):
    bh = tl.program_id(0)
    b = bh // HV
    h = bh % HV
    hk = h // GQA
    ov = tl.arange(0, DVP)
    vm = ov < DV
    seq = _seq_of(IDX, b, HAS_IDX)
    if HAS_IDX:
        if seq <= 0:
            tl.store(O + b * sO + h * DV + ov, tl.zeros((DVP,), tl.float32).to(O.dtype.element_ty), mask=vm)
            return
    ok = tl.arange(0, DKP)
    km = ok < DK
    n = tl.load(NPTR + seq * sN + h) & (M - 1)                      # this head's copy of the counter

    # ---- prep ----
    c = tl.load(CM + b * sC + hk * DK + ok, mask=km, other=0.0).to(tl.float32)
    kx = tl.load(BM + b * sB + hk * DK + ok, mask=km, other=0.0).to(tl.float32)
    dt = tl.load(DT + b * sDT + h).to(tl.float32)
    if HAS_DT_BIAS:
        dt = dt + tl.load(DTB + h).to(tl.float32)
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)      # vLLM / mamba_ssm (threshold 20)
    if DT_CLAMP:
        dt = tl.minimum(tl.maximum(dt, DT_MIN), DT_MAX)
    g = _mul_rn(tl.load(A + h).to(tl.float32), dt)                 # not contracted into gcur + g (exact as torch)
    a = libdevice.exp(g)
    Xh = X + b * sX + h * DV
    u = dt * tl.load(Xh + ov, mask=vm, other=0.0).to(tl.float32)  # the buffered value, whole row
    bc = tl.sum(kx * c)
    if HAS_D:
        dd = tl.load(DP + h).to(tl.float32)
    else:
        dd = 0.0

    kbase = KB + seq * sKB + hk * (M * DK)
    ksc = KBS + seq * sKBS + hk * M
    gb = GB + seq * sGB + h * M
    ub = UB + seq * sUB + h * (M * DV)
    ubs = UBS + seq * sUBS + h * M
    gcur = tl.load(GCUR + seq * sGC + h)
    alpha = libdevice.exp(gcur)
    NG: tl.constexpr = DK // GROUP
    if BITS == 4:
        Sb = S + seq * sS + h * (DV * (DK // 2))
    else:
        Sb = S + seq * sS + h * (DV * DK)
    if AXK:
        SCh = SC + seq * sSC + h * (DV * NG)
    else:
        SCh = SC + seq * sSC + h * ((DV // GROUP) * DK)

    # c[i] = (alpha_{n-1} / alpha_i) B_i . C, slots i < n (dead slots, possibly garbage, discarded)
    om = tl.arange(0, M)
    kb = _tof(tl.load(kbase + om[:, None] * DK + ok[None, :], mask=km[None, :], other=0))
    if QBUF:
        kb = kb * tl.load(ksc + om).to(tl.float32)[:, None]
    live = om < n
    cq = tl.where(live, libdevice.exp(gcur - tl.load(gb + om)) * tl.sum(kb * c[None, :], 1), 0.0)

    # ---- state: value blocks of SBV rows ----
    oh = tl.arange(0, DKP // 2)
    hm = oh < DK // 2
    if BITS == 4:
        ce_ = tl.load(CM + b * sC + hk * DK + 2 * oh, mask=hm, other=0.0).to(tl.float32)
        co_ = tl.load(CM + b * sC + hk * DK + 2 * oh + 1, mask=hm, other=0.0).to(tl.float32)
    NSB: tl.constexpr = (DV + SBV - 1) // SBV
    NGP: tl.constexpr = DKP // GROUP
    og = tl.arange(0, NGP)
    for iv in range(0, NSB):
        fv = iv * SBV + tl.arange(0, SBV)
        fm = fv < DV
        if BITS == 4:
            m2 = fm[:, None] & hm[None, :]
            by = tl.load(Sb + fv[:, None] * (DK // 2) + oh[None, :], mask=m2, other=136).to(tl.int32)
            ce = _nib(by & 0xF)
            co = _nib((by >> 4) & 0xF)
            if AXK:                          # per key group: (codes . C) summed, times the group's scale
                pr = tl.reshape(ce * ce_[None, :] + co * co_[None, :], (SBV, NGP, GROUP // 2))
                sq = tl.sum(tl.sum(pr, 2) * _ksc(SCh, fv, fm, og, NG), 1)
            elif GROUP % SBV == 0:           # the block lies in one value group: one scale row, folded into C
                sp = SCh + (iv * SBV // GROUP) * DK + 2 * oh
                sce = tl.load(sp, mask=hm, other=0.0).to(tl.float32)
                sco = tl.load(sp + 1, mask=hm, other=0.0).to(tl.float32)
                sq = tl.sum(ce * (ce_ * sce)[None, :] + co * (co_ * sco)[None, :], 1)
            else:
                sp = SCh + (fv[:, None] // GROUP) * DK + 2 * oh[None, :]
                sce = tl.load(sp, mask=m2, other=0.0).to(tl.float32)
                sco = tl.load(sp + 1, mask=m2, other=0.0).to(tl.float32)
                sq = tl.sum(ce * sce * ce_[None, :] + co * sco * co_[None, :], 1)
        else:
            m2 = fm[:, None] & km[None, :]
            cs = _tof(tl.load(Sb + fv[:, None] * DK + ok[None, :], mask=m2, other=0))
            if BITS == 8 and AXK:
                pr = tl.reshape(cs * c[None, :], (SBV, NGP, GROUP))
                sq = tl.sum(tl.sum(pr, 2) * _ksc(SCh, fv, fm, og, NG), 1)
            elif BITS == 8 and GROUP % SBV == 0:
                sq = tl.sum(cs * (c * tl.load(SCh + (iv * SBV // GROUP) * DK + ok, mask=km,
                                              other=0.0).to(tl.float32))[None, :], 1)
            elif BITS == 8:
                cs = cs * tl.load(SCh + (fv[:, None] // GROUP) * DK + ok[None, :], mask=m2, other=0.0).to(tl.float32)
                sq = tl.sum(cs * c[None, :], 1)
            else:
                sq = tl.sum(cs * c[None, :], 1)
        ubt = _tof(tl.load(ub + om[:, None] * DV + fv[None, :], mask=fm[None, :], other=0))
        if QBUF:
            ubt = ubt * tl.load(ubs + om).to(tl.float32)[:, None]
        ubt = tl.where(live[:, None], ubt, 0.0)
        sq = alpha * sq + tl.sum(ubt * cq[:, None], 0)
        xr = tl.load(Xh + fv, mask=fm, other=0.0).to(tl.float32)
        y = tl.fma(bc, dt * xr, a * sq)
        if HAS_D:
            y = tl.fma(dd, xr, y)
        tl.store(O + b * sO + h * DV + fv, y.to(O.dtype.element_ty), mask=fm)

    # ---- append: decay, value (int8: one scale over the whole row), key (group's first head only) ----
    gnew = gcur + g
    tl.store(gb + n, gnew)
    if QBUF:
        us = _fp16_scale(tl.max(tl.abs(u)), 127)
        tl.store(ubs + n, us.to(tl.float16))
        tl.store(ub + n * DV + ov, _round_buf(u, us).to(tl.int8), mask=vm)
        ks = _fp16_scale(tl.max(tl.abs(kx)), 127)
    else:
        tl.store(ub + n * DV + ov, u.to(UB.dtype.element_ty), mask=vm)
        ks = 1.0
    if h % GQA == 0:
        if QBUF:
            tl.store(ksc + n, ks.to(tl.float16))
            tl.store(kbase + n * DK + ok, _round_buf(kx, ks).to(tl.int8), mask=km)
        else:
            tl.store(kbase + n * DK + ok, kx.to(KB.dtype.element_ty), mask=km)

    # ---- fold ----
    full = n == M - 1
    if full:
        tl.debug_barrier()                  # the state pass's reads and the value store complete before the fold
        _m2_fold(Sb, SCh, kbase, ksc, gb, ub, ubs, kx, ks, gnew, DK, DV, DKP, FBV, M, BITS, AXK, GROUP, QBUF)

    # ---- advance ----
    tl.store(GCUR + seq * sGC + h, tl.where(full, 0.0, gnew))
    tl.store(NPTR + seq * sN + h, tl.where(full, 0, n + 1))


@triton.jit
def _ksc(SCh, fv, fm, og, NG: tl.constexpr):
    """Key-axis scales of value rows fv, [rows, NGP] (padded groups 0)."""
    return tl.load(SCh + fv[:, None] * NG + og[None, :], mask=fm[:, None] & (og < NG)[None, :], other=0.0).to(tl.float32)


@triton.jit
def _mul_rn(x, y):
    """x * y rounded on its own (an fma contraction with a later add would round differently from torch)."""
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', '=r,r,r', [x, y], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _m2_fold(Sb, SCh, kbase, ksc, gb, ub, ubs, kx, ks, gnew, DK: tl.constexpr, DV: tl.constexpr, DKP: tl.constexpr,
             FBV: tl.constexpr, M: tl.constexpr, BITS: tl.constexpr, AXK: tl.constexpr, GROUP: tl.constexpr,
             QBUF: tl.constexpr):
    """S = alpha_m S_0 + sum_i (alpha_m/alpha_i) B_i u_i^T over the full buffer, re-rounded in place, FBV value rows
    at a time. Slots 0..m-2 from memory; slot m-1: the value as just stored by this program, the key from
    registers. 4-bit codes: even / odd key columns (the nibbles); 8 / 16 bit: whole padded rows."""
    NG: tl.constexpr = DK // GROUP
    NGP: tl.constexpr = DKP // GROUP
    QMAX: tl.constexpr = 2 ** (BITS - 1) - 1
    alphan = libdevice.exp(gnew)
    om = tl.arange(0, M)
    last = om == M - 1
    old = om < M - 1
    dec = libdevice.exp(gnew - tl.where(last, gnew, tl.load(gb + om, mask=old, other=0.0)))
    if QBUF:
        usv = tl.load(ubs + om).to(tl.float32)
        ksv = tl.load(ksc + om, mask=old, other=0.0).to(tl.float32)
    og = tl.arange(0, NGP)
    if BITS == 4:
        oh = tl.arange(0, DKP // 2)
        hm = oh < DK // 2
        kxe, kxo = tl.split(tl.reshape(kx, (DKP // 2, 2)))
        if QBUF:
            kre = _round_buf(kxe, ks) * ks                            # = stored int8 * fp16 scale, exactly
            kro = _round_buf(kxo, ks) * ks
        else:
            kre = kxe.to(tl.bfloat16).to(tl.float32)
            kro = kxo.to(tl.bfloat16).to(tl.float32)
        mk = old[:, None] & hm[None, :]
        Ke = _tof(tl.load(kbase + om[:, None] * DK + 2 * oh[None, :], mask=mk, other=0))
        Ko = _tof(tl.load(kbase + om[:, None] * DK + 2 * oh[None, :] + 1, mask=mk, other=0))
        if QBUF:
            Ke = Ke * ksv[:, None]
            Ko = Ko * ksv[:, None]
        KeT = tl.trans(tl.where(last[:, None], kre[None, :], Ke))
        KoT = tl.trans(tl.where(last[:, None], kro[None, :], Ko))
        for iv in range(0, (DV + FBV - 1) // FBV):
            fv = iv * FBV + tl.arange(0, FBV)
            fm = fv < DV
            m2 = fm[:, None] & hm[None, :]
            by = tl.load(Sb + fv[:, None] * (DK // 2) + oh[None, :], mask=m2, other=136).to(tl.int32)
            acce = _nib(by & 0xF)
            acco = _nib((by >> 4) & 0xF)
            if AXK:
                msc = fm[:, None] & (og < NG)[None, :]
                fsc = tl.load(SCh + fv[:, None] * NG + og[None, :], mask=msc, other=0.0).to(tl.float32)[:, :, None]
                acce = tl.reshape(tl.reshape(acce, (FBV, NGP, GROUP // 2)) * fsc, (FBV, DKP // 2))
                acco = tl.reshape(tl.reshape(acco, (FBV, NGP, GROUP // 2)) * fsc, (FBV, DKP // 2))
            else:
                fsb = SCh + (iv * FBV // GROUP) * DK                      # FBV == GROUP
                acce = acce * tl.load(fsb + 2 * oh, mask=hm, other=0.0).to(tl.float32)[None, :]
                acco = acco * tl.load(fsb + 2 * oh + 1, mask=hm, other=0.0).to(tl.float32)[None, :]
            acce = alphan * acce
            acco = alphan * acco
            U = _tof(tl.load(ub + om[:, None] * DV + fv[None, :], mask=fm[None, :], other=0))
            if QBUF:
                U = U * usv[:, None]
            acce, acco = _fold_acc(acce, acco, tl.trans(dec[:, None] * U), KeT, KoT, M, True)
            if AXK:
                ae = tl.reshape(acce, (FBV, NGP, GROUP // 2))
                ao = tl.reshape(acco, (FBV, NGP, GROUP // 2))
                sc = _fp16_scale(tl.maximum(tl.max(tl.abs(ae), 2), tl.max(tl.abs(ao), 2)), QMAX)
                tl.store(SCh + fv[:, None] * NG + og[None, :], sc.to(tl.float16), mask=msc)
                fce = tl.reshape(_rint_quot(ae, sc[:, :, None], tl.div_rn(1.0, sc)[:, :, None]), (FBV, DKP // 2))
                fco = tl.reshape(_rint_quot(ao, sc[:, :, None], tl.div_rn(1.0, sc)[:, :, None]), (FBV, DKP // 2))
            else:
                se = _fp16_scale(tl.max(tl.abs(acce), 0), QMAX)
                so = _fp16_scale(tl.max(tl.abs(acco), 0), QMAX)
                tl.store(fsb + 2 * oh, se.to(tl.float16), mask=hm)
                tl.store(fsb + 2 * oh + 1, so.to(tl.float16), mask=hm)
                fce = _rint_quot(acce, se[None, :], tl.div_rn(1.0, se)[None, :])
                fco = _rint_quot(acco, so[None, :], tl.div_rn(1.0, so)[None, :])
            fce = tl.minimum(tl.maximum(fce, -QMAX * 1.0), QMAX * 1.0)
            fco = tl.minimum(tl.maximum(fco, -QMAX * 1.0), QMAX * 1.0)
            tl.store(Sb + fv[:, None] * (DK // 2) + oh[None, :], _pack4(fce, fco), mask=m2)
    else:
        ok = tl.arange(0, DKP)
        km = ok < DK
        if QBUF:
            kr = _round_buf(kx, ks) * ks
        else:
            kr = kx.to(tl.bfloat16).to(tl.float32)
        Kf = _tof(tl.load(kbase + om[:, None] * DK + ok[None, :], mask=old[:, None] & km[None, :], other=0))
        if QBUF:
            Kf = Kf * ksv[:, None]
        KfT = tl.trans(tl.where(last[:, None], kr[None, :], Kf))
        for iv in range(0, (DV + FBV - 1) // FBV):
            fv = iv * FBV + tl.arange(0, FBV)
            fm = fv < DV
            m2 = fm[:, None] & km[None, :]
            base = Sb + fv[:, None] * DK + ok[None, :]
            acc = _tof(tl.load(base, mask=m2, other=0))
            if BITS == 8:
                if AXK:
                    msc = fm[:, None] & (og < NG)[None, :]
                    fsc = tl.load(SCh + fv[:, None] * NG + og[None, :], mask=msc, other=0.0).to(tl.float32)[:, :, None]
                    acc = tl.reshape(tl.reshape(acc, (FBV, NGP, GROUP)) * fsc, (FBV, DKP))
                else:
                    fsb = SCh + (iv * FBV // GROUP) * DK                  # FBV == GROUP
                    acc = acc * tl.load(fsb + ok, mask=km, other=0.0).to(tl.float32)[None, :]
            acc = alphan * acc
            U = _tof(tl.load(ub + om[:, None] * DV + fv[None, :], mask=fm[None, :], other=0))
            if QBUF:
                U = U * usv[:, None]
            acc, unused = _fold_acc(acc, acc, tl.trans(dec[:, None] * U), KfT, KfT, M, False)
            if BITS == 16:
                tl.store(base, acc.to(tl.bfloat16), mask=m2)
            else:
                if AXK:
                    a3 = tl.reshape(acc, (FBV, NGP, GROUP))
                    sc = _fp16_scale(tl.max(tl.abs(a3), 2), QMAX)
                    tl.store(SCh + fv[:, None] * NG + og[None, :], sc.to(tl.float16), mask=msc)
                    fc = tl.reshape(_rint_quot(a3, sc[:, :, None], tl.div_rn(1.0, sc)[:, :, None]), (FBV, DKP))
                else:
                    sc = _fp16_scale(tl.max(tl.abs(acc), 0), QMAX)
                    tl.store(fsb + ok, sc.to(tl.float16), mask=km)
                    fc = _rint_quot(acc, sc[None, :], tl.div_rn(1.0, sc)[None, :])
                fc = tl.minimum(tl.maximum(fc, -QMAX * 1.0), QMAX * 1.0)
                tl.store(base, _f2i8(fc), mask=m2)
