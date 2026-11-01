"""Triton kernels of the packed GGA / STEP state with the PW-S8 window (arXiv 2610.39931), one fused decode step.

Per-slot blob of one layer (byte offsets; ``integrations/vllm_gga.GGAPacked`` builds the table):

    0           int32 [4]      n (queued pairs), head (ring index of the oldest pair)
    OFFS[h]     head h's state, by its frozen width b_h:
                  b = 16       fp16 [d_k, d_v]                         (pivot: the full update in fp16)
                  b = 4, 6, 8  codes [d_k, d_v * b / 8] (u = z + qmax, packed little-endian along d_v: 4 bit 2 per
                               byte, 6 bit 4 per 3 bytes, 8 bit 1), row factors r fp16 [d_k], column factors c fp16 [d_v]
                  b = 2        codes [d_k, d_v / 4] (u = (z + 3) / 2, 4 per byte), group rows fp16 [d_k, d_v / 32],
                               c fp16 [d_v]
                the state is r * c * z, STEP's dual-axis fit (``_stepquant_triton._fit_parts``)
    QOFF + h * QSZ  the window of head h (PW): P e4m3 [m, d_k], V e4m3 [m, d_v], PS fp32 [m], VS fp32 [m] (ring)

``_gga_step``: one program per (decode row, value head) holds the head's whole [d_k, d_v] tile; reads the codes,
O_pre = A O, propagates and re-encodes the queued left factors, reads out o = O_pre^T q + sum_i (p_i^T q) v_i +
(beta k^T q) v, merges the evicted pair, re-fits and re-packs in place and enqueues (R(beta k), R(v)) (Alg. 1);
without PW (or for pivots) the new write goes into the state. ``_gga_advance`` moves the counters after it.
``_gga_store`` / ``_gga_load``: fit + pack / unpack whole states (prefill side).
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ._stepquant_triton import _fit_parts

GROUP = tl.constexpr(32)


# -- codes ----------------------------------------------------------------------------------------------------------
@triton.jit
def _split4(u, DK: tl.constexpr, DV: tl.constexpr):
    """[DK, DV] -> its elements 4g, 4g+1, 4g+2, 4g+3 as four [DK, DV/4]."""
    a, b = tl.split(tl.reshape(u, (DK, DV // 4, 2, 2)))      # a: 4g, 4g+2   b: 4g+1, 4g+3
    e0, e2 = tl.split(a)
    e1, e3 = tl.split(b)
    return e0, e1, e2, e3


@triton.jit
def _join4(e0, e1, e2, e3, DK: tl.constexpr, DV: tl.constexpr):
    y = tl.join(tl.join(e0, e1), tl.join(e2, e3))           # [DK, DV/4, t, s] = element 2s + t
    return tl.reshape(tl.permute(y, (0, 1, 3, 2)), (DK, DV))


@triton.jit
def _store_codes(P, z, B: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr):
    ok = tl.arange(0, DK)
    if B == 2:
        u = ((z + 3.) * 0.5).to(tl.int32)
    else:
        u = (z + (2 ** (B - 1) - 1)).to(tl.int32)
    if B == 8:
        ov = tl.arange(0, DV)
        tl.store(P + ok[:, None] * DV + ov[None, :], u.to(tl.uint8))
    elif B == 4:
        lo, hi = tl.split(tl.reshape(u, (DK, DV // 2, 2)))
        oj = tl.arange(0, DV // 2)
        tl.store(P + ok[:, None] * (DV // 2) + oj[None, :], (lo | (hi << 4)).to(tl.uint8))
    elif B == 2:
        e0, e1, e2, e3 = _split4(u, DK, DV)
        og = tl.arange(0, DV // 4)
        tl.store(P + ok[:, None] * (DV // 4) + og[None, :], (e0 | (e1 << 2) | (e2 << 4) | (e3 << 6)).to(tl.uint8))
    else:
        tl.static_assert(B == 6)
        e0, e1, e2, e3 = _split4(u, DK, DV)
        wd = e0 | (e1 << 6) | (e2 << 12) | (e3 << 18)
        og = tl.arange(0, DV // 4)
        base = P + ok[:, None] * (DV // 4 * 3) + og[None, :] * 3
        tl.store(base, (wd & 255).to(tl.uint8))
        tl.store(base + 1, ((wd >> 8) & 255).to(tl.uint8))
        tl.store(base + 2, (wd >> 16).to(tl.uint8))


@triton.jit
def _load_codes(P, B: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr):
    ok = tl.arange(0, DK)
    if B == 8:
        ov = tl.arange(0, DV)
        u = tl.load(P + ok[:, None] * DV + ov[None, :]).to(tl.int32)
    elif B == 4:
        oj = tl.arange(0, DV // 2)
        by = tl.load(P + ok[:, None] * (DV // 2) + oj[None, :]).to(tl.int32)
        u = tl.reshape(tl.join(by & 15, by >> 4), (DK, DV))
    elif B == 2:
        og = tl.arange(0, DV // 4)
        by = tl.load(P + ok[:, None] * (DV // 4) + og[None, :]).to(tl.int32)
        u = _join4(by & 3, (by >> 2) & 3, (by >> 4) & 3, (by >> 6) & 3, DK, DV)
    else:
        og = tl.arange(0, DV // 4)
        base = P + ok[:, None] * (DV // 4 * 3) + og[None, :] * 3
        wd = tl.load(base).to(tl.int32) | (tl.load(base + 1).to(tl.int32) << 8) | (tl.load(base + 2).to(tl.int32) << 16)
        u = _join4(wd & 63, (wd >> 6) & 63, (wd >> 12) & 63, (wd >> 18) & 63, DK, DV)
    if B == 2:
        return u.to(tl.float32) * 2. - 3.
    return (u - (2 ** (B - 1) - 1)).to(tl.float32)


@triton.jit
def _f16(p):
    return p.to(tl.pointer_type(tl.float16))


@triton.jit
def _load_int(S, B: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr):
    """Head state of width B (2, 4, 6, 8) at byte pointer S -> fp32 [DK, DV]."""
    ok = tl.arange(0, DK)
    ov = tl.arange(0, DV)
    z = _load_codes(S, B, DK, DV)
    R = _f16(S + DK * DV * B // 8)
    if B == 2:
        NG: tl.constexpr = DV // GROUP
        r = tl.load(R + ok[:, None] * NG + tl.arange(0, NG)[None, :]).to(tl.float32)
        rf = tl.reshape(tl.broadcast_to(r[:, :, None], (DK, NG, GROUP)), (DK, DV))
        C = _f16(S + DK * DV // 4 + DK * NG * 2)
    else:
        rf = tl.broadcast_to(tl.load(R + ok).to(tl.float32)[:, None], (DK, DV))
        C = _f16(S + DK * DV * B // 8 + DK * 2)
    c = tl.load(C + ov).to(tl.float32)
    return rf * c[None, :] * z


@triton.jit
def _store_int(S, x, w, B: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr):
    """Fit x [DK, DV] at width B (STEP's dual-axis fit) and pack it at byte pointer S."""
    ok = tl.arange(0, DK)
    ov = tl.arange(0, DV)
    m = (ok[:, None] < DK) & (ov[None, :] < DV)
    if B == 2:
        r, rf, c, z = _fit_parts(x, w, m, 3., DK, DV, DK, DV, True, GROUP)
    else:
        r, rf, c, z = _fit_parts(x, w, m, 2. ** (B - 1) - 1, DK, DV, DK, DV, False, GROUP)
    _store_codes(S, z, B, DK, DV)
    R = _f16(S + DK * DV * B // 8)
    if B == 2:
        NG: tl.constexpr = DV // GROUP
        tl.store(R + ok[:, None] * NG + tl.arange(0, NG)[None, :], r.to(tl.float16))
        C = _f16(S + DK * DV // 4 + DK * NG * 2)
    else:
        tl.store(R + ok, r.to(tl.float16))
        C = _f16(S + DK * DV * B // 8 + DK * 2)
    tl.store(C + ov, c.to(tl.float16))


@triton.jit
def _load_head(S, b, DK: tl.constexpr, DV: tl.constexpr):
    ok = tl.arange(0, DK)
    ov = tl.arange(0, DV)
    if b == 16:
        x = tl.load(_f16(S) + ok[:, None] * DV + ov[None, :]).to(tl.float32)
    elif b == 2:
        x = _load_int(S, 2, DK, DV)
    elif b == 4:
        x = _load_int(S, 4, DK, DV)
    elif b == 6:
        x = _load_int(S, 6, DK, DV)
    else:
        x = _load_int(S, 8, DK, DV)
    return x


@triton.jit
def _store_head(S, x, w, b, DK: tl.constexpr, DV: tl.constexpr):
    ok = tl.arange(0, DK)
    ov = tl.arange(0, DV)
    if b == 16:
        tl.store(_f16(S) + ok[:, None] * DV + ov[None, :], x.to(tl.float16))
    elif b == 2:
        _store_int(S, x, w, 2, DK, DV)
    elif b == 4:
        _store_int(S, x, w, 4, DK, DV)
    elif b == 6:
        _store_int(S, x, w, 6, DK, DV)
    else:
        _store_int(S, x, w, 8, DK, DV)


# -- FP8 vectors ------------------------------------------------------------------------------------------------------
@triton.jit
def _e4m3_rtne(y):
    """Round |y| <= 448 to the E4M3 grid, nearest-even (Triton's fp32 -> e4m3 cast truncates; torch's rounds to
    nearest even): spacing 2^(e - 3) for exponent e >= -6 (subnormals: 2^-9)."""
    e = ((y.to(tl.int32, bitcast=True) >> 23) & 255) - 127
    e = tl.maximum(e, -6)
    ulp = (((e - 3) + 127) << 23).to(tl.float32, bitcast=True)
    return libdevice.rint(y / ulp) * ulp


@triton.jit
def _fp8_enc(x):
    """R_FP8 (App. B.4): (codes e4m3, scale s = amax / 448, reconstruction); x / s clamped to +-448 (saturating)."""
    s = tl.div_rn(tl.max(tl.abs(x), 0), 448.)
    safe = tl.where(s > 0, s, 1.)
    y = _e4m3_rtne(tl.minimum(tl.maximum(tl.div_rn(x, safe), -448.), 448.))
    code = y.to(tl.float8e4nv)                                      # exact: y is on the grid
    rec = tl.where(s > 0, code.to(tl.float32) * safe, 0.)
    return code, s, rec


@triton.jit
def _fp8_dec(P, Sc, n):
    return tl.load(P + n).to(tl.float8e4nv, bitcast=True).to(tl.float32) * tl.load(Sc)


# -- the decode step ---------------------------------------------------------------------------------------------------
@triton.jit
def _gga_step(BLOB, sB, IDX, OFFS, BITS, W, Q, K, sQ, sK, V, sV, A, BT, sA, sBT, ALOG, DTB, O, SCALE, QOFF, QSZ,
              DK: tl.constexpr, DV: tl.constexpr, HV: tl.constexpr, GQA: tl.constexpr, M: tl.constexpr,
              PW: tl.constexpr):
    bh = tl.program_id(0)
    b = bh // HV
    h = bh % HV
    ok = tl.arange(0, DK)
    ov = tl.arange(0, DV)
    seq = tl.load(IDX + b).to(tl.int64)
    if seq <= 0:
        tl.store(O + bh * DV + ov, tl.zeros((DV,), tl.float32).to(O.dtype.element_ty))
        return
    base = BLOB + seq * sB
    hk = h // GQA
    q = tl.load(Q + b * sQ + hk * DK + ok).to(tl.float32)
    k = tl.load(K + b * sK + hk * DK + ok).to(tl.float32)
    q = tl.div_rn(q, tl.sqrt_rn(tl.sum(q * q) + 1e-6)) * SCALE
    k = tl.div_rn(k, tl.sqrt_rn(tl.sum(k * k) + 1e-6))
    x = tl.load(A + b * sA + h).to(tl.float32) + tl.load(DTB + h).to(tl.float32)
    sp = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    a = libdevice.exp(-tl.exp(tl.load(ALOG + h).to(tl.float32)) * sp)
    beta = tl.sigmoid(tl.load(BT + b * sBT + h).to(tl.float32))
    v = tl.load(V + b * sV + h * DV + ov).to(tl.float32)
    bits = tl.load(BITS + h)
    S = base + tl.load(OFFS + h).to(tl.int64)
    St = _load_head(S, bits, DK, DV)
    kO = tl.sum(k[:, None] * St, 0)
    AO = a * (St - (beta * k)[:, None] * kO[None, :])
    bk = beta * k
    o = tl.sum(AO * q[:, None], 0) + tl.sum(bk * q) * v
    Om = AO + bk[:, None] * v[None, :]
    if PW:
        if bits != 16:
            Om, o = _window(base, h, ok, ov, a, bk, k, q, v, AO, o, QOFF, QSZ, DK, DV, M)
    tl.store(O + bh * DV + ov, o.to(O.dtype.element_ty))
    w = tl.load(W + h * DK + ok)
    _store_head(S, Om, w, bits, DK, DV)


@triton.jit
def _fp8_enc_rows(X):
    """``_fp8_enc`` of every row of X [R, N]: (codes, scales [R], reconstruction)."""
    s = tl.div_rn(tl.max(tl.abs(X), 1), 448.)
    safe = tl.where(s > 0, s, 1.)
    y = _e4m3_rtne(tl.minimum(tl.maximum(tl.div_rn(X, safe[:, None]), -448.), 448.))
    code = y.to(tl.float8e4nv)
    rec = tl.where((s > 0)[:, None], code.to(tl.float32) * safe[:, None], 0.)
    return code, s, rec


@triton.jit
def _window(base, h, ok, ov, a, bk, k, q, v, AO, o, QOFF, QSZ, DK: tl.constexpr, DV: tl.constexpr, M: tl.constexpr):
    """PW-S``M`` for one head, the window as one [M, d] tile in age order: propagate + re-encode the queued left
    factors, their readout, the eviction merge (oldest pair, window full) and the new pair; returns (the state to
    re-round, the output)."""
    CT = base.to(tl.pointer_type(tl.int32))
    n = tl.load(CT)
    hd = tl.load(CT + 1)
    qb = base + QOFF + h * QSZ
    PP = qb
    VP = qb + M * DK
    PS = (qb + M * (DK + DV)).to(tl.pointer_type(tl.float32))
    VS = PS + M
    oi = tl.arange(0, M)
    jj = (hd + oi) % M                                             # ring slot of the i-th oldest pair
    live = oi < n
    P = tl.load(PP + jj[:, None] * DK + ok[None, :]).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    P = P * tl.load(PS + jj)[:, None]
    Vt = tl.load(VP + jj[:, None] * DV + ov[None, :]).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    Vt = Vt * tl.load(VS + jj)[:, None]
    AP = a * (P - bk[None, :] * tl.sum(k[None, :] * P, 1)[:, None])
    code, sc, R = _fp8_enc_rows(AP)
    tl.store(PP + jj[:, None] * DK + ok[None, :], code.to(tl.uint8, bitcast=True), mask=live[:, None])
    tl.store(PS + jj, sc, mask=live)
    coef = tl.where(live, tl.sum(R * q[None, :], 1), 0.)
    o += tl.sum(coef[:, None] * tl.where(live[:, None], Vt, 0.), 0)
    full = n == M
    first = (oi == 0)[:, None]
    r0 = tl.sum(tl.where(first, R, 0.), 0)
    v0 = tl.sum(tl.where(first, Vt, 0.), 0)
    Om = AO + tl.where(full, r0[:, None] * v0[None, :], 0.)
    jn = tl.where(n < M, (hd + n) % M, hd)                       # full: the evicted slot takes the new pair
    cp, sp_, rp = _fp8_enc(bk)
    cv, sv, rv = _fp8_enc(v)
    tl.store(PP + jn * DK + ok, cp.to(tl.uint8, bitcast=True))
    tl.store(PS + jn, sp_)
    tl.store(VP + jn * DV + ov, cv.to(tl.uint8, bitcast=True))
    tl.store(VS + jn, sv)
    return Om, o


@triton.jit
def _gga_advance(BLOB, sB, IDX, M: tl.constexpr):
    b = tl.program_id(0)
    seq = tl.load(IDX + b).to(tl.int64)
    if seq <= 0:
        return
    CT = (BLOB + seq * sB).to(tl.pointer_type(tl.int32))
    n = tl.load(CT)
    hd = tl.load(CT + 1)
    tl.store(CT, tl.minimum(n + 1, M))
    tl.store(CT + 1, tl.where(n == M, (hd + 1) % M, hd))


# -- prefill side --------------------------------------------------------------------------------------------------------
@triton.jit
def _gga_store(X, BLOB, sB, IDX, OFFS, BITS, W, DK: tl.constexpr, DV: tl.constexpr, HV: tl.constexpr):
    """Fit + pack canonical states X [P, HV, DK, DV] into slots IDX; zero the counters."""
    bh = tl.program_id(0)
    p = bh // HV
    h = bh % HV
    seq = tl.load(IDX + p).to(tl.int64)
    ok = tl.arange(0, DK)
    ov = tl.arange(0, DV)
    x = tl.load(X + bh.to(tl.int64) * (DK * DV) + ok[:, None] * DV + ov[None, :])
    base = BLOB + seq * sB
    _store_head(base + tl.load(OFFS + h).to(tl.int64), x, tl.load(W + h * DK + ok), tl.load(BITS + h), DK, DV)
    if h == 0:
        CT = base.to(tl.pointer_type(tl.int32))
        tl.store(CT, 0)
        tl.store(CT + 1, 0)


@triton.jit
def _gga_load(Y, BLOB, sB, IDX, OFFS, BITS, DK: tl.constexpr, DV: tl.constexpr, HV: tl.constexpr):
    """Unpack slots IDX into canonical fp32 Y [P, HV, DK, DV] (the stored state, without the window)."""
    bh = tl.program_id(0)
    p = bh // HV
    h = bh % HV
    seq = tl.load(IDX + p).to(tl.int64)
    ok = tl.arange(0, DK)
    ov = tl.arange(0, DV)
    x = _load_head(BLOB + seq * sB + tl.load(OFFS + h).to(tl.int64), tl.load(BITS + h), DK, DV)
    tl.store(Y + bh.to(tl.int64) * (DK * DV) + ok[:, None] * DV + ov[None, :], x)
