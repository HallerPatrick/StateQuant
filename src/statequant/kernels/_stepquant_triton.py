"""Triton kernel of STEPQuant's dual-axis fit (``statequant.step_quant.fit``): one program per (sequence, head) holds
the whole [d_k, d_v] head state, so the fit is one read and one write of the state instead of ~30 elementwise passes.

Per head of width b (from the per-head plan): b = 16 is an fp16 pivot (the state cast to fp16); otherwise the row
factors r_i = half(sqrt(max(mean_j |x_ij|, 1e-20) / w_i)) (2 bits: one per 32 value columns), column factors
c_j = half(max_i |x_ij / r_i| / qmax), the codes, one weighted least-squares refit of c with the codes fixed, and the
codes again. half(.) clamps to [2^-24, 65504] and rounds to fp16. Divisions and square roots are IEEE (``div_rn``,
``sqrt_rn``) and rounding is half-to-even (``rint``) as in torch; only the order of the reductions differs from the
torch reference, which can move a code across a rounding threshold (tests/test_stepquant_kernel.py).

Tiles are padded to powers of two (DKP x DVP) and masked, so any d_k, d_v work (2-bit heads need 32 | d_v).
"""
import triton
import triton.language as tl


@triton.jit
def _half(x):
    return tl.minimum(tl.maximum(x, 5.960464477539063e-08), 65504.).to(tl.float16).to(tl.float32)


@triton.jit
def _encode(y, qmax, TWO: tl.constexpr):
    if TWO:
        return tl.where(y >= 0, 1., -1.) * tl.where(tl.abs(y) >= 2, 3., 1.)
    z = tl.extra.cuda.libdevice.rint(y)
    return tl.minimum(tl.maximum(z, -qmax), qmax)


@triton.jit
def _fit_head(x, w, m, qmax, DK, DV, DKP: tl.constexpr, DVP: tl.constexpr, TWO: tl.constexpr, GROUP: tl.constexpr):
    ax = tl.where(m, tl.abs(x), 0.)
    if TWO:
        NG: tl.constexpr = DVP // GROUP
        mg = tl.sum(tl.reshape(ax, (DKP, NG, GROUP)), 2) * (1. / GROUP)                      # [DKP, NG]
        r = _half(tl.sqrt_rn(tl.div_rn(tl.maximum(mg, 1e-20), w[:, None])))
        rf = tl.reshape(tl.broadcast_to(r[:, :, None], (DKP, NG, GROUP)), (DKP, DVP))
    else:
        mr = tl.sum(ax, 1) * (1. / DV)                                               # torch: sum * 1/N
        r = _half(tl.sqrt_rn(tl.div_rn(tl.maximum(mr, 1e-20), w)))
        rf = tl.broadcast_to(r[:, None], (DKP, DVP))
    rf = tl.where(m, rf, 1.)
    y0 = tl.div_rn(x, rf)
    c = _half(tl.max(tl.where(m, tl.div_rn(tl.abs(y0), qmax), 0.), 0))                 # [DVP]
    z = _encode(tl.div_rn(y0, c[None, :]), qmax, TWO)
    vv = rf * z
    ww = (w * w)[:, None]
    num = tl.sum(tl.where(m, ww * vv * x, 0.), 0)
    den = tl.sum(tl.where(m, ww * (vv * vv), 0.), 0)
    c = _half(tl.where(den > 0, tl.div_rn(num, tl.maximum(den, 1e-30)), c))
    z = _encode(tl.div_rn(y0, c[None, :]), qmax, TWO)
    return rf * c[None, :] * z


@triton.jit
def _stepquant_fit(X, Y, BITS, W, H, DK, DV: tl.constexpr, DKP: tl.constexpr, DVP: tl.constexpr, GROUP: tl.constexpr,
                   HAS_TWO: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    ok = tl.arange(0, DKP)
    ov = tl.arange(0, DVP)
    m = (ok[:, None] < DK) & (ov[None, :] < DV)
    off = pid.to(tl.int64) * DK * DV + ok[:, None] * DV + ov[None, :]
    x = tl.load(X + off, mask=m, other=0.)
    b = tl.load(BITS + h)
    if b == 16:
        tl.store(Y + off, x.to(tl.float16).to(tl.float32), mask=m)
        return
    w = tl.load(W + h * DK + ok, mask=ok < DK, other=1.)
    if HAS_TWO:
        if b == 2:
            tl.store(Y + off, _fit_head(x, w, m, 3., DK, DV, DKP, DVP, True, GROUP), mask=m)
            return
    qmax = tl.exp2((tl.minimum(b, 8) - 1).to(tl.float32)) - 1
    tl.store(Y + off, _fit_head(x, w, m, qmax, DK, DV, DKP, DVP, False, GROUP), mask=m)


def fit(x, bits, w, group=32, has_two=None):
    """x [B, H, d_k, d_v] fp32 contiguous (CUDA); bits [H] int32 on the device; w [H, d_k] fp32 on the device."""
    import torch
    B, H, DK, DV = x.shape
    x = x.float().contiguous()
    y = torch.empty_like(x)
    if has_two is None:
        has_two = bool((bits == 2).any())
    if has_two:
        assert DV % group == 0, f'2-bit heads need {group} | d_v'
    DKP, DVP = triton.next_power_of_2(DK), max(triton.next_power_of_2(DV), group)
    _stepquant_fit[(B * H,)](x, y, bits, w, H, DK, DV, DKP=DKP, DVP=DVP, GROUP=group, HAS_TWO=has_two,
                             num_warps=8 if DKP * DVP >= 128 * 128 else 4)
    return y
