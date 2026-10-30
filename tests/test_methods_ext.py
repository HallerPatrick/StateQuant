"""CPU tests for statequant.methods_ext: BDR / LRB (2610.15709), LeapQuant (2609.38166), Gramian allocation
(2609.30950), DAMP paper recipe (2608.27513). A tiny gated-delta 'model' with a module-level torch op drives the
stateful methods through statequant.apply exactly as on a real model."""
import math
import sys

import pytest
import torch

import statequant as sq
from statequant.families import BY_NAME, Piece
from statequant.methods_ext import (AffineGrid, DampPaper, GramianAlloc, LeapQuant, LRB, allocate, bdr_round,
                                    clip_quantize, compensators, randomized_svd, sr_candidates, sr_forward,
                                    sr_inverse)
from statequant.methods_ext.gramian import ema_update


# ------------------------------------------------------------------------------------------------------------------
# a tiny GDN layer whose op is a module-level function, as in transformers' Qwen3.5 modeling file
# ------------------------------------------------------------------------------------------------------------------
def torch_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=None, output_final_state=False,
                                     use_qk_l2norm_in_kernel=False, **kw):
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    if use_qk_l2norm_in_kernel:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q * q.shape[-1] ** -0.5
    B, T, H, dk = k.shape
    S = torch.zeros(B, H, dk, v.shape[-1]) if initial_state is None else initial_state.float().clone()
    out = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt = k[:, t]
        delta = (v[:, t] - torch.einsum('bhkv,bhk->bhv', S, kt)) * beta[:, t, :, None]
        S = S + kt[..., :, None] * delta[..., None, :]
        out.append(torch.einsum('bhkv,bhk->bhv', S, q[:, t]))
    return torch.stack(out, 1), S


class TinyGatedDeltaNet(torch.nn.Module):
    def __init__(self, H=2, dk=128, dv=128, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.H, self.dk, self.dv = H, dk, dv
        self.head_k_dim, self.head_v_dim, self.num_v_heads, self.num_k_heads = dk, dv, H, H
        self.decay = torch.nn.Parameter(torch.tensor([0.9995, 0.97])[:H].log(), requires_grad=False)
        self.gen = g

    def forward(self, x, state=None):
        q, k, v, beta = x                                          # [B, T, H, d] ...
        B, T = q.shape[:2]
        g = self.decay.expand(B, T, self.H)
        mod = sys.modules[__name__]
        return mod.torch_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=state, output_final_state=True,
                                                    use_qk_l2norm_in_kernel=True)


class Tiny(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.layer = TinyGatedDeltaNet(**kw)

    def forward(self, x, state=None):
        return self.layer(x, state)


def inputs(T=64, B=1, H=2, dk=128, dv=128, seed=1):
    g = torch.Generator().manual_seed(seed)
    q, k = torch.randn(B, T, H, dk, generator=g), torch.randn(B, T, H, dk, generator=g)
    v = torch.randn(B, T, H, dv, generator=g) * (1 + 4 * (torch.rand(1, 1, 1, dv, generator=g) > 0.9))
    beta = torch.rand(B, T, H, generator=g)
    return q, k, v, beta


def run(model, x, method=None, schedule=None, decode=0):
    """Prompt of len(x) - decode tokens in one call, then ``decode`` single-token calls; outputs and final state."""
    T = x[0].shape[1]
    P = T - decode
    import contextlib
    cm = sq.apply(model, method, schedule) if method is not None else contextlib.nullcontext()
    with cm, torch.no_grad():
        o, S = model(tuple(t[:, :P] for t in x))
        outs = [o]
        for t in range(P, T):
            o, S = model(tuple(z[:, t:t + 1] for z in x), S)
            outs.append(o)
    return torch.cat(outs, 1), S


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


# ------------------------------------------------------------------------------------------------------------------
# BDR
# ------------------------------------------------------------------------------------------------------------------
def state(seed=0, B=2, H=2, K=128, V=128):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, H, K, V, generator=g) * (1 + 3 * torch.rand(1, 1, 1, V, generator=g))


def test_affine_grid_matches_quantizer():
    S = state()
    ref = sq.Quantizer(4, 'k', 128, affine=True, zero_point='int').fake(S)
    out = AffineGrid(4, 128).fake(S)
    assert (out - ref).abs().max() < 1e-5 * S.abs().max()
    assert AffineGrid(4, 128).bits_per_value(128, 128) == 4.5


FORMATS = [AffineGrid(4, 128), AffineGrid(4, 32), sq.Quantizer(4, 'k', 32), sq.Quantizer(4, 'v', 32, affine=True),
           sq.Quantizer(4, 'k', 64, rotation='hadamard'), sq.Quantizer(8, 'v', 32)]


@pytest.mark.parametrize('fmt', FORMATS, ids=repr)
def test_bdr_inf_threshold_is_plain_rounding(fmt):
    s = fmt.fake(state(1))
    S = s + 0.01 * state(2)
    assert torch.equal(bdr_round(fmt, S, s, math.inf), fmt.fake(S))


@pytest.mark.parametrize('fmt', [f for f in FORMATS if getattr(f, 'rotation', None) is None], ids=repr)
def test_bdr_direction(fmt):
    """After BDR no element with |d| > eps_B Delta moves against its update unless the corrected point is off the
    grid; corrected elements follow d, overshoot it (Lemma 1 (ii)) and stay below |d| / eps_B (Lemma 1 (iv))."""
    eps = 0.3
    s = 0.9 * fmt.fake(state(3))                                   # a decayed stored state: off the new grid
    d = 0.05 * state(4)
    S = s + d
    plain = fmt.fake(S)
    out, st = bdr_round(fmt, S, s, eps, return_stats=True)
    assert st['corrected'] > 0 and st['reversed'] >= st['corrected']
    if isinstance(fmt, AffineGrid):
        step = fmt.step_size(S)
    else:
        X = fmt._groups(S)
        from statequant.methods_ext.bdr_lrb import _QuantizerGrid
        sc = _QuantizerGrid(fmt).grid(X)[1].expand_as(X)
        step = fmt._ungroup(sc, S.shape)
    dt, dt0 = out - s, plain - s
    changed = out != plain
    assert changed.any()
    assert (torch.sign(dt[changed]) == torch.sign(d[changed])).all()               # follows the update
    assert (dt[changed].abs() >= d[changed].abs() + step[changed] / 2 - 1e-5).all()  # overshoot (Lemma 1 ii)
    assert (dt[changed].abs() < d[changed].abs() / eps + 1e-6).all()               # bounded (Lemma 1 iv)
    assert (dt0[changed] * d[changed] < 0).all()                                   # only reversals were changed
    # changed elements moved exactly one grid step
    assert torch.allclose((out - plain)[changed].abs(), step[changed], rtol=1e-4)
    big = (d.abs() > eps * step) & ~changed
    assert ((dt[big] * d[big] >= 0) | ((out - plain)[big] == 0)).all()


def test_bdr_method_per_token_runs_and_differs():
    m, x = Tiny(), inputs(T=40)
    ref, _ = run(m, x)
    o_int4, _ = run(m, x, sq.int4_g128())
    o_bdr, _ = run(m, x, sq.bdr_int4())
    assert rel(o_int4, ref) > 1e-3 and not torch.equal(o_int4, o_bdr)
    # BDR with an infinite threshold is INT4
    o_inf, _ = run(m, x, sq.BDR(AffineGrid(4, 128), math.inf), sq.per_token())
    assert torch.equal(o_inf, o_int4)


# ------------------------------------------------------------------------------------------------------------------
# LRB
# ------------------------------------------------------------------------------------------------------------------
def test_randomized_svd_exact_on_low_rank():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(3, 128, 2, generator=g) @ torch.randn(3, 2, 64, generator=g)
    P, Q = randomized_svd(A, 2, generator=torch.Generator().manual_seed(1))
    assert P.shape == (3, 128, 2) and Q.shape == (3, 64, 2)
    assert rel(P @ Q.transpose(-1, -2), A) < 1e-5
    A2 = A + 1e-3 * torch.randn(3, 128, 64, generator=g)
    P, Q = randomized_svd(A2, 1, generator=torch.Generator().manual_seed(1))
    U, Sv, Vh = torch.linalg.svd(A2)
    best = Sv[:, :1, None] * U[:, :, :1] @ Vh[:, :1]
    assert rel(P @ Q.transpose(-1, -2), best) < 1e-3


def _merge_setup(eps_L, k=1, eps_B=None, fmt=None, seed=5):
    m = LRB(k=k, m=15, eps_L=eps_L, eps_B=eps_B, fmt=fmt)
    m.reset()
    Sq = (fmt or m.fmt).fake(state(seed))
    g = torch.Generator().manual_seed(seed + 1)
    L = sum(torch.randn(2, 2, 128, 1, generator=g) @ torch.randn(2, 2, 1, 128, generator=g) * 0.02 * 0.5 ** i
            for i in range(15))                                    # 15 updates with a decaying spectrum
    st = {'Sq': Sq, 'acc': torch.full((2, 2, 1, 1), 0.95), 'K': None, 'V': None}
    return m, st, L


def test_lrb_merge_rank_and_threshold():
    # eps_L = 0: nothing stays in the bypass, D = L
    m, st, L = _merge_setup(0.)
    s = st['acc'] * st['Sq']
    out = m._merge(st, L)
    assert m.stats['kept'] == 0 and st['K'].shape[-1] == 1 and torch.equal(st['K'], torch.zeros_like(st['K']))
    assert torch.equal(out, m.fmt.fake(s + L))
    # a large threshold keeps everything: the bypass is the best rank-1 approximation of L (k = 1 columns, fp16)
    m, st, L = _merge_setup(1e9)
    s = st['acc'] * st['Sq']
    out = m._merge(st, L)
    assert m.stats['kept'] == 1. and st['K'].shape[-1] == 1 and st['K'].dtype == torch.float16
    Lp = st['K'].float() @ st['V'].float().transpose(-1, -2)
    U, Sv, Vh = torch.linalg.svd(L)
    assert rel(Lp, Sv[..., :1, None] * U[..., :1] @ Vh[..., :1, :]) < 1e-2      # randomised SVD ~ truncated SVD
    assert torch.allclose(out - Lp, st['Sq']) and torch.equal(st['Sq'], m.fmt.fake(s + (L - Lp)))
    # the mask is exactly |L| < eps_L * Delta(s + L)
    m, st, L = _merge_setup(0.5)
    L = 20 * L
    s = st['acc'] * st['Sq']
    mask = L.abs() < 0.5 * m.fmt.step_size(s + L)
    m._merge(st, L)
    assert abs(m.stats['kept'] - mask.float().mean().item()) < 1e-9 and 0 < m.stats['kept'] < 1
    # k = 0 leaves no bypass
    m, st, L = _merge_setup(0.5, k=0)
    s = st['acc'] * st['Sq']
    out = m._merge(st, L)
    assert st['K'].shape[-1] == 0 and torch.equal(out, m.fmt.fake(s + L))


def test_lrb_bypass_holds_k_plus_m_columns():
    m, x = Tiny(), inputs(T=40)
    q = LRB(k=1, m=15, eps_L=0.5, eps_B=0.3)
    cols = []
    orig = q.step

    def spy(ctx):
        out = orig(ctx)
        cols.append(q.layers[0]['K'].shape[-1])
        return out
    q.step = spy
    run(m, x, q, decode=8)
    # columns after each token: 1..15 before the first merge, then k=1 after it and k + j after j more tokens
    assert cols[:16] == list(range(1, 15)) + [1, 2] and max(cols) <= 16 and cols[29] == 1


def test_lrb_fine_grid_tracks_fp32_and_records_agree():
    """With a 15-bit grid LRB is close to the fp32 trajectory; fp16 and fp32 records give nearly the same result."""
    m, x = Tiny(), inputs(T=48)
    ref, Sref = run(m, x, decode=8)
    fine = AffineGrid(15, 128)
    o16, S16 = run(m, x, LRB(1, 15, 0.5, 0.3, fmt=fine, records='fp16'), decode=8)
    o32, S32 = run(m, x, LRB(1, 15, 0.5, 0.3, fmt=fine, records='fp32'), decode=8)
    assert rel(o16, ref) < 2e-3 and rel(o32, ref) < 2e-3 and rel(o16, o32) < 2e-3
    q4 = LRB(1, 15, 0.5, 0.3, records='fp32')
    o4, _ = run(m, x, q4, decode=8)
    oi, _ = run(m, x, sq.int4_g128(), decode=8)
    assert rel(o4, ref) < rel(oi, ref)                       # the bypass helps on this toy too


def test_lrb_accounting_matches_paper():
    """Table 6: GDN / RetNet (d_k 256, d_v 512) avg 5.25 peak 6.00; GLA (per-row decay) 5.31 / 6.07."""
    q = sq.lrb_bdr('gdn')
    assert round(q.bits_per_value(256, 512), 2) == 5.25 and round(q.peak_bits_per_value(256, 512), 2) == 6.00
    gla = LRB(**sq.methods_ext.LRB_CONFIGS['gla'], decay_scalars=256)
    assert round(gla.bits_per_value(256, 512), 2) == 5.31 and abs(gla.peak_bits_per_value(256, 512) - 6.07) < 0.01
    # Table 8/9 rows (GDN): k=0,m=16 -> 5.20; k=8,m=8 -> 5.58; k+m=8 -> 4.88 / 5.25; k+m=24 -> 5.63 / 6.76
    for k, mm, avg in ((0, 16, 5.20), (2, 14, 5.30), (4, 12, 5.39), (8, 8, 5.58), (1, 7, 4.88), (1, 23, 5.63)):
        assert abs(LRB(k, mm).bits_per_value(256, 512) - avg) < 0.006, (k, mm)


# ------------------------------------------------------------------------------------------------------------------
# LeapQuant
# ------------------------------------------------------------------------------------------------------------------
def outlier_state(seed=0, B=1, H=2, K=128, V=128):
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(B, H, K, V, generator=g) * 0.1
    S = S + 5 * torch.randn(B, H, K, 1, generator=g) * torch.randn(B, H, 1, V, generator=g)    # a strong rank-one part
    S[..., 3, :] *= 8                                                                            # an outlier row
    return S


def test_compensators_reduce_residual_range():
    S = outlier_state()
    for r in (1, 4, 8):
        K, U = compensators(S, r, iters=4, generator=torch.Generator().manual_seed(0))
        R = S - K.float() @ U.float().transpose(-1, -2)
        assert R.abs().amax() < 0.1 * S.abs().amax()
    U0, Sv, Vh = torch.linalg.svd(S)
    K, U = compensators(S, 4, iters=8, generator=torch.Generator().manual_seed(0))
    best = (S - (U0[..., :4] * Sv[..., None, :4]) @ Vh[..., :4, :]).norm()
    assert (S - K.float() @ U.float().transpose(-1, -2)).norm() < 1.01 * best


def test_leapquant_boundary_error_below_plain_int8():
    S = outlier_state(1)
    lq = LeapQuant(8)
    lq.reset()
    out, _ = lq.quantize_state(S)
    plain = sq.Quantizer(8, 'k', 128).fake(S)
    assert (out - S).norm() < 0.5 * (plain - S).norm()
    lq0 = LeapQuant(8, r=0)
    lq0.reset()
    smooth_only, _ = lq0.quantize_state(S)
    assert (out - S).norm() < (smooth_only - S).norm()


def test_leapquant_accounting():
    assert LeapQuant(8).bits_per_value(128, 128) / 8 == 1.1875                 # "1.19 bytes per element"
    assert LeapQuant(4).r == 8 and LeapQuant(6).r == 4
    assert abs(LeapQuant(8).buffer_bits_per_value(128, 128) * 128 * 128 / 8 / 1024 - 8.03) < 0.01   # ~8 KiB/head


def test_leapquant_windows_and_records():
    """Windows close every p decode tokens and at the prompt end; with a fine grid the method is close to fp32 and
    fp16 / fp32 records agree."""
    m, x = Tiny(), inputs(T=72)
    ref, _ = run(m, x, decode=40)
    q = LeapQuant(15, r=4, records='fp16')
    seen = []
    orig = q._boundary
    q._boundary = lambda st, Sp: (seen.append(True), orig(st, Sp))[1]
    o16, _ = run(m, x, q, decode=40)
    assert len(seen) == 1 + 40 // 16
    o32, _ = run(m, x, LeapQuant(15, r=4, records='fp32'), decode=40)
    assert rel(o16, ref) < 2e-3 and rel(o32, ref) < 1e-4 and rel(o16, o32) < 2e-3
    o8, _ = run(m, x, LeapQuant(8), decode=40)
    oi8, _ = run(m, x, sq.rtn(8, 'k', 128), decode=40)
    assert rel(o8, ref) < rel(oi8, ref)
    assert torch.equal(o8[:, :32], ref[:, :32])              # the prompt itself runs in full precision


# ------------------------------------------------------------------------------------------------------------------
# Gramian allocation
# ------------------------------------------------------------------------------------------------------------------
def test_allocation_respects_budget():
    g = torch.Generator().manual_seed(0)
    for mean_bits, lo, hi in ((4, 2, 16), (4, 2, 8), (2, 1, 8), (6, 2, 16), (3, 2, 8)):
        c = torch.rand(5, 16, generator=g).pow(8) * 1e3
        b = allocate(c, mean_bits, lo, hi)
        assert (b.sum(-1) == mean_bits * 16).all() and (b >= lo).all() and (b <= hi).all()
        order = c.argsort(-1)
        assert (b.gather(-1, order).diff(dim=-1) >= 0).all()                 # larger weight -> no fewer bits
    b = allocate(torch.ones(1, 16), 4, 2, 16)
    assert (b == 4).all()
    # the unconstrained rule: c ratio 16 -> 2 bits apart
    b = allocate(torch.tensor([[16., 1.]]), 4, 2, 16)
    assert b.tolist() == [[5, 3]]


def test_clip_search_and_one_bit():
    X = state(0).transpose(-1, -2).reshape(2, 2, 128, 4, 32)
    for b in (2, 3, 4, 8):
        q = clip_quantize(X, b)
        n = 2 ** (b - 1) - 1
        step = (X.abs().amax(-1, keepdim=True) / n).half().float()
        q1 = torch.round(X / step).clamp(-n, n) * step                       # f = 1
        assert ((q - X).square().sum(-1) <= (q1 - X).square().sum(-1) + 1e-6).all()
    q = clip_quantize(X, 1)
    assert torch.allclose(q.abs(), X.abs().mean(-1, keepdim=True).half().float().expand_as(X))
    assert (torch.sign(q) == torch.sign(X)).all()


def test_ema_chunk_invariant():
    x = torch.rand(2, 70, 3, dtype=torch.float64)
    a = 2 / 65
    n1, d1 = ema_update(0., 0., x, a)
    n2, d2 = 0., 0.
    for i in range(0, 70, 7):
        n2, d2 = ema_update(n2, d2, x[:, i:i + 7], a)
    assert torch.allclose(n1 / d1, n2 / d2)
    n3, d3 = ema_update(0., 0., x[:, :1], a)
    assert torch.allclose(n3 / d3, x[:, 0])                                  # debiased


def test_gramian_slow_heads_get_more_bits():
    m, x = Tiny(), inputs(T=20)                                             # head 0 decays slowly, head 1 fast
    q = GramianAlloc(4, C=1)
    run(m, x, q)
    w = q.widths[0]
    assert w.sum().item() == 8 and w[0, 0] > w[0, 1]
    assert q.bits_per_value(128, 128) == 4 + 0.5 + 4 / 128 ** 2
    assert GramianAlloc(4, C=64).bits_per_value(128, 128) == 4 + 0.5 + 3 / 128 ** 2
    o_g, _ = run(m, x, q)
    ref, _ = run(m, x)
    o_u, _ = run(m, x, GramianAlloc(4, C=1, allocation='uniform'))
    assert rel(o_g, ref) > 0 and rel(o_u, ref) > 0


# ------------------------------------------------------------------------------------------------------------------
# DAMP paper recipe
# ------------------------------------------------------------------------------------------------------------------
def test_sr_exact_inverse():
    S = state(0)
    rms = S.square().mean((0, 2)).sqrt()
    for i, (c, perm) in enumerate(sr_candidates(rms)):
        T = sr_forward(S, c, perm)
        assert torch.allclose(sr_inverse(T, c, perm), S, rtol=1e-6, atol=0)
        # with an identity storage format the whole SR round trip is exact up to float rounding of the scaling
        assert torch.allclose(sr_inverse(sq.Identity().fake(T), c, perm), S, rtol=1e-6, atol=0)
        # the order sorts value dims by rms / c
        if i > 0:                                                          # 0 is the identity candidate
            r = torch.gather(rms / c, -1, perm)
            assert (r.diff(dim=-1) <= 0).all()
    c, perm = sr_candidates(rms, (None,))[0]
    assert torch.equal(sr_forward(S, c, perm), S)


def test_damp_paper_calibration_and_accounting():
    torch.manual_seed(0)
    q = DampPaper(8, anchors=(256, 512))
    for seq in range(4):
        S = state(seq, B=1)
        S[:, :, 7] *= 30                                                    # key channel 7 dominates the error
        for pos in (256, 512, 768):
            lg = torch.full((1, 256, 2), math.log(0.999))
            pc = Piece({}, BY_NAME['gdn'], 0, None, torch.tensor([pos]))
            q.observe(0, S * (1 + pos / 1000), lg, pc)
    q.finalize(device='cpu')
    assert q.masks[0].shape == (2, 128) and (q.masks[0].sum(-1) == 16).all() and q.masks[0][:, 7].all()
    S = state(9, B=1)
    out = q.quantize(S, 0)
    assert rel(out, S) < 1e-2
    assert q.bits_per_value(128, 128) == 9.875 and DampPaper(4).bits_per_value(128, 128) == 7.25
    assert q.schedule.decode == 1 and q.schedule.prompt_end


def test_bdr_on_a_periodic_schedule():
    """BDR is a rounding rule for any schedule: with an infinite threshold it is the plain format on that schedule;
    s is the state stored at the previous rounding decayed since, so a finite threshold changes the result."""
    m, x = Tiny(), inputs(T=48)
    fmt = sq.Quantizer(4, 'k', 32)
    sch = sq.Schedule(prefill=16, decode=4, buffer_bits=0)
    plain, _ = run(m, x, sq.Uniform(fmt), sch, decode=12)
    inf, _ = run(m, x, sq.BDR(fmt, math.inf), sch, decode=12)
    assert torch.equal(plain, inf)
    b, _ = run(m, x, sq.BDR(fmt, 0.3), sch, decode=12)
    assert not torch.equal(plain, b)
    assert torch.equal(b[:, :16], plain[:, :16])                       # identical until the first rounding is read


def test_damp_ragged_value_groups():
    """d_v = 80 (Nemotron-3-Nano's Mamba2 head_dim): groups 32, 32, 16; padding by the last value is exact."""
    T = state(0, V=80)
    for fmt, n in ((sq.Quantizer(8, 'v', 32), 9.0), (sq.Quantizer(4, 'v', 32, affine=True, zero_point='fp'), 6.0)):
        out = DampPaper._fake(fmt, T)
        f16 = sq.Quantizer(fmt.bits, 'v', 16, affine=fmt.affine, zero_point=fmt.zero_point)
        ref = torch.cat([fmt.fake(T[..., :64]), f16.fake(T[..., 64:])], -1)
        assert torch.equal(out, ref)
    q = DampPaper(8)
    assert abs(q.bits_per_value(128, 80) - (16 / 128 * 16 + 112 / 128 * (8 + 32 * 3 / 80))) < 1e-12
    assert q.bits_per_value(128, 128) == 9.875


def _allocate_sequential(c, mean_bits, b_min, b_max):
    """The one-bit-at-a-time greedy (reference for ``allocate``)."""
    c = c.double().clamp_min(1e-300)
    n = c.shape[-1]
    lc = c.log2()
    b = (mean_bits + 0.5 * (lc - lc.mean(-1, keepdim=True))).round().clamp(b_min, b_max)
    budget = int(round(mean_bits * n))
    for _ in range(n * (b_max - b_min) + 1):
        tot = b.sum(-1)
        over, under = tot > budget, tot < budget
        if not bool(over.any() or under.any()):
            break
        cost = torch.where(b > b_min, c * 3 * 4. ** (-b), torch.full_like(c, math.inf))
        gain = torch.where(b < b_max, c * 0.75 * 4. ** (-b), torch.full_like(c, -math.inf))
        dec = torch.nn.functional.one_hot(cost.argmin(-1), n).bool() & over[..., None]
        inc = torch.nn.functional.one_hot(gain.argmax(-1), n).bool() & under[..., None]
        b = b - dec.double() + inc.double()
    return b.long()


def test_allocate_equals_sequential_greedy():
    g = torch.Generator().manual_seed(7)
    for mean_bits, lo, hi, n in ((4, 2, 16, 16), (4, 1, 8, 300), (2, 1, 8, 64), (6, 2, 16, 18), (3, 2, 8, 5)):
        for p in (1, 4, 12):
            c = torch.rand(6, n, generator=g).pow(p) * 10 ** torch.randint(-3, 3, (6, 1), generator=g).double()
            c[0, : n // 2] = c[0, 0]                                         # ties
            assert torch.equal(allocate(c, mean_bits, lo, hi), _allocate_sequential(c, mean_bits, lo, hi))
