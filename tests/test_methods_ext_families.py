"""The methods of statequant.methods_ext on every recurrent-state family (tiny models from ``_tiny.py``): GDN
(Qwen3-Next), KDA (Kimi Linear), Mamba2 (Mamba2, Nemotron-H), MiniMax lightning; RWKV-7 (fla) must refuse.

(a) every method runs, prompt + cached decoding (Mamba2's decode is the in-place step op);
(b) identity-equivalent settings reproduce the full-precision logits (exactly for BDR with an Identity grid; up to
    float rounding where the method re-adds a decomposed state);
(c) rounding / merge / window positions are the schedule's (BDR, Gramian), every m tokens with the family's LRB
    configuration, and the prompt end + every p decode tokens (LeapQuant);
(d) the update u recovered for an additive family equals the op's own rank-one update; Gramian units per family;
    LeapQuant leaves the prompt in full precision.
"""
import importlib.util

import pytest
import torch

import statequant as sq
from statequant.methods_ext import BDR, LRB, DampPaper, GramianAlloc, LeapQuant
from statequant.methods_ext.bdr_lrb import FAMILY_ARCH, LRB_CONFIGS
from statequant.methods_ext.hooks import decay_shape, rank1_value, terms
from _tiny import FAMILY, V, build, run

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HAS_FLA = importlib.util.find_spec('fla') is not None
MODELS = ['qwen3_next', 'kimi_linear', 'mamba2', 'nemotron_h', 'minimax']
_CACHE = {}


def _model(name):
    if name.startswith('fla_') and not (HAS_FLA and DEV == 'cuda'):
        pytest.skip('fla layers need fla and a GPU')
    if DEV == 'cpu' and HAS_FLA and FAMILY[name] in ('gdn', 'kda'):
        pytest.skip('transformers dispatches this op to fla (Triton): needs a GPU')
    if name not in _CACHE:
        try:
            _CACHE[name] = build(name).to(DEV)
        except ImportError as e:
            pytest.skip(f'{name} not in this transformers: {e}')
    return _CACHE[name]


@pytest.fixture(params=MODELS)
def model(request):
    return request.param, _model(request.param)


def ids(T, B=1, seed=1):
    return torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(seed)).to(DEV)


def tol(name):
    fla = FAMILY[name] in ('gdn', 'kda')
    return 2e-2 if (DEV == 'cuda' and HAS_FLA and fla) else 1e-4


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def methods():
    return {'bdr': BDR(), 'lrb_bdr': sq.lrb_bdr(), 'leapquant8': sq.leapquant(8), 'gramian4': sq.gramian(4, 1),
            'gramian4_c64': sq.gramian(4, 64)}


@pytest.mark.parametrize('meth', list(methods()))
def test_runs(model, meth):
    name, m = model
    x = ids(40)
    ref = run(m, x, 12)
    q = methods()[meth]
    if meth == 'bdr':
        q.schedule = sq.per_token()
    with sq.apply(m, q):
        out = run(m, x, 12)
    assert torch.isfinite(out).all() and not torch.equal(out, ref)
    if meth == 'leapquant8':                         # the prompt runs in full precision, rounded at its end
        assert torch.equal(out[:, :40], ref[:, :40])
    r = sq.report(m, q)
    assert r['payload_bits_per_value'] > 0


def test_damp_paper_runs(model):
    name, m = model
    q = DampPaper(8, k_hi=4, horizon=32, anchors=(32, 64))
    sq.calibrate(m, q, ids(128, B=2, seed=3))
    assert all(int(mk.sum(-1).min()) == 4 for mk in q.masks.values())
    x = ids(40)
    ref = run(m, x, 8)
    with sq.apply(m, q):
        out = run(m, x, 8)
    assert torch.isfinite(out).all() and torch.equal(out[:, :40], ref[:, :40])     # fp32 prefill (one chunk)


# (b) --------------------------------------------------------------------------------------------------------------
def test_bdr_identity_grid_is_exact(model):
    name, m = model
    x = ids(40)
    sch = sq.Schedule(prefill=64, decode=4, buffer_bits=0)
    ref = run(m, x, 8)
    with sq.apply(m, BDR(sq.Identity(), 0.3), sch):
        assert torch.equal(run(m, x, 8), ref)


@pytest.mark.parametrize('records', ['fp32', 'fp16'])
def test_lrb_leapquant_damp_identity_equivalent(model, records):
    """Identity storage: LRB (state = decayed stored state + bypass, re-added at merges) and LeapQuant (compensators
    + smoothed residual) reproduce the identity carry on the same schedule up to float rounding (fp16 records: up to
    the fp16 rounding of the bypass factors / window records)."""
    name, m = model
    x = ids(40)
    eps = 1e-5 if records == 'fp32' else 2e-2
    sch = sq.Schedule(prefill=4, decode=4, buffer_bits=0) if records == 'fp32' else sq.per_token()
    with sq.apply(m, sq.Identity(), sch):
        base = run(m, x, 12)
    q = LRB(m=4, fmt=sq.Identity(), arch='auto', records=records)
    with sq.apply(m, q, sch):
        out = run(m, x, 12)
    assert rel(out, base) < eps, rel(out, base)
    lq = LeapQuant(None, r=2, p=4, records=records)
    with sq.apply(m, lq):
        out = run(m, x, 12)
    ref = run(m, x, 12)
    assert rel(out, ref) < max(eps, tol(name)), rel(out, ref)
    if records == 'fp32':
        dq = DampPaper(8, k_hi=0, horizon=32, anchors=(32,), lo=sq.Identity())
        sq.calibrate(m, dq, ids(64, B=2, seed=3))
        with sq.apply(m, dq):
            out = run(m, x, 12)
        assert rel(out, ref) < 1e-5


# (c) --------------------------------------------------------------------------------------------------------------
def _spy(q, record):
    orig = q.step

    def spy(ctx):
        record(ctx)
        return orig(ctx)
    q.step = spy
    return q


def test_bdr_and_gramian_round_at_schedule_positions(model):
    name, m = model
    x = ids(40)
    for q, sch, expect in ((BDR(), sq.Schedule(prefill=16, decode=4, buffer_bits=0), [16, 32, 44, 48]),
                           (GramianAlloc(4, C=64), None, list(range(41, 49))),
                           (GramianAlloc(4, C=1), None, list(range(1, 49)))):
        hits = {}
        _spy(q, lambda c: bool(c.hit.any()) and hits.setdefault(c.layer, []).append(int(c.pos[0])))
        with sq.apply(m, q, sch):
            run(m, x, 8)
        assert hits and all(h == expect for h in hits.values()), (type(q).__name__, hits)


def test_lrb_merges_every_m_with_the_family_config(model):
    name, m = model
    fam = FAMILY[name]
    mm = LRB_CONFIGS[FAMILY_ARCH[fam]]['m']
    q = sq.lrb_bdr()
    merges = {}
    orig = q._merge

    def mspy(st, L, cfg=None):
        merges.setdefault('cfg', cfg)
        merges['n'] = merges.get('n', 0) + 1
        return orig(st, L, cfg)
    q._merge = mspy
    with sq.apply(m, q):
        run(m, ids(40), 8)
    n_layers = len(sq.find_layers(m))
    assert merges['cfg']['m'] == mm and merges['n'] == n_layers * (48 // mm)
    assert q.peak_bits_per_value(16, 32) > q.bits_per_value(16, 32)


def test_leapquant_windows(model):
    name, m = model
    q = LeapQuant(8, r=2, p=16)
    seen = {}
    orig = q._boundary

    def bspy(st, Sp):
        seen['n'] = seen.get('n', 0) + 1
        return orig(st, Sp)
    q._boundary = bspy
    with sq.apply(m, q):
        run(m, ids(40), 30)                                # boundaries: prompt end 40, then 48, 64
    assert seen['n'] == 3 * len(sq.find_layers(m))


# (d) --------------------------------------------------------------------------------------------------------------
def test_written_value_matches_the_op(model):
    """For every family the update recovered by the methods is the op's own rank-one update S_t - A_t S_{t-1}."""
    name, m = model
    q = sq.lrb_bdr()
    errs = []

    def rec(ctx):
        lg, key, _, v, rule = terms(ctx.pc)
        if lg.shape[1] != 1:
            return
        U = ctx.S_out - decay_shape(lg[:, 0], ctx.S_out).exp() * ctx.S_in
        u = v[:, 0] if rule == 'additive' else rank1_value(U, key[:, 0])
        errs.append(((U - key[:, 0, :, :, None] * u[..., None, :]).norm() / U.norm().clamp_min(1e-30)).item())
    _spy(q, rec)
    with sq.apply(m, q):
        run(m, ids(24), 6)
    assert errs and max(errs) < 2e-2 if FAMILY[name] in ('gdn', 'kda') and DEV == 'cuda' else max(errs) < 1e-4


def test_gramian_units_per_family(model):
    name, m = model
    q = GramianAlloc(4, C=1)
    with sq.apply(m, q):
        run(m, ids(24), 4)
    fam = FAMILY[name]
    d_k, d_v, H, _ = sq.families.family_of(sq.find_layers(m)[0]).dims(sq.find_layers(m)[0])
    w = next(iter(q.widths.values()))
    n_units = H if fam == 'gdn' else H * d_k
    assert w.shape[-1] == n_units and (w.sum(-1) == 4 * n_units).all()
    assert w.min() >= (2 if fam == 'gdn' else 1) and w.max() <= 16
    expect = 4 + 0.5 + 4 / (d_k * d_v) if fam == 'gdn' else 4 + 20 / d_v
    assert abs(q.bits_per_value(d_k, d_v) - expect) < 1e-12


@pytest.mark.parametrize('meth', ['bdr', 'lrb_bdr', 'leapquant8', 'gramian4'])
def test_rwkv7_refuses(meth):
    m = _model('fla_rwkv7')
    q = methods()[meth]
    with pytest.raises(NotImplementedError, match='rwkv7'):
        with sq.apply(m, q):
            run(m, ids(8), 1)
    with pytest.raises(NotImplementedError, match='rwkv7'):
        sq.calibrate(m, DampPaper(8, k_hi=4, horizon=32, anchors=(32,)), ids(64))
