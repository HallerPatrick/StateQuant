"""Every recurrent-state family on the reference path, with tiny randomly initialised transformers models built from
the real config classes (``_tiny.py``): Gated DeltaNet (Qwen3-Next, Qwen4-Exp, OLMo-Hybrid with d_k 96 / d_v 192),
KDA (Kimi Linear, Bailing hybrid), Mamba2 (Mamba2, Nemotron-H), MiniMax lightning.

(a) Identity through ``sq.apply`` reproduces the unpatched logits exactly (prompt + cached decoding);
(b) a quantizing format changes the state only at the schedule's rounding points: the boundaries fall exactly on
    the scheduled positions, each piece starts from the previous boundary's rounded state, and cached decoding equals
    a manual loop that runs the unpatched model token by token and rounds the cached state in place;
(c) the rounding positions do not depend on how the sequence is split into calls (one prompt call, two prompt calls,
    prompt + single-token decoding);
(d) calibration + HeadMix / ChannelProtect / StepQuant run end to end; ``report`` reads the state shape.

Runs on the GPU if there is one, else on CPU. Where fla is installed transformers dispatches the GDN / KDA ops to its
Triton kernels, so those models are skipped on CPU in that environment (run the vLLM venv for the torch paths)."""
import importlib.util

import pytest
import torch

import statequant as sq
from statequant import families as fam_mod
from _tiny import BUILDERS, FAMILY, V, build, run

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HAS_FLA = importlib.util.find_spec('fla') is not None
NAMES = list(BUILDERS)


def _model(name):
    if name.startswith('fla_') and not (HAS_FLA and DEV == 'cuda'):
        pytest.skip('fla layers need fla and a GPU')
    if DEV == 'cpu' and HAS_FLA and FAMILY[name] in ('gdn', 'kda'):
        pytest.skip('transformers dispatches this op to fla (Triton): needs a GPU')
    try:
        return build(name).to(DEV)
    except ImportError as e:
        pytest.skip(f'{name} not in this transformers: {e}')


_CACHE = {}


@pytest.fixture(params=NAMES)
def model(request):
    if request.param not in _CACHE:
        _CACHE[request.param] = _model(request.param)
    return request.param, _CACHE[request.param]


def ids(T, B=2, seed=1):
    return torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(seed)).to(DEV)


def tol(name):
    """Kernel-rounding tolerance for comparisons between different call splits (relative to max |logit|)."""
    fla = name.startswith('fla_') or FAMILY[name] in ('gdn', 'kda')
    return 2e-2 if (DEV == 'cuda' and HAS_FLA and fla) else 1e-4


def fmt(name):
    return sq.Quantizer(3, 'k', 16)


def test_family_detected(model):
    name, m = model
    fams = sq.layer_families(m)
    assert fams and set(fams) == {FAMILY[name]}


# (a) --------------------------------------------------------------------------------------------------------------
def test_identity_exact(model):
    name, m = model
    x = ids(40)
    ref = run(m, x, 8)
    with sq.apply(m, sq.Identity(), sq.Schedule(prefill=64, decode=4, buffer_bits=0)):
        out = run(m, x, 8)
    assert torch.equal(out, ref)
    with sq.apply(m, sq.Identity(), sq.Schedule(prefill=16, decode=4, buffer_bits=0)):   # prompt split in 3 pieces
        out = run(m, x, 8)
    assert (out - ref).abs().max() <= tol(name) * ref.abs().max()
    assert torch.equal(run(m, x, 8), ref)                                                   # patch removed on exit


# (b) --------------------------------------------------------------------------------------------------------------
class Recorder(sq.StateQuantizer):
    """Rounds with ``fmt`` and logs every boundary: (layer, state handed in, rounded state, piece's start state)."""

    def __init__(self, fmt):
        self.fmt, self.log = fmt, []

    def quantize(self, S, layer):
        return self.fmt.fake(S)

    def bits_per_value(self, d_k, d_v):
        return self.fmt.bits_per_value(d_k, d_v)


def _recording_apply(m, rec, sch):
    from statequant import patch

    def boundary(S, layer, piece):
        Sq = rec.quantize(S, layer)
        rec.log.append((layer, S.clone(), Sq.clone(), piece.state_before, piece))
        return Sq
    return patch._patched(m, sch, boundary)


def test_boundaries_at_schedule_positions(model):
    name, m = model
    T, n, p, d = 40, 9, 16, 4
    rec = Recorder(fmt(name))
    with _recording_apply(m, rec, sq.Schedule(prefill=p, decode=d, buffer_bits=0)):
        run(m, ids(T), n)
    L = len(sq.find_layers(m))
    expect = [t for t in range(1, T + 1) if t % p == 0] + [t for t in range(T + 1, T + n + 1) if t % d == 0]
    assert len(rec.log) == L * len(expect)
    per_layer = {}
    for layer, S, Sq, before, piece in rec.log:
        per_layer.setdefault(layer, []).append((S, Sq, before))
    n_prompt = sum(t <= T for t in expect)
    for ev in per_layer.values():
        assert ev[0][2] is None                       # the first piece starts the sequence (zero state)
        for i in range(1, n_prompt):                  # each prompt piece starts from the previous rounded state
            assert torch.equal(ev[i][2], ev[i - 1][1])
        for S, Sq, _ in ev:
            assert not torch.equal(S, Sq)             # and the rounding changed it


def _cache_states(m, cache):
    """(family, get, put) of every recurrent layer's cached state (native layout), in find_layers order."""
    out = []
    for layer in sq.find_layers(m):
        fam = fam_mod.family_of(layer)
        i = layer.layer_idx
        if hasattr(cache, 'linear_cache'):                                   # MiniMaxCache
            get = lambda i=i: cache.linear_cache[i]                          # noqa: E731

            def put(S, i=i):
                cache.linear_cache[i] = S
        elif hasattr(cache, 'layers') and hasattr(cache.layers[i], 'recurrent_states'):   # transformers
            get = lambda i=i: cache.layers[i].recurrent_states[0]            # noqa: E731

            def put(S, i=i):
                cache.layers[i].recurrent_states[0].copy_(S)
        else:                                                                # fla
            get = lambda i=i: cache[i]['recurrent_state']                    # noqa: E731

            def put(S, i=i):
                cache[i]['recurrent_state'].copy_(S)
        kw = {'state_v_first': name_is_fla(layer) and fam.name in ('gdn', 'kda')}      # fla's GDN / KDA: v-first
        out.append((fam, get, put, kw))
    return out


def name_is_fla(layer):
    return type(layer).__module__.startswith('fla.')


@torch.no_grad()
def _manual_decode(m, x, forced, d, q):
    """Unpatched model: prompt in one call (no rounding), then one token per call, rounding every cached state in
    place after the update of every d-th token."""
    from _tiny import _cache
    out = m(input_ids=x, use_cache=True)
    logits = [out.logits]
    T = x.shape[1]
    for i in range(forced.shape[1]):
        name, c = _cache(out)
        out = m(input_ids=forced[:, i:i + 1], use_cache=True, **{name: c})
        logits.append(out.logits)
        if (T + i + 1) % d == 0:
            name, c = _cache(out)
            for fam, get, put, kw in _cache_states(m, c):
                S = get()
                put(fam.state_out(q.fake(fam.state_in(S, kw)), S, kw))
    return torch.cat(logits, 1).float()


def test_decode_equals_manual_loop(model):
    name, m = model
    x, forced, d = ids(24), ids(13, seed=2), 4
    q = fmt(name)
    man = _manual_decode(m, x, forced, d, q)
    with sq.apply(m, q, sq.Schedule(prefill=1000, decode=d, buffer_bits=0)):
        pat = run(m, x, forced.shape[1], forced)
    assert torch.equal(pat, man)
    assert (pat - run(m, x, forced.shape[1], forced)).abs().max() > 0         # the rounding did something


# (c) --------------------------------------------------------------------------------------------------------------
# transformers' own cached decoding of these tiny models disagrees with one-call prefill even without statequant
# (Bamba 5.19 / 5.20: also with attention layers only), so split invariance cannot be checked on them
HF_SPLIT_BROKEN = {'bamba'}


def test_positions_independent_of_split(model):
    name, m = model
    if name in HF_SPLIT_BROKEN:
        pytest.skip(f'unpatched {name} is not split-invariant in transformers')
    x = ids(48)
    sch = sq.Schedule(prefill=8, decode=8, buffer_bits=0)
    pos, out = {}, {}
    for mode in ('one', 'two', 'dec'):
        rec = Recorder(fmt(name))
        with _recording_apply(m, rec, sch):
            out[mode] = (run(m, x) if mode == 'one' else run(m, x, split=21) if mode == 'two' else
                         run(m, x[:, :29], 19, forced=x[:, 29:]))
        pos[mode] = sorted((e[0], int(e[4].position[0])) for e in rec.log)
    L = len(sq.find_layers(m))
    assert pos['one'] == sorted((layer, t) for layer in range(L) for t in range(8, 49, 8))
    assert pos['two'] == pos['one'] and pos['dec'] == pos['one']
    ref = {'one': run(m, x), 'two': run(m, x, split=21), 'dec': run(m, x[:, :29], 19, forced=x[:, 29:])}
    eff = (out['one'] - ref['one']).norm()                                    # the effect of the rounding itself
    assert eff > 0
    for mode in ('two', 'dec'):
        # equal up to kernel rounding (which may flip a few codes) and transformers' own split discrepancy
        base = (ref[mode] - ref['one']).norm()
        assert (out[mode] - out['one']).norm() < 0.1 * eff + 5 * base


def test_decode_buffer_rounding_runs(model):
    name, m = model
    x = ids(24)
    with sq.apply(m, q := fmt(name), sq.Schedule(prefill=8, decode=4, buffer_bits=8)):
        a = run(m, x, 6)
    with sq.apply(m, q, sq.Schedule(prefill=8, decode=4, buffer_bits=0)):
        b = run(m, x, 6)
    assert torch.isfinite(a).all() and torch.equal(a[:, :24], b[:, :24])      # the buffer only rounds decoded tokens


# (d) --------------------------------------------------------------------------------------------------------------
def test_calibrate_headmix_channelprotect(model):
    name, m = model
    lo, hi = sq.Quantizer(4, 'k', 16), sq.Quantizer(8, 'k', 16)
    L = len(sq.find_layers(m))
    hm = sq.calibrate(m, sq.HeadMix(lo, hi, frac=0.5), ids(64), period=16)
    assert set(hm.masks) == set(range(L))
    H = sq.report(m, hm)['heads']
    assert all(int(v.sum()) == max(1, round(0.5 * H)) for v in hm.masks.values())
    assert all(torch.isfinite(s).all() for s in hm.scores.values())
    with sq.apply(m, hm, sq.Schedule(16, 4, 8)):
        assert torch.isfinite(run(m, ids(32), 4)).all()
    cp = sq.calibrate(m, sq.ChannelProtect(lo, hi, k=4), ids(64), period=16)
    with sq.apply(m, cp, sq.Schedule(16, 4, 8)):
        assert torch.isfinite(run(m, ids(32), 4)).all()
    r = sq.report(m, hm)
    d_k, d_v, h_v, _ = fam_mod.family_of(sq.find_layers(m)[0]).dims(sq.find_layers(m)[0])
    assert (r['d_k'], r['d_v'], r['heads']) == (d_k, d_v, h_v)


def test_calibrate_stepquant(model):
    name, m = model
    layers = sq.find_layers(m)
    n_heads = sum(fam_mod.family_of(x).dims(x)[2] for x in layers)
    piv = 1 if n_heads >= 4 else 0                        # a pivot costs 8 bits of a 4-bit budget per head
    if FAMILY[name] == 'rwkv7':
        with pytest.raises(NotImplementedError):
            sq.calibrate(m, sq.StepQuant(4, pivots=piv), ids(32))
        return
    q = sq.calibrate(m, sq.StepQuant(4, pivots=piv), ids(32))
    assert len(q.plan) == len(layers)
    bits = torch.cat([q.plan[i][0] for i in range(len(layers))])
    assert int((bits == 16).sum()) == piv and set(bits.tolist()) <= {2, 4, 6, 8, 16}
    assert bits[bits != 16].sum() <= 4 * n_heads - 8 * piv
    with sq.apply(m, q, sq.per_token()):
        out = run(m, ids(20), 3)
    assert torch.isfinite(out).all()


def test_canonical_layout(model):
    """Every boundary sees [B, H, d_k, d_v] with the adapter's (d_k, d_v, H)."""
    name, m = model
    rec = Recorder(sq.Identity())
    with _recording_apply(m, rec, sq.Schedule(8, 4, 0)):
        run(m, ids(16), 4)
    layers = sq.find_layers(m)
    for layer, S, *_ in rec.log:
        d_k, d_v, h, _ = fam_mod.family_of(layers[layer]).dims(layers[layer])
        assert tuple(S.shape) == (2, h, d_k, d_v) and S.dtype == torch.float32


def test_piece_inputs(model):
    """The adapter's canonical per-token inputs have the documented shapes, and the log decay is negative."""
    name, m = model
    rec = Recorder(sq.Identity())
    with _recording_apply(m, rec, sq.Schedule(8, 4, 0)):
        run(m, ids(16), 4)
    layers = sq.find_layers(m)
    for layer, S, _, _, piece in rec.log:
        x = piece.inputs()
        d_k, d_v, h, _ = fam_mod.family_of(layers[layer]).dims(layers[layer])
        B, T = x['q'].shape[:2]
        assert x['q'].shape == x['k'].shape == (B, T, h, d_k) and x['v'].shape == (B, T, h, d_v)
        lg = x['log_decay']
        assert lg.shape in ((B, T, h), (B, T, h, d_k)) and (lg <= 0).all()
        assert (lg.dim() == 4) == (fam_mod.family_of(layers[layer]).decay == 'channel')
        assert x['beta'] is None or x['beta'].shape == (B, T, h)


def test_packed_backend_refuses_other_families(model):
    name, m = model
    if FAMILY[name] == 'gdn':
        pytest.skip('gdn has a packed backend')
    with pytest.raises(NotImplementedError):
        with sq.apply(m, sq.Quantizer(4, 'k', 16), sq.Schedule(16, 4, 8), backend='packed'):
            pass
