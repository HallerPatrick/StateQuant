"""The fused single-launch decode step (``_fused_triton._fused_step``) against the five-kernel step it replaces:
lockstep (``PackedGDNState(fused=True)`` vs ``fused=False``) and slot-addressed (``SlotLayout.step(fused=...)``,
random slot subsets with padding rows), GQA 2, CUDA-graph replay. Needs a GPU and Triton.

Every step compares the outputs and every stored byte (codes, scales, key / delta-value buffers and their scales,
decays, counters). The fused kernel is written instruction for instruction like the unfused ones; where Triton picks
a different reduction tree, fp32 sums can differ in the last bit, which can flip a rounding (a code or a buffer
entry) at a tie. Such differences are counted (run with ``-s``) and the unfused copy is re-synced to the fused one,
so a flip is counted once and not inherited."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('needs a GPU', allow_module_level=True)
pytest.importorskip('triton')

import statequant as sq  # noqa: E402
from statequant.kernels import PackedGDNState, SlotLayout  # noqa: E402
from statequant.kernels.fold import use_fused  # noqa: E402

DEV = 'cuda'
FORMATS = {'int4-k-g64': sq.Quantizer(4, 'k', 64), 'int4-v-g32': sq.Quantizer(4, 'v', 32),
           'int8-k-g32': sq.Quantizer(8, 'k', 32), 'bf16': 'bf16'}
B, HV, HK, D = 2, 4, 2, 128                                  # GQA 2
BH = B * HV
STATE = ('codes', 'scales', 'KB', 'UB', 'KBS', 'UBS', 'GB', 'GCUR')


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _inputs(T, seed, per_key):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=DEV, generator=g)  # noqa: E731
    if per_key:
        q, k = r(T, B * HK, D), r(T, B * HK, D)
    else:                                                    # per value head, keys shared within a GQA group
        q = r(T, BH, D)
        k = r(T, B, HK, 1, D).expand(-1, -1, -1, HV // HK, -1).reshape(T, BH, D).contiguous()
    v = r(T, BH, D)
    gl = -torch.rand(T, BH, device=DEV, generator=g) * 0.08
    beta = torch.rand(T, BH, device=DEV, generator=g).sigmoid()
    return q, k, v, gl, beta


CASES = [(f, m, bb) for f in FORMATS for m in (1, 8, 16) for bb in (8, 16)]


@pytest.mark.parametrize('per_key', [True, False])
@pytest.mark.parametrize('name,m,bb', CASES)
def test_lockstep_fused_matches_unfused(name, m, bb, per_key):
    torch.manual_seed(1)
    S0 = torch.randn(BH, D, D, device=DEV) * 0.5
    kw = dict(m=m, hv=HV, hk=HK, buffer_bits=bb, out_dtype=torch.float32)
    fu = PackedGDNState(S0, FORMATS[name], fused=True, **kw)
    un = PackedGDNState(S0, FORMATS[name], fused=False, **kw)
    assert fu.fused and not un.fused
    T = 40
    q, k, v, g, beta = _inputs(T, 7, per_key)
    worst, o_diff, diff = 0., 0, {n: 0 for n in STATE}
    for t in range(T):
        of = fu.step(q[t], k[t], v[t], g[t], beta[t], D ** -0.5).clone()
        ou = un.step(q[t], k[t], v[t], g[t], beta[t], D ** -0.5).clone()
        worst = max(worst, _rel(of, ou))
        o_diff += (of != ou).sum().item()
        assert fu.n == un.n == (t + 1) % m
        assert torch.equal(fu.NPTR, torch.full((B, HV), fu.n, device=DEV, dtype=torch.int32))  # every head's copy
        for n_ in STATE:
            a, b_ = getattr(fu, n_), getattr(un, n_)
            diff[n_] += (a != b_).sum().item()
            b_.copy_(a)                                      # re-sync
    n_codes = (T // m) * fu.codes.numel()
    print(f'\n[lockstep {name} m={m} buf{bb} per_key={per_key}] max rel |o_fused - o| {worst:.2e}, '
          f'output entries differing {o_diff} of {T * BH * D}; differing entries: '
          + ', '.join(f'{n_} {c}' for n_, c in diff.items()))
    assert worst < 1e-5, worst
    assert diff['GB'] == 0 and diff['GCUR'] == 0
    assert diff['codes'] <= 1e-5 * n_codes and diff['scales'] <= 1e-5 * n_codes
    n_buf = T * (B * HK * D + BH * D)
    assert diff['KB'] + diff['UB'] <= 1e-4 * n_buf and diff['KBS'] + diff['UBS'] <= 1e-3 * T * (B * HK + BH)


# ---------------------------------------------------------------------------------------------------------------
def _setup(name, m, bb, slots, n_slots=10, seed=0):
    torch.manual_seed(seed)
    lay = SlotLayout(FORMATS[name], HV, HK, D, D, m=m, buffer_bits=bb)
    raw = torch.randint(0, 256, (n_slots, lay.nbytes + 4096), device=DEV, dtype=torch.uint8)
    blob = raw[:, 2048:2048 + lay.nbytes]
    views = lay.views(blob)
    S0 = torch.randn(len(slots), HV, D, D, device=DEV) * 0.5
    lay.write(views, torch.tensor(slots, device=DEV), S0)
    return lay, raw, blob, views


def _token_batch(gen, rows, slots):
    Bn = len(rows)
    q, k = torch.zeros(Bn, HK, D, device=DEV), torch.zeros(Bn, HK, D, device=DEV)
    v = torch.zeros(Bn, HV, D, device=DEV)
    g, beta = torch.zeros(Bn, HV, device=DEV), torch.zeros(Bn, HV, device=DEV)
    for b, r in enumerate(rows):
        if isinstance(r, int):
            q[b], k[b] = torch.randn(HK, D, device=DEV, generator=gen), torch.randn(HK, D, device=DEV, generator=gen)
            v[b] = torch.randn(HV, D, device=DEV, generator=gen)
            g[b] = -torch.rand(HV, device=DEV, generator=gen) * 0.08
            beta[b] = torch.rand(HV, device=DEV, generator=gen).sigmoid()
    idx = torch.tensor([slots[r] if isinstance(r, int) else (0 if r == 'pad0' else -1) for r in rows],
                       device=DEV, dtype=torch.int32)
    return idx, q, k, v, g, beta


@pytest.mark.parametrize('name,m,bb', CASES)
def test_slot_fused_matches_unfused(name, m, bb):
    slots = [5, 2, 7, 9, 3]
    lay, raw, blob, views = _setup(name, m, bb, slots)
    raw_u = raw.clone()
    blob_u = raw_u[:, 2048:2048 + lay.nbytes]
    views_u = lay.views(blob_u)
    sched = torch.Generator().manual_seed(1)
    gen = torch.Generator(device=DEV).manual_seed(4)
    worst, o_diff, byte_diff, n_bytes = 0., 0, {}, 0
    for t in range(40):
        live = [i for i in torch.randperm(len(slots), generator=sched).tolist()
                if torch.rand(1, generator=sched).item() < 0.7]
        rows = live + ['pad0', 'padneg']
        rows = [rows[j] for j in torch.randperm(len(rows), generator=sched).tolist()]
        idx, q, k, v, g, beta = _token_batch(gen, rows, slots)
        of = torch.full((len(rows), HV, D), 7., device=DEV)
        ou = torch.full((len(rows), HV, D), 7., device=DEV)
        lay.step(views, idx, q, k, v, of, D ** -0.5, g=g, beta=beta, fused=True)
        lay.step(views_u, idx, q, k, v, ou, D ** -0.5, g=g, beta=beta, fused=False)
        worst = max(worst, _rel(of, ou))
        o_diff += (of != ou).sum().item()
        for b, r in enumerate(rows):
            if not isinstance(r, int):
                assert torch.equal(of[b], torch.zeros_like(of[b]))
        for f, (off, _, _, nb) in lay.fields.items():          # bytes (garbage slots may hold NaN patterns)
            d = (blob[:, off:off + nb] != blob_u[:, off:off + nb]).sum().item()
            byte_diff[f] = byte_diff.get(f, 0) + d
        assert torch.equal(views['NPTR'], views_u['NPTR'])     # counters (every head's copy) agree
        n_bytes += len(live)
        raw_u.copy_(raw)                                       # re-sync
    n_codes = 40 * len(slots) * views['S'][0].numel() // m
    print(f'\n[slots {name} m={m} buf{bb}] max rel |o_fused - o| {worst:.2e}, output entries differing {o_diff}; '
          f'differing bytes per field: ' + ', '.join(f'{f} {c}' for f, c in byte_diff.items()))
    assert worst < 1e-5, worst
    assert byte_diff['GB'] == 0 and byte_diff['GCUR'] == 0 and byte_diff['NPTR'] == 0
    assert byte_diff['S'] <= max(2, 1e-5 * n_codes)
    n_buf = 40 * len(slots) * (HK + HV) * D
    assert byte_diff['KB'] + byte_diff['UB'] <= 1e-4 * n_buf


def test_slot_untouched_outside_live_slots():
    """The fused step writes nothing outside the live slots' blobs (padding rows, unused slots, page padding)."""
    slots = [5, 2, 7]
    lay, raw, blob, views = _setup('int4-k-g64', 8, 8, slots)
    outside = raw.clone()
    gen = torch.Generator(device=DEV).manual_seed(0)
    for t in range(20):
        idx, q, k, v, g, beta = _token_batch(gen, [0, 'pad0', 2, 'padneg', 1], slots)
        out = torch.empty(5, HV, D, device=DEV)
        lay.step(views, idx, q, k, v, out, D ** -0.5, g=g, beta=beta, fused=True)
    keep = torch.ones_like(raw, dtype=torch.bool)
    for s in slots:
        keep[s, 2048:2048 + lay.nbytes] = False
    assert torch.equal(raw[keep], outside[keep])


def test_slot_mixing_fused_and_unfused_calls():
    """Auto mode can pick either step per call (batch size); both keep the blob in the same state machine."""
    slots = [5, 2, 7]
    lay, raw, blob, views = _setup('int4-k-g64', 8, 8, slots)
    raw_u = raw.clone()
    views_u = lay.views(raw_u[:, 2048:2048 + lay.nbytes])
    gen = torch.Generator(device=DEV).manual_seed(0)
    worst = 0.
    for t in range(30):
        idx, q, k, v, g, beta = _token_batch(gen, [0, 1, 2], slots)
        oa, ob = torch.empty(3, HV, D, device=DEV), torch.empty(3, HV, D, device=DEV)
        lay.step(views, idx, q, k, v, oa, D ** -0.5, g=g, beta=beta, fused=bool(t % 3))
        lay.step(views_u, idx, q, k, v, ob, D ** -0.5, g=g, beta=beta, fused=False)
        worst = max(worst, _rel(oa, ob))
        assert torch.equal(views['NPTR'], views_u['NPTR'])
        raw_u.copy_(raw)
    assert worst < 1e-5


def test_in_kernel_gating_fused():
    slots = [5, 2, 7]
    lay, raw, blob, views = _setup('int4-k-g64', 8, 8, slots)
    gen = torch.Generator(device=DEV).manual_seed(3)
    Bn = len(slots)
    q, k, v = (torch.randn(Bn, h, D, device=DEV, generator=gen) for h in (HK, HK, HV))
    a = torch.randn(Bn, HV, device=DEV, generator=gen).to(torch.bfloat16)
    bb = torch.randn(Bn, HV, device=DEV, generator=gen).to(torch.bfloat16)
    A_log = torch.randn(HV, device=DEV, generator=gen) * 0.5
    dt_bias = torch.randn(HV, device=DEV, generator=gen).to(torch.bfloat16)
    blob2 = blob.clone()
    v2 = lay.views(blob2)
    idx = torch.tensor(slots, device=DEV, dtype=torch.int32)
    o1, o2 = torch.empty(Bn, HV, D, device=DEV), torch.empty(Bn, HV, D, device=DEV)
    for _ in range(9):                                        # crosses one fold
        lay.step(views, idx, q, k, v, o1, D ** -0.5, gate=(a, bb, A_log, dt_bias), fused=True)
        lay.step(v2, idx, q, k, v, o2, D ** -0.5, gate=(a, bb, A_log, dt_bias), fused=False)
        assert _rel(o1, o2) < 1e-5
        blob2.copy_(blob)


# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('name', ['int4-k-g64', 'int4-v-g32', 'bf16'])
def test_cuda_graph_replay_fused_lockstep(name):
    torch.manual_seed(2)
    S0 = torch.randn(BH, D, D, device=DEV)
    m, T = 8, 20
    eager = PackedGDNState(S0, FORMATS[name], m=m, hv=HV, hk=HK, fused=True)
    graph = PackedGDNState(S0, FORMATS[name], m=m, hv=HV, hk=HK, fused=True)
    q, k, v, g, beta = [x.to(torch.bfloat16) if i < 3 else x for i, x in enumerate(_inputs(T, 2, True))]
    sx = [x[0].clone() for x in (q, k, v, g, beta)]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph.step(*sx, D ** -0.5)
    torch.cuda.current_stream().wait_stream(s)
    graph.load(S0)
    torch.cuda.synchronize()
    G = torch.cuda.CUDAGraph()
    with torch.cuda.graph(G):
        out = graph.step(*sx, D ** -0.5)
    graph.load(S0)
    torch.cuda.synchronize()
    for t in range(T):
        for dst, src in zip(sx, (q, k, v, g, beta)):
            dst.copy_(src[t])
        G.replay()
        oe = eager.step(q[t], k[t], v[t], g[t], beta[t], D ** -0.5)
        torch.cuda.synchronize()
        assert torch.equal(out, oe), t
    for n_ in STATE + ('NPTR',):
        assert torch.equal(getattr(graph, n_), getattr(eager, n_)), n_
    assert graph.n == eager.n == T % m


def test_cuda_graph_replay_fused_slots():
    """One captured padded batch, replayed with changing slot tables (incl. padding) and diverging counters."""
    m = 8
    slots = [5, 2, 7]
    lay, raw, blob, views = _setup('int4-k-g64', m, 8, slots)
    blob_e = blob.clone()
    ve = lay.views(blob_e)
    Bn = 4
    idx = torch.zeros(Bn, device=DEV, dtype=torch.int32)
    q, k = torch.zeros(Bn, HK, D, device=DEV), torch.zeros(Bn, HK, D, device=DEV)
    v, out = torch.zeros(Bn, HV, D, device=DEV), torch.zeros(Bn, HV, D, device=DEV)
    g, beta = torch.zeros(Bn, HV, device=DEV), torch.zeros(Bn, HV, device=DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        lay.step(views, idx, q, k, v, out, D ** -0.5, g=g, beta=beta, fused=True)
    torch.cuda.current_stream().wait_stream(s)
    G = torch.cuda.CUDAGraph()
    with torch.cuda.graph(G):
        lay.step(views, idx, q, k, v, out, D ** -0.5, g=g, beta=beta, fused=True)
    gen = torch.Generator(device=DEV).manual_seed(5)
    for tab in [[5, 2, 7, 0], [2, 0, -1, 0], [7, 5, 0, 0], [5, 7, 2, -1]] * 5:
        idx.copy_(torch.tensor(tab, dtype=torch.int32))
        for x in (q, k, v):
            x.copy_(torch.randn(x.shape, device=DEV, generator=gen))
        g.copy_(-torch.rand(g.shape, device=DEV, generator=gen) * 0.08)
        beta.copy_(torch.rand(beta.shape, device=DEV, generator=gen))
        G.replay()
        oe = torch.zeros_like(out)
        lay.step(ve, idx, q, k, v, oe, D ** -0.5, g=g, beta=beta, fused=True)
        torch.cuda.synchronize()
        assert torch.equal(out, oe)
    assert torch.equal(blob, blob_e)
    assert sorted(int(views['NPTR'][s_, 0]) for s_ in slots) != [0, 0, 0]


def test_auto_policy():
    from statequant.kernels import fold
    assert use_fused(True, 10 ** 6) and not use_fused(False, 1)
    assert use_fused(None, fold.FUSED_MAX_BH_AXIS_V, True) and not use_fused(None, fold.FUSED_MAX_BH_AXIS_V + 1, True)
    assert use_fused(None, 10 ** 6) == (fold.FUSED_MAX_BH is None)
    S0 = torch.zeros(BH, D, D, device=DEV)
    assert PackedGDNState(S0, 'bf16', hv=HV, hk=HK).fused == use_fused(None, BH)
