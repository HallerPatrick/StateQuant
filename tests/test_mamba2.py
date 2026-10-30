"""Packed Mamba2 / SSD decode (statequant.kernels.mamba2): the Triton step against the torch reference of the same
arithmetic (lockstep), the slot-addressed step against the lockstep one (random slot subsets with padding rows), the
fp32 controls against the plain Mamba2 recurrence, in-kernel gating, GQA groups, non-power-of-two head dims, the
prefill-side read / write, CUDA-graph replay. Needs a GPU and Triton.

Exactness standard (as ``test_kernels.py`` for gated delta): outputs to fp32 rounding (rel 2e-5: the kernel's fp32 sums
associate differently from torch's), decays bit-exact, buffer entries and state codes / scales after every fold
bit-exact except at rounding ties of those fp32 sums (counted; the reference adopts the kernel's entry so a flip is
not inherited). Run with ``-s`` for the per-config numbers."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('needs a GPU', allow_module_level=True)
pytest.importorskip('triton')

import statequant as sq  # noqa: E402
from statequant.kernels import (Mamba2SlotLayout, PackedMamba2State, ReferenceMamba2State,  # noqa: E402
                                mamba2_recurrence, unpack_vk)

DEV = 'cuda'
# (heads, groups, d_state = d_k, head_dim = d_v): powers of two, and Nemotron-Nano-like non-powers (d_v 80, d_k 96)
SHAPES = {'p2': (8, 2, 128, 64), 'np2': (6, 2, 96, 80)}


def formats(shape):
    H, G, N, P = SHAPES[shape]
    return {'int4-k-g32': sq.Quantizer(4, 'k', 32), 'int4-v-g16': sq.Quantizer(4, 'v', 16),
            'int8-k-g32': sq.Quantizer(8, 'k', 32), 'int8-v-g16': sq.Quantizer(8, 'v', 16), 'bf16': 'bf16',
            **({'int4-k-g64': sq.Quantizer(4, 'k', 64), 'int4-v-g32': sq.Quantizer(4, 'v', 32)} if shape == 'p2'
               else {})}


def params(H, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    A = -(torch.rand(H, device=DEV, generator=g) * 3 + 0.2)              # A = -exp(A_log)
    D = torch.randn(H, device=DEV, generator=g)
    dt_bias = torch.randn(H, device=DEV, generator=g) - 2
    return A, D, dt_bias


def tokens(T, Bn, H, G, N, P, seed, raw=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=DEV, generator=g)              # noqa: E731
    x, Bm, Cm = r(T, Bn, H, P), r(T, Bn, G, N), r(T, Bn, G, N)
    dt = r(T, Bn, H) if raw else torch.rand(T, Bn, H, device=DEV, generator=g) * 0.15
    return x, dt, Bm, Cm


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


# ---------------------------------------------------------------------------------------------------------------
def run_decode(shape, fmt, m, bb, T=40, Bn=2, seed=1):
    H, G, N, P = SHAPES[shape]
    torch.manual_seed(seed)
    S0 = torch.randn(Bn, H, P, N, device=DEV) * 0.5
    ker = PackedMamba2State(S0, fmt, m=m, hv=H, hk=G, buffer_bits=bb, out_dtype=torch.float32)
    ref = ReferenceMamba2State(S0, fmt, m=m, hv=H, hk=G, buffer_bits=bb, auto_fold=False)
    assert torch.equal(ker.dequantize(), ref.dequantize())
    A, D, _ = params(H)
    x, dt, Bm, Cm = tokens(T, Bn, H, G, N, P, seed)
    worst, folds, flips = 0., [], 0
    for t in range(T):
        slot = ker.n
        o = ker.step(x[t], dt[t], Bm[t], Cm[t], A, D).clone().reshape(-1, P)
        r = ref.step(x[t], dt[t], Bm[t], Cm[t], A, D)
        worst = max(worst, _rel(o, r))
        kb, ub = ker.KB[:, slot].float(), ker.UB[:, slot].float()
        if bb == 8:
            kb, ub = kb * ker.KBS[:, slot, None].float(), ub * ker.UBS[:, slot, None].float()
        flips += (kb != ref.K[:, slot]).sum().item() + (ub != ref.U[:, slot]).sum().item()
        assert torch.equal(ker.GB[:, slot], ref.G[:, slot])
        ref.K[:, slot], ref.U[:, slot] = kb, ub
        if ref.n == m:
            ref.fold()
        assert ker.n == ref.n and torch.equal(ker.NPTR, torch.full_like(ker.NPTR, ker.n))
        if ref.n == 0:
            mism = (ker.codes != ref.codes).sum().item()
            smism = 0 if ref.fmt is None else (ker.scales != ref.scales).sum().item()
            Sk = ker.dequantize()
            raw = _rel(Sk, ref.S_raw)
            folds.append((mism, smism, raw))
            ref.codes, ref.scales, ref.S0 = ker.codes.clone(), ker.scales.clone(), Sk
        assert torch.equal(ker.GCUR, ref.gcur)
    return worst, folds, flips, ker


CASES = [(s, f, m, bb) for s in SHAPES for f in formats(s) for m in (1, 8, 16) for bb in (8, 16)]
CASES += [('p2', 'int4-k-g64', 4, 8), ('np2', 'int8-k-g32', 4, 16), ('p2', 'bf16', 4, 8)]


@pytest.mark.parametrize('shape,name,m,bb', CASES)
def test_decode_vs_reference(shape, name, m, bb):
    H, G, N, P = SHAPES[shape]
    T, Bn = 40, 2
    worst, folds, flips, ker = run_decode(shape, formats(shape)[name], m, bb, T, Bn)
    mism = sum(f[0] for f in folds)
    smism = sum(f[1] for f in folds)
    n_codes = ker.codes.numel() * len(folds)
    raw = max(f[2] for f in folds)
    n_buf = T * Bn * (G * N + H * P)
    print(f'\n[{shape} {name} m={m} buf{bb}] {len(folds)} folds: max rel |y - y_ref| {worst:.2e}; codes differing '
          f'after folds {mism} of {n_codes}, scales differing {smism}; max rel |S - S_raw_ref| {raw:.2e}; buffer '
          f'entries flipped {flips} of {n_buf}')
    assert len(folds) == T // m
    assert worst < 2e-5, worst
    assert flips <= 1e-4 * n_buf, flips
    assert mism <= 1e-4 * n_codes, mism
    assert smism <= 1e-4 * n_codes, smism
    assert raw < {'bf16': 1e-2}.get(name, 0.2), raw


def test_large_gqa_one_group():
    """Mamba-2 / Falcon-H1 / Granite style: one group shared by every head (GQA = H)."""
    H, G, N, P, m = 64, 1, 128, 64, 8
    torch.manual_seed(0)
    S0 = torch.randn(2, H, P, N, device=DEV) * 0.3
    fmt = sq.Quantizer(4, 'k', 64)
    ker = PackedMamba2State(S0, fmt, m=m, hv=H, hk=G, out_dtype=torch.float32)
    ref = ReferenceMamba2State(S0, fmt, m=m, hv=H, hk=G)
    A, D, _ = params(H)
    x, dt, Bm, Cm = tokens(12, 2, H, G, N, P, 3)
    worst = 0.
    for t in range(12):
        worst = max(worst, _rel(ker.step(x[t], dt[t], Bm[t], Cm[t], A, D).reshape(-1, P),
                                ref.step(x[t], dt[t], Bm[t], Cm[t], A, D)))
        if ref.n == 0:                                    # adopt the fold (ties)
            ref.codes, ref.scales, ref.S0 = ker.codes.clone(), ker.scales.clone(), ker.dequantize()
    assert worst < 2e-5, worst


# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('shape', list(SHAPES))
def test_identity_reference_is_the_recurrence(shape):
    """The buffer algebra itself is exact: an fp32 state, fp32 buffer reference == the plain recurrence."""
    H, G, N, P = SHAPES[shape]
    S0 = torch.randn(2, H, P, N, device=DEV)
    S = S0.clone()
    ref = ReferenceMamba2State(S0, sq.Identity(), m=8, hv=H, hk=G, buffer_bits=32)
    A, D, dtb = params(H)
    x, dt, Bm, Cm = tokens(30, 2, H, G, N, P, 4, raw=True)
    worst = 0.
    for t in range(30):
        y = mamba2_recurrence(S, x[t], dt[t], Bm[t], Cm[t], A, D, dtb, dt_softplus=True)
        r = ref.step(x[t], dt[t], Bm[t], Cm[t], A, D, dtb, dt_softplus=True)
        worst = max(worst, _rel(r, y.reshape(-1, P)))
    st = _rel(ref.state(), S.reshape(-1, P, N))
    print(f'\n[{shape}] identity reference vs fp32 recurrence: max rel |y| {worst:.2e}, state {st:.2e}')
    assert worst < 1e-5 and st < 1e-5


@pytest.mark.parametrize('shape', list(SHAPES))
@pytest.mark.parametrize('bb', [8, 16])
def test_bf16_control_vs_fp32_recurrence(shape, bb):
    """The bf16 packed state (the plugin's control arm) tracks the fp32 recurrence to bf16 / int8-buffer accuracy."""
    H, G, N, P = SHAPES[shape]
    S0 = torch.randn(2, H, P, N, device=DEV)
    S = S0.to(torch.bfloat16).float()
    ker = PackedMamba2State(S0, 'bf16', m=8, hv=H, hk=G, buffer_bits=bb, out_dtype=torch.float32)
    A, D, dtb = params(H)
    x, dt, Bm, Cm = tokens(64, 2, H, G, N, P, 5, raw=True)
    worst, rel2 = 0., 0.
    for t in range(64):
        y = mamba2_recurrence(S, x[t], dt[t], Bm[t], Cm[t], A, D, dtb, dt_softplus=True)
        o = ker.step(x[t], dt[t], Bm[t], Cm[t], A, D, dtb, dt_softplus=True)
        worst = max(worst, _rel(o, y))
        rel2 = max(rel2, ((o - y).norm() / y.norm()).item())
    st = ((ker.dequantize().view_as(S) - S).norm() / S.norm()).item()
    print(f'\n[{shape} bf16 buf{bb}] vs fp32 recurrence over 64 tokens: max rel {worst:.2e}, rel l2 {rel2:.2e}, '
          f'state rel l2 {st:.2e}')
    assert rel2 < (1.5e-2 if bb == 8 else 5e-3) and st < 1e-2      # int8 buffer: ~1/254 per value


# ---------------------------------------------------------------------------------------------------------------
def test_in_kernel_gating():
    """dt_raw + dt_bias, softplus (threshold 20) and dt_limit in the kernel == the same gate formed in torch."""
    H, G, N, P = SHAPES['np2']
    S0 = torch.randn(2, H, P, N, device=DEV)
    fmt = sq.Quantizer(4, 'k', 32)
    a = PackedMamba2State(S0, fmt, m=8, hv=H, hk=G, out_dtype=torch.float32)
    b = PackedMamba2State(S0, fmt, m=8, hv=H, hk=G, out_dtype=torch.float32)
    A, D, dtb = params(H)
    x, dt, Bm, Cm = tokens(9, 2, H, G, N, P, 6, raw=True)
    dt[:, 0, 0] = 25.                                       # the softplus threshold branch
    for lim in [None, (0.01, 0.5)]:
        a.load(S0), b.load(S0)
        for t in range(9):
            d = dt[t] + dtb
            d = torch.where(d <= 20., torch.log(1. + torch.exp(d)), d)
            if lim is not None:
                d = d.clamp(*lim)
            oa = a.step(x[t], dt[t], Bm[t], Cm[t], A, D, dtb, dt_softplus=True, dt_limit=lim)
            ob = b.step(x[t], d, Bm[t], Cm[t], A, D)
            assert _rel(oa, ob) < 1e-5
            assert _rel(a.dequantize(), b.dequantize()) < 1e-2   # (a 1-ulp dt difference can flip a code)


# ---------------------------------------------------------------------------------------------------------------
def _setup_slots(shape, fmt, m, bb, slots, n_slots=10, seed=0):
    H, G, N, P = SHAPES[shape]
    torch.manual_seed(seed)
    lay = Mamba2SlotLayout(fmt, H, G, N, P, m=m, buffer_bits=bb)
    raw = torch.randint(0, 256, (n_slots, lay.nbytes + 4096), device=DEV, dtype=torch.uint8)
    blob = raw[:, 2048:2048 + lay.nbytes]
    views = lay.views(blob)
    S0 = [torch.randn(H, P, N, device=DEV) * 0.5 for _ in slots]
    refs = [PackedMamba2State(s, fmt, m=m, hv=H, hk=G, buffer_bits=bb, out_dtype=torch.float32) for s in S0]
    lay.write(views, torch.tensor(slots, device=DEV), torch.stack(S0))
    return lay, raw, blob, views, refs


def _adopt(views, slot, ref):
    for k in ('S', 'SC', 'KB', 'UB', 'KBS', 'UBS', 'GB', 'GCUR'):
        if k in views:
            ref.T[k].copy_(views[k][slot].reshape(ref.T[k].shape))


SLOT_CASES = [('p2', 'int4-k-g64', 8, 8), ('p2', 'int4-v-g32', 8, 8), ('p2', 'bf16', 4, 16), ('np2', 'int4-k-g32', 8, 8),
              ('np2', 'int8-v-g16', 1, 8), ('np2', 'bf16', 16, 16), ('np2', 'int8-k-g32', 4, 16)]


@pytest.mark.parametrize('shape,name,m,bb', SLOT_CASES)
def test_slot_step_matches_lockstep(shape, name, m, bb):
    H, G, N, P = SHAPES[shape]
    fmt = formats(shape)[name]
    slots = [5, 2, 7]
    lay, raw, blob, views, refs = _setup_slots(shape, fmt, m, bb, slots)
    for i, r in enumerate(refs):
        assert torch.equal(views['S'][slots[i]], r.codes.view(H, *r.codes.shape[1:]))
    A, D, dtb = params(H)
    outside = raw.clone()
    sched = torch.Generator().manual_seed(1)
    gen = torch.Generator(device=DEV).manual_seed(2)
    steps = [0] * len(slots)
    worst, flips = 0., 0
    for t in range(30):
        live = [i for i in torch.randperm(len(slots), generator=sched).tolist()
                if torch.rand(1, generator=sched).item() < 0.7]
        rows = live + ['pad0', 'padneg']
        rows = [rows[j] for j in torch.randperm(len(rows), generator=sched).tolist()]
        R = len(rows)
        idx = torch.tensor([slots[r] if isinstance(r, int) else (0 if r == 'pad0' else -1) for r in rows],
                           device=DEV, dtype=torch.int32)
        x = torch.randn(R, H, P, device=DEV, generator=gen)
        dt = torch.randn(R, H, device=DEV, generator=gen)
        Bm, Cm = torch.randn(R, G, N, device=DEV, generator=gen), torch.randn(R, G, N, device=DEV, generator=gen)
        out = torch.full((R, H, P), 7., device=DEV)
        lay.step(views, idx, x, dt, Bm, Cm, out, A, D, dtb)
        for b, r in enumerate(rows):
            if not isinstance(r, int):
                assert torch.equal(out[b], torch.zeros_like(out[b]))
                continue
            want = refs[r].step(x[b:b + 1], dt[b:b + 1], Bm[b:b + 1], Cm[b:b + 1], A, D, dtb, dt_softplus=True)
            steps[r] += 1
            worst = max(worst, _rel(out[b], want[0]))
            assert int(views['NPTR'][slots[r], 0]) == refs[r].n == steps[r] % m
            assert torch.equal(views['NPTR'][slots[r], :H], refs[r].NPTR[0])
            flips += (views['S'][slots[r]] != refs[r].codes.view(H, *refs[r].codes.shape[1:])).sum().item()
            _adopt(views, slots[r], refs[r])
    n_codes = 30 * len(slots) * views['S'][0].numel() // m
    print(f'\n[slots {shape} {name} m={m} buf{bb}] max rel |y - y_lockstep| {worst:.2e}; code entries differing '
          f'{flips}')
    assert worst < 1e-5, worst
    assert flips <= max(2, 1e-5 * n_codes), flips
    keep = torch.ones_like(raw, dtype=torch.bool)
    for s in slots:
        keep[s, 2048:2048 + lay.nbytes] = False
    assert torch.equal(raw[keep], outside[keep])


@pytest.mark.parametrize('shape,name', [('p2', 'int4-k-g64'), ('np2', 'bf16'), ('np2', 'int4-v-g16')])
def test_read_is_the_state_the_next_step_reads(shape, name):
    """read() (stored state advanced across the pending buffer) predicts the next step's output; write() of the
    read state reproduces it up to one rounding (the prefill round trip of the vLLM plugin)."""
    H, G, N, P = SHAPES[shape]
    fmt = formats(shape)[name]
    slots = [5, 2, 7]
    lay, raw, blob, views, refs = _setup_slots(shape, fmt, 8, 8, slots)
    A, D, dtb = params(H)
    idx = torch.tensor(slots, device=DEV, dtype=torch.int32)
    x, dt, Bm, Cm = tokens(13, 3, H, G, N, P, 7, raw=True)
    out = torch.empty(3, H, P, device=DEV)
    for t in range(13):
        if t == 12:
            S = lay.read(views, idx).clone()                    # [3, H, d_v, d_k]
            pred = mamba2_recurrence(S, x[t], dt[t], Bm[t], Cm[t], A, D, dtb, dt_softplus=True)
        lay.step(views, idx, x[t], dt[t], Bm[t], Cm[t], out, A, D, dtb)
    assert int(views['NPTR'][slots[0], 0]) == 13 % 8
    rel = ((out - pred).norm() / pred.norm()).item()
    print(f'\n[{shape} {name}] read() + fp32 step vs kernel step: rel err {rel:.2e}')
    assert rel < 1e-5, rel
    S = lay.read(views, idx)
    lay.write(views, idx, S)
    assert views['NPTR'][idx.long(), 0].tolist() == [0, 0, 0]
    S2 = lay.read(views, idx)
    from statequant.kernels import pack_vk
    c, s = pack_vk(S.reshape(-1, P, N), lay.fmt)
    assert torch.equal(S2.reshape(-1, P, N), unpack_vk(c, s, lay.fmt, N))


def test_layout_bytes_and_stock_ratio():
    lay = Mamba2SlotLayout(sq.Quantizer(4, 'k', 64), 112, 8, 128, 64, m=8, buffer_bits=8)   # Nemotron-H-4B
    b = lay.breakdown()
    assert b['S'] == 112 * 64 * 64 and b['SC'] == 112 * 64 * 2 * 2
    assert b['KB'] == 8 * 8 * 128 and b['UB'] == 112 * 8 * 64
    assert lay.nbytes == sum(-(-x // 16) * 16 for x in b.values())
    print('\n', lay, b, f'stock fp32 {112 * 64 * 128 * 4} B, bf16 {112 * 64 * 128 * 2} B')
    lay = Mamba2SlotLayout('bf16', 96, 8, 128, 80, m=8, buffer_bits=8)                     # Nemotron-3-Nano-4B
    assert lay.breakdown()['S'] == 96 * 80 * 128 * 2


# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('name', ['int4-k-g32', 'int4-v-g16', 'bf16'])
def test_cuda_graph_replay_lockstep(name):
    H, G, N, P = SHAPES['np2']
    fmt = formats('np2')[name]
    S0 = torch.randn(2, H, P, N, device=DEV)
    eager = PackedMamba2State(S0, fmt, m=8, hv=H, hk=G)
    graph = PackedMamba2State(S0, fmt, m=8, hv=H, hk=G)
    A, D, dtb = params(H)
    T = 20
    x, dt, Bm, Cm = [t.to(torch.bfloat16) if i != 1 else t for i, t in enumerate(tokens(T, 2, H, G, N, P, 8, True))]
    sx = [t[0].clone() for t in (x, dt, Bm, Cm)]
    kw = dict(D=D, dt_bias=dtb, dt_softplus=True)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph.step(*sx, A, **kw)
    torch.cuda.current_stream().wait_stream(s)
    graph.load(S0)
    torch.cuda.synchronize()
    Gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(Gr):
        out = graph.step(*sx, A, **kw)
    graph.load(S0)
    for t in range(T):
        for dst, src in zip(sx, (x, dt, Bm, Cm)):
            dst.copy_(src[t])
        Gr.replay()
        oe = eager.step(x[t], dt[t], Bm[t], Cm[t], A, **kw)
        torch.cuda.synchronize()
        assert torch.equal(out, oe), t
    for k in ('S', 'SC', 'KB', 'UB', 'KBS', 'UBS', 'GB', 'GCUR', 'NPTR'):
        assert torch.equal(graph.T[k], eager.T[k]), k
    assert graph.n == eager.n == T % 8


def test_cuda_graph_replay_slots():
    """One captured padded batch, replayed with changing slot tables (incl. padding) and diverging counters; inputs
    as vLLM hands them (column slices of one conv output, dt a slice of the projection)."""
    H, G, N, P = SHAPES['np2']
    fmt = formats('np2')['int4-k-g32']
    slots = [5, 2, 7]
    lay, raw, blob, views, _ = _setup_slots('np2', fmt, 8, 8, slots)
    blob_e = blob.clone()
    ve = lay.views(blob_e)
    A, D, dtb = params(H)
    R = 4
    idx = torch.zeros(R, device=DEV, dtype=torch.int32)
    conv = torch.zeros(R, H * P + 2 * G * N, device=DEV, dtype=torch.bfloat16)
    proj = torch.zeros(R, 3 + H, device=DEV, dtype=torch.bfloat16)
    x = conv[:, :H * P].view(R, H, P)
    Bm = conv[:, H * P:H * P + G * N].view(R, G, N)
    Cm = conv[:, H * P + G * N:].view(R, G, N)
    dt = proj[:, 3:]
    out = torch.zeros(R, H, P, device=DEV, dtype=torch.bfloat16)
    call = lambda vw, o: lay.step(vw, idx, x, dt, Bm, Cm, o, A, D, dtb)   # noqa: E731
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call(views, out)                                   # idx all padding: no state touched
    torch.cuda.current_stream().wait_stream(s)
    Gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(Gr):
        call(views, out)
    gen = torch.Generator(device=DEV).manual_seed(5)
    for tab in [[5, 2, 7, 0], [2, 0, -1, 0], [7, 5, 0, 0], [5, 7, 2, -1]] * 5:
        idx.copy_(torch.tensor(tab, dtype=torch.int32))
        conv.copy_(torch.randn(conv.shape, device=DEV, generator=gen))
        proj.copy_(torch.randn(proj.shape, device=DEV, generator=gen))
        Gr.replay()
        oe = torch.zeros_like(out)
        call(ve, oe)
        torch.cuda.synchronize()
        assert torch.equal(out, oe)
    assert torch.equal(blob, blob_e)
    assert sorted(int(views['NPTR'][s_, 0]) for s_ in slots) != [0, 0, 0]
