"""Packed KDA decode (statequant.kernels.kda): the Triton step against the torch reference of the same arithmetic
(lockstep), the slot-addressed step against the lockstep one (random slot subsets with padding rows), the fp32
controls against the plain KDA recurrence, the prefill-side read / write, CUDA-graph replay. Needs a GPU and Triton.

Exactness standard (as ``test_mamba2.py``): outputs to fp32 rounding (rel 2e-5), decays bit-exact, buffer entries and
state codes / scales after every fold bit-exact except at rounding ties of fp32 sums (counted; the reference adopts
the kernel's entry so a flip is not inherited). Run with ``-s`` for the per-config numbers."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('needs a GPU', allow_module_level=True)
pytest.importorskip('triton')

import statequant as sq  # noqa: E402
from statequant.kernels import KDASlotLayout, PackedKDAState, ReferenceKDAState, kda_recurrence  # noqa: E402
from statequant.kernels import pack_vk, unpack_vk  # noqa: E402

DEV = 'cuda'
# (heads, d_k, d_v): Ling-3.0 / Kimi Linear head dims (128 x 128), and a small one (one value block, d_v 32)
SHAPES = {'ling': (4, 128, 128), 'small': (3, 64, 32), 'wide': (2, 128, 256)}


def formats(shape):
    H, K, Vd = SHAPES[shape]
    f = {'int4-k-g32': sq.Quantizer(4, 'k', 32), 'int4-v-g16': sq.Quantizer(4, 'v', 16),
         'int8-k-g32': sq.Quantizer(8, 'k', 32), 'int8-v-g16': sq.Quantizer(8, 'v', 16), 'bf16': 'bf16'}
    if K == 128:
        f.update({'int4-k-g64': sq.Quantizer(4, 'k', 64), 'int4-v-g32': sq.Quantizer(4, 'v', 32),
                  'int8-v-g32': sq.Quantizer(8, 'v', 32)})
    return f


def tokens(T, Bn, H, K, Vd, seed, lower_bound=-5.):
    """Raw q, k (normalised in the step), v, a KDA gate (safe gate: lower_bound * sigmoid(...)) and beta."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=DEV, generator=g)              # noqa: E731
    q, k, v = r(T, Bn, H, K), r(T, Bn, H, K), r(T, Bn, H, Vd)
    gate = lower_bound * torch.sigmoid(r(T, Bn, H, K) * 2 - 3)
    beta = torch.sigmoid(r(T, Bn, H))
    return q, k, v, gate, beta


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _flat(t):
    return t.reshape(-1, t.shape[-1])


# ---------------------------------------------------------------------------------------------------------------
def run_decode(shape, fmt, m, bb, T=40, Bn=2, seed=1):
    H, K, Vd = SHAPES[shape]
    torch.manual_seed(seed)
    S0 = torch.randn(Bn, H, Vd, K, device=DEV) * 0.5
    ker = PackedKDAState(S0, fmt, m=m, hv=H, buffer_bits=bb, out_dtype=torch.float32)
    ref = ReferenceKDAState(S0, fmt, m=m, hv=H, buffer_bits=bb, auto_fold=False)
    assert torch.equal(ker.dequantize(), ref.dequantize())
    q, k, v, g, beta = tokens(T, Bn, H, K, Vd, seed)
    worst, folds, flips = 0., [], 0
    for t in range(T):
        slot = ker.n
        o = ker.step(q[t], k[t], v[t], g[t], beta[t]).clone().reshape(-1, Vd)
        r = ref.step(_flat(q[t]), _flat(k[t]), _flat(v[t]), _flat(g[t]), beta[t].reshape(-1))
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


CASES = [(s, f, m, bb) for s in ('ling', 'small') for f in formats(s) for m in (1, 8, 16) for bb in (8, 16)]
CASES += [('ling', 'int4-k-g64', 4, 8), ('ling', 'int8-k-g32', 32, 8), ('ling', 'bf16', 2, 16),
          ('wide', 'int4-k-g64', 8, 8), ('wide', 'int8-v-g32', 8, 16), ('wide', 'bf16', 8, 8)]


@pytest.mark.parametrize('shape,name,m,bb', CASES)
def test_decode_vs_reference(shape, name, m, bb):
    H, K, Vd = SHAPES[shape]
    T, Bn = 40, 2
    worst, folds, flips, ker = run_decode(shape, formats(shape)[name], m, bb, T, Bn)
    mism = sum(f[0] for f in folds)
    smism = sum(f[1] for f in folds)
    n_codes = ker.codes.numel() * len(folds)
    raw = max(f[2] for f in folds)
    n_buf = T * Bn * H * (K + Vd)
    print(f'\n[{shape} {name} m={m} buf{bb}] {len(folds)} folds: max rel |o - o_ref| {worst:.2e}; codes differing '
          f'after folds {mism} of {n_codes}, scales differing {smism}; max rel |S - S_raw_ref| {raw:.2e}; buffer '
          f'entries flipped {flips} of {n_buf}')
    assert len(folds) == T // m
    assert worst < 2e-5, worst
    assert flips <= 1e-4 * n_buf, flips
    assert mism <= 1e-4 * n_codes, mism
    assert smism <= 1e-4 * n_codes, smism
    assert raw < {'bf16': 1e-2}.get(name, 0.2), raw


# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('shape', ['ling', 'small'])
def test_identity_reference_is_the_recurrence(shape):
    """The buffer algebra with per-channel decays is exact: fp32 state + fp32 buffer reference == the recurrence."""
    H, K, Vd = SHAPES[shape]
    S0 = torch.randn(2, H, Vd, K, device=DEV)
    S = S0.clone().reshape(-1, Vd, K)
    ref = ReferenceKDAState(S0, sq.Identity(), m=8, hv=H, buffer_bits=32)
    q, k, v, g, beta = tokens(30, 2, H, K, Vd, 4)
    worst = 0.
    for t in range(30):
        y = kda_recurrence(S, _flat(q[t]), _flat(k[t]), _flat(v[t]), _flat(g[t]), beta[t].reshape(-1))
        r = ref.step(_flat(q[t]), _flat(k[t]), _flat(v[t]), _flat(g[t]), beta[t].reshape(-1))
        worst = max(worst, _rel(r, y))
    st = _rel(ref.state(), S)
    print(f'\n[{shape}] identity reference vs fp32 recurrence: max rel |o| {worst:.2e}, state {st:.2e}')
    assert worst < 1e-5 and st < 1e-5


def test_recurrence_matches_fla():
    """``kda_recurrence`` is fla's KDA (fused_recurrent_kda, state [B, H, K, V]) on the transposed state."""
    fla = pytest.importorskip('fla.ops.kda')
    H, K, Vd = SHAPES['ling']
    S0 = torch.randn(2, H, Vd, K, device=DEV)
    q, k, v, g, beta = tokens(16, 2, H, K, Vd, 9)
    S = S0.clone().reshape(-1, Vd, K)
    ys = [kda_recurrence(S, _flat(q[t]), _flat(k[t]), _flat(v[t]), _flat(g[t]), beta[t].reshape(-1))
          for t in range(16)]
    o, St = fla.fused_recurrent_kda(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), g.transpose(0, 1),
                                    beta.transpose(0, 1), initial_state=S0.transpose(-1, -2).contiguous(),
                                    output_final_state=True, use_qk_l2norm_in_kernel=True)
    y = torch.stack(ys).reshape(16, 2, H, Vd).transpose(0, 1)
    print(f'\nkda_recurrence vs fla fused_recurrent_kda: o {_rel(y, o):.2e}, state '
          f'{_rel(S.reshape(2, H, Vd, K), St.transpose(-1, -2)):.2e}')
    assert _rel(y, o) < 1e-4 and _rel(S.reshape(2, H, Vd, K), St.transpose(-1, -2)) < 1e-4


@pytest.mark.parametrize('bb', [8, 16])
def test_bf16_control_vs_fp32_recurrence(bb):
    """The bf16 packed state (the plugin's control arm) tracks the fp32 recurrence to bf16 / int8-buffer accuracy."""
    H, K, Vd = SHAPES['ling']
    S0 = torch.randn(2, H, Vd, K, device=DEV)
    S = S0.to(torch.bfloat16).float().reshape(-1, Vd, K)
    ker = PackedKDAState(S0, 'bf16', m=8, hv=H, buffer_bits=bb, out_dtype=torch.float32)
    q, k, v, g, beta = tokens(64, 2, H, K, Vd, 5)
    rel2 = 0.
    for t in range(64):
        y = kda_recurrence(S, _flat(q[t]), _flat(k[t]), _flat(v[t]), _flat(g[t]), beta[t].reshape(-1))
        o = ker.step(q[t], k[t], v[t], g[t], beta[t]).reshape(-1, Vd)
        rel2 = max(rel2, ((o - y).norm() / y.norm()).item())
    st = ((ker.dequantize() - S).norm() / S.norm()).item()
    print(f'\n[bf16 buf{bb}] vs fp32 recurrence over 64 tokens: rel l2 {rel2:.2e}, state rel l2 {st:.2e}')
    assert rel2 < (1.5e-2 if bb == 8 else 5e-3) and st < 1e-2


# ---------------------------------------------------------------------------------------------------------------
def _setup_slots(shape, fmt, m, bb, slots, n_slots=10, seed=0):
    H, K, Vd = SHAPES[shape]
    torch.manual_seed(seed)
    lay = KDASlotLayout(fmt, H, K, Vd, m=m, buffer_bits=bb)
    raw = torch.randint(0, 256, (n_slots, lay.nbytes + 4096), device=DEV, dtype=torch.uint8)
    blob = raw[:, 2048:2048 + lay.nbytes]
    views = lay.views(blob)
    S0 = [torch.randn(H, Vd, K, device=DEV) * 0.5 for _ in slots]
    refs = [PackedKDAState(s, fmt, m=m, hv=H, buffer_bits=bb, out_dtype=torch.float32) for s in S0]
    lay.write(views, torch.tensor(slots, device=DEV), torch.stack(S0))
    return lay, raw, blob, views, refs


def _adopt(views, slot, ref):
    for k in ('S', 'SC', 'KB', 'UB', 'KBS', 'UBS', 'GB', 'GCUR'):
        if k in views:
            ref.T[k].copy_(views[k][slot].reshape(ref.T[k].shape))


SLOT_CASES = [('ling', 'int4-k-g64', 8, 8), ('ling', 'int4-v-g32', 8, 8), ('ling', 'bf16', 4, 16),
              ('ling', 'int8-k-g32', 8, 8), ('small', 'int8-v-g16', 1, 8), ('small', 'bf16', 16, 16),
              ('small', 'int4-k-g32', 4, 16)]


@pytest.mark.parametrize('shape,name,m,bb', SLOT_CASES)
def test_slot_step_matches_lockstep(shape, name, m, bb):
    H, K, Vd = SHAPES[shape]
    fmt = formats(shape)[name]
    slots = [5, 2, 7]
    lay, raw, blob, views, refs = _setup_slots(shape, fmt, m, bb, slots)
    for i, r in enumerate(refs):
        assert torch.equal(views['S'][slots[i]], r.codes.view(H, *r.codes.shape[1:]))
    outside = raw.clone()
    sched = torch.Generator().manual_seed(1)
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
        q, k, v, g, beta = [x[0] for x in tokens(1, R, H, K, Vd, 100 + t)]
        out = torch.full((R, H, Vd), 7., device=DEV)
        lay.step(views, idx, q, k, v, g, beta, out)
        for b, r in enumerate(rows):
            if not isinstance(r, int):
                assert torch.equal(out[b], torch.zeros_like(out[b]))
                continue
            want = refs[r].step(q[b:b + 1], k[b:b + 1], v[b:b + 1], g[b:b + 1], beta[b:b + 1])
            steps[r] += 1
            worst = max(worst, _rel(out[b], want[0]))
            assert int(views['NPTR'][slots[r], 0]) == refs[r].n == steps[r] % m
            assert torch.equal(views['NPTR'][slots[r], :H], refs[r].NPTR[0])
            flips += (views['S'][slots[r]] != refs[r].codes.view(H, *refs[r].codes.shape[1:])).sum().item()
            _adopt(views, slots[r], refs[r])
    n_codes = 30 * len(slots) * views['S'][0].numel() // m
    print(f'\n[slots {shape} {name} m={m} buf{bb}] max rel |o - o_lockstep| {worst:.2e}; code entries differing '
          f'{flips}')
    assert worst < 1e-5, worst
    assert flips <= max(2, 1e-5 * n_codes), flips
    keep = torch.ones_like(raw, dtype=torch.bool)
    for s in slots:
        keep[s, 2048:2048 + lay.nbytes] = False
    assert torch.equal(raw[keep], outside[keep])


@pytest.mark.parametrize('shape,name', [('ling', 'int4-k-g64'), ('ling', 'bf16'), ('small', 'int4-v-g16'),
                                        ('ling', 'int8-k-g32')])
def test_read_is_the_state_the_next_step_reads(shape, name):
    """read() (stored state advanced across the pending buffer, per-channel decays) predicts the next step's output;
    write() of the read state reproduces it up to one rounding (the prefill round trip of the vLLM plugin)."""
    H, K, Vd = SHAPES[shape]
    fmt = formats(shape)[name]
    slots = [5, 2, 7]
    lay, raw, blob, views, refs = _setup_slots(shape, fmt, 8, 8, slots)
    idx = torch.tensor(slots, device=DEV, dtype=torch.int32)
    q, k, v, g, beta = tokens(13, 3, H, K, Vd, 7)
    out = torch.empty(3, H, Vd, device=DEV)
    for t in range(13):
        if t == 12:
            S = lay.read(views, idx).clone().reshape(-1, Vd, K)
            pred = kda_recurrence(S, _flat(q[t]), _flat(k[t]), _flat(v[t]), _flat(g[t]), beta[t].reshape(-1))
        lay.step(views, idx, q[t], k[t], v[t], g[t], beta[t], out)
    assert int(views['NPTR'][slots[0], 0]) == 13 % 8
    rel = ((out.reshape(-1, Vd) - pred).norm() / pred.norm()).item()
    print(f'\n[{shape} {name}] read() + fp32 step vs kernel step: rel err {rel:.2e}')
    assert rel < 1e-5, rel
    S = lay.read(views, idx)
    lay.write(views, idx, S)
    assert views['NPTR'][idx.long(), 0].tolist() == [0, 0, 0]
    assert float(views['GCUR'][idx.long()].abs().max()) == 0.
    S2 = lay.read(views, idx)
    c, s = pack_vk(S.reshape(-1, Vd, K), lay.fmt)
    assert torch.equal(S2.reshape(-1, Vd, K), unpack_vk(c, s, lay.fmt, K))


def test_layout_bytes():
    lay = KDASlotLayout(sq.Quantizer(4, 'k', 64), 16, 128, 128, m=8, buffer_bits=8)        # Ling-3.0-tiny layer
    b = lay.breakdown()
    assert b['S'] == 16 * 128 * 64 and b['SC'] == 16 * 128 * 2 * 2
    assert b['KB'] == 16 * 8 * 128 and b['UB'] == 16 * 8 * 128
    assert b['GB'] == 16 * 8 * 128 * 4 and b['GCUR'] == 16 * 128 * 4
    assert lay.nbytes == sum(-(-x // 16) * 16 for x in b.values())
    print('\n', lay, b, f'stock fp32 {16 * 128 * 128 * 4} B, bf16 {16 * 128 * 128 * 2} B')


# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('name', ['int4-k-g64', 'int4-v-g32', 'bf16'])
def test_cuda_graph_replay_lockstep(name):
    H, K, Vd = SHAPES['ling']
    fmt = formats('ling')[name]
    S0 = torch.randn(2, H, Vd, K, device=DEV)
    eager = PackedKDAState(S0, fmt, m=8, hv=H)
    graph = PackedKDAState(S0, fmt, m=8, hv=H)
    T = 20
    q, k, v, g, beta = tokens(T, 2, H, K, Vd, 8)
    q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    sx = [t[0].clone() for t in (q, k, v, g, beta)]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph.step(*sx)
    torch.cuda.current_stream().wait_stream(s)
    graph.load(S0)
    torch.cuda.synchronize()
    Gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(Gr):
        out = graph.step(*sx)
    graph.load(S0)
    for t in range(T):
        for dst, src in zip(sx, (q, k, v, g, beta)):
            dst.copy_(src[t])
        Gr.replay()
        oe = eager.step(q[t], k[t], v[t], g[t], beta[t])
        torch.cuda.synchronize()
        assert torch.equal(out, oe), t
    for key in ('S', 'SC', 'KB', 'UB', 'KBS', 'UBS', 'GB', 'GCUR', 'NPTR'):
        assert torch.equal(graph.T[key], eager.T[key]), key
    assert graph.n == eager.n == T % 8


def test_cuda_graph_replay_slots():
    """One captured padded batch, replayed with changing slot tables (incl. padding) and diverging counters; inputs
    as vLLM hands them (q / k / v column slices of one projection, the gate fp32 [rows, H, d_k], beta fp32)."""
    H, K, Vd = SHAPES['ling']
    fmt = formats('ling')['int4-k-g64']
    slots = [5, 2, 7]
    lay, raw, blob, views, _ = _setup_slots('ling', fmt, 8, 8, slots)
    blob_e = blob.clone()
    ve = lay.views(blob_e)
    R = 4
    idx = torch.zeros(R, device=DEV, dtype=torch.int32)
    proj = torch.zeros(R, 2 * H * K + H * Vd, device=DEV, dtype=torch.bfloat16)
    q = proj[:, :H * K].view(R, H, K)
    k = proj[:, H * K:2 * H * K].view(R, H, K)
    v = proj[:, 2 * H * K:].view(R, H, Vd)
    g = torch.zeros(R, H, K, device=DEV)
    beta = torch.zeros(R, H, device=DEV)
    out = torch.zeros(R, H, Vd, device=DEV, dtype=torch.bfloat16)
    call = lambda vw, o: lay.step(vw, idx, q, k, v, g, beta, o)   # noqa: E731
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
        proj.copy_(torch.randn(proj.shape, device=DEV, generator=gen))
        g.copy_(-5 * torch.sigmoid(torch.randn(g.shape, device=DEV, generator=gen) * 2 - 3))
        beta.copy_(torch.sigmoid(torch.randn(beta.shape, device=DEV, generator=gen)))
        Gr.replay()
        oe = torch.zeros_like(out)
        call(ve, oe)
        torch.cuda.synchronize()
        assert torch.equal(out, oe)
    assert torch.equal(blob, blob_e)
    assert sorted(int(views['NPTR'][s_, 0]) for s_ in slots) != [0, 0, 0]
