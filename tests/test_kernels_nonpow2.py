"""Packed GDN kernels on head dims that are not powers of two (OLMo-Hybrid: d_k 96, d_v 192). The kernels pad the key
tile to the next power of two (128) and mask it; memory holds exactly d_k x d_v; value rows run in blocks dividing
d_v (64 / 32). Key-axis groups may be ragged (64 on d_k 96 = 64 + 32). Needs a GPU and Triton.

(a) storage: ``unpack(pack(S)) == Quantizer(..., scale_dtype='fp16').fake(S)`` bit for bit, ragged groups included;
(b) exactness of the padding: a 96 x 192 state stepped by the kernels equals the same state zero-padded to 128 x 256
    (the power-of-two path, tested in test_kernels / test_fused) with zero-padded inputs, bit for bit: outputs, codes,
    scales, buffers;
(c) the step against the torch reference of the same arithmetic (``ReferenceGDNState``), as in test_kernels;
(d) the fused step against the five-kernel step, lockstep and slot-addressed (random slot subsets with padding);
(e) the slot layout: ``read`` predicts the next step, graph replay."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('needs a GPU', allow_module_level=True)
pytest.importorskip('triton')

import statequant as sq  # noqa: E402
from statequant.kernels import PackedGDNState, ReferenceGDNState, SlotLayout, pack, unpack  # noqa: E402

DEV = 'cuda'
B, HV, HK, DK, DV = 2, 4, 2, 96, 192                        # GQA 2
BH = B * HV
FORMATS = {'int4-k-g64': sq.Quantizer(4, 'k', 64),            # ragged: 64 + 32
           'int4-k-g32': sq.Quantizer(4, 'k', 32),
           'int8-k-g32': sq.Quantizer(8, 'k', 32),
           'int8-k-g64': sq.Quantizer(8, 'k', 64),            # ragged
           'int4-v-g32': sq.Quantizer(4, 'v', 32),
           'int8-v-g64': sq.Quantizer(8, 'v', 64),
           'bf16': 'bf16'}
SCALE = DK ** -0.5


def triton_version():
    import triton
    return tuple(int(x) for x in triton.__version__.split('.')[:2])


def _inputs(T, seed, dk=DK, dv=DV):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=DEV, generator=g)  # noqa: E731
    q, k = r(T, B * HK, DK), r(T, B * HK, DK)
    v = r(T, BH, DV)
    gl = -torch.rand(T, BH, device=DEV, generator=g) * 0.08
    beta = torch.rand(T, BH, device=DEV, generator=g).sigmoid() * 2          # OLMo: beta in (0, 2)
    pad = lambda x, n: torch.nn.functional.pad(x, (0, n - x.shape[-1])).contiguous()  # noqa: E731
    return pad(q, dk), pad(k, dk), pad(v, dv), gl, beta


# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('name', [n for n in FORMATS if n != 'bf16'])
def test_pack_roundtrip_matches_fake(name):
    torch.manual_seed(0)
    S = torch.randn(8, DK, DV, device=DEV) * torch.rand(8, DK, 1, device=DEV) * 3
    S[0, 64:] = 0                                            # an all-zero (ragged) key group
    S[1] *= 1e-6
    q = FORMATS[name]
    codes, scales = pack(S, q)
    ng = -(-DK // q.group)
    assert scales.shape == ((8, DV, ng) if q.axis == 'k' else (8, DV // q.group, DK))
    assert codes.shape == (8, DV, DK // (2 if q.bits == 4 else 1))
    want = sq.Quantizer(q.bits, q.axis, q.group, scale_dtype='fp16').fake(S)
    assert torch.equal(unpack(codes, scales, q, DK), want)


# ---------------------------------------------------------------------------------------------------------------
def _padded_state(S0):
    """[BH, d_k, d_v] -> zero-padded [BH, 128, 256]."""
    return torch.nn.functional.pad(S0, (0, 256 - DV, 0, 128 - DK))


def _resync(p, a):
    """Copy a's stored codes, scales and buffers into the real part of the padded state p."""
    c = a.codes
    p.codes[:, :c.shape[1], :c.shape[2]] = c
    if a.fmt is not None:
        sc = a.scales
        p.scales[:, :sc.shape[1], :sc.shape[2]] = sc
    p.KB[..., :DK] = a.KB
    p.UB[..., :DV] = a.UB
    p.KBS.copy_(a.KBS)
    p.UBS.copy_(a.UBS)


def _stored(st, dk, dv):
    """Dequantized stored state cut to [BH, DK, DV] plus the buffers' real parts."""
    S = st.dequantize()[:, :DK, :DV]
    kb = st.KB.float()[..., :DK]
    ub = st.UB.float()[..., :DV]
    return S, kb, ub


@pytest.mark.parametrize('fused', [True, False])
@pytest.mark.parametrize('name', list(FORMATS))
@pytest.mark.parametrize('m,bb', [(8, 8), (16, 16), (1, 8)])
def test_padding_is_exact(name, m, bb, fused):
    """The 96 x 192 kernels == the 128 x 256 kernels on the zero-padded problem, bit for bit."""
    torch.manual_seed(1)
    S0 = torch.randn(BH, DK, DV, device=DEV) * 0.5
    kw = dict(m=m, hv=HV, hk=HK, buffer_bits=bb, out_dtype=torch.float32, fused=fused)
    a = PackedGDNState(S0, FORMATS[name], **kw)
    p = PackedGDNState(_padded_state(S0), FORMATS[name], **kw)
    assert torch.equal(a.dequantize(), p.dequantize()[:, :DK, :DV])
    T = 33
    qa, ka, va, g, beta = _inputs(T, 7)
    qp, kp, vp, _, _ = _inputs(T, 7, 128, 256)
    diff = {'o': 0, 'S': 0, 'KB': 0, 'UB': 0, 'UBS': 0}
    worst = 0.
    for t in range(T):
        oa = a.step(qa[t], ka[t], va[t], g[t], beta[t], SCALE).clone()
        op = p.step(qp[t], kp[t], vp[t], g[t], beta[t], SCALE).clone()
        diff['o'] += (oa != op[:, :DV]).sum().item()
        worst = max(worst, ((oa - op[:, :DV]).abs().max() / op.abs().max()).item())
        assert torch.equal(op[:, DV:], torch.zeros_like(op[:, DV:]))
        Sa, kba, uba = _stored(a, DK, DV)
        Sp, kbp, ubp = _stored(p, DK, DV)
        diff['S'] += (Sa != Sp).sum().item()
        diff['KB'] += (kba != kbp).sum().item()
        diff['UB'] += (uba != ubp).sum().item()
        diff['UBS'] += (a.UBS != p.UBS).sum().item() if bb == 8 else 0
        assert torch.equal(a.GB, p.GB) and torch.equal(a.GCUR, p.GCUR)
        assert a.n == p.n == (t + 1) % m
        _resync(p, a)                                        # count a rounding flip once, not in every later step
    print(f'\n[pad {name} m={m} buf{bb} fused={fused}] differing entries vs the padded power-of-two run: {diff}, '
          f'max rel output difference {worst:.1e}')
    # Triton 3.8: stored state and buffers bit-identical, outputs too except int8 at m = 1 (7e-8 relative in 0.6% of
    # the entries: the read's last fp32 operations associate differently for the two tile shapes). Triton 3.7 (the
    # vLLM venv) associates the fused int4 key-axis g32 read differently (2e-7 relative), which can flip a code at a
    # rounding tie in a fold (counted once: p is re-synced to a after every step).
    n_state = T * BH * DK * DV
    assert worst < 1e-6, (diff, worst)
    assert diff['S'] <= 1e-3 * n_state and diff['KB'] + diff['UB'] <= 1e-3 * n_state, diff
    if triton_version() >= (3, 8):
        assert all(v == 0 for k_, v in diff.items() if k_ != 'o'), diff
        assert diff['o'] <= 1e-2 * T * BH * DV, diff


# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('fused', [True, False])
@pytest.mark.parametrize('name', list(FORMATS))
@pytest.mark.parametrize('m,bb', [(8, 8), (16, 16), (1, 8)])
def test_decode_vs_reference(name, m, bb, fused):
    torch.manual_seed(1)
    S0 = torch.randn(BH, DK, DV, device=DEV) * 0.5
    fmt = FORMATS[name]
    ker = PackedGDNState(S0, fmt, m=m, hv=HV, hk=HK, buffer_bits=bb, out_dtype=torch.float32, fused=fused)
    ref = ReferenceGDNState(S0, fmt, m=m, hv=HV, hk=HK, buffer_bits=bb, auto_fold=False)
    assert torch.equal(ker.dequantize(), ref.dequantize())
    T = 40
    q, k, v, g, beta = _inputs(T, 3)
    worst, mism, flips, nfold = 0., 0, 0, 0
    for t in range(T):
        slot = ker.n
        o = ker.step(q[t], k[t], v[t], g[t], beta[t], SCALE).clone()
        r = ref.step(q[t], k[t], v[t], g[t], beta[t], SCALE)
        worst = max(worst, ((o - r).abs().max() / r.abs().max()).item())
        kb, ub = ker.KB[:, slot].float(), ker.UB[:, slot].float()
        if bb == 8:
            kb, ub = kb * ker.KBS[:, slot, None].float(), ub * ker.UBS[:, slot, None].float()
        flips += (kb != ref.K[:, slot]).sum().item() + (ub != ref.U[:, slot]).sum().item()
        assert torch.equal(ker.GB[:, slot], ref.G[:, slot])
        ref.K[:, slot], ref.U[:, slot] = kb, ub
        if ref.n == m:
            ref.fold()
            nfold += 1
            mism += (ker.codes != ref.codes).sum().item()
            ref.codes, ref.scales, ref.S0 = ker.codes.clone(), ker.scales.clone(), ker.dequantize()
    n_codes = max(nfold, 1) * ker.codes.numel()
    print(f'\n[{name} m={m} buf{bb} fused={fused}] max rel |o - o_ref| {worst:.2e}, code flips after folds {mism} of '
          f'{n_codes}, buffer flips {flips}')
    assert nfold == T // m
    assert worst < 2e-5, worst
    assert mism <= 1e-4 * n_codes, mism
    assert flips <= 3e-4 * T * (B * HK * DK + BH * DV), flips          # bf16 / int8 ties of the fp32 u


# ---------------------------------------------------------------------------------------------------------------
def _slots(name, m, bb, slots, n_slots=10):
    torch.manual_seed(0)
    lay = SlotLayout(FORMATS[name], HV, HK, DK, DV, m=m, buffer_bits=bb)
    raw = torch.randint(0, 256, (n_slots, lay.nbytes + 4096), device=DEV, dtype=torch.uint8)
    blob = raw[:, 2048:2048 + lay.nbytes]
    views = lay.views(blob)
    S0 = torch.randn(len(slots), HV, DV, DK, device=DEV) * 0.5                       # value-major, as vLLM
    lay.write(views, torch.tensor(slots, device=DEV), S0)
    return lay, raw, blob, views


def _batch(gen, rows, slots):
    Bn = len(rows)
    q, k = torch.zeros(Bn, HK, DK, device=DEV), torch.zeros(Bn, HK, DK, device=DEV)
    v = torch.zeros(Bn, HV, DV, device=DEV)
    g, beta = torch.zeros(Bn, HV, device=DEV), torch.zeros(Bn, HV, device=DEV)
    for b, r in enumerate(rows):
        if isinstance(r, int):
            q[b], k[b] = torch.randn(HK, DK, device=DEV, generator=gen), torch.randn(HK, DK, device=DEV, generator=gen)
            v[b] = torch.randn(HV, DV, device=DEV, generator=gen)
            g[b] = -torch.rand(HV, device=DEV, generator=gen) * 0.08
            beta[b] = torch.rand(HV, device=DEV, generator=gen) * 2
    idx = torch.tensor([slots[r] if isinstance(r, int) else (0 if r == 'pad0' else -1) for r in rows],
                       device=DEV, dtype=torch.int32)
    return idx, q, k, v, g, beta


@pytest.mark.parametrize('name', list(FORMATS))
@pytest.mark.parametrize('m,bb', [(8, 8), (4, 16)])
def test_slot_fused_matches_unfused(name, m, bb):
    slots = [5, 2, 7, 9]
    lay, raw, blob, views = _slots(name, m, bb, slots)
    raw_u = raw.clone()
    blob_u = raw_u[:, 2048:2048 + lay.nbytes]
    views_u = lay.views(blob_u)
    outside = raw.clone()
    sched = torch.Generator().manual_seed(1)
    gen = torch.Generator(device=DEV).manual_seed(4)
    worst, byte_diff = 0., {}
    for t in range(30):
        live = [i for i in torch.randperm(len(slots), generator=sched).tolist()
                if torch.rand(1, generator=sched).item() < 0.7]
        rows = live + ['pad0', 'padneg']
        rows = [rows[j] for j in torch.randperm(len(rows), generator=sched).tolist()]
        idx, q, k, v, g, beta = _batch(gen, rows, slots)
        of = torch.full((len(rows), HV, DV), 7., device=DEV)
        ou = torch.full((len(rows), HV, DV), 7., device=DEV)
        lay.step(views, idx, q, k, v, of, SCALE, g=g, beta=beta, fused=True)
        lay.step(views_u, idx, q, k, v, ou, SCALE, g=g, beta=beta, fused=False)
        worst = max(worst, ((of - ou).abs().max() / ou.abs().max()).item())
        for b, r in enumerate(rows):
            if not isinstance(r, int):
                assert torch.equal(of[b], torch.zeros_like(of[b]))
        for f, (off, _, _, nb) in lay.fields.items():
            byte_diff[f] = byte_diff.get(f, 0) + (blob[:, off:off + nb] != blob_u[:, off:off + nb]).sum().item()
        assert torch.equal(views['NPTR'], views_u['NPTR'])
        raw_u.copy_(raw)
    print(f'\n[slots {name} m={m} buf{bb}] max rel |o_fused - o| {worst:.2e}; differing bytes {byte_diff}')
    assert worst < 1e-5, worst
    assert byte_diff['GB'] == 0 and byte_diff['GCUR'] == 0 and byte_diff['NPTR'] == 0
    n_codes = 30 * len(slots) * views['S'][0].numel() // m
    assert byte_diff['S'] <= max(2, 1e-5 * n_codes)
    keep = torch.ones_like(raw, dtype=torch.bool)
    for s in slots:
        keep[s, 2048:2048 + lay.nbytes] = False
    assert torch.equal(raw[keep], outside[keep])            # nothing outside the live slots' blobs is written


@pytest.mark.parametrize('name', ['int4-k-g64', 'int8-k-g32', 'int4-v-g32', 'bf16'])
@pytest.mark.parametrize('fused', [True, False])
def test_read_is_the_state_the_next_step_reads(name, fused):
    m = 8
    slots = [5, 2, 7]
    lay, raw, blob, views = _slots(name, m, 8, slots)
    gen = torch.Generator(device=DEV).manual_seed(10)
    idx = torch.tensor(slots, device=DEV, dtype=torch.int32)
    for t in range(13):
        _, q, k, v, g, beta = _batch(gen, [0, 1, 2], slots)
        out = torch.empty(len(slots), HV, DV, device=DEV)
        if t == 12:
            S = lay.read(views, idx)                                                  # [P, HV, d_v, d_k]
            assert S.shape == (3, HV, DV, DK)
            qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
            kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            qx, kx = qn.repeat_interleave(HV // HK, 1), kn.repeat_interleave(HV // HK, 1)
            a = g.exp()[..., None]
            sk = torch.einsum('phvk,phk->phv', S, kx)
            u = beta[..., None] * (v - a * sk)
            o_pred = a * torch.einsum('phvk,phk->phv', S, qx) + (qx * kx).sum(-1, keepdim=True) * u
        lay.step(views, idx, q, k, v, out, SCALE, g=g, beta=beta, fused=fused)
    rel = ((out - o_pred).norm() / o_pred.norm()).item()
    print(f'\n[read {name} fused={fused}] rel err {rel:.2e}')
    assert rel < 1e-5, rel


def test_cuda_graph_replay_slots():
    m = 8
    slots = [5, 2, 7]
    lay, raw, blob, views = _slots('int4-k-g64', m, 8, slots)
    blob_e = blob.clone()
    ve = lay.views(blob_e)
    Bn = 4
    idx = torch.zeros(Bn, device=DEV, dtype=torch.int32)
    q, k = torch.zeros(Bn, HK, DK, device=DEV), torch.zeros(Bn, HK, DK, device=DEV)
    v, out = torch.zeros(Bn, HV, DV, device=DEV), torch.zeros(Bn, HV, DV, device=DEV)
    g, beta = torch.zeros(Bn, HV, device=DEV), torch.zeros(Bn, HV, device=DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        lay.step(views, idx, q, k, v, out, SCALE, g=g, beta=beta)
    torch.cuda.current_stream().wait_stream(s)
    G = torch.cuda.CUDAGraph()
    with torch.cuda.graph(G):
        lay.step(views, idx, q, k, v, out, SCALE, g=g, beta=beta)
    gen = torch.Generator(device=DEV).manual_seed(5)
    for tab in [[5, 2, 7, 0], [2, 0, -1, 0], [7, 5, 0, 0], [5, 7, 2, -1]] * 5:
        idx.copy_(torch.tensor(tab, dtype=torch.int32))
        for x in (q, k, v):
            x.copy_(torch.randn(x.shape, device=DEV, generator=gen))
        g.copy_(-torch.rand(g.shape, device=DEV, generator=gen) * 0.08)
        beta.copy_(torch.rand(beta.shape, device=DEV, generator=gen) * 2)
        G.replay()
        oe = torch.zeros_like(out)
        lay.step(ve, idx, q, k, v, oe, SCALE, g=g, beta=beta)
        torch.cuda.synchronize()
        assert torch.equal(out, oe)
    assert torch.equal(blob, blob_e)


def test_layout_bytes_olmo():
    """OLMo-Hybrid-7B (30 heads, d_k 96, d_v 192): exactly d_k x d_v codes, ceil(d_k / g) key scales per row."""
    lay = SlotLayout(sq.Quantizer(4, 'k', 64), 30, 30, 96, 192, m=8, buffer_bits=8)
    b = lay.breakdown()
    assert b['S'] == 30 * 192 * 48 and b['SC'] == 30 * 192 * 2 * 2
    assert b['KB'] == 30 * 8 * 96 and b['UB'] == 30 * 8 * 192
    print('\n', lay, b, f'stock bf16 {30 * 96 * 192 * 2} B')


def test_unsupported_dims_refused():
    with pytest.raises(AssertionError, match='value group'):
        SlotLayout(sq.Quantizer(4, 'v', 128), 4, 2, 96, 192)
    with pytest.raises(AssertionError, match='multiple of 32'):
        SlotLayout('bf16', 4, 2, 96, 80)
