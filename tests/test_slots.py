"""Slot-addressed packed states (statequant.kernels.SlotLayout): sequences at their own positions in one batch, slot
indirection, padding rows, per-sequence buffer counters, the torch read / write used around prefill, in-kernel
gating, CUDA-graph capture. The reference is ``PackedGDNState`` run per sequence (the same kernels in lockstep
mode), so outputs and stored codes must agree bit for bit. Needs a GPU and Triton."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('needs a GPU', allow_module_level=True)
pytest.importorskip('triton')

import statequant as sq  # noqa: E402
from statequant.kernels import PackedGDNState, SlotLayout, pack_vk  # noqa: E402

DEV = 'cuda'
HV, HK, D = 4, 2, 128
FORMATS = {'int4-k-g64': sq.Quantizer(4, 'k', 64), 'int4-v-g32': sq.Quantizer(4, 'v', 32),
           'int8-k-g32': sq.Quantizer(8, 'k', 32), 'bf16': 'bf16'}


def token(gen):
    r = lambda *s: torch.randn(*s, device=DEV, generator=gen)          # noqa: E731
    q, k, v = r(HK, D), r(HK, D), r(HV, D)
    g = -torch.rand(HV, device=DEV, generator=gen) * 0.08
    beta = torch.rand(HV, device=DEV, generator=gen).sigmoid()
    return q, k, v, g, beta


def _setup(name, m, bb, n_seq=3, n_slots=8, seed=0):
    torch.manual_seed(seed)
    lay = SlotLayout(FORMATS[name], HV, HK, D, D, m=m, buffer_bits=bb)
    # a blob with a slot stride larger than the layout (as in vLLM's padded pages) and garbage elsewhere
    raw = torch.randint(0, 256, (n_slots, lay.nbytes + 4096), device=DEV, dtype=torch.uint8)
    blob = raw[:, 2048:2048 + lay.nbytes]
    views = lay.views(blob)
    slots = [5, 2, 7][:n_seq]
    S0 = [torch.randn(HV, D, D, device=DEV) * 0.5 for _ in slots]                    # [HV, d_k, d_v]
    refs = [PackedGDNState(s, FORMATS[name], m=m, hv=HV, hk=HK, buffer_bits=bb, out_dtype=torch.float32)
            for s in S0]
    lay.write(views, torch.tensor(slots, device=DEV), torch.stack([s.transpose(-1, -2) for s in S0]))
    return lay, raw, blob, views, slots, refs


def adopt(views, slot, ref):
    """Copy a slot's stored state and buffer into the lockstep reference (re-sync after a comparison)."""
    ref.codes.copy_(views['S'][slot].reshape(ref.codes.shape))
    if 'SC' in views:
        ref.scales.copy_(views['SC'][slot].reshape(ref.scales.shape))
    for a, b in (('KB', 'KB'), ('UB', 'UB'), ('KBS', 'KBS'), ('UBS', 'UBS'), ('GB', 'GB'), ('GCUR', 'GCUR')):
        if a in views:
            getattr(ref, b).copy_(views[a][slot].reshape(getattr(ref, b).shape))


@pytest.mark.parametrize('name', list(FORMATS))
@pytest.mark.parametrize('m,bb', [(8, 8), (4, 16), (1, 8)])
def test_slot_step_matches_lockstep_per_sequence(name, m, bb):
    lay, raw, blob, views, slots, refs = _setup(name, m, bb)
    gens = [torch.Generator(device=DEV).manual_seed(10 + i) for i in range(len(slots))]
    for i, r in enumerate(refs):                                   # stored state identical after write
        assert torch.equal(views['S'][slots[i]], r.codes.view(HV, *r.codes.shape[1:]))
    outside = raw.clone()
    sched = torch.Generator().manual_seed(1)
    steps = [0] * len(slots)
    worst, flips = 0., 0
    for t in range(30):
        # a random subset of the sequences, in random order, plus padding rows (slot 0 and -1)
        live = [i for i in torch.randperm(len(slots), generator=sched).tolist()
                if torch.rand(1, generator=sched).item() < 0.7]
        rows = live + ['pad0', 'padneg']
        perm = torch.randperm(len(rows), generator=sched).tolist()
        rows = [rows[j] for j in perm]
        idx = torch.tensor([slots[r] if isinstance(r, int) else (0 if r == 'pad0' else -1) for r in rows],
                           device=DEV, dtype=torch.int32)
        B = len(rows)
        q = torch.zeros(B, HK, D, device=DEV)
        k = torch.zeros(B, HK, D, device=DEV)
        v = torch.zeros(B, HV, D, device=DEV)
        g = torch.zeros(B, HV, device=DEV)
        beta = torch.zeros(B, HV, device=DEV)
        want = {}
        for b, r in enumerate(rows):
            if not isinstance(r, int):
                continue
            q[b], k[b], v[b], g[b], beta[b] = token(gens[r])
            want[b] = refs[r].step(q[b].contiguous(), k[b].contiguous(), v[b].contiguous(), g[b].contiguous(),
                                   beta[b].contiguous(), D ** -0.5).clone()
            steps[r] += 1
        out = torch.full((B, HV, D), 7., device=DEV)
        lay.step(views, idx, q, k, v, out, D ** -0.5, g=g, beta=beta)
        for b, r in enumerate(rows):
            if isinstance(r, int):
                # same kernels, but Triton specialises on the strides (alignment -> load vectorisation ->
                # reduction order), so allow fp32-ulp differences and count code flips at rounding ties
                o, w = out[b], want[b].view(HV, D)
                worst = max(worst, ((o - w).abs().max() / w.abs().max()).item())
                assert int(views['NPTR'][slots[r], 0]) == refs[r].n == steps[r] % m
                flips += (views['S'][slots[r]] != refs[r].codes.view(HV, *refs[r].codes.shape[1:])).sum().item()
                adopt(views, slots[r], refs[r])
            else:
                assert torch.equal(out[b], torch.zeros_like(out[b]))
    n_codes = 30 * len(slots) * HV * D * D
    print(f'\n[{name} m={m} buf{bb}] max rel |o - o_lockstep| {worst:.2e}; code entries differing {flips}')
    assert worst < 1e-5, worst
    assert flips <= 1e-5 * n_codes, flips
    # nothing outside the live slots' blobs was touched (slot 0, unused slots, the page padding)
    keep = torch.ones_like(raw, dtype=torch.bool)
    for s in slots:
        keep[s, 2048:2048 + lay.nbytes] = False
    assert torch.equal(raw[keep], outside[keep])


@pytest.mark.parametrize('name', ['int4-k-g64', 'bf16'])
def test_read_is_the_state_the_next_step_reads(name):
    """read() (stored state advanced over the pending buffer, in torch) predicts the next step's output."""
    m = 8
    lay, raw, blob, views, slots, refs = _setup(name, m, 8)
    gens = [torch.Generator(device=DEV).manual_seed(10 + i) for i in range(len(slots))]
    idx = torch.tensor(slots, device=DEV, dtype=torch.int32)
    for t in range(13):                                            # leaves n = 5 in every slot
        toks = [token(gn) for gn in gens]
        q, k, v, g, beta = (torch.stack(x) for x in zip(*toks))
        out = torch.empty(len(slots), HV, D, device=DEV)
        if t == 12:
            S = lay.read(views, idx)                               # [P, HV, d_v, d_k], before the 13th token
            qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * D ** -0.5
            kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            qx, kx = qn.repeat_interleave(HV // HK, 1), kn.repeat_interleave(HV // HK, 1)
            a = g.exp()[..., None]
            sk = torch.einsum('phvk,phk->phv', S, kx)
            u = beta[..., None] * (v - a * sk)
            o_pred = a * torch.einsum('phvk,phk->phv', S, qx) + (qx * kx).sum(-1, keepdim=True) * u
        lay.step(views, idx, q, k, v, out, D ** -0.5, g=g, beta=beta)
    assert int(views['NPTR'][slots[0], 0]) == 13 % m
    rel = ((out - o_pred).norm() / o_pred.norm()).item()
    print(f'\n[{name}] read() vs kernel step: rel err {rel:.2e}')
    assert rel < 1e-5, rel


def test_write_read_roundtrip_and_reset():
    lay = SlotLayout(FORMATS['int4-k-g64'], HV, HK, D, D, m=8)
    blob = torch.randint(0, 256, (4, lay.nbytes), device=DEV, dtype=torch.uint8)  # garbage counters / buffers
    views = lay.views(blob)
    S = torch.randn(2, HV, D, D, device=DEV)
    idx = torch.tensor([3, 1], device=DEV)
    lay.write(views, idx, S)
    got = lay.read(views, idx)
    from statequant.kernels import unpack_vk
    c, s = pack_vk(S.reshape(-1, D, D), FORMATS['int4-k-g64'])
    assert torch.equal(got, unpack_vk(c, s, FORMATS['int4-k-g64'], D).view_as(S))
    assert views['NPTR'][idx, 0].tolist() == [0, 0] and torch.equal(views['GCUR'][idx], torch.zeros(2, HV, device=DEV))


def test_in_kernel_gating():
    lay, raw, blob, views, slots, refs = _setup('int4-k-g64', 8, 8)
    gen = torch.Generator(device=DEV).manual_seed(3)
    B = len(slots)
    q, k, v = (torch.randn(B, h, D, device=DEV, generator=gen) for h in (HK, HK, HV))
    a = torch.randn(B, HV, device=DEV, generator=gen).to(torch.bfloat16)
    bb = torch.randn(B, HV, device=DEV, generator=gen).to(torch.bfloat16)
    A_log = torch.randn(HV, device=DEV, generator=gen) * 0.5
    dt_bias = torch.randn(HV, device=DEV, generator=gen).to(torch.bfloat16)
    g = -A_log.exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())
    beta = bb.float().sigmoid()
    blob2 = blob.clone()
    v2 = lay.views(blob2)
    idx = torch.tensor(slots, device=DEV, dtype=torch.int32)
    o1 = torch.empty(B, HV, D, device=DEV)
    o2 = torch.empty(B, HV, D, device=DEV)
    lay.step(views, idx, q, k, v, o1, D ** -0.5, g=g, beta=beta)
    lay.step(v2, idx, q, k, v, o2, D ** -0.5, gate=(a, bb, A_log, dt_bias))
    assert torch.allclose(o1, o2, rtol=1e-5, atol=1e-5), (o1 - o2).abs().max()


def test_cuda_graph_capture_slot_step():
    """Capture one padded batch; replay with changing slot tables (incl. padding) and per-sequence counters."""
    m = 8
    lay, raw, blob, views, slots, refs = _setup('int4-k-g64', m, 8)
    blob_e = blob.clone()
    ve = lay.views(blob_e)
    B = 4
    idx = torch.zeros(B, device=DEV, dtype=torch.int32)
    q, k = torch.zeros(B, HK, D, device=DEV), torch.zeros(B, HK, D, device=DEV)
    v, out = torch.zeros(B, HV, D, device=DEV), torch.zeros(B, HV, D, device=DEV)
    g, beta = torch.zeros(B, HV, device=DEV), torch.zeros(B, HV, device=DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):                                     # compile (idx all padding: no state touched)
        lay.step(views, idx, q, k, v, out, D ** -0.5, g=g, beta=beta)
    torch.cuda.current_stream().wait_stream(s)
    G = torch.cuda.CUDAGraph()
    with torch.cuda.graph(G):
        lay.step(views, idx, q, k, v, out, D ** -0.5, g=g, beta=beta)
    gen = torch.Generator(device=DEV).manual_seed(5)
    tables = [[5, 2, 7, 0], [2, 0, 0, 0], [7, 5, 0, 0], [5, 7, 2, 0]] * 5
    for tab in tables:
        idx.copy_(torch.tensor(tab, dtype=torch.int32))
        for x in (q, k, v):
            x.copy_(torch.randn(x.shape, device=DEV, generator=gen))
        g.copy_(-torch.rand(g.shape, device=DEV, generator=gen) * 0.08)
        beta.copy_(torch.rand(beta.shape, device=DEV, generator=gen))
        G.replay()
        oe = torch.zeros_like(out)
        lay.step(ve, idx, q, k, v, oe, D ** -0.5, g=g, beta=beta)
        torch.cuda.synchronize()
        assert torch.equal(out, oe)
    assert torch.equal(blob, blob_e)
    assert sorted(int(views['NPTR'][s_, 0]) for s_ in slots) != [0, 0, 0]       # counters diverged


def test_layout_bytes():
    lay = SlotLayout(sq.Quantizer(4, 'k', 64), 32, 16, 128, 128, m=8, buffer_bits=8)
    b = lay.breakdown()
    assert b['S'] == 32 * 128 * 64 and b['SC'] == 32 * 128 * 2 * 2
    assert b['KB'] == 16 * 8 * 128 and b['UB'] == 32 * 8 * 128
    assert lay.nbytes == sum(-(-x // 16) * 16 for x in b.values())
    print('\n', lay, b)
