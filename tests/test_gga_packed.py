"""Packed GGA / STEP storage with the PW-S8 window (``integrations.vllm_gga.GGAPacked``, ``kernels/_gga_triton.py``)
against the fp32 emulation (``GGALayout``, itself checked against the reference path in test_vllm_gga.py): pack /
unpack of every width, the fused decode step from the same stored state (outputs, stored state, window), padding
rows, CUDA-graph capture, and the storage accounting. Needs a GPU with FP8 conversions (sm_89+)."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('needs a GPU', allow_module_level=True)
pytest.importorskip('triton')
if torch.cuda.get_device_capability() < (8, 9):
    pytest.skip('FP8 e4m3 conversions need sm_89+', allow_module_level=True)

import statequant as sq  # noqa: E402
from statequant.integrations.vllm_gga import GGALayout, GGAPacked  # noqa: E402

DEV = 'cuda'
B, HV, HK, D = 3, 8, 4, 128


def method(layers=2):
    q = sq.GGA(4, pivots=1)
    g = torch.Generator().manual_seed(0)
    widths = [[2, 4, 6, 8, 16, 2, 4, 6], [8, 6, 4, 2, 2, 16, 8, 4]]
    q.plan = {i: (torch.tensor(widths[i]), torch.rand(HV, D, generator=g) + 0.5) for i in range(layers)}
    q.masks = {i: b == 16 for i, (b, _) in q.plan.items()}
    return q


def tokens(n, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=DEV, generator=g)      # noqa: E731
    return [(r(B, HK, D), r(B, HK, D), r(B, HV, D), r(B, HV) * 0.5 - 2, r(B, HV)) for _ in range(n)]


@pytest.mark.parametrize('layer', [0, 1])
def test_pack_unpack_equals_fit(layer):
    q = method()
    pk, em = GGAPacked(q, HV, HK, D, D, pw=True), GGALayout(q, HV, HK, D, D, pw=True)
    Vp, Ve = pk.views(pk.alloc(6)), em.views(em.alloc(6))
    idx = torch.tensor([4, 2, 5], device=DEV, dtype=torch.int32)
    S = torch.randn(B, HV, D, D, device=DEV) * torch.logspace(-3, 0, D, device=DEV)
    pk.write(Vp, idx, S.transpose(-1, -2), layer=layer)
    em.write(Ve, idx, S.transpose(-1, -2), layer=layer)
    got, want = pk.stored(Vp, idx, layer), Ve['O'][idx.long()].transpose(-1, -2)
    assert torch.equal(got, want)
    assert torch.equal(pk.read(Vp, idx, layer=layer), em.read(Ve, idx, layer=layer))


@pytest.mark.parametrize('pw', [False, True])
def test_step_matches_emulation(pw):
    """Every step from the same stored state and window (the packed slot's, copied into the emulation)."""
    q = method()
    layer = 0
    pk, em = GGAPacked(q, HV, HK, D, D, pw=pw), GGALayout(q, HV, HK, D, D, pw=pw)
    Vp, Ve = pk.views(pk.alloc(6)), em.views(em.alloc(6))
    idx = torch.tensor([4, 0, 5], device=DEV, dtype=torch.int32)       # row 1 is padding
    li = idx.long()
    ok = idx > 0
    S0 = torch.randn(B, HV, D, D, device=DEV) * 0.3
    pk.write(Vp, idx[ok], S0[ok].transpose(-1, -2), layer=layer)
    A_log, dt = torch.randn(HV, device=DEV) * 0.3, torch.randn(HV, device=DEV) * 0.3
    piv = q.plan[layer][0] == 16
    for t, (qq, kk, vv, a, b) in enumerate(tokens(21)):
        Ve['O'][li[ok]] = pk.stored(Vp, idx[ok], layer).transpose(-1, -2)
        if pw:
            P, V, n = pk.window(Vp, idx[ok])
            Ve['P'][li[ok]], Ve['V'][li[ok]] = P, V
            Ve['NQ'][li[ok]] = n[:, None].expand(-1, 4).to(torch.int32)
        op, oe = torch.empty(B, HV, D, device=DEV), torch.empty(B, HV, D, device=DEV)
        pk.step(Vp, idx, qq, kk, vv, op, D ** -0.5, gate=(a, b, A_log, dt), layer=layer)
        em.step(Ve, idx, qq, kk, vv, oe, D ** -0.5, gate=(a, b, A_log, dt), layer=layer)
        assert float(op[1].abs().max()) == 0.
        assert ((op - oe).norm() / oe.norm()).item() < 1e-5, t
        Sp = pk.stored(Vp, idx[ok], layer)
        Se = Ve['O'][li[ok]].transpose(-1, -2)
        flips = ((Sp - Se).abs() > 1e-3 * Se.abs().amax()).float().mean().item()
        assert flips < 2e-3, (t, flips)
        if pw:
            P, V, n = pk.window(Vp, idx[ok])
            assert torch.equal(n, Ve['NQ'][li[ok], 0]), t
            nonp = (~piv).to(DEV)[None, None, :, None]
            assert ((P * nonp - Ve["P"][li[ok]]).abs() > 1e-6 * P.abs().amax()).float().mean().item() < 1e-3, t
            dv = (V * nonp != Ve['V'][li[ok]] * nonp)
            assert not dv.any(), (t, dv.float().mean().item(), (V * nonp)[dv][:4].tolist(),
                                                     (Ve['V'][li[ok]] * nonp)[dv][:4].tolist())


def test_cuda_graph_capture():
    q = method()
    pk = GGAPacked(q, HV, HK, D, D, pw=True)
    Vp = pk.views(pk.alloc(6))
    idx = torch.tensor([4, 0, 5], device=DEV, dtype=torch.int32)
    pk.write(Vp, idx[[0, 2]], torch.randn(2, HV, D, D, device=DEV) * 0.3, layer=0)
    snap = Vp['blob'].clone()
    toks = tokens(10)
    A_log, dt = torch.zeros(HV, device=DEV), torch.zeros(HV, device=DEV)
    args = [t.clone() for t in toks[0]]
    out = torch.empty(B, HV, D, device=DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        pk.step(Vp, idx, *args[:3], out, D ** -0.5, gate=(args[3], args[4], A_log, dt), layer=0)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        pk.step(Vp, idx, *args[:3], out, D ** -0.5, gate=(args[3], args[4], A_log, dt), layer=0)
    Vp['blob'].copy_(snap)
    ev = pk.views(snap.clone())
    oe = torch.empty_like(out)
    for tk in toks:
        for a_, b_ in zip(args, tk):
            a_.copy_(b_)
        gr.replay()
        pk.step(ev, idx, *tk[:3], oe, D ** -0.5, gate=(tk[3], tk[4], A_log, dt), layer=0)
        assert torch.equal(out, oe)
    assert torch.equal(Vp['blob'], ev['blob'])


def test_storage_accounting():
    q = method()
    for pw in (False, True):
        pk = GGAPacked(q, HV, HK, D, D, pw=pw)
        ref = sq.pure_write(q).bits_per_value(D, D) if pw else q.bits_per_value(D, D)
        assert pk.bits_per_value() == pytest.approx(ref, abs=1e-9)
        assert pk.nbytes >= max(pk.bytes_per_layer.values())
