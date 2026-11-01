"""The vLLM gga mode's slot step (``integrations.vllm_gga.GGALayout``) against the reference path
(``methods_ext.gga.PureWrite`` / the StepQuant base): same plan, same tokens, outputs and stored state step by step,
with and without PW; capture in a CUDA graph. Needs a GPU (no vLLM)."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('needs a GPU', allow_module_level=True)
pytest.importorskip('triton')

import statequant as sq  # noqa: E402
from statequant.families import BY_NAME, Piece  # noqa: E402
from statequant.integrations.vllm_gga import GGALayout  # noqa: E402

GDN = BY_NAME['gdn']
DEV = 'cuda'
B, HV, HK, D = 3, 4, 2, 64


def method():
    q = sq.GGA(4, pivots=1)
    g = torch.Generator().manual_seed(0)
    q.plan = {0: (torch.tensor([2, 4, 16, 8]), torch.rand(HV, D, generator=g) + 0.5)}
    q.masks = {0: q.plan[0][0] == 16}
    return q


def tokens(n, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=DEV, generator=g)      # noqa: E731
    return [(r(B, HK, D), r(B, HK, D), r(B, HV, D), r(B, HV) * 0.5 - 2, r(B, HV)) for _ in range(n)]


def kw_of(qq, kk, vv, a, b):
    return {'q': qq[:, None], 'k': kk[:, None], 'v': vv[:, None], 'g': a[:, None], 'beta': b[:, None],
            'A_log': torch.zeros(HV, device=DEV), 'dt_bias': torch.zeros(HV, device=DEV),
            'use_gate_in_kernel': True, 'use_beta_sigmoid_in_kernel': True, 'use_qk_l2norm_in_kernel': True,
            'scale': D ** -0.5}


def delta(S, kw):
    x = GDN.inputs(kw)
    Sa = x['log_decay'][:, 0].exp()[..., None, None] * S
    u = x['beta'][:, 0][..., None] * (x['v'][:, 0] - torch.einsum('bhkv,bhk->bhv', Sa, x['k'][:, 0]))
    Sn = Sa + x['k'][:, 0][..., None] * u[:, :, None]
    return torch.einsum('bhkv,bhk->bhv', Sn, x['q'][:, 0]), Sn


@pytest.mark.parametrize('pw', [False, True])
def test_slot_step_matches_reference(pw):
    """Every step from the same stored state (the slot's, copied into the reference): the outputs and the new stored
    state agree to float rounding. (Whole trajectories diverge: per-token low-bit re-rounding, 2-bit heads above
    all, turns float-level differences into code flips.)"""
    q = method()
    lay = GGALayout(q, HV, HK, D, D, pw=pw, m=8)
    V = lay.views(lay.alloc(6))
    idx = torch.tensor([4, 2, 5], device=DEV, dtype=torch.int32)
    li = idx.long()
    S0 = torch.randn(B, HV, D, D, device=DEV) * 0.3
    lay.write(V, idx, S0.transpose(-1, -2), layer=0)
    toks = tokens(19)
    ref = sq.pure_write(q, m=8) if pw else q
    if pw:
        ref.reset()
        ref.layers[0] = {'O': None, 'Q': [], 'R': []}
    A_log, dt = torch.zeros(HV, device=DEV), torch.zeros(HV, device=DEV)
    for t, tk in enumerate(toks):
        kw = kw_of(*tk)
        O = V['O'][li].transpose(-1, -2).clone()
        if pw:
            n = int(V['NQ'][li[0], 0])
            st = ref.layers[0]
            st['O'] = O
            st['Q'] = [(V['P'][li, i].clone(), V['V'][li, i].clone()) for i in range(n)]
            S = ref._logical(st)
            S = ref.prepare(S, 0, Piece(kw, GDN, 0, None, torch.full((B,), 3 + t)))
            o, Sn = delta(S, kw)
            d = ref.output_delta(0)
            o = o if d is None else o + d[:, 0]
            Snew = ref.piece(Sn, 0, Piece(kw, GDN, 0, S, torch.full((B,), 3 + t), hit=torch.ones(B, dtype=torch.bool)))
            Oref = ref.layers[0]['O']
        else:
            o, Sn = delta(O, kw)
            Oref = Snew = q.quantize(Sn, 0)
        out = torch.empty(B, HV, D, device=DEV)
        qq, kk, vv, a, b = tk
        lay.step(V, idx, qq, kk, vv, out, D ** -0.5, gate=(a, b, A_log, dt), layer=0)
        assert ((out - o).norm() / o.norm()).item() < 2e-5, t
        Ol = V['O'][li].transpose(-1, -2)
        flips = ((Ol - Oref).abs() > 1e-3 * Oref.abs().amax()).float().mean().item()
        assert flips < 1e-3, (t, flips)
        assert ((lay.read(V, idx).transpose(-1, -2) - Snew).norm() / Snew.norm()).item() < 2e-3, t


def test_cuda_graph_capture():
    q = method()
    lay = GGALayout(q, HV, HK, D, D, pw=True, m=8)
    V = lay.views(lay.alloc(6))
    idx = torch.tensor([4, 0, 5], device=DEV, dtype=torch.int32)         # with a padding row
    lay.write(V, idx[[0, 2]], torch.randn(2, HV, D, D, device=DEV) * 0.3, layer=0)
    toks = tokens(4)
    A_log, dt = torch.zeros(HV, device=DEV), torch.zeros(HV, device=DEV)
    args = [t.clone() for t in toks[0]]
    out = torch.empty(B, HV, D, device=DEV)
    snap = {n: x.clone() for n, x in V.items()}
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        lay.step(V, idx, *args[:3], out, D ** -0.5, gate=(args[3], args[4], A_log, dt), layer=0)
    torch.cuda.current_stream().wait_stream(s)
    for n in V:
        V[n].copy_(snap[n])
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        lay.step(V, idx, *args[:3], out, D ** -0.5, gate=(args[3], args[4], A_log, dt), layer=0)
    for n in V:
        V[n].copy_(snap[n])
    eager = {n: x.clone() for n, x in V.items()}
    ev = lay.views(torch.zeros(6, lay.nbytes, dtype=torch.uint8, device=DEV))
    for n in ev:
        ev[n].copy_(snap[n])
    oe = torch.empty_like(out)
    for tk in toks:
        for a_, b_ in zip(args, tk):
            a_.copy_(b_)
        gr.replay()
        lay.step(ev, idx, *tk[:3], oe, D ** -0.5, gate=(tk[3], tk[4], A_log, dt), layer=0)
        assert torch.allclose(out, oe, atol=1e-5) and float(out[1].abs().max()) == 0.
    for n in V:
        assert torch.allclose(V[n].float(), ev[n].float(), atol=1e-5), n
    del eager
