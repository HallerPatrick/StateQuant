"""GGA and PW-S8 (arXiv 2610.39931): Table 3 storage from the Table 2 head counts; the writeback adjoint (Eq. 6) vs
autograd; PW driven through the piece hooks vs a literal transcription of Algorithm 1; PW with an identity quantizer
and an exact queue reproduces the unquantized model (App. A); GGA calibration end to end on a tiny model."""
import pytest
import torch

import statequant as sq
from statequant.families import BY_NAME, Piece
from statequant.methods_ext.gga import fp8_vec, writeback_adjoints

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GDN = BY_NAME['gdn']


def plan(counts, heads=32, layers=24):
    b = sum(([w] * n for w, n in zip([2, 4, 6, 8, 16], counts)), [])
    assert len(b) == heads * layers
    return {i: (torch.tensor(b[i * heads:(i + 1) * heads]), torch.ones(heads, 128)) for i in range(layers)}


@pytest.mark.parametrize('name,bits,counts,ref,ref_pw', [
    ('STEP@4 4B', 4, [380, 119, 158, 79, 32], 4.758, 5.790), ('GGA@4 4B', 4, [369, 139, 151, 77, 32], 4.753, 5.784),
    ('STEP@4 9B', 4, [372, 132, 156, 76, 32], 4.755, 5.786), ('GGA@4 9B', 4, [361, 167, 119, 89, 32], 4.749, 5.780),
    ('GGA@5 4B', 5, [215, 171, 165, 185, 32], 5.678, None), ('STEP@5 4B', 5, [214, 185, 140, 197, 32], 5.677, None),
    ('GGA@6 4B', 6, [0, 293, 182, 261, 32], 6.573, 7.604)])
def test_table3_storage(name, bits, counts, ref, ref_pw):
    q = sq.GGA(bits)
    q.plan = plan(counts)
    assert q.bits_per_value(128, 128) == pytest.approx(ref, abs=5e-4), name
    if ref_pw is not None:
        assert sq.pure_write(q).bits_per_value(128, 128) == pytest.approx(ref_pw, abs=5e-4), name


def test_writeback_adjoint_vs_autograd():
    torch.manual_seed(0)
    T, H, dk, dv = 24, 2, 8, 6
    q, k = torch.randn(T, H, dk), torch.nn.functional.normalize(torch.randn(T, H, dk), dim=-1)
    v, g = torch.randn(T, H, dv), torch.randn(T, H, dv)
    beta, a = torch.rand(T, H), torch.rand(T, H) * 0.5 + 0.5
    E = [torch.zeros(H, dk, dv, requires_grad=True) for _ in range(T)]
    S, L = torch.zeros(H, dk, dv), 0.
    for t in range(T):                       # S_t = A_t S_{t-1} + beta k v^T, readout, then the writeback perturbation
        Sa = a[t][:, None, None] * S
        S = Sa + beta[t][:, None, None] * k[t][:, :, None] * (v[t] - torch.einsum('hkv,hk->hv', Sa, k[t]))[:, None]
        L = L + (torch.einsum('hkv,hk->hv', S, q[t]) * g[t]).sum()
        S = S + E[t]
    L.backward()
    adj = writeback_adjoints(q, k, beta, a, g, 4)
    for t in range(4, T + 1, 4):
        want = E[t - 1].grad if E[t - 1].grad is not None else torch.zeros(H, dk, dv)   # t = T: no future use
        assert torch.allclose(adj[t], want, atol=1e-5, rtol=1e-4), t


class _Base(sq.StepQuant):
    """A STEP plan with a fixed quantizer (or identity) for driving PureWrite."""

    def __init__(self, bits, fmt=None):
        super().__init__(4)
        self.fmt = fmt
        self.plan = {0: (bits, torch.ones(len(bits), 32))}

    def quantize(self, S, layer):
        b = self.plan[layer][0].to(S.device)
        Q = S if self.fmt is None else self.fmt.fake(S)
        return torch.where((b == 16)[None, :, None, None], S.half().float(), Q)


def _alg1(O, R, toks, base, m, enc):
    """Algorithm 1 for every head of one layer (lockstep batch): returns the outputs and the final (O, R)."""
    outs = []
    piv = base.plan[0][0] == 16
    for q, k, v, beta, a in toks:
        A = lambda x: a[..., None] * (x - beta[..., None] * k * (k * x).sum(-1, keepdim=True))  # noqa: E731
        AO = a[..., None, None] * (O - beta[..., None, None] * k[..., None] * torch.einsum('bhk,bhkv->bhv', k, O)[:, :, None])
        Scand = AO + (beta[..., None] * k)[..., None] * v[..., None, :]
        R = [(enc(A(p)), vv) for p, vv in R]
        o = torch.einsum('bhkv,bhk->bhv', AO, q) + (beta[..., None] * k * q).sum(-1, keepdim=True) * v
        for p, vv in R:
            o = o + (p * q).sum(-1, keepdim=True) * vv
        Om = AO
        if len(R) == m:
            p, vv = R.pop(0)
            Om = Om + p[..., None] * vv[..., None, :]
        O = torch.where(piv[None, :, None, None], Scand.half().float(), base.quantize(Om, 0))
        live = (~piv)[None, :, None].float()
        R.append((enc(beta[..., None] * k) * live, enc(v) * live))
        outs.append(torch.where(piv[None, :, None], torch.einsum('bhkv,bhk->bhv', Scand, q), o))
    return outs, O, R


@pytest.mark.parametrize('extreme', [False, True])
def test_pw_matches_algorithm1(extreme):
    """Also with decays near 0 and beta near 1 (extreme), where A is (nearly) singular."""
    torch.manual_seed(0)
    B, H, dk, dv, m, steps = 2, 3, 32, 32, 4, 11
    base = _Base(torch.tensor([4, 16, 2]), sq.Quantizer(4, 'k', 32))
    pw = sq.pure_write(base, m=m)
    pw.reset()
    S0 = torch.randn(B, H, dk, dv) * 0.3

    def kw_of(q, k, v, beta, g):
        return {'q': q[:, None], 'k': k[:, None], 'v': v[:, None], 'g': g[:, None], 'beta': beta[:, None],
                'scale': 1.0}
    toks = []
    for _ in range(steps):
        q, k = torch.randn(B, H, dk), torch.nn.functional.normalize(torch.randn(B, H, dk), dim=-1)
        if extreme:
            toks.append((q, k, torch.randn(B, H, dv), 1 - torch.rand(B, H) * 1e-7, -torch.rand(B, H) * 60))
        else:
            toks.append((q, k, torch.randn(B, H, dv), torch.rand(B, H) * 0.9, -torch.rand(B, H) * 0.2))
    two = {n: torch.cat([x, x], 1) for n, x in kw_of(*toks[0]).items() if torch.is_tensor(x)}
    O = pw.piece(S0, 0, Piece({**two, 'scale': 1.0}, GDN, 0, None, torch.full((B,), 2), hit=torch.ones(B, dtype=torch.bool),
                              start=True, prompt_end=True))
    ref_outs, _, _ = _alg1(base.quantize(S0, 0), [], [(q, k, v, b, g.exp()) for q, k, v, b, g in toks], base, m, fp8_vec)
    S = O
    for t, (q, k, v, b, g) in enumerate(toks):
        kw = kw_of(q, k, v, b, g)
        S = pw.prepare(S, 0, Piece(kw, GDN, 0, None, torch.full((B,), 3 + t)))
        x = GDN.inputs(kw)
        Sa = x['log_decay'][:, 0].exp()[..., None, None] * S
        u = x['beta'][:, 0][..., None] * (x['v'][:, 0] - torch.einsum('bhkv,bhk->bhv', Sa, x['k'][:, 0]))
        Sn = Sa + x['k'][:, 0][..., None] * u[:, :, None]
        o = torch.einsum('bhkv,bhk->bhv', Sn, x['q'][:, 0])
        d = pw.output_delta(0)
        if d is not None:
            o = o + d[:, 0]
        S = pw.piece(Sn, 0, Piece(kw, GDN, 0, S, torch.full((B,), 3 + t), hit=torch.ones(B, dtype=torch.bool)))
        assert torch.isfinite(o).all() and torch.isfinite(S).all()
        err = ((o - ref_outs[t]).norm() / ref_outs[t].norm()).item()
        assert err < 1e-4, (t, err)


@pytest.fixture(scope='module')
def tiny():
    from _tiny import build
    if DEV == 'cpu':
        pytest.skip('needs a GPU (fla)')
    return build('qwen3_next').to(DEV)


def test_pw_identity_is_exact(tiny):
    """App. A: identity old-state quantizer and an exact (fp32) queue reproduce the unquantized decode."""
    from _tiny import V, run
    lays = sq.find_layers(tiny)
    H = 4
    base = _Base(torch.tensor([4] * H))
    base.plan = {i: (torch.tensor([4] * H), torch.ones(H, 32)) for i in range(len(lays))}
    pw = sq.methods_ext.PureWrite(base, m=3, queue='fp32')
    x = torch.randint(0, V, (2, 20), generator=torch.Generator().manual_seed(1)).to(DEV)
    f = torch.randint(0, V, (2, 12), generator=torch.Generator().manual_seed(2)).to(DEV)
    ref = run(tiny, x, 12, forced=f)
    with sq.apply(tiny, pw):
        out = run(tiny, x, 12, forced=f)
    assert ((out - ref).norm() / ref.norm()).item() < 2e-3


def test_gga_calibrate_and_apply(tiny):
    from _tiny import V, run
    q = sq.gga(4, pivots=1)
    q.sample_every, q.max_snapshots = 4, 8
    ids = torch.randint(0, V, (2, 32), generator=torch.Generator().manual_seed(0))
    sq.calibrate(tiny, q, ids)
    assert set(q.plan) == {0} and int((q.plan[0][0] == 16).sum()) == 1
    b = q.plan[0][0]
    assert set(b.tolist()) <= {2, 4, 6, 8, 16} and int(b[b != 16].sum()) <= 4 * len(b) - 8
    x = torch.randint(0, V, (1, 20), generator=torch.Generator().manual_seed(1)).to(DEV)
    for m in (q, sq.pure_write(q)):
        with sq.apply(tiny, m):
            out = run(tiny, x, 10)
        assert torch.isfinite(out).all()
    r = sq.report(tiny, sq.pure_write(q))
    assert r['payload_bits_per_value'] == pytest.approx(q.bits_per_value(32, 32) + 8 * 8 * (32 + 32 + 8) / 32 ** 2)
