"""Packed decode path (statequant.kernels): storage format vs Quantizer.fake, the Triton step vs the torch reference
of the same arithmetic, and CUDA-graph capture. Needs a GPU and Triton.

Run with ``-s`` to see the per-config numbers (output error, code mismatches after each fold)."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('needs a GPU', allow_module_level=True)
pytest.importorskip('triton')

import statequant as sq  # noqa: E402
from statequant.kernels import PackedGDNState, ReferenceGDNState, pack, packed_format, unpack  # noqa: E402

DEV = 'cuda'
FORMATS = {
    'int4-v-g32': sq.Quantizer(4, 'v', 32),
    'int4-k-g64': sq.Quantizer(4, 'k', 64),
    'int8-k-g32': sq.Quantizer(8, 'k', 32),
    'int8-v-g64': sq.Quantizer(8, 'v', 64),
    'bf16': 'bf16',
}


# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('bits', [4, 8])
@pytest.mark.parametrize('axis', ['k', 'v'])
@pytest.mark.parametrize('group', [32, 64])
def test_pack_roundtrip_matches_fake(bits, axis, group):
    torch.manual_seed(0)
    S = torch.randn(8, 128, 128, device=DEV) * torch.rand(8, 128, 1, device=DEV) * 3
    S[0, :, :64] = 0                                         # all-zero groups
    S[1] *= 1e-6                                             # scales in the fp16 subnormal range
    q = sq.Quantizer(bits, axis, group)
    codes, scales = pack(S, q)
    assert scales.dtype == torch.float16
    assert codes.dtype == (torch.uint8 if bits == 4 else torch.int8)
    assert codes.shape == (8, 128, 128 // (2 if bits == 4 else 1))
    got = unpack(codes, scales, q, 128)
    want = sq.Quantizer(bits, axis, group, scale_dtype='fp16').fake(S)
    assert torch.equal(got, want)
    assert torch.equal(packed_format(q).fake(S), want)
    # the fp32-scale Quantizer is a different format: fp16 scale storage does change values
    n_diff = (q.fake(S) != want).sum().item()
    print(f'\n[{bits}b {axis} g{group}] entries where fp16-scale storage differs from the fp32-scale fake: '
          f'{n_diff} of {S.numel()}')
    # a 4-d state in statequant's [B, H, d_k, d_v] layout packs the same
    c4, s4 = pack(S.reshape(2, 4, 128, 128), q)
    assert torch.equal(c4, codes) and torch.equal(s4, scales)


def test_bf16_roundtrip():
    S = torch.randn(4, 128, 128, device=DEV)
    c, s = pack(S, 'bf16')
    assert torch.equal(unpack(c, s, 'bf16', 128), S.to(torch.bfloat16).float())


# ---------------------------------------------------------------------------------------------------------------
B, HV, HK, D = 2, 4, 2, 128                                  # BH = 8, GQA 2
BH = B * HV


def inputs(T, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=DEV, generator=g)
    q = r(T, BH, D)
    kk = r(T, B, HK, 1, D).expand(-1, -1, -1, HV // HK, -1).reshape(T, BH, D).contiguous()
    v = r(T, BH, D)
    gl = -torch.rand(T, BH, device=DEV, generator=g) * 0.08
    beta = torch.rand(T, BH, device=DEV, generator=g).sigmoid()
    return q, kk, v, gl, beta


def run_decode(fmt, m, buffer_bits, T=40, seed=1):
    torch.manual_seed(seed)
    S0 = torch.randn(BH, D, D, device=DEV) * 0.5
    ker = PackedGDNState(S0, fmt, m=m, hv=HV, hk=HK, buffer_bits=buffer_bits, out_dtype=torch.float32)
    ref = ReferenceGDNState(S0, fmt, m=m, hv=HV, hk=HK, buffer_bits=buffer_bits, auto_fold=False)
    assert torch.equal(ker.dequantize(), ref.dequantize())
    q, k, v, g, beta = inputs(T, seed)
    scale = D ** -0.5
    worst_o, folds, buf_flips = 0., [], 0
    for t in range(T):
        slot = ker.n
        o = ker.step(q[t], k[t], v[t], g[t], beta[t], scale).clone()
        r = ref.step(q[t], k[t], v[t], g[t], beta[t], scale)
        worst_o = max(worst_o, ((o - r).abs().max() / r.abs().max()).item())
        # the appended (key, delta value): rounding to the buffer width can flip where the fp32 u differs in its
        # last bits; count those and adopt the kernel's entry so the comparison does not inherit them
        kb, ub = ker.KB[:, slot].float(), ker.UB[:, slot].float()
        if buffer_bits == 8:
            kb, ub = kb * ker.KBS[:, slot, None].float(), ub * ker.UBS[:, slot, None].float()
        buf_flips += (kb != ref.K[:, slot]).sum().item() + (ub != ref.U[:, slot]).sum().item()
        assert torch.equal(ker.GB[:, slot], ref.G[:, slot])
        ref.K[:, slot], ref.U[:, slot] = kb, ub
        if ref.n == m:
            ref.fold()
        assert ker.n == ref.n
        if ref.n == 0:                                       # a fold happened this step
            Sk, Sr = ker.dequantize(), ref.dequantize()
            mism = (ker.codes != ref.codes).sum().item() if fmt != 'bf16' else (Sk != Sr).sum().item()
            smism = 0 if fmt == 'bf16' else (ker.scales != ref.scales).sum().item()
            # where the codes agree, the dequantized states agree exactly (scales agree too)
            same = Sk == Sr
            # kernel state vs the reference's UNROUNDED fold, relative to the state's magnitude
            raw_err = ((Sk - ref.S_raw).abs().max() / ref.S_raw.abs().max()).item()
            folds.append((mism, smism, (~same).sum().item(), raw_err))
            # re-sync, so later steps measure the step arithmetic and not an inherited rounding flip
            ref.codes, ref.scales, ref.S0 = ker.codes.clone(), ker.scales.clone(), Sk
    return worst_o, folds, buf_flips


CASES = [(f, m, bb) for f in FORMATS for m in (1, 8, 16) for bb in (8, 16)]


@pytest.mark.parametrize('name,m,bb', CASES)
def test_decode_vs_reference(name, m, bb):
    worst_o, folds, buf_flips = run_decode(FORMATS[name], m, bb)
    mism = sum(f[0] for f in folds)
    smism = sum(f[1] for f in folds)
    n_codes = BH * D * D * len(folds)
    raw = max(f[3] for f in folds)
    vdiff = sum(f[2] for f in folds)
    print(f'\n[{name} m={m} buf{bb}] 40 steps, {len(folds)} folds: max rel |o - o_ref| {worst_o:.2e}; '
          f'state codes differing after folds {mism} of {n_codes} ({100 * mism / max(n_codes, 1):.5f}%), '
          f'scales differing {smism}, dequantized entries differing {vdiff}; max rel |S - S_raw_ref| {raw:.2e}; buffer entries flipped {buf_flips} '
          f'of {40 * (B * HK + BH) * D}')
    assert len(folds) == 40 // m
    assert worst_o < 2e-5, worst_o
    assert buf_flips <= 1e-4 * 40 * (B * HK + BH) * D, buf_flips
    # codes after a fold: exact in the vast majority of entries (flips only at rounding ties of fp32 sums)
    assert mism <= 1e-4 * n_codes, mism
    # the folded state is the reference's unrounded fold up to one rounding of the format
    assert raw < {'bf16': 1e-2}.get(name, 0.2), raw


def test_q_k_per_key_head():
    """q, k may be handed per key head [B * hk, d_k] instead of expanded per value head."""
    torch.manual_seed(3)
    S0 = torch.randn(BH, D, D, device=DEV)
    a = PackedGDNState(S0, FORMATS['int4-k-g64'], m=8, hv=HV, hk=HK, out_dtype=torch.float32)
    b = PackedGDNState(S0, FORMATS['int4-k-g64'], m=8, hv=HV, hk=HK, out_dtype=torch.float32)
    q, k, v, g, beta = inputs(10, 3)
    for t in range(10):
        qh = q[t].reshape(B, HK, HV // HK, D)[:, :, 0].reshape(B * HK, D).contiguous()
        kh = k[t].reshape(B, HK, HV // HK, D)[:, :, 0].reshape(B * HK, D).contiguous()
        q_full = qh.reshape(B, HK, 1, D).expand(-1, -1, HV // HK, -1).reshape(BH, D).contiguous()
        oa = a.step(q_full, k[t], v[t], g[t], beta[t], D ** -0.5).clone()
        ob = b.step(qh, kh, v[t], g[t], beta[t], D ** -0.5).clone()
        assert torch.equal(oa, ob)


# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('name', ['int4-v-g32', 'int4-k-g64', 'bf16'])
def test_cuda_graph_replay_matches_eager(name):
    torch.manual_seed(2)
    S0 = torch.randn(BH, D, D, device=DEV)
    m, T = 8, 20
    eager = PackedGDNState(S0, FORMATS[name], m=m, hv=HV, hk=HK)
    graph = PackedGDNState(S0, FORMATS[name], m=m, hv=HV, hk=HK)
    q, k, v, g, beta = [x.to(torch.bfloat16) if i < 3 else x for i, x in enumerate(inputs(T, 2))]
    sx = [x[0].clone() for x in (q, k, v, g, beta)]
    scale = D ** -0.5

    # warm up (compiles), then restore the state, then capture one step
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph.step(*sx, scale)
    torch.cuda.current_stream().wait_stream(s)
    graph.load(S0)
    torch.cuda.synchronize()
    G = torch.cuda.CUDAGraph()
    with torch.cuda.graph(G):
        out = graph.step(*sx, scale)
    graph.load(S0)                                           # capture does not execute, but be explicit
    torch.cuda.synchronize()

    for t in range(T):
        for dst, src in zip(sx, (q, k, v, g, beta)):
            dst.copy_(src[t])
        G.replay()
        oe = eager.step(q[t], k[t], v[t], g[t], beta[t], scale)
        torch.cuda.synchronize()
        assert torch.equal(out, oe), t
    assert torch.equal(graph.codes, eager.codes) and torch.equal(graph.scales, eager.scales)
    assert graph.n == eager.n == T % m


def test_stored_bytes():
    S0 = torch.zeros(BH, D, D, device=DEV)
    st = PackedGDNState(S0, FORMATS['int4-k-g64'], m=8, hv=HV, hk=HK)
    b = st.stored_bytes()
    assert b['state'] == BH * D * D // 2 + BH * D * (D // 64) * 2
    assert b['buffer'] == B * HK * 8 * D + BH * 8 * D + 2 * (B * HK * 8 + BH * 8)
    assert torch.equal(st.dequantize(), S0)
