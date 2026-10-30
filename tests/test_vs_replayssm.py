"""ReplaySSM (``statequant.ReplaySSM``) against the paper's own emulation kernels (github.com/Johnny-Liou/ReplaySSM,
``emulation/``, Apache-2.0): the GDN decode outputs of every precision mode, step by step over three flush windows.
The method is driven through its piece hooks exactly as ``statequant.apply``'s carry drives it, with a naive fp32
delta-rule step as the op. The deterministic modes (FP32 checkpoint, RTN read copy) match to float rounding (the flush token's own output to
BF16 rounding); the SR
modes draw different uniforms than the kernels' Philox, so they are compared by error size against the FP32 run.
Set REPLAYSSM_SRC to the directory that holds ``emulation/``; needs a GPU and triton."""
import os
import sys

import pytest
import torch

SRC = os.environ.get('REPLAYSSM_SRC')
if not SRC or not os.path.isdir(os.path.join(SRC, 'emulation')):
    pytest.skip('set REPLAYSSM_SRC to a ReplaySSM checkout (the directory holding emulation/)', allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip('needs a GPU', allow_module_level=True)
pytest.importorskip('triton')
sys.path.insert(0, SRC)
from emulation import GDNReplay  # noqa: E402

import statequant as sq  # noqa: E402
from statequant.families import BY_NAME, Piece  # noqa: E402
from statequant.methods_ext.replay import MODES, cast_state  # noqa: E402

GDN = BY_NAME['gdn']
B, H, HV, K, V, STEPS = 2, 2, 4, 128, 128, 48


def case(seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    state = torch.randn(B, HV, V, K, device='cuda', generator=g) * 0.1          # kernels' layout [B, HV, V, K]
    steps = [(torch.randn(B, 2 * H * K + HV * V, device='cuda', generator=g).bfloat16(),
              torch.randn(B, HV, device='cuda', generator=g), torch.randn(B, HV, device='cuda', generator=g))
             for _ in range(STEPS)]
    A_log = torch.randn(HV, device='cuda', generator=g) * 0.5 - 1
    dt_bias = torch.randn(HV, device='cuda', generator=g) * 0.5
    return state, steps, A_log, dt_bias


def kernel_outputs(mode, state, steps, A_log, dt_bias):
    r = GDNReplay(state.clone(), H, mode)
    return [r.step(qkv, a, b, A_log, dt_bias).clone() for qkv, a, b in steps]


def piece_kw(qkv, a, b, A_log, dt_bias):
    q = qkv[:, :H * K].view(B, 1, H, K)
    k = qkv[:, H * K:2 * H * K].view(B, 1, H, K)
    v = qkv[:, 2 * H * K:].view(B, 1, HV, V).float()
    g = (-A_log.exp() * torch.nn.functional.softplus(a + dt_bias))[:, None]
    beta = b.sigmoid().bfloat16().float()[:, None]                                # the kernel's BF16 beta
    return {'q': q, 'k': k, 'v': v, 'g': g, 'beta': beta, 'use_qk_l2norm_in_kernel': True, 'scale': K ** -0.5}


def delta_step(S, kw):
    """Naive fp32 delta rule on the canonical state [B, HV, K, V]: (o [B, HV, V], S_new)."""
    x = GDN.inputs(kw)
    q, k, v, beta, lg = x['q'][:, 0], x['k'][:, 0], x['v'][:, 0], x['beta'][:, 0], x['log_decay'][:, 0]
    Sa = lg.exp()[..., None, None] * S
    u = beta[..., None] * (v - torch.einsum('bhkv,bhk->bhv', Sa, k))
    Sn = Sa + k[..., None] * u[..., None, :]
    return torch.einsum('bhkv,bhk->bhv', Sn, q), Sn


def method_outputs(method, state, steps, A_log, dt_bias):
    """Drive the method's hooks as statequant.apply's carry does: a prompt end, then single-token pieces."""
    method.reset()
    S = state.transpose(-1, -2).contiguous()                                       # canonical [B, HV, K, V]
    kw0 = piece_kw(*steps[0], A_log, dt_bias)
    pre = {n: (torch.cat([x, x], 1) if torch.is_tensor(x) else x) for n, x in kw0.items()}   # a 2-token prompt piece
    S = method.piece(S, 0, Piece(pre, GDN, 0, None, torch.full((B,), 2), hit=torch.ones(B, dtype=torch.bool),
                                 start=True, prompt_end=True))
    outs = []
    for t, (qkv, a, b) in enumerate(steps):
        kw = piece_kw(qkv, a, b, A_log, dt_bias)
        pos = torch.full((B,), 3 + t)
        S = method.prepare(S, 0, Piece(kw, GDN, 0, None, pos))
        o, Sn = delta_step(S, kw)
        S = method.piece(Sn, 0, Piece(kw, GDN, 0, S, pos, hit=torch.ones(B, dtype=torch.bool)))
        outs.append(o)
    return outs


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize('mode', ['s32_rs', 'tp8_hi32', 'tp4_hi32'])
def test_deterministic_modes_match_kernel(mode):
    c = case()
    ref = kernel_outputs(mode, *c)
    out = method_outputs(sq.ReplaySSM(mode), *c)
    errs = [rel(o, r) for o, r in zip(out, ref)]
    # flush tokens: the kernel rounds the token's own residual to BF16, the op here does not; the state after the
    # flush (and so every later output) matches
    flush = [e for t, e in enumerate(errs) if t % 16 == 15]
    other = [e for t, e in enumerate(errs) if t % 16 != 15]
    assert max(other) < 2e-4 and max(flush) < 5e-3, [f'{e:.1e}' for e in errs]


@pytest.mark.parametrize('mode', ['f16_sr_rs', 'f8_sr_rs', 'f4_sr_rs', 'tp8_hi16_sr', 'tp4_hi16_sr', 'tp4_hi8_sr'])
def test_sr_modes_match_kernel_in_error_size(mode):
    """SR draws differ from the kernels': the deviation from the FP32 run has the same size (per window), and both
    stay close to each other relative to that deviation's scale."""
    c = case()
    fp32 = kernel_outputs('s32_rs', *c)
    ref = kernel_outputs(mode, *c)
    out = method_outputs(sq.ReplaySSM(mode), *c)
    e_ref = torch.tensor([rel(r, f) for r, f in zip(ref, fp32)])
    e_out = torch.tensor([rel(o, f) for o, f in zip(out, fp32)])
    for w in range(STEPS // 16):
        a, b = e_out[16 * w:16 * w + 16].mean(), e_ref[16 * w:16 * w + 16].mean()
        assert 0.5 < (a / b).item() < 2.0, (w, a.item(), b.item())


def test_formats():
    g = torch.Generator(device='cuda').manual_seed(0)
    S = torch.randn(2, 3, 128, 64, device='cuda', generator=g) * torch.logspace(-3, 1, 64, device='cuda')
    # FP8 RTN: E4M3 of x / (amax / 448) per 128 key entries of a value column (torch's cast is RTN, ties to even)
    sc = S.abs().amax(-2, keepdim=True) / 448
    assert torch.equal(cast_state(S, 'fp8'), (S / sc).to(torch.float8_e4m3fn).float() * sc)
    assert torch.equal(cast_state(S, 'fp16'), S.half().float())
    Q = cast_state(S, 'nvfp4')
    X = Q.transpose(-1, -2).reshape(2, 3, 64, 8, 16)
    Xs = S.transpose(-1, -2).reshape(2, 3, 64, 8, 16)
    s_g = S.abs().flatten(1).amax(-1).view(-1, 1, 1, 1, 1) / (6 * 448)
    s_b = (Xs.abs().amax(-1, keepdim=True) / (6 * s_g)).to(torch.float8_e4m3fn).float()
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device='cuda')
    codes = (X / (s_b * s_g)).abs()
    assert torch.allclose(codes, grid[(codes[..., None] - grid).abs().argmin(-1)], atol=1e-5)
    # SR is unbiased
    x = torch.full((1, 1, 128, 1 << 14), 0.3, device='cuda')
    u = torch.rand(x.shape, device='cuda', generator=g)
    assert abs(cast_state(x, 'nvfp4', u).mean().item() - 0.3) < 2e-3
    assert abs(cast_state(x, 'fp16', u).mean().item() - 0.3) < 1e-6


def test_bits():
    m = sq.replayssm('tp4_hi16_sr')
    assert m.bits_per_value(128, 128) == pytest.approx(16 + 4.5 + 32 / 128 ** 2, abs=1e-6)
    assert m.buffer_bits_per_value(128, 128, heads_per_key=2) == pytest.approx(5.0625)
    assert sq.replayssm('f4_sr_rs').bits_per_value(128, 128) == pytest.approx(4.5 + 32 / 128 ** 2, abs=1e-6)
    assert set(MODES) == {k[len('replay_'):] for k in sq.PRESETS if k.startswith('replay_')}
