"""statequant formats and methods against the PACE repo's reference implementations (bit-exact where the arithmetic is
the same). Needs the PACE source tree (set PACE_SRC to its src/ directory); run with its venv:
    PACE_SRC=<pace>/src PYTHONPATH=src <pace venv>/bin/python -m pytest tests/test_vs_pace.py
"""
import os
import sys

import pytest
import torch

PACE = os.environ.get('PACE_SRC', '')
if not os.path.isdir(PACE):
    pytest.skip('PACE source tree not available', allow_module_level=True)
sys.path.insert(0, PACE)
import damp  # noqa: E402
import state_quant  # noqa: E402

import statequant as sq  # noqa: E402


def state(seed=0, B=2, H=4, K=128, V=128):
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(B, H, K, V, generator=g)
    # GDN-like structure: column outliers (persistent update channels) and a per-head scale
    S = S * (1 + 6 * (torch.rand(1, 1, 1, V, generator=g) > 0.95)) * torch.rand(1, H, 1, 1, generator=g)
    return S


@pytest.mark.parametrize('bits,axis,group,ref_group', [(4, 'k', 64, 'kg64'), (4, 'k', 32, 'kg32'), (4, 'v', 32, 'g32'),
                                                       (8, 'v', 32, 'g32'), (3, 'k', 32, 'kg32')])
def test_symmetric(bits, axis, group, ref_group):
    S = state()
    ref = state_quant.fake_quant_state(S, bits, 'rtn', ste=False, group=ref_group)
    assert torch.equal(sq.Quantizer(bits, axis, group).fake(S), ref)


def test_symmetric_head():
    S = state()
    ref = state_quant.fake_quant_state(S, 4, 'rtn', ste=False, group='head')
    assert torch.equal(sq.Quantizer(4, 'head').fake(S), ref)


@pytest.mark.parametrize('bits', [4, 8])
@pytest.mark.parametrize('axis', ['k', 'v'])
@pytest.mark.parametrize('zp', ['int', 'fp'])
@pytest.mark.parametrize('had', [False, True])
def test_affine(bits, axis, zp, had):
    S = state(1)
    ref = damp._affine_block_quant(S, bits, 32, 16, 'full' if had else 'none', zp, 'rtn', axis)
    ours = sq.Quantizer(bits, axis, 32, affine=True, zero_point=zp, rotation='hadamard' if had else None).fake(S)
    tol = 0
    assert (ours - ref).abs().max().item() <= tol * S.abs().max().item()


def test_affine_scale8():
    S = state(2)
    ref = damp._affine_block_quant(S, 4, 32, 8, 'none', 'int', 'rtn', 'v')
    assert torch.equal(sq.Quantizer(4, 'v', 32, affine=True, scale_bits=8).fake(S), ref)


def test_dsq():
    S = state(3)
    assert torch.equal(sq.DSQ(4).quantize(S, 0), damp.dsq_quant_state(S, 4))


def test_damp_protect():
    S = state(4)
    g = torch.Generator().manual_seed(0)
    mask = torch.rand(S.shape[1], S.shape[2], generator=g) > 0.875
    for hi_bits in (0, 8):
        ref = damp.damp_quant_state(S, mask, 4, 32, 16, 'full', 'int', 'rtn', hi_bits)
        hi = sq.FloatCast('fp16') if not hi_bits else sq.Quantizer(8, 'v', 32, affine=True, rotation='hadamard')
        m = sq.ChannelProtect(sq.Quantizer(4, 'v', 32, affine=True, rotation='hadamard'), hi, k=16)
        m.masks = {0: mask}
        d = (m.quantize(S, 0) - ref).abs().max().item()
        assert d <= 1e-5 * S.abs().max().item(), d


def _feed(cal_ref, ours, n=6):
    g = torch.Generator().manual_seed(5)
    for layer in range(2):
        for i in range(n):
            S = state(10 * layer + i, B=1)
            lg = -torch.rand(1, 256, S.shape[1], generator=g) * (0.002 + 0.2 * torch.arange(S.shape[1]) / S.shape[1])
            cal_ref.observe_state(layer, S)
            cal_ref.observe_decay(layer, lg.exp())
            ours.observe(layer, S, lg)
    ours.finalize()


def test_damp_scores_and_masks():
    cal = damp.DampCalibrator(8)                     # default: zp fp, value axis, no rotation
    q = sq.damp_int8()
    _feed(cal, q)
    for layer in range(2):
        ref = damp.protect_mask(cal.scores(layer, 128), 16, 'per_head')
        assert torch.equal(q.masks[layer], ref)


@pytest.mark.parametrize('score,sel', [('retained', 'product'), ('energy', 'energy'), ('slow', 'slow')])
def test_headmix_scores(score, sel):
    cal = damp.DampCalibrator(4)
    q = sq.pace48(paper=True)
    q.score = score
    _feed(cal, q)
    for layer in range(2):
        ref = damp.head_scores(cal, layer, sel, 256)
        torch.testing.assert_close(q.scores[layer], ref.float(), rtol=1e-5, atol=0)


def test_bits_accounting():
    assert sq.pace4().bits_per_value(128, 128) == 4.25
    assert abs(sq.pace4().schedule.buffer_bits_per_value(128, 128, 2) - 0.78125) < 1e-9   # paper: 5.03 resident
    assert sq.Quantizer(4, 'k', 32, affine=True).bits_per_value(128, 128) == 4.625          # paper: 4.625 / 8.75
    assert sq.Quantizer(8, 'k', 32, affine=True).bits_per_value(128, 128) == 8.75
    m = sq.pace48()
    assert m.bits_per_value(128, 128) == 0.25 * 8.75 + 0.75 * 4.625                       # 5.656, = DAMP INT4
    assert abs(sq.damp_int4().bits_per_value(128, 128) - 5.65625) < 1e-9
