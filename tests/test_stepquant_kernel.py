"""The fused Triton STEPQuant fit against the torch ``step_quant.fit`` (itself bit-identical to the authors' code,
tests/test_vs_stepquant.py): every width incl. 2-bit groups and fp16 pivots, power-of-two and padded head dims. Only
the reduction order differs, so a few codes / column factors may move by one step; the chunk-prefill preset rounds
at prompt ends and every 2048 tokens."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('needs a GPU', allow_module_level=True)
pytest.importorskip('triton')

import statequant as sq  # noqa: E402
from statequant.kernels._stepquant_triton import fit as fit_triton  # noqa: E402
from statequant.step_quant import fit  # noqa: E402


@pytest.mark.parametrize('shape', [(16, 32, 128, 128), (3, 8, 128, 64), (2, 4, 96, 192), (2, 4, 64, 32)])
def test_matches_torch_fit(shape):
    B, H, K, V = shape
    g = torch.Generator().manual_seed(0)
    bits = torch.tensor([2, 4, 6, 8, 16] * 8)[torch.randperm(40, generator=g)][:H]
    w = torch.rand(H, K, generator=g) + 0.5
    S = (torch.randn(B, H, K, V, generator=g) * torch.logspace(-3, 0, V)).cuda()
    ref = fit(S, bits, w)
    out = fit_triton(S, bits.int().cuda(), w.cuda())
    assert (out != ref).float().mean().item() < 1e-3
    assert ((out - ref).norm() / ref.norm()).item() < 1e-3
    piv = bits == 16
    assert torch.equal(out[:, piv.cuda()], S[:, piv.cuda()].half().float())


def test_quantize_backends_agree():
    q = sq.StepQuant(4)
    H, K, V = 8, 128, 128
    q.plan = {0: (torch.tensor([2, 4, 6, 8, 16, 4, 4, 8]), torch.rand(H, K) + 0.5)}
    S = torch.randn(4, H, K, V, device='cuda') * 0.1
    a = q.quantize(S, 0)
    q.backend = 'torch'
    b = q.quantize(S, 0)
    assert ((a - b).norm() / b.norm()).item() < 1e-3


def test_chunk_preset_schedule():
    q = sq.stepquant(4, prefill='chunk')
    s = q.schedule
    assert (s.prefill, s.decode, s.buffer_bits, s.prompt_end) == (2048, 1, 0, True)
    assert sq.PRESETS['stepquant4']().schedule == sq.per_token()
