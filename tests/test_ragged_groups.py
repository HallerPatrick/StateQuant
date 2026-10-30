"""Quantizer groups that do not divide their axis (OLMo-Hybrid: PACE-4's 64-row key groups on d_k 96): the last
group is shorter, which is the same as quantizing each group on its own; bits per value count ceil(n / group)
scales. CPU only."""
import pytest
import torch

import statequant as sq


@pytest.mark.parametrize('q', [sq.Quantizer(4, 'k', 64), sq.Quantizer(4, 'k', 64, affine=True),
                               sq.Quantizer(8, 'v', 128, affine=True, zero_point='fp'),
                               sq.Quantizer(4, 'k', 64, scale_dtype='fp16'),
                               sq.Quantizer(4, 'k', 64, affine=True, rotation='hadamard')],
                         ids=['sym-k64', 'affine-k64', 'affine-fp-v128', 'sym-k64-fp16', 'affine-k64-hadamard'])
def test_ragged_group_is_per_group_quantization(q):
    torch.manual_seed(0)
    S = torch.randn(2, 3, 96, 192) * torch.rand(2, 3, 96, 1)
    ax = -2 if q.axis == 'k' else -1
    x = q._rot(S.float()) if q.rotation else S
    parts = []
    for s0 in range(0, S.shape[ax], q.group):
        e = min(S.shape[ax], s0 + q.group)
        sl = [slice(None)] * 4
        sl[ax] = slice(s0, e)
        parts.append(sq.Quantizer(q.bits, q.axis, e - s0, q.affine, q.zero_point,
                                  scale_dtype=q.scale_dtype).fake(x[tuple(sl)]))
    want = torch.cat(parts, ax)
    if q.rotation:
        want = q._rot(want, inverse=True)
    assert torch.equal(q.fake(S), want)


def test_bits_per_value_ragged():
    assert sq.Quantizer(4, 'k', 64).bits_per_value(96, 192) == 4 + 16 * 2 / 96
    assert sq.Quantizer(4, 'k', 64).bits_per_value(128, 128) == 4.25
    assert sq.Quantizer(4, 'v', 32).bits_per_value(96, 192) == 4.5
