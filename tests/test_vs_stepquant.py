"""statequant.StepQuant against the STEPQuant authors' reference code (github.com/Dreamer-Toby/STEPQuant), used as an
oracle only. Set STEPQUANT_SRC to the clone's root.
"""
import os
import sys

import pytest
import torch

ROOT = os.environ.get('STEPQUANT_SRC', '')
if not os.path.isdir(os.path.join(ROOT, 'stepquant')):
    pytest.skip('STEPQuant reference not available', allow_module_level=True)
sys.path.insert(0, ROOT)
from stepquant import allocation as ref_alloc  # noqa: E402
from stepquant import calibration as ref_cal  # noqa: E402
from stepquant import core as ref_core  # noqa: E402
from stepquant import quantization as ref_q  # noqa: E402
sys.path.remove(ROOT)
for _m in [m for m in sys.modules if m == 'stepquant' or m.startswith('stepquant.')]:
    sys.modules['_ref_' + _m] = sys.modules.pop(_m)                 # keep our own module names free

from statequant import step_quant as ours  # noqa: E402


def state(seed=0, B=2, H=6, K=128, V=128):
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(B, H, K, V, generator=g) * (1 + 5 * (torch.rand(1, 1, 1, V, generator=g) > 0.95))
    return S * torch.rand(1, H, 1, 1, generator=g) * (1 + torch.rand(1, 1, K, 1, generator=g))


def impact(seed=1, H=6, K=128):
    return ref_core.impact_factors(torch.rand(H, K, generator=torch.Generator().manual_seed(seed)) + 0.01)


def test_impact_and_lifetime():
    om = torch.rand(6, 128) + 1e-3
    assert torch.equal(ours.impact_factors(om), ref_core.impact_factors(om))
    ld = -torch.rand(6) * 0.1
    ld[0] = 0.
    assert torch.equal(ours.lifetime(ld, 2048), ref_core.lifetime_weight(ld, 2048))


def test_row_impact():
    g = torch.Generator().manual_seed(2)
    q, k = torch.randn(3, 6, 128, generator=g), torch.randn(3, 6, 128, generator=g)
    lg, beta = -torch.rand(3, 6, generator=g), torch.rand(3, 6, generator=g)
    assert torch.equal(ours.row_impact(q, k, lg, beta), ref_core.row_impact(q, k, lg, beta))


@pytest.mark.parametrize('widths', [(4, 4, 4, 4, 4, 4), (2, 4, 6, 8, 16, 2), (8, 6, 16, 4, 2, 2)])
def test_fit(widths):
    S, w = state(), impact()
    bits = torch.tensor(widths)
    plan = ref_q.QuantizationPlan('gdn', bits[:, None].expand(-1, 128).clone(), w)
    ref = ref_q.fit_state(S, plan)[-1]
    assert torch.equal(ours.fit(S, bits, w), ref)


def test_dp():
    g = torch.Generator().manual_seed(3)
    cost = torch.rand(40, 4, generator=g).sort(-1, descending=True).values
    for budget in (4 * 40, 4 * 40 - 8 * 2, 3 * 40):
        assert torch.equal(ours.allocate_dp(cost, [2, 4, 6, 8], budget),
                           ref_alloc.allocate_dp(cost, [2, 4, 6, 8], budget))


def test_calibrate_plan():
    """Same statistics -> same per-head widths and pivots as the reference's calibrate()."""
    g = torch.Generator().manual_seed(4)
    stats, q = {}, ours.StepQuant(4, pivots=3)
    for layer in range(3):
        om, ld = torch.rand(6, 128, generator=g) + 1e-3, -torch.rand(6, generator=g) * 0.05
        snaps = [state(10 * layer + i, B=1)[0] for i in range(4)]
        stats[f'layers.{layer}'] = dict(architecture='gdn', omega=om, log_decay=ld, snapshots=torch.stack(snaps),
                                        tokens=1, sampled_states=4)
        tr = ours._Trace(8, 64)
        tr.omega, tr.log_decay, tr.count, tr.snapshots = om.double(), ld.double(), 1, snaps
        q.traces[f'layers.{layer}'] = tr
    ref = ref_cal.calibrate(stats, nominal_bits=4, pivots=3)
    q.finalize()
    for name, p in ref['plans'].items():
        assert torch.equal(q.plan[name][0], p['bits'][:, 0]), name
        torch.testing.assert_close(q.plan[name][1], p['impact'])
