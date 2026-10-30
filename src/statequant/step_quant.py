"""STEPQuant (Yao et al., arXiv 2609.38169), the gated-delta (GDN) path, generalised to every family with a delta or
additive state update (``statequant.families``: GDN, KDA, Mamba2, lightning, RWKV-6; not RWKV-7). Reimplemented from the paper and checked
against the authors' reference code (github.com/Dreamer-Toby/STEPQuant, tests/test_vs_stepquant.py).

Spatial: key-row-aware dual-axis scales. X_hat_ij = r_i c_j z_ij with r_i = sqrt(mean_j |X_ij| / w_i) per key row and
c_j per value column, first the tightest absmax bound and then one least-squares refit with the codes held fixed,
weighted by w_i^2 (Eq. 12-14, App. C.1). w is the normalised row impact w_i = omega_i^(gamma/2) / GM(omega^(gamma/2)),
omega_i = E[(a_t (q_t - beta_t k_t (k_t^T q_t)))_i^2], the squared read of key row i through one transition.
Two-bit heads use signed levels {-3,-1,1,3} and one row factor per 32 value coordinates.

Temporal: per-head widths from {2,4,6,8} (budget 4) or {4,6,8} (budget 6) by an exact multiple-choice DP over all heads
of all layers, minimising sum_u L_u d_u(b_u), with lifetime L_u = sum_{j<horizon} a_u^(2j) and d_u the w^2-weighted
squared error of the dual-axis fit on calibration state snapshots; the riskiest heads (after a first DP) are kept in
fp16 as pivots, each charged 8 bits against the budget, and the rest re-allocated (App. C.2).

STEPQuant rounds after every state update (``schedule = per_token()``); calibration traces the model token by token at
full precision and snapshots the state every 8 updates (reservoir of 64 per layer, seed 0).

Other families (not in the paper): the row impact uses the family's read vector q (Mamba2: C; lightning: q; RWKV-6:
r) and beta = 0 where there is no delta rule; with a per-key-channel decay (KDA, RWKV-6) the lifetime is per key row,
L_hi = sum_j a_hi^(2j), and a head's cost is the mean over its key rows of L_hi times the row's weighted error.
"""
from __future__ import annotations

import numpy as np
import torch

from .methods import StateQuantizer


def _half(x):
    return x.clamp(2 ** -24, 65504).half().float()


def row_impact(q, k, log_decay, beta):
    """[B, H, d_k] squared transported read for one token; q, k already normalised (q scaled by d_k^-1/2)."""
    a = log_decay.float().exp()
    if a.dim() == q.dim() - 1:
        a = a.unsqueeze(-1)
    q, k = q.float(), k.float()
    return (a * (q - beta.float().unsqueeze(-1) * k * (k * q).sum(-1, keepdim=True))).square()


def impact_factors(omega, gamma=0.25, floor=1e-20):
    lw = omega.double().clamp_min(floor).log() * (gamma / 2)
    return (lw - lw.mean(-1, keepdim=True)).exp().float()


def lifetime(mean_log_decay, horizon=2048):
    """sum_{j < horizon} exp(2 j l), stable at l = 0."""
    x = 2 * mean_log_decay.double()
    assert (x <= 0).all(), 'mean log retention must be <= 0'
    safe = torch.where(x == 0, torch.full_like(x, -1.), x)
    out = torch.expm1(horizon * safe) / torch.expm1(safe)
    return torch.where(x == 0, torch.full_like(out, float(horizon)), out)


def fit(x, bits, w, group=32):
    """Dual-axis fit. x [B, H, K, V]; bits [H] in {2,4,6,8,16} (16 = fp16 pivot); w [H, K] impact factors.
    Returns the reconstruction."""
    x = x.float()
    b = bits.to(x.device)
    wb = w.to(x.device)[None, :, :, None]
    integer = (b != 16)[None, :, None, None]
    qmax = (2.0 ** (b.clamp_max(8).float() - 1) - 1)[None, :, None, None].expand(1, -1, x.shape[2], 1).clone()
    r = _half((x.abs().mean(-1, keepdim=True).clamp_min(1e-20) / wb).sqrt())
    r_full = r.expand_as(x).clone()
    two = b == 2
    if two.any():
        assert x.shape[-1] % group == 0
        xg = x[:, two].reshape(*x[:, two].shape[:-1], -1, group)
        r2 = _half((xg.abs().mean(-1).clamp_min(1e-20) / wb[:, two]).sqrt())
        r_full[:, two] = r2.repeat_interleave(group, -1)
        qmax[:, two] = 3
    y0 = x / r_full
    c = _half(torch.where(integer, y0.abs() / qmax, 0).amax(-2, keepdim=True))

    def encode(cols):
        y = y0 / cols
        z = torch.round(y).maximum(-qmax).minimum(qmax)
        if two.any():
            y2 = y[:, two]
            z[:, two] = torch.where(y2 >= 0, 1., -1.) * torch.where(y2.abs() >= 2, 3., 1.)
        return z
    z = encode(c)
    vv = r_full * z
    ww = wb.square() * integer
    num = (ww * vv * x).sum(-2, keepdim=True)
    den = (ww * vv.square()).sum(-2, keepdim=True)
    c = _half(torch.where(den > 0, num / den.clamp_min(1e-30), c))
    z = encode(c)
    return torch.where(integer, r_full * c * z, x.half().float())


def allocate_dp(cost, candidates, budget):
    """Exact multiple-choice knapsack: one candidate per unit, sum of widths <= budget, minimum total cost."""
    cost = np.asarray(cost, dtype=np.float64)
    cand = np.asarray(candidates, dtype=int)
    n = cost.shape[0]
    assert budget >= n * cand.min(), 'integer budget is infeasible'
    budget = min(int(budget), n * int(cand.max()))
    dp = np.full(budget + 1, np.inf)
    dp[0] = 0.
    back = np.full((n, budget + 1), -1, dtype=np.int8)
    for i in range(n):
        nxt = np.full_like(dp, np.inf)
        for j, b in enumerate(cand):
            trial = dp[:budget + 1 - b] + cost[i, j]
            better = trial < nxt[b:]
            nxt[b:][better] = trial[better]
            back[i, b:][better] = j
        dp = nxt
    rem = int(dp.argmin())
    out = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        j = back[i, rem]
        out[i] = cand[j]
        rem -= cand[j]
    return torch.from_numpy(out)


class _Trace:
    """Per-layer calibration statistics, collected token by token."""

    def __init__(self, sample_every, max_snapshots):
        self.omega = self.log_decay = None
        self.count = self.updates = self.sample_count = 0
        self.snapshots = []
        self.every, self.max = sample_every, max_snapshots
        self.rng = torch.Generator().manual_seed(0)

    def token(self, q, k, lg, beta):
        om = row_impact(q, k, lg, beta).double().sum(0).cpu()
        ld = lg.double().sum(0).cpu()
        self.omega = om if self.omega is None else self.omega + om
        self.log_decay = ld if self.log_decay is None else self.log_decay + ld
        self.count += q.shape[0]
        self.updates += 1

    def snapshot(self, S):
        for s in S:
            self.sample_count += 1
            if len(self.snapshots) < self.max:
                self.snapshots.append(s.float().cpu().clone())
            else:
                i = torch.randint(self.sample_count, (), generator=self.rng).item()
                if i < self.max:
                    self.snapshots[i] = s.float().cpu().clone()


def _normalised_inputs(kw):
    """q, k (in the kernel's space), log decay, beta per token for one piece; beta = 0 for families without a delta
    rule (the transition is then diag(a_t), and the row impact reduces to (a_t q_t)^2). ``kw`` is a
    ``families.Piece`` (``statequant.calibrate``) or a plain dict of gated-delta op arguments."""
    if not hasattr(kw, 'inputs'):
        from .families import delta_inputs, gdn_log_decay
        return delta_inputs(kw, gdn_log_decay)
    if kw.family.name == 'gdn':                       # the original GDN arithmetic, bit for bit
        from .families import delta_inputs, gdn_log_decay
        return delta_inputs(kw, gdn_log_decay)
    x = kw.inputs()
    if x['rule'] not in ('delta', 'additive'):
        raise NotImplementedError(f"StepQuant's row impact needs a delta or additive state update; the "
                                  f"{kw.family.name} family's transition is {x['rule']!r}")
    beta = x['beta'] if x['beta'] is not None else torch.zeros_like(x['q'][..., 0])
    return x['q'], x['k'], x['log_decay'], beta


class StepQuant(StateQuantizer):
    """STEPQuant at a nominal budget of ``bits`` (4 or 6) per state value (scales and pivots not included, as in the
    paper's 'nominal' budget; ``bits_per_value`` reports the stored total).

    pivots      number of fp16 heads over the whole model; None scales the paper's 32 of 2,304 to this model
    horizon     lifetime horizon in tokens (2,048)
    """
    needs_calibration = True
    calibration_period = 8                       # snapshot every 8 updates (the boundary period of the trace)

    def __init__(self, bits: int = 4, pivots: int | None = None, horizon: int = 2048, gamma: float = 0.25,
                 sample_every: int = 8, max_snapshots: int = 64, group: int = 32):
        assert bits in (4, 6)
        self.bits, self.pivots, self.horizon, self.gamma, self.group = bits, pivots, horizon, gamma, group
        self.candidates = [2, 4, 6, 8] if bits == 4 else [4, 6, 8]
        self.sample_every, self.max_snapshots = sample_every, max_snapshots
        self.traces, self.plan, self.masks = {}, {}, {}
        self.name = f'STEPQuant-{bits}'

    # calibration: called per piece of ``sample_every`` tokens with the full-precision state at its end
    def observe(self, layer, S, log_decay_, kw=None):
        assert kw is not None, 'StepQuant needs the per-token op inputs; calibrate through statequant.calibrate'
        tr = self.traces.setdefault(layer, _Trace(self.sample_every, self.max_snapshots))
        q, k, lg, beta = _normalised_inputs(kw)
        for t in range(q.shape[1]):
            tr.token(q[:, t], k[:, t], lg[:, t], beta[:, t])
        if tr.updates % self.sample_every == 0:
            tr.snapshot(S)

    def finalize(self, device='cpu'):
        names = sorted(self.traces)
        info, costs = {}, []
        for name in names:
            tr = self.traces[name]
            assert tr.snapshots, 'no state snapshots; calibrate on longer sequences'
            w = impact_factors(tr.omega / tr.count, self.gamma).to(device)
            L = lifetime(tr.log_decay / tr.count, self.horizon)                       # [H] or [H, d_k]
            wsq = w[None, :, :, None].square()
            d = []
            for b in self.candidates:
                bits = torch.full((w.shape[0],), b, dtype=torch.long, device=device)
                tot = 0
                for s in tr.snapshots:
                    x = s[None].to(device)
                    e = (fit(x, bits, w, self.group) - x).square() * wsq
                    if L.dim() == 1:
                        tot = tot + e.mean((0, 2, 3)).cpu().double()
                    else:                    # per-channel decay (KDA, RWKV-6): each key row's error lives L_i
                        tot = tot + (e.mean((0, 3)).cpu().double() * L).mean(-1)
                d.append(tot / len(tr.snapshots))
            d = torch.stack(d, -1)                                                    # [H, C]
            info[name] = (w.cpu(), d.shape[0])
            costs.append(d * L[:, None] if L.dim() == 1 else d)
        cost = torch.cat(costs)
        n = len(cost)
        piv = round(n * 32 / 2304) if self.pivots is None else self.pivots
        budget = self.bits * n - 8 * piv
        first = allocate_dp(cost, self.candidates, self.bits * n)
        risk = cost.gather(1, torch.searchsorted(torch.tensor(self.candidates), first)[:, None]).flatten()
        pm = torch.zeros(n, dtype=torch.bool)
        pm[torch.argsort(risk, descending=True, stable=True)[:piv]] = True
        assign = torch.full((n,), 16, dtype=torch.long)
        assign[~pm] = allocate_dp(cost[~pm], self.candidates, budget)
        off = 0
        for name in names:
            w, h = info[name]
            self.plan[name] = (assign[off:off + h].clone(), w)
            self.masks[name] = assign[off:off + h] == 16
            off += h
        self.n_pivots = piv
        self.traces = {}

    def quantize(self, S, layer):
        bits, w = self.plan[layer]
        return fit(S, bits, w, self.group).to(S.dtype)

    def bits_per_value(self, d_k, d_v):
        """Stored bits per value: assigned widths (fp16 for pivots) + one fp16 row and column vector per head (+ the
        per-group row factors of 2-bit heads)."""
        if not self.plan:
            return self.bits + 16 * (d_k + d_v) / (d_k * d_v)
        b = torch.cat([p[0] for p in self.plan.values()]).float()
        side = torch.where(b == 16, 0., 16 * (d_k + d_v) / (d_k * d_v))
        side = side + torch.where(b == 2, 16 * d_k * (d_v // self.group) / (d_k * d_v), 0.)   # 2-bit group rows
        return (b + side).mean().item()
