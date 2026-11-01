"""GGA and PW-S8: "Low-Bit GDN Recurrent-State Quantization: Global Gradient-aware Allocation and Pure-Write Sliding
Windows" (Huang, Huang, Yin, arXiv 2610.39931v1). No code was released; implemented from the paper.

GGA (Sec. 4.2, App. B.1-B.3). STEPQuant's candidate quantizers (``step_quant.fit``: dual-axis fit with row impact,
2-bit heads with group-32 rows, fp16 scales), menu ({2,4,6,8} at @4 / @5, {4,6,8} at @6), 32 fp16 pivots and the
two-stage global DP (preliminary allocation over all heads at budget b|U|, pivots = the P largest residual risks,
re-allocation of the rest at B_int = b|U| - 8P), with the candidate cost (Eq. 8)

    C_{u,b} = 1 / (2 N_u) sum_{t in T_u} || G^wb_{u,t} (.) (Q_b(S_{u,t}) - S_{u,t}) ||_F^2

instead of STEP's L_u d_u(b). G^wb is the writeback adjoint of the summed next-token cross-entropy on the
unquantized calibration trajectory (Eq. 6): with g_t = dL / do_t the gradient of the head's raw readout
o_t = S_t^T q_t,  J_t = q_t g_t^T + G^wb_t,  G^wb_{t-1} = A_t^T J_t,  G^wb_T = 0,  A_t = gamma_t (I - beta_t k_t k_t^T).
States and adjoints are paired at every 8th token, reservoir-sampled (64 per layer, seed 0) as STEP's snapshots;
the row impact w of the fit is STEP's. Calibration (``statequant.calibrate``, which calls ``calibrate_model``) runs
one forward + backward per sequence (the paper: 32 x 2048 WikiText-2 train tokens) and the two recurrences in fp32.
The paper also divides all costs by their median positive value, which changes neither the plan nor the pivots.

PW-S8 (Sec. 4.3, Alg. 1, App. B.4). Per non-pivot head the old state O is re-rounded with the head's frozen width
at every decoded token and the last m = 8 rank-one writes are kept as FP8 E4M3 vector pairs (p, v), each vector
with an FP32 scale amax / 448, propagated p <- R(A_t p) and re-encoded at every step; the readout uses
A O + sum R(A p) v^T + beta k v^T, then the oldest pair (if m are queued) is merged into O, O is re-rounded and
(R(beta k), R(v)) is enqueued. FP16 pivots keep the full update in fp16 without a queue. Prefill follows the base
path (the queue starts empty at the first decoded token; the post-prefill state is rounded at the prompt end), and
every request decodes in lockstep. The op is handed S' = O + sum_j A^-1 R(A p_j) v_j^T, so that its own update
and readout give exactly Alg. 1's (A is invertible for beta < 1, e.g. sigmoid gates; done in float64).

Storage (App. B.5, Table 3): STEP accounting (``StepQuant.bits_per_value``) plus, with PW, a queue for every head
(pivots included): 8 m (d_k + d_v + 8) / (d_k d_v) = 1.03125 b/value at d_k = d_v = 128, m = 8.
GGA@6 + PW-S8 in the paper uses a plan re-calibrated on a PW reference trajectory (App. B.3); ``gga(6, pw=True)``
uses the GGA@6 plan.
"""
from __future__ import annotations

import torch

from ..schedule import Schedule
from ..step_quant import StepQuant, _normalised_inputs, fit, impact_factors, row_impact
from .hooks import StatefulMethod, terms


# ------------------------------------------------------------------------------------------------------------------
# GGA
# ------------------------------------------------------------------------------------------------------------------
class _Reservoir:
    """STEP's snapshot reservoir (``step_quant._Trace.snapshot``), holding (state, adjoint) pairs."""

    def __init__(self, size, seed=0):
        self.size, self.count, self.items = size, 0, []
        self.rng = torch.Generator().manual_seed(seed)

    def offer(self, make):
        self.count += 1
        if len(self.items) < self.size:
            self.items.append(make())
        else:
            i = torch.randint(self.count, (), generator=self.rng).item()
            if i < self.size:
                self.items[i] = make()


@torch.no_grad()
def states_at(q, k, v, beta, a, every):
    """fp32 delta-rule states S_t (after t tokens) at t = every, 2 every, ...: {t: [H, d_k, d_v]}.
    q, k [T, H, d_k] (normalised), v [T, H, d_v], beta [T, H], a [T, H] (decay)."""
    T, H, dk = k.shape
    S = torch.zeros(H, dk, v.shape[-1], device=k.device)
    out = {}
    for t in range(T):
        Sa = a[t][:, None, None] * S
        u = beta[t][:, None] * (v[t] - torch.einsum('hkv,hk->hv', Sa, k[t]))
        S = Sa + k[t][:, :, None] * u[:, None, :]
        if (t + 1) % every == 0:
            out[t + 1] = S.clone()
    return out


@torch.no_grad()
def writeback_adjoints(q, k, beta, a, g, every):
    """Eq. 6: G^wb_T = 0, J_t = q_t g_t^T + G^wb_t, G^wb_{t-1} = A_t^T J_t with A_t^T X = a_t (X - beta_t k_t (k_t^T X)).
    g [T, H, d_v] = dL / do_t. Returns {t: G^wb_t} at t = every, 2 every, ... (t = tokens processed)."""
    T, H, dk = k.shape
    G = torch.zeros(H, dk, g.shape[-1], device=k.device)
    out = {}
    for t in range(T, 0, -1):                                          # G holds G^wb_t
        if t % every == 0:
            out[t] = G.clone()
        J = q[t - 1][:, :, None] * g[t - 1][:, None, :] + G
        kk = k[t - 1]
        J = J - beta[t - 1][:, None, None] * kk[:, :, None] * torch.einsum('hk,hkv->hv', kk, J)[:, None, :]
        G = a[t - 1][:, None, None] * J
    return out


class GGA(StepQuant):
    """bits 4 | 5 | 6 (nominal), menu {2,4,6,8} for 4 and 5, {4,6,8} for 6; ``pivots`` fp16 heads (32)."""
    needs_calibration = True

    def __init__(self, bits: int = 4, pivots: int = 32, sample_every: int = 8, max_snapshots: int = 64,
                 group: int = 32, gamma: float = 0.25, backend: str = 'auto'):
        assert bits in (4, 5, 6)
        super().__init__(4 if bits == 4 else 6, pivots=pivots, sample_every=sample_every,
                         max_snapshots=max_snapshots, group=group, gamma=gamma, backend=backend)
        self.bits = bits
        self.candidates = [2, 4, 6, 8] if bits in (4, 5) else [4, 6, 8]
        self.name = f'GGA@{bits}'
        self.omega, self.count, self.res = {}, {}, {}

    # -- calibration ------------------------------------------------------------------------------------------------
    def observe(self, *a, **kw):
        raise RuntimeError('GGA needs gradients: calibrate with statequant.calibrate(model, method, ids), which '
                           'calls GGA.calibrate_model')

    def calibrate_model(self, model, input_ids, batch_size=1):
        from ..patch import _LAYER, _patched
        assert batch_size == 1
        recs = {}

        class Recorder:
            def __init__(self, orig, sch, bnd, pos, fam, op):
                self.orig, self.fam, self.op = orig, fam, op

            def __call__(self, *args, **kw):
                out = self.orig(*args, **kw)                           # as called (op argument names vary)
                kw = {**dict(zip(self.op.positional, args)), **kw}     # the canonical names, for the inputs
                o = out[0] if isinstance(out, tuple) else out
                assert o.requires_grad, 'GGA calibration: the op output has no autograd graph'
                o.retain_grad()
                recs[_LAYER.get()] = (kw, o, self.fam)
                return out

        emb = model.get_input_embeddings()
        dev = next(model.parameters()).device
        params = [p for p in model.parameters()]
        req = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(False)
        try:
            with torch.enable_grad(), _patched(model, Schedule(prefill=1 << 30, decode=1, buffer_bits=0), None,
                                               carry=lambda *a: Recorder(*a)):
                for i in range(len(input_ids)):
                    recs.clear()
                    ids = input_ids[i:i + 1].to(dev)
                    x = emb(ids).detach().requires_grad_(True)
                    logits = model(inputs_embeds=x, use_cache=False).logits
                    loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction='sum')
                    loss.backward()
                    self._sequence(recs)
                    del logits, loss, x
        finally:
            for p, r in zip(params, req):
                p.requires_grad_(r)
        self.finalize()
        return self

    @torch.no_grad()
    def _sequence(self, recs):
        """One calibration sequence: per layer the fp32 state at every ``sample_every``-th token and its writeback
        adjoint (Eq. 6), offered to the layer's reservoir; the row impact accumulated as STEP's."""
        from ..families import Piece
        for layer in sorted(recs):
            kw, o, fam = recs[layer]
            pc = Piece(kw, fam, layer)
            assert fam.rule == 'delta', f'GGA is defined for the delta rule (GDN), not {fam.name}'
            q, k, lg, beta = _normalised_inputs(pc)                    # [1, T, H, d_k], [1, T, H], [1, T, H]
            v = pc.inputs()['v'].float()
            g = o.grad.float()                                         # [1, T, H, d_v]
            om = row_impact(q[0], k[0], lg[0], beta[0]).double().sum(0).cpu()     # [H, d_k]
            self.omega[layer] = self.omega.get(layer, 0) + om
            self.count[layer] = self.count.get(layer, 0) + q.shape[1]
            a = lg[0].float().exp()                                    # [T, H]
            assert a.dim() == 2, 'GGA: scalar decay per head (GDN)'
            q0, k0, b0, v0, g0 = q[0].float(), k[0].float(), beta[0].float(), v[0], g[0]
            snaps = states_at(q0, k0, v0, b0, a, self.sample_every)
            adj = writeback_adjoints(q0, k0, b0, a, g0, self.sample_every)
            r = self.res.setdefault(layer, _Reservoir(self.max_snapshots))
            for t in sorted(snaps):
                r.offer(lambda t=t: (snaps[t].cpu(), adj[t].cpu()))

    def finalize(self, device=None):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        names = sorted(self.res)
        info, costs = {}, []
        for name in names:
            w = impact_factors(self.omega[name] / self.count[name], self.gamma).to(device)
            d = []
            for b in self.candidates:
                bits = torch.full((w.shape[0],), b, dtype=torch.long, device=device)
                tot = 0
                for S, G in self.res[name].items:
                    x, Gd = S[None].to(device), G[None].to(device)
                    tot = tot + (Gd * (fit(x, bits, w, self.group) - x)).square().sum((0, 2, 3)).double().cpu()
                d.append(tot / (2 * len(self.res[name].items)))
            costs.append(torch.stack(d, -1))                           # [H, C]
            info[name] = (w.cpu(), costs[-1].shape[0])
        self._plan_from_costs(names, info, torch.cat(costs))
        self.res, self.omega, self.count = {}, {}, {}

    def _plan_from_costs(self, names, info, cost):
        from ..step_quant import allocate_dp
        pos = cost[cost > 0]
        if pos.numel():
            cost = cost / pos.median()                                 # the paper's common rescaling (no-op)
        n = len(cost)
        piv = self.pivots
        first = allocate_dp(cost, self.candidates, self.bits * n)
        risk = cost.gather(1, torch.searchsorted(torch.tensor(self.candidates), first)[:, None]).flatten()
        pm = torch.zeros(n, dtype=torch.bool)
        pm[torch.argsort(risk, descending=True, stable=True)[:piv]] = True
        assign = torch.full((n,), 16, dtype=torch.long)
        assign[~pm] = allocate_dp(cost[~pm], self.candidates, self.bits * n - 8 * piv)
        off = 0
        for name in names:
            w, h = info[name]
            self.plan[name] = (assign[off:off + h].clone(), w)
            self.masks[name] = assign[off:off + h] == 16
            off += h
        self.n_pivots = piv
        self.costs = cost


# ------------------------------------------------------------------------------------------------------------------
# PW-S8
# ------------------------------------------------------------------------------------------------------------------
def fp8_vec(x):
    """R_FP8 of App. B.4 per vector (last dim): s = amax / 448, s * fp32(e4m3fn(x / s)); 0 -> 0."""
    s = x.abs().amax(-1, keepdim=True) / 448.
    safe = torch.where(s > 0, s, torch.ones_like(s))
    return torch.where(s > 0, (x / safe).to(torch.float8_e4m3fn).float() * safe, torch.zeros_like(x))


class PureWrite(StatefulMethod):
    """PW-S``m`` around a calibrated STEP / GGA plan (``base``): old state re-rounded every decoded token by the
    base's per-head widths, the last m writes as FP8 (p, v) pairs. ``prefill``: 'end' (base path: full precision,
    rounded at the prompt end) or 'token' (the base rounds every prompt token too)."""
    rules = ('delta',)

    def __init__(self, base: StepQuant, m: int = 8, prefill: str = 'end', queue: str = 'fp8',
                 name: str | None = None):
        assert prefill in ('end', 'token') and m >= 1 and queue in ('fp8', 'fp32')
        self.base, self.m, self.prefill = base, m, prefill
        self.enc = fp8_vec if queue == 'fp8' else (lambda x: x.float())
        self.layers = {}
        self.name = name or f'{base.name}+PW-S{m}'
        self.schedule = Schedule(prefill=1 << 30 if prefill == 'end' else 1, decode=1, buffer_bits=0,
                                 prompt_end=True)

    needs_calibration = property(lambda self: self.base.needs_calibration)
    masks = property(lambda self: self.base.masks)
    calibrated = property(lambda self: bool(getattr(self.base, 'plan', None)))
    calibration_period = property(lambda self: getattr(self.base, 'calibration_period', 8))

    def observe(self, *a, **kw):
        return self.base.observe(*a, **kw)

    def finalize(self):
        return self.base.finalize()

    def __getattr__(self, name):                                       # calibrate_model and friends of the base
        if name in ('calibrate_model',) and hasattr(self.__dict__.get('base'), name):
            return getattr(self.base, name)
        raise AttributeError(name)

    def _pivots(self, layer, dev):
        return (self.base.plan[layer][0] == 16).to(dev)                # [H]

    @staticmethod
    def _A(x, k, beta, a):
        """A x = a (x - beta k (k^T x)) for x [B, H, d_k] or [B, H, d_k, d_v]."""
        if x.dim() == 3:
            return a[..., None] * (x - beta[..., None] * k * (k * x).sum(-1, keepdim=True))
        return a[..., None, None] * (x - beta[..., None, None] * k[..., None] * torch.einsum('bhk,bhkv->bhv', k, x)[:, :, None])

    @staticmethod
    def _Ainv(x, k, beta, a):
        """A^-1 x for x [B, H, d_k] (float64): (1/a) (x + beta / (1 - beta |k|^2) k (k^T x))."""
        x, k, beta, a = x.double(), k.double(), beta.double(), a.double()
        kk = (k * k).sum(-1, keepdim=True)
        c = beta[..., None] / (1 - beta[..., None] * kk)
        return ((x + c * k * (k * x).sum(-1, keepdim=True)) / a[..., None]).float()

    def _logical(self, st):
        S = st['O'].clone()
        for p, v in st['Q']:
            S = S + p[..., None] * v[..., None, :]
        return S

    def prepare(self, S, layer, pc):
        st = (self.layers or {}).get(layer)
        if st is None or pc.inputs()['k'].shape[1] != 1:              # the prompt (or no decode state yet)
            return S
        lg, k, beta, _, _ = terms(pc)
        a, k, beta = lg[:, 0].exp(), k[:, 0], beta[:, 0].float()
        st['R'] = [self.enc(self._A(p, k, beta, a)) for p, _ in st['Q']]   # propagated + re-encoded left factors
        Sp = st['O'].clone()
        for (p, v), r in zip(st['Q'], st['R']):
            Sp = Sp + self._Ainv(r, k, beta, a)[..., None] * v[..., None, :]
        return Sp

    def step(self, ctx):
        pc, layer, S = ctx.pc, ctx.layer, ctx.S_out
        T = pc.inputs()['k'].shape[1]
        if ctx.start:
            self.layers.pop(layer, None)
        st = self.layers.get(layer)
        if st is None:                                                 # the prompt: the base path
            if not ctx.prompt_end:
                return S if T > 1 else self.base.quantize(S, layer)    # 'end': full precision; 'token': rounded
            O = self.base.quantize(S, layer)                           # prompt end: the queue starts empty
            self.layers[layer] = {'O': O, 'Q': [], 'R': []}
            return O
        lg, k, beta, v, _ = terms(pc)
        a, k, beta, v = lg[:, 0].exp(), k[:, 0], beta[:, 0].float(), v[:, 0]
        piv = self._pivots(layer, S.device)
        Q = list(zip(st.get('R', []), [vv for _, vv in st['Q']]))
        Om = self._A(st['O'], k, beta, a)                              # O_pre = A O
        if len(Q) == self.m:
            p_old, v_old = Q.pop(0)
            Om = Om + p_old[..., None] * v_old[..., None, :]
        O = self.base.quantize(Om, layer)
        O = torch.where(piv[None, :, None, None], S.half().float(), O)  # pivots: fp16 of the full update
        live = (~piv)[None, :, None].float()
        Q.append((self.enc(beta[..., None] * k) * live, self.enc(v) * live))
        st['O'], st['Q'], st['R'] = O, Q, []
        return self._logical(st)

    def reset(self):
        self.layers = {}

    def bits_per_value(self, d_k, d_v):
        return self.base.bits_per_value(d_k, d_v) + 8 * self.m * (d_k + d_v + 8) / (d_k * d_v)
