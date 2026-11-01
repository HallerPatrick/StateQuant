"""vLLM plugin mode ``STATEQUANT_VLLM=gga:<plan.pt>[:pw]``: GGA / STEP plans (``StepQuant.save``) and PW-S8 of
arXiv 2610.39931 on Qwen3.5 / Qwen3-Next GDN layers, as an emulation: the slot holds the old state as fp32 VALUES
on each head's frozen width (re-fitted with STEP's dual-axis fit at every decoded token, fp16 for pivots) and, with
``:pw``, the window of the last m (STATEQUANT_VLLM_PW_M, 8) writes as reconstructed fp32 values of the FP8 vector
pairs. Outputs follow the reference path (``statequant.methods_ext.gga``, Algorithm 1); the stored bytes are not
the paper's packed layout (``report()`` gives that storage), so vLLM's KV planner sees a larger slot.

Decode (per row, all value heads at once, torch + statequant's Triton STEP fit; no host syncs, so CUDA graphs can
capture it):  A X = a (X - beta k (k^T X)),  O_pre = A O,  p_i <- R(A p_i) for queued pairs,
o = O_pre^T q + sum_i v_i (p_i^T q) + v (beta k^T q);  O <- Q_b(O_pre + evicted p v^T) (pivots: fp16(O_pre +
beta k v^T), no queue);  enqueue (R(beta k), R(v)).  Without ``:pw``: O <- Q_b(A O + beta k v^T) every token.
Prefill: vLLM's chunk kernel from the logical state (O + queue), the result rounded with the plan at the end of every
prefill chunk and the queue emptied (the paper's base path: the queue starts at the first decoded token).

Per-slot blob: O fp32 [HV, d_v, d_k] (value-major, as vLLM's GDN state), P fp32 [m, HV, d_k], V fp32 [m, HV, d_v],
NQ int32 [4] (queued pairs, oldest first).
"""
from __future__ import annotations

import math

import torch

from ..kernels.slots import _ALIGN


def fp8_vec(x):
    s = x.abs().amax(-1, keepdim=True) / 448.
    safe = torch.where(s > 0, s, torch.ones_like(s))
    return torch.where(s > 0, (x / safe).to(torch.float8_e4m3fn).float() * safe, torch.zeros_like(x))


class GGALayout:
    per_layer = True

    def __init__(self, method, hv, hk, dk, dv, pw=False, m=8):
        self.method, self.HV, self.HK, self.DK, self.DV = method, hv, hk, dk, dv
        self.pw, self.m, self.gqa = pw, (m if pw else 1), hv // hk
        fields = [('O', torch.float32, (hv, dv, dk)), ('P', torch.float32, (self.m, hv, dk)),
                  ('V', torch.float32, (self.m, hv, dv)), ('NQ', torch.int32, (4,))]
        self.fields, off = {}, 0
        for name, dt, shape in fields:
            nb = math.prod(shape) * torch.empty((), dtype=dt).element_size()
            self.fields[name] = (off, dt, shape, nb)
            off += -(-nb // _ALIGN) * _ALIGN
        self.nbytes = off
        self._dev = {}
        if torch.cuda.is_available():                          # plans on the device now: no copies during capture
            dev = torch.device('cuda', torch.cuda.current_device())
            for layer in method.plan:
                self._plan(layer, dev)

    def __repr__(self):
        return (f'GGALayout({self.method.name}{"+PW-S%d" % self.m if self.pw else ""}, hv={self.HV}, hk={self.HK}, '
                f'dk={self.DK}, dv={self.DV}, {self.nbytes} B/slot emulated)')

    def views(self, blob):
        assert blob.dtype == torch.uint8 and blob.dim() == 2 and blob.shape[1] >= self.nbytes
        N = blob.shape[0]
        return {n: blob[:, o:o + nb].view(dt).view(N, *sh) for n, (o, dt, sh, nb) in self.fields.items()}

    def alloc(self, num_slots, device='cuda'):
        return torch.zeros(num_slots, self.nbytes, dtype=torch.uint8, device=device)

    # -- the plan on the device -------------------------------------------------------------------------------
    def _plan(self, layer, dev):
        key = (layer, dev)
        if key not in self._dev:
            b, w = self.method.plan[layer]
            assert len(b) == self.HV, (len(b), self.HV)
            self._dev[key] = (b.to(dev, torch.int32), w.to(dev, torch.float32).contiguous(), bool((b == 2).any()),
                              (b == 16).to(dev))
        return self._dev[key]

    def _quant(self, S, layer):
        """Q_b of canonical states [B, HV, d_k, d_v] with the layer's widths (pivots: fp16)."""
        from ..kernels._stepquant_triton import fit
        b, w, two, _ = self._plan(layer, S.device)
        if S.shape[0] == 0:
            return S
        return fit(S.contiguous(), b, w, self.method.group, two)

    # -- decode -----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, views, idx, q, k, v, out, scale, *, g=None, beta=None, gate=None, layer=None, fused=None):
        B = idx.shape[0]
        if B == 0:
            return out
        HV, DK, DV = self.HV, self.DK, self.DV
        li = idx.long()
        valid = idx > 0
        qf = q.float()
        kf = k.float()
        qf = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * scale
        kf = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
        qf, kf = qf.repeat_interleave(self.gqa, 1), kf.repeat_interleave(self.gqa, 1)       # [B, HV, d_k]
        if gate is not None:
            a_, b_, A_log, dt_bias = gate
            gg = -A_log.float().exp() * torch.nn.functional.softplus(a_[:B].float() + dt_bias.float())
            bt = b_[:B].float().sigmoid()
        else:
            gg, bt = g[:B].float(), beta[:B].float()
        a = gg.exp()                                                                         # [B, HV]
        vf = v.float()
        O = views['O'][li].transpose(-1, -2)                                                 # canonical [B, HV, dk, dv]
        AO = a[..., None, None] * (O - bt[..., None, None] * kf[..., None]
                                   * torch.einsum('bhk,bhkv->bhv', kf, O)[:, :, None])
        bk = bt[..., None] * kf
        new = bk[..., None] * vf[..., None, :]
        o = torch.einsum('bhkv,bhk->bhv', AO, qf) + (bk * qf).sum(-1, keepdim=True) * vf
        _, _, _, piv = self._plan(layer, q.device)
        if self.pw:
            m = self.m
            P, Vq = views['P'][li], views['V'][li]                                           # [B, m, HV, d]
            n = views['NQ'][li, 0].clamp(0, m)                                               # [B]
            live = (torch.arange(m, device=q.device)[None, :] < n[:, None])[..., None, None]  # [B, m, 1, 1]
            AP = a[:, None, :, None] * (P - bt[:, None, :, None] * kf[:, None]
                                        * (kf[:, None] * P).sum(-1, keepdim=True))
            P = torch.where(live, fp8_vec(AP), 0.)
            o = o + torch.einsum('bmhk,bhk,bmhv->bhv', P, qf, Vq)
            full = (n == m)[:, None, None, None]
            Om = AO + torch.where(full, P[:, 0, :, :, None] * Vq[:, 0, :, None, :], 0.)
            On = self._quant(Om, layer)
            On = torch.where(piv[None, :, None, None], (AO + new).half().float(), On)
            nonp = (~piv)[None, :, None].float()
            pn, vn = fp8_vec(bk) * nonp, fp8_vec(vf) * nonp
            P_shift = torch.cat([P[:, 1:], pn[:, None]], 1)
            V_shift = torch.cat([Vq[:, 1:], vn[:, None]], 1)
            at = (torch.arange(m, device=q.device)[None, :] == n[:, None])[..., None, None]  # [B, m, 1, 1]
            P_app = torch.where(at, pn[:, None], P)
            V_app = torch.where(at, vn[:, None], Vq)
            P_new = torch.where(full, P_shift, P_app)
            V_new = torch.where(full, V_shift, V_app)
            ok = valid[:, None, None, None]
            views['P'][li] = torch.where(ok, P_new, views['P'][li])
            views['V'][li] = torch.where(ok, V_new, views['V'][li])
            nq = views['NQ'][li]
            views['NQ'][li] = torch.where(valid[:, None], (n + 1).clamp_max(m).to(torch.int32)[:, None].expand_as(nq), nq)
        else:
            On = self._quant(AO + new, layer)
        views['O'][li] = torch.where(valid[:, None, None, None], On.transpose(-1, -2), views['O'][li])
        out[:B] = torch.where(valid[:, None, None], o, 0.).to(out.dtype)
        return out

    # -- prefill side -----------------------------------------------------------------------------------------
    @torch.no_grad()
    def read(self, views, idx, layer=None):
        """Logical state O + sum of the queued pairs, value-major fp32 [P, HV, d_v, d_k]."""
        li = idx.long()
        S = views['O'][li].clone()
        if self.pw:
            n = views['NQ'][li, 0].clamp(0, self.m)
            live = (torch.arange(self.m, device=li.device)[None, :] < n[:, None])[..., None, None]
            S = S + torch.einsum('bmhk,bmhv->bhvk', torch.where(live, views['P'][li], 0.),
                                 torch.where(live, views['V'][li], 0.))
        return S

    @torch.no_grad()
    def write(self, views, idx, S, layer=None):
        """Round value-major states S [P, HV, d_v, d_k] with the plan, store them, empty the queue."""
        li = idx.long()
        Sq = self._quant(S.float().transpose(-1, -2), layer).transpose(-1, -2)
        views['O'][li] = Sq
        views['NQ'][li] = 0
        if self.pw:
            views['P'][li] = 0.
            views['V'][li] = 0.
