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
import os

import torch

from ..kernels.slots import _ALIGN
from ..methods_ext.gga import fp8_vec


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


# ------------------------------------------------------------------------------------------------------------------
# packed storage (kernels/_gga_triton.py)
# ------------------------------------------------------------------------------------------------------------------
def head_bytes(b, dk, dv, group=32):
    """Stored bytes of one head at width b: codes + row factors (2 bit: group rows) + column factors; fp16 pivots."""
    if b == 16:
        return 2 * dk * dv
    if b == 2:
        return dk * dv // 4 + 2 * dk * (dv // group) + 2 * dv
    return dk * dv * b // 8 + 2 * dk + 2 * dv


def _al(n):
    return -(-n // _ALIGN) * _ALIGN


class GGAPacked:
    """The packed slot (layout in ``kernels/_gga_triton.py``) for every GDN layer of a plan: the per-head widths
    differ between layers, so each layer has its own offset table; vLLM sizes one slot for all layers, so ``nbytes``
    is the largest layer's (``bytes_per_layer`` and ``bits_per_value`` give the per-layer and mean figures)."""
    per_layer = True

    def __init__(self, method, hv, hk, dk, dv, pw=False, m=8):
        assert dk & (dk - 1) == 0 and dv & (dv - 1) == 0 and dv % 32 == 0, (dk, dv)
        self.method, self.HV, self.HK, self.DK, self.DV = method, hv, hk, dk, dv
        self.pw, self.m, self.gqa = pw, m, hv // hk
        self.warps = int(os.environ.get('STATEQUANT_GGA_WARPS', '16'))
        self.tables, self.bytes_per_layer = {}, {}
        for layer, (b, w) in method.plan.items():
            b = [int(x) for x in b]
            assert len(b) == hv and set(b) <= {2, 4, 6, 8, 16}, b
            offs, off = [], 16                                  # counters first
            for x in b:
                offs.append(off)
                off += _al(head_bytes(x, dk, dv, method.group))
            self.tables[layer] = (offs, b, w)
            self.bytes_per_layer[layer] = off
        self.QOFF = _al(max(self.bytes_per_layer.values()))
        self.QSZ = m * (dk + dv + 8)
        self.nbytes = self.QOFF + (hv * self.QSZ if pw else 0)
        self._dev = {}
        if torch.cuda.is_available():
            dev = torch.device('cuda', torch.cuda.current_device())
            for layer in self.tables:
                self._plan(layer, dev)

    def __repr__(self):
        return (f'GGAPacked({self.method.name}{"+PW-S%d" % self.m if self.pw else ""}, hv={self.HV}, hk={self.HK}, '
                f'dk={self.DK}, dv={self.DV}, {self.nbytes} B/slot = max over layers; mean '
                f'{self.bits_per_value():.3f} b/value)')

    def bits_per_value(self):
        """Mean stored bits per state value over the layers (without the 16-byte counters), queue included."""
        n = self.HV * self.DK * self.DV
        st = sum(sum(head_bytes(x, self.DK, self.DV, self.method.group) for x in t[1]) for t in self.tables.values())
        q = self.HV * self.QSZ * len(self.tables) if self.pw else 0
        return 8 * (st + q) / (n * len(self.tables))

    def _plan(self, layer, dev):
        key = (layer, dev)
        if key not in self._dev:
            offs, b, w = self.tables[layer]
            self._dev[key] = (torch.tensor(offs, dtype=torch.int32, device=dev),
                              torch.tensor(b, dtype=torch.int32, device=dev),
                              w.to(dev, torch.float32).contiguous())
        return self._dev[key]

    def views(self, blob):
        assert blob.dtype == torch.uint8 and blob.dim() == 2 and blob.shape[1] >= self.nbytes
        assert blob.stride(1) == 1 and blob.stride(0) % _ALIGN == 0 and blob.data_ptr() % _ALIGN == 0
        return {'blob': blob}

    def alloc(self, num_slots, device='cuda'):
        return torch.zeros(num_slots, self.nbytes, dtype=torch.uint8, device=device)

    def step(self, views, idx, q, k, v, out, scale, *, g=None, beta=None, gate=None, layer=None, fused=None):
        from ..kernels import _gga_triton as G
        assert gate is not None, 'GGAPacked.step takes the gate inputs (a, b, A_log, dt_bias)'
        B = idx.shape[0]
        if B == 0:
            return out
        assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1 and v.stride(1) == self.DV
        assert out.is_contiguous() and idx.dtype == torch.int32 and idx.is_contiguous()
        a_, b_, A_log, dt_bias = gate
        offs, bits, w = self._plan(layer, q.device)
        blob = views['blob']
        G._gga_step[(B * self.HV,)](blob, blob.stride(0), idx, offs, bits, w, q, k, q.stride(0), k.stride(0), v,
                                    v.stride(0), a_, b_, a_.stride(0), b_.stride(0), A_log, dt_bias, out,
                                    float(scale), self.QOFF, self.QSZ, DK=self.DK, DV=self.DV, HV=self.HV,
                                    GQA=self.gqa, M=self.m, PW=self.pw, num_warps=self.warps)
        if self.pw:
            G._gga_advance[(B,)](blob, blob.stride(0), idx, M=self.m, num_warps=1)
        return out

    @torch.no_grad()
    def stored(self, views, idx, layer):
        """The stored states of slots idx, canonical fp32 [P, HV, d_k, d_v] (without the window)."""
        from ..kernels import _gga_triton as G
        idx32 = idx.to(torch.int32).contiguous()
        P = idx32.shape[0]
        Y = torch.empty(P, self.HV, self.DK, self.DV, device=idx.device)
        if P:
            offs, bits, _ = self._plan(layer, idx.device)
            blob = views['blob']
            G._gga_load[(P * self.HV,)](Y, blob, blob.stride(0), idx32, offs, bits, DK=self.DK, DV=self.DV,
                                        HV=self.HV, num_warps=8)
        return Y

    @torch.no_grad()
    def window(self, views, idx):
        """The queued pairs of slots idx as reconstructed fp32 (P [n, m, HV, d_k], V [n, m, HV, d_v]) in ring order
        from the oldest, and the counts n [n]; dead entries zeroed."""
        li = idx.long()
        blob = views['blob'][li]
        m, HV, DK, DV = self.m, self.HV, self.DK, self.DV
        ct = blob[:, :16].view(torch.int32)
        n, hd = ct[:, 0], ct[:, 1]
        Qb = blob[:, self.QOFF:self.QOFF + HV * self.QSZ].reshape(-1, HV, self.QSZ)
        Pc = Qb[..., :m * DK].contiguous().view(torch.float8_e4m3fn).float().view(-1, HV, m, DK)
        Vc = Qb[..., m * DK:m * (DK + DV)].contiguous().view(torch.float8_e4m3fn).float().view(-1, HV, m, DV)
        sc = Qb[..., m * (DK + DV):].contiguous().view(torch.float32).view(-1, HV, 2, m)
        Pf = Pc * sc[:, :, 0, :, None]
        Vf = Vc * sc[:, :, 1, :, None]
        order = (hd[:, None] + torch.arange(m, device=li.device)[None, :]) % m          # oldest first
        Pf = torch.gather(Pf, 2, order[:, None, :, None].expand(-1, HV, -1, DK)).transpose(1, 2)
        Vf = torch.gather(Vf, 2, order[:, None, :, None].expand(-1, HV, -1, DV)).transpose(1, 2)
        live = (torch.arange(m, device=li.device)[None, :] < n[:, None])[:, :, None, None]
        return torch.where(live, Pf, 0.), torch.where(live, Vf, 0.), n

    @torch.no_grad()
    def read(self, views, idx, layer=None):
        """Logical state (stored + window), value-major fp32 [P, HV, d_v, d_k]."""
        S = self.stored(views, idx, layer)
        if self.pw and idx.shape[0]:
            P, V, _ = self.window(views, idx)
            _, bits, _ = self._plan(layer, idx.device)
            nonp = (bits != 16)[None, :, None, None].float()
            S = S + torch.einsum('bmhk,bmhv->bhkv', P, V) * nonp
        return S.transpose(-1, -2)

    @torch.no_grad()
    def write(self, views, idx, S, layer=None):
        """Fit + pack value-major states S [P, HV, d_v, d_k] into slots idx; empty the window."""
        from ..kernels import _gga_triton as G
        idx32 = idx.to(torch.int32).contiguous()
        P = idx32.shape[0]
        if P == 0:
            return
        X = S.float().transpose(-1, -2).contiguous()
        offs, bits, w = self._plan(layer, S.device)
        blob = views['blob']
        G._gga_store[(P * self.HV,)](X, blob, blob.stride(0), idx32, offs, bits, w, DK=self.DK, DV=self.DV,
                                     HV=self.HV, num_warps=8)
