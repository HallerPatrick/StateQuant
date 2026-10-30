"""Slot-addressed packed GDN states: one fixed-size byte blob per sequence slot, sequences at their own positions.

A serving engine (vLLM) keeps the recurrent state of every running sequence in a slot of a preallocated cache and
hands each step a table of slot indices. ``SlotLayout`` defines the per-slot blob for one layer and runs the packed
decode step (the kernels of ``_fold_triton`` with slot indirection and a per-sequence buffer counter) on a batch of
slots, plus the torch-side read (dequantize + apply the pending buffer) and write (quantize, empty the buffer) used
around prefill.

Per-slot blob (one layer; offsets aligned to 16 bytes, fields in this order):

    GB      fp32  [HV, m]           cumulative log decay of each buffered token
    GCUR    fp32  [HV]              running log decay since the last fold
    NPTR    int32 [max(HV, 4)]      the buffer fill counter n, one copy per value head (all equal between steps:
                                    the five-kernel step reads element 0 and writes all, the fused step's program
                                    for head h reads and writes element h)
    SC      fp16  scales            axis 'k': [HV, d_v, d_k/g]; axis 'v': [HV, d_v/g, d_k]; bf16 state: none
    KBS     fp16  [HK, m]           buffered key scales (int8 buffer only)
    UBS     fp16  [HV, m]           buffered delta-value scales (int8 buffer only)
    S       codes                   int4: uint8 [HV, d_v, d_k/2]; int8: int8 [HV, d_v, d_k]; bf16: bf16 [HV, d_v, d_k]
    KB      int8 / bf16 [HK, m, d_k]  buffered normalised keys, per KEY head
    UB      int8 / bf16 [HV, m, d_v]  buffered delta values, per value head

``views(blob)`` turns a uint8 tensor [num_slots, >= nbytes] (any slot stride, unit inner stride; e.g. vLLM's
mamba page view) into typed views [num_slots, *shape] of every field. Slot index 0 and negative indices are treated
as padding by the step (vLLM's NULL_BLOCK_ID = 0 / PAD_SLOT_ID = -1): nothing is read or written there and the
output row is zeroed.
"""
from __future__ import annotations

import math

import torch

from .fold import _bits, _fused_config, check_dims, pack_vk, packed_format, pow2, unpack_vk, use_fused

_ALIGN = 16


class SlotLayout:
    """The packed per-slot layout of one GDN layer and the operations on a batch of slots.

    fmt          ``Quantizer(4 or 8, 'k' or 'v', group)`` or 'bf16' (see ``fold.packed_format``)
    hv, hk       value / key heads of the layer (on this rank); hv % hk == 0
    dk, dv       head dims (any even d_k >= 16, d_v a multiple of 32; see ``fold.check_dims``)
    m            buffer length (power of two)
    buffer_bits  8 (int8 + fp16 scale per vector) or 16 (bf16)
    fused        decode step as one fused launch (True), the five-kernel step (False), or None (auto, per call:
                 ``fold.use_fused``). Both act on the same blob, so they mix freely.
    """

    def __init__(self, fmt, hv, hk, dk, dv, m=8, buffer_bits=8, fused=None):
        self.fused = fused
        self.fmt = packed_format(fmt)
        self.bits = _bits(self.fmt)
        self.HV, self.HK, self.DK, self.DV, self.m = hv, hk, dk, dv, m
        assert hv % hk == 0
        self.gqa = hv // hk
        assert m >= 1 and m & (m - 1) == 0, 'm must be a power of two'
        check_dims(self.fmt, dk, dv)
        self.DKP, self.DVP = pow2(dk), pow2(dv)
        assert buffer_bits in (8, 16)
        self.buffer_bits, self.qbuf = buffer_bits, buffer_bits == 8
        self.axk = self.fmt is not None and self.fmt.axis == 'k'
        self.group = self.fmt.group if self.fmt is not None else 32
        self.BV = self.group if (self.fmt is not None and not self.axk) else 32
        assert dv % self.BV == 0
        self.warps = 2 if self.BV <= 32 else 4
        self.fold_warps = 4 if self.BV <= 32 else 8
        self.DK, self.DV = dk, dv
        self.fused_sbv, self.fused_bv, self.fused_warps = _fused_config(self)

        bdt = torch.int8 if self.qbuf else torch.bfloat16
        if self.fmt is None:
            sdt, sshape, scshape = torch.bfloat16, (hv, dv, dk), None
        else:
            sdt = torch.uint8 if self.bits == 4 else torch.int8
            sshape = (hv, dv, dk // 2) if self.bits == 4 else (hv, dv, dk)
            g = self.group
            scshape = (hv, dv, -(-dk // g)) if self.axk else (hv, dv // g, dk)          # ragged key group: ceil
        fields = [('GB', torch.float32, (hv, m)), ('GCUR', torch.float32, (hv,)), ('NPTR', torch.int32, (max(hv, 4),))]
        if scshape is not None:
            fields.append(('SC', torch.float16, scshape))
        if self.qbuf:
            fields += [('KBS', torch.float16, (hk, m)), ('UBS', torch.float16, (hv, m))]
        fields += [('S', sdt, sshape), ('KB', bdt, (hk, m, dk)), ('UB', bdt, (hv, m, dv))]
        self.fields, off = {}, 0
        for name, dt, shape in fields:
            nb = math.prod(shape) * torch.empty((), dtype=dt).element_size()
            self.fields[name] = (off, dt, shape, nb)
            off += -(-nb // _ALIGN) * _ALIGN
        self.nbytes = off

    def __repr__(self):
        f = 'bf16' if self.fmt is None else f'int{self.bits}-{self.fmt.axis}-g{self.group}'
        return (f'SlotLayout({f}, hv={self.HV}, hk={self.HK}, dk={self.DK}, dv={self.DV}, m={self.m}, '
                f'buffer_bits={self.buffer_bits}, {self.nbytes} B/slot)')

    def breakdown(self):
        """Bytes per slot by field (before alignment padding)."""
        return {n: f[3] for n, f in self.fields.items()}

    # -- views ------------------------------------------------------------------------------------------------
    def views(self, blob):
        """uint8 [num_slots, >= nbytes] (inner stride 1) -> {field: typed view [num_slots, *shape]}."""
        assert blob.dtype == torch.uint8 and blob.dim() == 2 and blob.shape[1] >= self.nbytes, blob.shape
        assert blob.stride(1) == 1 and blob.stride(0) % _ALIGN == 0 and blob.data_ptr() % _ALIGN == 0
        out = {}
        N = blob.shape[0]
        for name, (off, dt, shape, nb) in self.fields.items():
            v = blob[:, off:off + nb].view(dt)
            out[name] = v.view(N, *shape)
        return out

    def alloc(self, num_slots, device='cuda'):
        """A standalone zeroed blob [num_slots, nbytes] uint8 (tests, or engines without their own allocator)."""
        return torch.zeros(num_slots, self.nbytes, dtype=torch.uint8, device=device)

    # -- decode -----------------------------------------------------------------------------------------------
    def step(self, views, idx, q, k, v, out, scale, *, g=None, beta=None, gate=None, fused='layout'):
        """One decode token for every batch row b, on slot ``idx[b]`` (int32 [B]; <= 0 = padding row).

        q, k    [B, hk, d_k] per key head, raw (l2-normalised inside); any row stride, unit inner stride
        v       [B, hv, d_v]
        out     [B, hv, d_v] contiguous; written (zeros on padding rows)
        g, beta [B, hv] log decay and (sigmoided) beta;  or ``gate=(a, b, A_log, dt_bias)`` to form them in the
                kernel as g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b) (vLLM / fla's gating)
        fused   overrides the layout's ``fused`` option for this call (True / False / None = auto)
        Capture-safe: fixed launches, scratch from the caching allocator, no host syncs. Each row's buffer counter
        advances; rows whose buffer fills are folded and re-rounded in place.
        """
        from . import _fold_triton as K
        B = idx.shape[0]
        HV, HK, DK, DV, m = self.HV, self.HK, self.DK, self.DV, self.m
        BH = B * HV
        if B == 0:
            return out
        assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
        assert out.is_contiguous() and out.shape[0] >= B
        assert idx.dtype == torch.int32 and idx.is_contiguous()
        dev = q.device
        fused = use_fused(self.fused if fused == 'layout' else fused, BH, self.fmt is not None and not self.axk)
        f32 = torch.empty(0 if fused else BH * (2 * DK + 2 * m + 4 + DV), device=dev, dtype=torch.float32)
        o_ = 0

        def take(n):
            nonlocal o_
            t = f32[o_:o_ + n]
            o_ += n
            return t
        QN, KN, CQ, CK = take(BH * DK), take(BH * DK), take(BH * m), take(BH * m)
        KQ, ALPHA, GS, BS, UCUR = take(BH), take(BH), take(BH), take(BH), take(BH * DV)
        V = views
        st = lambda name: V[name].stride(0)                   # noqa: E731
        S, KB, UB, GB, GCUR, NPTR = V['S'], V['KB'], V['UB'], V['GB'], V['GCUR'], V['NPTR']
        SC = V.get('SC', GB)
        KBS, UBS = V.get('KBS', GB), V.get('UBS', GB)
        sSC = st('SC') if 'SC' in V else 0
        sKBS = st('KBS') if 'KBS' in V else 0
        sUBS = st('UBS') if 'UBS' in V else 0
        if gate is not None:
            a, bt, A_log, dt_bias = gate
            assert a.stride(-1) == 1 and bt.stride(-1) == 1
            G = BETA = a
            sG = sBETA = 0
            A, BT, sA, sBT, ALOG, DTB = a, bt, a.stride(0), bt.stride(0), A_log, dt_bias
            gate_on = True
        else:
            assert g.stride(-1) == 1 and beta.stride(-1) == 1
            G, BETA, sG, sBETA = g, beta, g.stride(0), beta.stride(0)
            A = BT = ALOG = DTB = g
            sA = sBT = 0
            gate_on = False
        if fused:
            from . import _fused_triton as F
            F._fused_step[(BH,)](
                q, k, q.stride(0), k.stride(0), v, v.stride(0), G, BETA, sG, sBETA, A, BT, sA, sBT, ALOG, DTB, idx,
                S, st('S'), SC, sSC, KB, st('KB'), KBS, sKBS, UB, st('UB'), UBS, sUBS, GB, st('GB'),
                GCUR, st('GCUR'), NPTR, st('NPTR'), out, float(scale),
                DK=DK, DV=DV, DKP=self.DKP, DVP=self.DVP, SBV=self.fused_sbv, FBV=self.fused_bv, M=m, BITS=self.bits, AXK=self.axk,
                GROUP=self.group, QBUF=self.qbuf, HV=HV, HK=HK, GQA=self.gqa, QK_PER_KEY_HEAD=True, HAS_IDX=True,
                GATE=gate_on, num_warps=self.fused_warps)
            return out
        cst = dict(DK=DK, M=m, HV=HV, HK=HK, GQA=self.gqa, QBUF=self.qbuf, HAS_IDX=True)
        stc = dict(DK=DK, DKP=self.DKP, DV=DV, BV=self.BV, M=m, BITS=self.bits, AXK=self.axk, GROUP=self.group,
                   QBUF=self.qbuf, HV=HV, HAS_IDX=True)
        K._prep[(BH,)](q, k, q.stride(0), k.stride(0), G, BETA, sG, sBETA, A, BT, sA, sBT, ALOG, DTB, idx,
                       GB, st('GB'), GCUR, st('GCUR'), KB, st('KB'), KBS, sKBS, NPTR, st('NPTR'),
                       QN, KN, CQ, CK, KQ, ALPHA, GS, BS, float(scale),
                       BM=min(m, 16), QK_PER_KEY_HEAD=True, GATE=gate_on, DKP=self.DKP, num_warps=4, **cst)
        K._state[(BH, DV // self.BV)](
            S, st('S'), SC, sSC, QN, KN, UB, st('UB'), UBS, sUBS, CQ, CK, ALPHA, v, v.stride(0), GS, BS, KQ,
            out, UCUR, NPTR, st('NPTR'), idx, num_warps=self.warps, **stc)
        K._append[(BH,)](UCUR, KN, GS, idx, GB, st('GB'), GCUR, st('GCUR'), KB, st('KB'), KBS, sKBS,
                         UB, st('UB'), UBS, sUBS, NPTR, st('NPTR'), DV=DV, DKP=self.DKP, DVP=self.DVP,
                         num_warps=4, **cst)
        K._fold[(BH, DV // self.BV)](
            S, st('S'), SC, sSC, UB, st('UB'), UBS, sUBS, KB, st('KB'), KBS, sKBS, GB, st('GB'),
            GCUR, st('GCUR'), NPTR, st('NPTR'), idx, HK=HK, GQA=self.gqa, num_warps=self.fold_warps, **stc)
        K._advance_seq[(B,)](idx, GCUR, st('GCUR'), NPTR, st('NPTR'), M=m, HV=HV,
                             BHV=triton_next_pow2(HV), num_warps=1)
        return out

    # -- prefill side (torch) ---------------------------------------------------------------------------------
    @torch.no_grad()
    def read(self, views, idx):
        """The current state of slots ``idx`` (long/int [P]) as fp32 [P, HV, d_v, d_k]: the stored state advanced
        across its pending buffer, S_n = a_n S_0 + sum_{i<n} (a_n/a_i) u_i k_i^T (value-major, as vLLM keeps it).
        Slots that hold garbage give garbage (mask the result with ``torch.where``, not a product)."""
        idx = idx.long()
        P, HV, HK, DK, DV, m = idx.shape[0], self.HV, self.HK, self.DK, self.DV, self.m
        V = views
        codes = V['S'][idx].reshape(P * HV, *V['S'].shape[2:])
        sc = V['SC'][idx].reshape(P * HV, *V['SC'].shape[2:]) if 'SC' in V else None
        S0 = unpack_vk(codes, sc, self.fmt, DK).view(P, HV, DV, DK)
        Kb = V['KB'][idx].float()                                             # [P, HK, m, DK]
        Ub = V['UB'][idx].float()                                             # [P, HV, m, DV]
        if self.qbuf:
            Kb = Kb * V['KBS'][idx].float()[..., None]
            Ub = Ub * V['UBS'][idx].float()[..., None]
        n = V['NPTR'][idx, 0] & (m - 1)                                       # [P]
        gcur = V['GCUR'][idx]                                                 # [P, HV]
        live = (torch.arange(m, device=idx.device)[None, :] < n[:, None])[:, None, :]       # [P, 1, m]
        dec = torch.where(live, torch.exp(gcur[..., None] - torch.where(live, V['GB'][idx], 0.)), 0.)
        Ub = torch.where(live[..., None], Ub, 0.)
        Kb = torch.where(live.expand(-1, HK, -1)[..., None], Kb, 0.)
        Kx = Kb.repeat_interleave(self.gqa, dim=1)                           # [P, HV, m, DK]
        S = torch.exp(gcur)[..., None, None] * S0 + torch.einsum('phmv,phmk->phvk', Ub * dec[..., None], Kx)
        return S

    @torch.no_grad()
    def write(self, views, idx, S):
        """Store fp32/bf16 states S [P, HV, d_v, d_k] (value-major) packed into slots ``idx`` and empty their
        buffers (n = 0, running decay 0)."""
        idx = idx.long()
        P = idx.shape[0]
        codes, sc = pack_vk(S.reshape(P * self.HV, self.DV, self.DK), self.fmt)
        V = views
        V['S'][idx] = codes.view(P, *V['S'].shape[1:])
        if 'SC' in V:
            V['SC'][idx] = sc.view(P, *V['SC'].shape[1:])
        V['NPTR'][idx] = 0
        V['GCUR'][idx] = 0.

    def stored(self, views, idx):
        """(codes, scales) of slots ``idx`` (for tests)."""
        idx = idx.long()
        return V_get(views, 'S', idx), V_get(views, 'SC', idx)


def V_get(views, name, idx):
    return views[name][idx] if name in views else None


def triton_next_pow2(n):
    return 1 << max(0, (n - 1).bit_length())
