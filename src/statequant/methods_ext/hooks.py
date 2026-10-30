"""What the methods in ``statequant.methods_ext`` need from the layer patch and the family adapters.

The fake-quant patch (``statequant.patch``) splits every recurrent op call at the schedule's positions. A plain
``StateQuantizer`` only sees the state at rounding points (``quantize(S, layer)``). The methods here need more, so
they set ``stateful = True``; ``statequant.apply`` then calls ``method.piece(S, layer, piece)`` after EVERY piece
(scan pieces and Mamba2's in-place decode steps alike) with

    S                     the family's canonical [B, H, d_k, d_v] fp32 state after the piece (op's full precision)
    piece                 ``families.Piece``: the piece's op arguments, and
      .state_before       canonical state the piece started from = what the method returned last (None: new sequence)
      .position           [B] tokens processed after the piece
      .hit                [B] the schedule rounds here (period boundary, or prompt end with Schedule(prompt_end=True))
      .start / .prompt_end
      .inputs()           q, k, v, beta, log_decay, rule (the family adapter, ``statequant.families``)
      .family             the adapter (name, decay = 'head' | 'channel', rule)

and carry on with the state it returns. Per method:

    method          per-token decay   write key k_t     update u_t                pre-update state     rounding points
    BDR             cumulative since  -                 -                         last rounded state   schedule hits
                    last rounding                                                 (method-held)
    LRB             per token (decays per token          delta: U^T k / |k|^2,     S_{t-1}              every m tokens
                    the key factors)                     U = S_t - A_t S_{t-1};    (.state_before)
                                                         additive: u = v
    LeapQuant       per token (fp16   per token          as LRB                    S_{t-1}              every p decode
                    log decay record)                                                                   tokens + prompt end
    GramianAlloc    per token (EMA)   per token (erase   -                         -                    schedule hits
                                      rate, delta only)
    DampPaper       calibration only: anchor states + the H following per-token decays (plain observe / quantize,
                    ``piece.position`` at calibration boundaries)

Family support (``check_family``): every method needs a diagonal transition plus a rank-one write, i.e. rule 'delta'
(GDN, KDA) or 'additive' (Mamba2, lightning, RWKV-6). RWKV-7's 'dplr' transition (diag + rank one acting on the
state) raises NotImplementedError: the decayed stored state A_{t0:t} S^_{t0} that BDR / LRB / LeapQuant hold fixed
between roundings is not a decay of the stored codes there, and the Gramian / DAMP retention weights have no term for
it.

The per-token methods (LRB, LeapQuant with fp16 records) need one-token pieces in the phases where they keep records
(``per_token()`` / decode = 1); with ``records='fp32'`` they run on period-m pieces and only reconstruct the
bypass / buffer at merges, exactly in fp32.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from ..methods import StateQuantizer


def check_family(family, method: str, rules=('delta', 'additive')):
    if family.rule not in rules:
        raise NotImplementedError(
            f"{method} needs a diagonal decay plus a rank-one write (rules {rules}); the {family.name} family's "
            f"transition is {family.rule!r} (diag + rank one acting on the state), so the stored state cannot be held "
            f"fixed and only decayed between roundings, and decay-only retention weights do not describe it")


def terms(pc):
    """(log_decay [B,T,H] or [B,T,H,d_k], write key [B,T,H,d_k], beta [B,T,H] or None, value [B,T,H,d_v], rule)."""
    x = pc.inputs()
    return x['log_decay'].float(), x['k'].float(), x['beta'], x['v'].float(), x['rule']


def decay_shape(lg_sum, S):
    """A summed log decay [B,H] or [B,H,d_k] as a factor broadcastable against S [B,H,d_k,d_v]."""
    if lg_sum.dim() == 2:
        return lg_sum[:, :, None, None]
    return lg_sum[..., None]


def rank1_value(U, key):
    """u with U = key u^T for a rank-one update U [B,H,d_k,d_v] and its key [B,H,d_k]: u = U^T key / |key|^2."""
    return torch.einsum('bhkv,bhk->bhv', U, key) / key.square().sum(-1, keepdim=True).clamp_min(1e-30)


def written_value(S_out, S_in, lg, key, v, rule):
    """The value u of this token's rank-one update S_out = A S_in + key u^T (one-token piece).
    delta: recovered from the op's own update (u = beta (v - (A S_in)^T key)); additive: the input value."""
    if rule == 'additive':
        return v[:, 0]
    return rank1_value(S_out - decay_shape(lg[:, 0], S_out).exp() * S_in, key[:, 0])


class StatefulMethod(StateQuantizer):
    """Base for methods that see every piece. Subclasses implement ``step(ctx) -> state`` (ctx: layer, S_in, S_out,
    pos, hit, start, prompt_end, pc = the Piece, family) and keep per-layer state in ``self.layers``; ``reset()``
    runs when ``statequant.apply`` is entered."""
    stateful = True
    needs_calibration = False
    rules = ('delta', 'additive')
    layers: dict = None

    def reset(self):
        self.layers = {}

    def piece(self, S, layer, pc):
        check_family(pc.family, type(self).__name__, self.rules)
        before = pc.state_before
        ctx = SimpleNamespace(layer=layer, S_out=S, S_in=torch.zeros_like(S) if before is None else before,
                              pos=pc.position, hit=pc.hit, start=pc.start or before is None,
                              prompt_end=pc.prompt_end, pc=pc, family=pc.family)
        return self.step(ctx)

    def _layer_state(self, ctx, init):
        """Per-layer state, re-initialised by ``init(ctx)`` at a sequence start (or if the batch size changed)."""
        if self.layers is None:
            self.layers = {}
        st = self.layers.get(ctx.layer)
        if ctx.start or st is None or st['B'] != ctx.S_out.shape[0]:
            st = init(ctx)
            st['B'] = ctx.S_out.shape[0]
            self.layers[ctx.layer] = st
        return st

    def quantize(self, S, layer):
        raise RuntimeError(f'{type(self).__name__} is stateful: it needs the piece hook of statequant.apply')
