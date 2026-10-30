"""Recurrent-state families: one adapter per state-update rule, so that the carry in ``statequant.patch`` and every
method in ``statequant.methods`` / ``statequant.step_quant`` are architecture-agnostic.

Canonical view (what every method sees)
    state      S: [B, H, d_k, d_v] float32, H = value heads. ``Quantizer`` axis 'k' groups run along d_k, 'v' along d_v.
    inputs     per token, head-expanded to H, in the space the kernel uses (normalised / scaled as it does):
                   q [B, T, H, d_k]   read vector              o_t = S_t^T q_t
                   k [B, T, H, d_k]   write key
                   v [B, T, H, d_v]   write value
                   beta [B, T, H] or None (no delta rule)
                   log_decay [B, T, H] (scalar per head) or [B, T, H, d_k] (per key channel)
                   rule  'delta'    S_t = (I - beta_t k_t k_t^T) diag(a_t) S_{t-1} + beta_t k_t v_t^T   (GDN, KDA)
                         'additive' S_t = diag(a_t) S_{t-1} + k_t v_t^T                               (Mamba2, lightning,
                                                                                                       RWKV-6)
                         'dplr'     S_t = (diag(a_t) + b_t a'_t^T) S_{t-1} + k_t v_t^T, with ``a``/``b``
                                    the extra rank-1 terms (RWKV-7; fla's a, b arguments)
               a_t = exp(log_decay_t) (broadcast over d_k for a scalar decay).

Family adapter (subclass ``Family``)
    layer_names / layer_suffixes   which modules are this family's mixers (matched on the class name)
    ops                             the ``OpSpec`` of every op the mixers call; looked up in the module that defines the
                                    layer class (``op_module``), or bound per instance (``op_attrs``, old transformers)
    dims(layer)                     (d_k, d_v, value heads, key heads) for memory accounting
    state_in(S, kw) / state_out(Sk, S, kw)
                                    native state <-> canonical [B, H, d_k, d_v] float32 (transpose views as needed)
    log_decay(kw)                   per-token log retention from the op's arguments
    inputs(kw)                      the canonical per-token dict above (q, k, v, beta, log_decay, rule)
    buffer_args                     which op arguments the schedule's decode buffer rounds ('k' -> key side, 'v' ->
                                    value side)
    call(op, orig, ...)             run the original op on one piece from state S, returning (o, S_end)
    patch_layer(layer)              optional: replace the layer's forward for families whose state update is inline
                                    torch (MiniMax lightning); returns an undo callable

Op kinds
    'scan'   o, S_T = op(..., <state_arg>=S_0, <final_arg>=True)  over a [B, T, ...] piece; the carry splits it at
             the schedule's rounding positions and replaces S at every boundary
    'step'   o = op(<state_arg>=S, ...): one token, S updated IN PLACE (Mamba2's selective_state_update); a boundary
             re-rounds S in place

Rounding points: positions count per layer along the whole sequence (a scan op called without a state starts a new
sequence), the state is rounded after the update of every ``period``-th token, period = ``schedule.prefill`` in
multi-token calls and ``schedule.decode`` in one-token calls. At a boundary the method receives the canonical state
after the piece's updates together with a ``Piece``: the op's arguments for the piece (``dict``), ``.state_before``
(the canonical state the piece started from: the previous rounded state for a prompt piece, the previous token's
state for a decode step; None at a sequence start), ``.position`` (tokens processed, [B]), ``.log_decay()`` and
``.inputs()``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OpSpec:
    name: str                                    # attribute name in the op module (or on the layer instance)
    positional: tuple[str, ...]                  # canonical names of positional arguments, in order
    kind: str = 'scan'                           # 'scan' | 'step' (see module docstring)
    state_arg: str = 'initial_state'
    final_arg: str | None = 'output_final_state'
    seq: tuple[str, ...] | None = None           # arguments with a time axis at dim 1; None: every tensor argument
                                                 # of dim >= 2 whose dim 1 is T (the original GDN rule)
    always_tuple: bool = True                    # the op returns (o, S) even when no final state was asked for


class Piece(dict):
    """The op arguments of one piece (time-sliced, decode-buffer rounded) plus the family's view of them.

    For stateful methods (``statequant.methods_ext``), which see every piece, also: ``hit`` ([B] bool, the schedule
    rounds here), ``start`` (the piece starts a new sequence), ``prompt_end`` (last piece of a multi-token call with
    ``Schedule(prompt_end=True)``)."""

    def __init__(self, kw, family, layer, state_before=None, position=None, hit=None, start=False, prompt_end=False):
        super().__init__(kw)
        self.family, self.layer, self._before = family, layer, state_before
        self.position = position          # [B] tokens of the sequence processed after this piece (the boundary)
        self.hit, self.start, self.prompt_end = hit, start, prompt_end

    def log_decay(self):
        return self.family.log_decay(self)

    def inputs(self):
        return self.family.inputs(self)

    @property
    def state_before(self):
        """Canonical [B, H, d_k, d_v] float32 state before this piece's updates (None at a sequence start = zeros)."""
        if self._before is None:
            return None
        return self.family.state_in(self._before, self)


def _get(layer, *names):
    return next((getattr(layer, n) for n in names if isinstance(getattr(layer, n, None), int)), None)


def _rep(x, h):
    """Repeat a [B, T, G, d] tensor's head axis to h heads (GQA / grouped B, C)."""
    return x if x.shape[2] == h else x.repeat_interleave(h // x.shape[2], 2)


class Family:
    name = 'family'
    layer_names: tuple[str, ...] = ()
    layer_suffixes: tuple[str, ...] = ()
    ops: tuple[OpSpec, ...] = ()
    op_attrs: tuple[str, ...] = ()
    buffer_args = {'k': 'k', 'v': 'v'}
    decay = 'head'                                 # 'head' | 'channel' (per key channel)
    rule = 'delta'

    def matches(self, layer) -> bool:
        n = type(layer).__name__
        return n in self.layer_names or (bool(self.layer_suffixes) and n.endswith(self.layer_suffixes))

    def op(self, name) -> OpSpec:
        return next(o for o in self.ops if o.name == name)

    def op_module(self, layer):
        return sys.modules[type(layer).__module__]

    def patch_layer(self, layer):
        return None

    def dims(self, layer):
        d_k, d_v = _get(layer, 'head_k_dim', 'head_dim'), _get(layer, 'head_v_dim', 'head_dim')
        h_v = _get(layer, 'num_v_heads', 'num_heads')
        h_k = _get(layer, 'num_k_heads', 'num_heads') or h_v
        assert d_k and d_v and h_v, f'cannot read state dimensions from {type(layer).__name__}'
        return d_k, d_v, h_v, h_k

    # -- state layout --------------------------------------------------------------------------------------------
    def state_in(self, S, kw):
        """native -> canonical [B, H, d_k, d_v] float32"""
        return S.float().transpose(-1, -2) if kw.get('state_v_first') else S.float()

    def state_out(self, Sk, S, kw):
        """canonical -> the native layout and dtype of S"""
        return (Sk.transpose(-1, -2) if kw.get('state_v_first') else Sk).to(S.dtype)

    # -- per-token quantities ------------------------------------------------------------------------------------
    def log_decay(self, kw):
        raise NotImplementedError(f'{self.name}: log decay')

    def inputs(self, kw):
        raise NotImplementedError(f'{self.name}: canonical inputs')

    # -- the op ----------------------------------------------------------------------------------------------------
    def seq_len(self, op, kw):
        if op.kind == 'step':
            return 1
        return kw[op.seq[0] if op.seq else op.positional[0]].shape[1]

    def batch(self, op, kw):
        return kw[op.state_arg].shape[0] if op.kind == 'step' else kw[op.seq[0] if op.seq else op.positional[0]].shape[0]

    def seq_args(self, op, kw, T):
        if op.seq is None:
            return [n for n, x in kw.items() if torch.is_tensor(x) and x.dim() >= 2 and x.shape[1] == T]
        return [n for n in op.seq if torch.is_tensor(kw.get(n))]

    @staticmethod
    def _invoke(op, orig, kw, npos):
        """Call ``orig`` with the first ``npos`` arguments positionally (as the model did: transformers' torch
        fallbacks name them query/key/value, fla q/k/v) and the rest by keyword."""
        if not npos:
            return orig(**kw)
        kw = dict(kw)
        return orig(*[kw.pop(n) for n in op.positional[:npos]], **kw)

    def call(self, op, orig, piece, keep, S, npos, raw=None):
        """Run the op on ``piece`` from state ``S`` -> (o, S_end) (a 'step' op updates S in place and S_end is S).
        ``raw``: the piece before decode-buffer rounding (None when nothing was rounded)."""
        kw = {**piece, **keep, op.state_arg: S}
        if op.final_arg:
            kw[op.final_arg] = True
        out = self._invoke(op, orig, kw, npos)
        return (out, S) if op.kind == 'step' else out

    def result(self, op, o, S, asked):
        return (o, S) if (op.always_tuple or asked) else o


# ------------------------------------------------------------------------------------------------------------------
# G1  Gated DeltaNet: scalar decay per head, delta rule
# ------------------------------------------------------------------------------------------------------------------
_GDN_POSITIONAL = ('q', 'k', 'v', 'g', 'beta', 'scale', 'initial_state', 'output_final_state')


def gdn_log_decay(kw) -> torch.Tensor | None:
    """Per-step log retention log a_t [B, T, H] from the op arguments (the gate is fused in fla's kernel)."""
    g = kw.get('g')
    if g is None or not torch.is_tensor(g):
        return None
    A_log = kw.get('A_log')
    if kw.get('use_gate_in_kernel') and torch.is_tensor(A_log):
        dt = kw.get('dt_bias')
        z = g.float() + (dt.float() if torch.is_tensor(dt) else 0.)
        return -A_log.float().exp() * torch.nn.functional.softplus(z)
    return g.float()


def delta_inputs(kw, log_decay_fn):
    """q, k (l2-normalised as the kernel does, q scaled), log decay, beta per token from a delta-rule op's arguments;
    GQA keys/queries repeated to the value heads."""
    q, k, v, beta = kw['q'].float(), kw['k'].float(), kw['v'], kw['beta'].float()
    if kw.get('use_qk_l2norm_in_kernel'):
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    sc = kw.get('scale')
    q = q * (q.shape[-1] ** -0.5 if sc is None else sc)
    if q.shape[2] < v.shape[2]:
        rep = v.shape[2] // q.shape[2]
        q, k = q.repeat_interleave(rep, 2), k.repeat_interleave(rep, 2)
    if kw.get('use_beta_sigmoid_in_kernel'):
        beta = beta.sigmoid()
    if kw.get('allow_neg_eigval'):
        beta = beta * 2
    return q, k, log_decay_fn(kw), beta


class GatedDeltaNet(Family):
    """Qwen3.5 / Qwen3.5-MoE / Qwen3-Next / Qwen4-Exp / OLMo-Hybrid (transformers' ``torch_chunk_gated_delta_rule`` /
    ``torch_recurrent_gated_delta_rule``, which dispatch to fla's kernels when fla is installed) and fla's
    ``GatedDeltaNet`` (``chunk_gated_delta_rule`` / ``fused_recurrent_gated_delta_rule``, gate fused in the kernel,
    ``state_v_first``). State [B, H, d_k, d_v] (fla v-first: [B, H, d_v, d_k])."""
    name = 'gdn'
    layer_suffixes = ('GatedDeltaNet',)
    ops = tuple(OpSpec(n, _GDN_POSITIONAL) for n in (
        'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
        'chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule'))
    op_attrs = ('chunk_gated_delta_rule', 'recurrent_gated_delta_rule')   # transformers <= 5.9: bound per instance

    def op(self, name):
        return OpSpec(name, _GDN_POSITIONAL) if name in self.op_attrs else super().op(name)

    def log_decay(self, kw):
        return gdn_log_decay(kw)

    def inputs(self, kw):
        q, k, lg, beta = delta_inputs(kw, gdn_log_decay)
        return {'q': q, 'k': k, 'v': kw['v'].float(), 'beta': beta, 'log_decay': lg, 'rule': 'delta'}


# ------------------------------------------------------------------------------------------------------------------
# G2  KDA (Kimi Delta Attention): per-key-channel decay, delta rule
# ------------------------------------------------------------------------------------------------------------------
def kda_log_decay(kw) -> torch.Tensor | None:
    """[B, T, H, d_k]. With the gate fused in fla's kernel: -exp(A_log) * softplus(g + dt_bias), or with
    ``lower_bound``: lower_bound * sigmoid(exp(A_log) * (g + dt_bias)) (fla's ``kda_gate``)."""
    g = kw.get('g')
    if g is None or not torch.is_tensor(g):
        return None
    if not kw.get('use_gate_in_kernel'):
        return g.float()
    H, K = g.shape[-2:]
    z = g.float()
    dt = kw.get('dt_bias')
    if torch.is_tensor(dt):
        z = z + dt.float().view(H, -1)
    A_log = kw.get('A_log')
    a = A_log.float().exp().view(H, 1) if torch.is_tensor(A_log) else None
    lb = kw.get('lower_bound')
    if lb is not None:
        return lb * torch.sigmoid(a * z if a is not None else z)
    return -a * F.softplus(z)


class KDA(Family):
    """Kimi Linear (``KimiLinearDeltaAttention``), GLM-5-Next (``Glm5NextTextLinearAttention``), Bailing hybrid
    (``BailingHybridDeltaAttention``): transformers' ``chunk_kimi_delta_attention`` / ``recurrent_kimi_delta_attention``
    (dispatching to fla's ``chunk_kda`` / ``fused_recurrent_kda`` when installed), gate computed in the layer; fla's
    ``KimiDeltaAttention`` (gate fused in the kernel, ``state_v_first``). The decay is per key channel."""
    name = 'kda'
    decay = 'channel'
    layer_names = ('KimiLinearDeltaAttention', 'Glm5NextTextLinearAttention', 'BailingHybridDeltaAttention',
                   'KimiDeltaAttention')
    _seq = ('q', 'k', 'v', 'g', 'beta')
    ops = (OpSpec('chunk_kimi_delta_attention', _seq, seq=_seq),
           OpSpec('recurrent_kimi_delta_attention', _seq, seq=_seq),
           OpSpec('chunk_kda', _seq, seq=_seq),
           OpSpec('fused_recurrent_kda', _seq, seq=_seq))

    def log_decay(self, kw):
        return kda_log_decay(kw)

    def inputs(self, kw):
        q, k, lg, beta = delta_inputs(kw, kda_log_decay)
        return {'q': q, 'k': k, 'v': kw['v'].float(), 'beta': beta, 'log_decay': lg, 'rule': 'delta'}


# ------------------------------------------------------------------------------------------------------------------
# G3  Mamba2 / SSD: scalar decay per head, additive (no delta rule)
# ------------------------------------------------------------------------------------------------------------------
class Mamba2(Family):
    """Mamba-2 mixers of transformers (``Mamba2Mixer``, ``NemotronHMamba2Mixer``, ``FalconH1Mixer``,
    ``GraniteMoeHybridMambaLayer``, ``BambaMixer``, ``Zamba2MambaMixer``): prefill ``mamba2_chunk_scan(x, dt, A, B,
    C, chunk_size, D, ..., initial_states, return_final_states)``, decode ``mamba2_selective_state_update(state, x,
    dt, A, B, C, D, ...)`` which updates the state in place. Native state [B, H, P, N] (P = head_dim, N = d_state);
    canonical d_k = N (the B / C side), d_v = P (the x side). log a_t = A * clamp(softplus(dt + dt_bias), dt_limit).
    The decode buffer rounds B (key side) and x (value side; dt stays exact); the D skip uses the unrounded x."""
    name = 'mamba2'
    rule = 'additive'
    layer_names = ('Mamba2Mixer', 'NemotronHMamba2Mixer', 'FalconH1Mixer', 'GraniteMoeHybridMambaLayer',
                   'BambaMixer', 'Zamba2MambaMixer')
    buffer_args = {'k': 'B', 'v': 'hidden_states'}
    ops = (OpSpec('mamba2_chunk_scan', ('hidden_states', 'dt', 'A', 'B', 'C', 'chunk_size', 'D'),
                  state_arg='initial_states', final_arg='return_final_states',
                  seq=('hidden_states', 'dt', 'B', 'C', 'z', 'seq_idx'), always_tuple=False),
           OpSpec('mamba2_selective_state_update', ('state', 'hidden_states', 'dt', 'A', 'B', 'C', 'D'),
                  kind='step', state_arg='state', final_arg=None))

    def dims(self, layer):
        P, N, H = layer.head_dim, layer.ssm_state_size, layer.num_heads
        return N, P, H, layer.n_groups

    def state_in(self, S, kw):
        return S.float().transpose(-1, -2)

    def state_out(self, Sk, S, kw):
        return Sk.transpose(-1, -2).to(S.dtype)

    @staticmethod
    def _dt(kw, step):
        dt = kw['dt'].float()
        bias = kw.get('dt_bias')
        if step:                                             # [B, H, P] expanded over P; bias [H, P]
            dt = dt[..., 0]
            bias = bias[..., 0] if torch.is_tensor(bias) and bias.dim() == 2 else bias
        if torch.is_tensor(bias):
            dt = dt + bias.float()
        if kw.get('dt_softplus'):
            dt = F.softplus(dt)
        lim = kw.get('dt_limit')
        if lim is not None and not step:
            dt = dt.clamp(min=lim[0], max=lim[1])
        return dt if not step else dt[:, None]               # [B, T, H]

    def log_decay(self, kw):
        step = 'state' in kw
        A = kw['A'].float()
        if step:                                             # [H, P, N] expanded
            A = A.reshape(A.shape[0], -1)[:, 0]
        return self._dt(kw, step) * A

    def inputs(self, kw):
        step = 'state' in kw
        x, B, C = kw['hidden_states'].float(), kw['B'].float(), kw['C'].float()
        if step:
            x, B, C = x[:, None], B[:, None], C[:, None]
        H = x.shape[2]
        dt = self._dt(kw, step)
        return {'q': _rep(C, H), 'k': _rep(B, H), 'v': x * dt[..., None], 'beta': None,
                'log_decay': self.log_decay(kw), 'rule': 'additive'}

    def call(self, op, orig, piece, keep, S, npos, raw=None):
        D = keep['D'] if 'D' in keep else piece.get('D')
        if raw is None or not torch.is_tensor(D):
            return super().call(op, orig, piece, keep, S, npos)
        # decode-buffer rounding of x: the skip connection D * x still reads the exact x
        piece, keep = ({**piece, 'D': None}, keep) if 'D' in piece else (piece, {**keep, 'D': None})
        o, S = super().call(op, orig, piece, keep, S, npos)
        return self.skip(o, D, raw['hidden_states']), S

    @staticmethod
    def skip(o, D, x):
        Dx = D.float() if D.dim() == 2 else D.float()[:, None]           # [H, P] or [H] -> [H, 1]
        return (o.float() + Dx * x.float()).to(o.dtype)

    def result(self, op, o, S, asked):
        return (o, S) if asked else o


# ------------------------------------------------------------------------------------------------------------------
# G5  Lightning attention (MiniMax): constant decay exp(-slope) per head, additive
# ------------------------------------------------------------------------------------------------------------------
def lightning_attention(q, k, v, slope_rate, block_size, query_decay, key_decay, diagonal_decay, initial_state=None,
                        output_final_state=True):
    """MiniMax's lightning attention as transformers computes it, on [B, T, H, d] inputs (time at dim 1), with an
    optional initial state [B, H, d_k, d_v]. Without a state: transformers' block algorithm from zeros; one token
    with a state: its recurrent step; several tokens with a state: the block algorithm from that state (transformers
    loops token by token there; equal up to float rounding). Returns (o [B, T, H, d_v], S_T)."""
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    B, H, T, d = v.shape
    S = initial_state
    out = []
    if S is not None and T == 1:
        ratio = torch.exp(-slope_rate)
        S = ratio * S + torch.matmul(k.transpose(-1, -2), v)
        out.append(torch.matmul(q, S))
    else:
        if S is None:
            S = torch.zeros(B, H, q.shape[-1], d, device=v.device, dtype=v.dtype)
        for i in range((T + block_size - 1) // block_size):
            s, e = i * block_size, min(i * block_size + block_size, T)
            n = e - s
            qb, kb, vb = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e]
            qd, kd, dd = query_decay[:, :n], key_decay[:, -n:], diagonal_decay[:, :, :n, :n]
            block_decay = torch.exp(-slope_rate * n)
            intra = torch.matmul(torch.matmul(qb, kb.transpose(-1, -2)) * dd, vb)
            inter = torch.matmul(qb * qd, S)
            out.append(inter + intra)
            S = S * block_decay + torch.matmul((kb * kd).transpose(-1, -2), vb)
    return torch.cat(out, dim=-2).transpose(1, 2), S


def _minimax_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    """``MiniMaxLightningAttention.forward`` (transformers 5.19 / 5.20) with the state update moved into the module
    level ``lightning_attention`` op (so the carry can wrap it)."""
    from transformers.models.minimax.modeling_minimax import apply_mask_to_padding_states
    batch_size, seq_len, hidden_size = hidden_states.shape
    qkv_states = self.act_fn(self.qkv_proj(hidden_states))
    qkv_states = apply_mask_to_padding_states(qkv_states, attention_mask)
    qkv_states = qkv_states.reshape(batch_size, seq_len, self.num_attention_heads, 3 * self.head_dim)
    query_states, key_states, value_states = torch.split(qkv_states, self.head_dim, dim=3)
    state = past_key_values.get_linear_cache(self.layer_idx) if past_key_values is not None else None
    attn_output, state = sys.modules[__name__].lightning_attention(
        query_states, key_states, value_states, slope_rate=self.slope_rate, block_size=self.block_size,
        query_decay=self.query_decay, key_decay=self.key_decay, diagonal_decay=self.diagonal_decay,
        initial_state=state, output_final_state=True)
    attn_output = attn_output.reshape(batch_size, seq_len, self.num_attention_heads * self.head_dim)
    attn_output = self.norm(attn_output)
    attn_output = F.sigmoid(self.output_gate(hidden_states)) * attn_output
    attn_output = self.out_proj(attn_output)
    if past_key_values is not None:
        past_key_values.set_linear_cache(self.layer_idx, state)
    return attn_output, state


class Lightning(Family):
    """MiniMax-Text-01 / MiniMax-M1 lightning attention (``MiniMaxLightningAttention``, inline torch in
    transformers): the layer's forward is replaced inside ``apply`` / ``calibrate`` by an equivalent one that calls
    ``statequant.families.lightning_attention``. State K^T V: [B, H, d_k, d_v] (key-first, canonical); log decay
    -slope_rate per head, constant in time. No beta, no normalisation, no scale."""
    name = 'lightning'
    rule = 'additive'
    layer_names = ('MiniMaxLightningAttention',)
    _pos = ('q', 'k', 'v')
    ops = (OpSpec('lightning_attention', _pos, seq=_pos),)

    def op_module(self, layer):
        return sys.modules[__name__]

    def patch_layer(self, layer):
        import types
        layer.forward = types.MethodType(_minimax_forward, layer)

        def undo():
            del layer.forward
        return undo

    def dims(self, layer):
        return layer.head_dim, layer.head_dim, layer.num_attention_heads, layer.num_attention_heads

    def log_decay(self, kw):
        q = kw['q']
        return (-kw['slope_rate'].float().reshape(1, 1, -1)).expand(q.shape[0], q.shape[1], -1)

    def inputs(self, kw):
        return {'q': kw['q'].float(), 'k': kw['k'].float(), 'v': kw['v'].float(), 'beta': None,
                'log_decay': self.log_decay(kw), 'rule': 'additive'}


# ------------------------------------------------------------------------------------------------------------------
# G6  RWKV-7 / RWKV-6 (fla only)
# ------------------------------------------------------------------------------------------------------------------
class RWKV7(Family):
    """fla's ``RWKV7Attention``: ``chunk_rwkv7(r, w, k, v, a, b, ...)`` / ``fused_mul_recurrent_rwkv7(r, w, k, v, kk,
    a, ...)``; state [B, H, K, V]; w is the log decay per key channel. The transition is diagonal plus rank one
    (``rule='dplr'``), which StepQuant's row impact does not cover."""
    name = 'rwkv7'
    decay = 'channel'
    rule = 'dplr'
    layer_names = ('RWKV7Attention',)
    ops = (OpSpec('chunk_rwkv7', ('r', 'w', 'k', 'v', 'a', 'b'), seq=('r', 'w', 'k', 'v', 'a', 'b')),
           OpSpec('fused_mul_recurrent_rwkv7', ('r', 'w', 'k', 'v', 'kk', 'a'),
                  seq=('r', 'w', 'k', 'v', 'kk', 'a')))

    def dims(self, layer):
        return layer.head_dim, layer.head_v_dim, layer.num_heads, layer.num_heads

    def log_decay(self, kw):
        return kw['w'].float()

    def inputs(self, kw):
        sc = kw.get('scale')
        r = kw['r'].float()
        r = r * (r.shape[-1] ** -0.5 if sc is None else sc)
        if 'kk' in kw:                                       # fused_mul_recurrent: a = -kk, b = kk * a
            a, b = -kw['kk'].float(), kw['kk'].float() * kw['a'].float()
        else:
            a, b = kw['a'].float(), kw['b'].float()
        return {'q': r, 'k': kw['k'].float(), 'v': kw['v'].float(), 'beta': None, 'log_decay': self.log_decay(kw),
                'a': a, 'b': b, 'rule': 'dplr'}


class RWKV6(Family):
    """fla's ``RWKV6Attention``: ``chunk_rwkv6`` / ``fused_recurrent_rwkv6(r, k, v, w, u, ...)``; state [B, H, K, V];
    w the log decay per key channel; the bonus u only enters the output."""
    name = 'rwkv6'
    decay = 'channel'
    rule = 'additive'
    layer_names = ('RWKV6Attention',)
    _seq = ('r', 'k', 'v', 'w')
    ops = (OpSpec('chunk_rwkv6', ('r', 'k', 'v', 'w', 'u'), seq=_seq),
           OpSpec('fused_recurrent_rwkv6', ('r', 'k', 'v', 'w', 'u'), seq=_seq))

    def dims(self, layer):
        return layer.head_k_dim, layer.head_v_dim, layer.num_heads, layer.num_heads

    def log_decay(self, kw):
        return kw['w'].float()

    def inputs(self, kw):
        sc = kw.get('scale')
        r = kw['r'].float()
        r = r * (r.shape[-1] ** -0.5 if sc is None else sc)
        return {'q': r, 'k': kw['k'].float(), 'v': kw['v'].float(), 'beta': None, 'log_decay': self.log_decay(kw),
                'rule': 'additive'}


FAMILIES: tuple[Family, ...] = (GatedDeltaNet(), KDA(), Mamba2(), Lightning(), RWKV7(), RWKV6())
BY_NAME = {f.name: f for f in FAMILIES}


def family_of(layer) -> Family | None:
    return next((f for f in FAMILIES if f.matches(layer)), None)


__all__ = ['FAMILIES', 'BY_NAME', 'Family', 'OpSpec', 'Piece', 'family_of', 'lightning_attention']
