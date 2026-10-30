"""vLLM plugin, Mamba2 / SSD part: keep every ``MambaMixer2`` layer's ssm state packed (statequant's int4 / int8 / bf16
slot layout with a decode buffer, ``kernels.mamba2.Mamba2SlotLayout``) and decode it with the Mamba2 Triton step.

Enabled by ``vllm_plugin.register`` when STATEQUANT_VLLM is set and STATEQUANT_VLLM_LAYERS includes ``mamba2``
(default: all supported layer types). Models: everything built on ``MambaMixer2`` (NemotronH / Nemotron-3, Falcon-H1,
Granite-4.0-H, Zamba2, Mamba2 / Codestral-Mamba). Restrictions (asserted): tensor parallel 1, no speculative
decoding, no prefix caching (``mamba_cache_mode='none'``), no ReplaySSM.

What is patched (installed vLLM files untouched):

  * ``MambaStateShapeCalculator.mamba2_state_shape`` / ``MambaStateDtypeCalculator.mamba2_state_dtype``: the ssm
    state (stock: [H, head_dim, d_state] in fp32 or the model dtype) becomes one uint8 blob of
    ``Mamba2SlotLayout.nbytes`` per slot, the conv state is unchanged. Scoped: only calls from ``MambaMixer2`` itself
    and from the model classes built on it (``_CALLERS``, checked on the caller's module) get the packed shape; any
    other caller of the calculators gets vLLM's stock answer. The model-level hooks
    (``get_mamba_state_shape_from_config``, used for the hybrid page-size alignment) and the layer's
    ``get_state_shape`` both resolve here, so the memory planner sizes the cache from the packed blob.
  * ``MambaMixer2.conv_ssm_forward``: the conv as vLLM; decode rows run ``Mamba2SlotLayout.step`` on their slots (dt
    bias + softplus in the kernel, D x added, padding rows with slot <= 0 skipped, per-sequence buffer counters,
    folds of exactly the sequences whose buffer filled; capturable in FULL CUDA graphs); prefill rows read their
    slot's state (dequantize + pending buffer, fp32), run vLLM's ``mamba_chunk_scan_combined_varlen`` from it with an
    fp32 state, and store the final state packed with an empty buffer.
  * ``MambaMixer2.__init__``: the asserts above; ``_warmup_ssd_kernels``: run with an fp32 state dtype.
"""
from __future__ import annotations

import sys

import torch

_CALLERS = frozenset('vllm.model_executor.' + m for m in (
    'layers.mamba.mamba_mixer2', 'models.nemotron_h', 'models.granitemoehybrid', 'models.zamba2',
    'models.falcon_h1', 'models.mamba2'))
_STATE = {}


def _scoped(depth=2):
    """True if the calculator was called from MambaMixer2 or a model built on it."""
    return sys._getframe(depth).f_globals.get('__name__') in _CALLERS


def _fmt_for(fmt, dk, dv):
    """The configured format, with its group shrunk (by halving) to divide its axis if it does not (e.g. a value-axis
    group 32 on head_dim 80 -> 16)."""
    import dataclasses
    from ..kernels.fold import packed_format
    f = packed_format(fmt)
    if f is None:
        return fmt
    n = dk if f.axis == 'k' else dv
    g = f.group
    while g > 2 and n % g:
        g //= 2
    return fmt if g == f.group else dataclasses.replace(fmt, group=g)


def layout(cfg, heads, groups, d_state, head_dim):
    from ..kernels.mamba2 import Mamba2SlotLayout
    fmt, m, bb, _ = cfg
    key = (heads, groups, d_state, head_dim)
    lays = _STATE.setdefault('layouts', {})
    if key not in lays:
        lays[key] = Mamba2SlotLayout(_fmt_for(fmt, d_state, head_dim), heads, groups, d_state, head_dim, m=m,
                                     buffer_bits=bb)
    return lays[key]


def patch(cfg):
    if _STATE.get('patched'):
        return
    _STATE['cfg'] = cfg
    _patch_calculators(cfg)
    _patch_layer(cfg)
    _STATE['patched'] = True


def _patch_calculators(cfg):
    from vllm.model_executor.layers.mamba import mamba_utils as mu
    shape0 = mu.MambaStateShapeCalculator.mamba2_state_shape
    dtype0 = mu.MambaStateDtypeCalculator.mamba2_state_dtype

    def mamba2_state_shape(cls, tp_world_size, intermediate_size, n_groups, num_heads, head_dim, state_size,
                           conv_kernel, num_spec=0):
        conv_shape, ssm_shape = shape0(tp_world_size=tp_world_size, intermediate_size=intermediate_size,
                                       n_groups=n_groups, num_heads=num_heads, head_dim=head_dim,
                                       state_size=state_size, conv_kernel=conv_kernel, num_spec=num_spec)
        if not _scoped():
            return conv_shape, ssm_shape
        assert tp_world_size == 1, 'statequant vLLM plugin (Mamba2): tensor parallel > 1 is not supported'
        assert num_spec == 0, 'statequant vLLM plugin (Mamba2): speculative decoding is not supported'
        H, P, N = ssm_shape
        return conv_shape, (layout(cfg, H, n_groups, N, P).nbytes,)

    def mamba2_state_dtype(cls, model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype):
        conv_dtype, ssm_dtype = dtype0(model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype)
        if not _scoped():
            return conv_dtype, ssm_dtype
        return conv_dtype, torch.uint8

    mu.MambaStateShapeCalculator.mamba2_state_shape = classmethod(mamba2_state_shape)
    mu.MambaStateDtypeCalculator.mamba2_state_dtype = classmethod(mamba2_state_dtype)


def _patch_layer(cfg):
    from vllm.model_executor.layers.mamba import mamba_mixer2 as M
    cls = M.MambaMixer2
    if getattr(cls, '_statequant_patched', False):
        return
    init0, warm0 = cls.__init__, cls._warmup_ssd_kernels

    def __init__(self, *args, **kw):
        init0(self, *args, **kw)
        vc = M.get_current_vllm_config()
        assert self.tp_size == 1, 'statequant vLLM plugin (Mamba2): tensor parallel > 1 is not supported'
        assert self.num_spec == 0 and vc.speculative_config is None, \
            'statequant vLLM plugin (Mamba2): speculative decoding is not supported'
        assert not self.use_replayssm, 'statequant vLLM plugin (Mamba2): ReplaySSM (--use-replayssm) is not supported'
        assert not vc.cache_config.enable_prefix_caching, \
            'statequant vLLM plugin (Mamba2): prefix caching is unsupported (pass enable_prefix_caching=False)'
        assert vc.cache_config.mamba_cache_mode == 'none', vc.cache_config.mamba_cache_mode
        self._sq_layout = layout(cfg, self.num_heads, self.n_groups, self.ssm_state_size, self.head_dim)
        self._sq_views = None
        if not _STATE.get('logged'):
            import statequant
            _STATE['logged'] = True
            lay = self._sq_layout
            M.logger.info('statequant (%s): Mamba2 ssm state stored as %s; stock state %d B/slot per layer in %s; '
                          'decode via statequant Triton slot kernel', statequant.__file__, lay,
                          lay.HV * lay.DV * lay.DK * 4, vc.cache_config.mamba_ssm_cache_dtype)

    def _warmup_ssd_kernels(self, projected_states):
        dt = type(self).get_state_dtype(self)
        self.get_state_dtype = lambda: (dt[0], torch.float32)
        try:
            return warm0(self, projected_states)
        finally:
            del self.get_state_dtype

    cls.__init__ = __init__
    cls._warmup_ssd_kernels = _warmup_ssd_kernels
    cls.conv_ssm_forward = conv_ssm_forward
    cls._statequant_patched = True
    _STATE['M'] = M


def _views(self):
    blob = self.kv_cache[1]
    key = (blob.data_ptr(), tuple(blob.shape), tuple(blob.stride()))
    if self._sq_views is None or self._sq_views[0] != key:
        self._sq_views = (key, self._sq_layout.views(blob))
    return self._sq_views[1]


def conv_ssm_forward(self, projected_states, output):
    """Replacement of ``MambaMixer2.conv_ssm_forward`` (no ReplaySSM, no speculative decoding, no checkpoints)."""
    M = _STATE['M']
    hidden_states_B_C, dt = torch.split(projected_states[..., self.tped_intermediate_size:],
                                        [self.tped_conv_size, self.tped_dt_size], dim=-1)
    md = M.get_forward_context().attn_metadata
    md = md.get(self.prefix) if isinstance(md, dict) else None
    if md is None:                                           # profile run (as vLLM)
        self._warmup_ssd_kernels(projected_states)
        hidden_states_B_C = hidden_states_B_C.transpose(0, 1).clone().transpose(0, 1).contiguous()
        hidden_states, _B, _C = self.split_hidden_states_B_C_fn(hidden_states_B_C)
        return hidden_states
    assert md.checkpoint_chunk_idx is None, 'statequant vLLM plugin (Mamba2): prefill checkpoints are unsupported'
    lay = self._sq_layout
    views = _views(self)
    conv_state = self.kv_cache[0] if M.is_conv_state_dim_first() else self.kv_cache[0].transpose(-1, -2)
    H, G, P, N = lay.HV, lay.HK, lay.DV, lay.DK
    nd, npf = md.num_decode_tokens, md.num_prefill_tokens
    T = nd + npf
    hbc_d, hbc_p = torch.split(hidden_states_B_C[:T], [nd, npf], dim=0)
    dt_d, dt_p = torch.split(dt[:T], [nd, npf], dim=0)
    out_d, out_p = torch.split(output[:T], [nd, npf], dim=0)

    if md.num_prefills > 0:
        hbc_p = M.causal_conv1d_fn(
            hbc_p.transpose(0, 1), self.conv_weights, self.conv1d.bias, activation=self.activation,
            conv_states=conv_state, has_initial_state=md.has_initial_states_p,
            cache_indices=md.state_indices_tensor_p, metadata=md,
            query_start_loc=md.query_start_loc_p).transpose(0, 1)[:npf]
        x_p, B_p, C_p = self.split_hidden_states_B_C_fn(hbc_p)
        pidx = md.state_indices_tensor_p
        init = None
        if md.has_initial_states_p is not None and md.prep_initial_states:
            init = lay.read(views, pidx)
            init = torch.where(md.has_initial_states_p[:, None, None, None], init, 0.).contiguous()
        final = M.mamba_chunk_scan_combined_varlen(
            x_p.view(npf, H, P), dt_p, self.A, B_p.view(npf, G, -1), C_p.view(npf, G, -1),
            chunk_size=md.chunk_size, D=self.D, z=None, dt_bias=self.dt_bias, seq_idx=md.seq_idx_p,
            cu_seqlens=md.query_start_loc_p, cu_chunk_seqlens=md.cu_chunk_seqlen_p,
            last_chunk_indices=md.last_chunk_indices_p, initial_states=init, return_intermediate_states=False,
            dt_softplus=True, dt_limit=(0.0, float('inf')), out=out_p.view(npf, -1, P), state_dtype=torch.float32)
        lay.write(views, pidx, final)

    if md.num_decodes > 0:
        sidx = md.state_indices_tensor_d
        hbc_d = M.causal_conv1d_update(
            hbc_d, conv_state, self.conv_weights, self.conv1d.bias, self.activation, conv_state_indices=sidx,
            num_accepted_tokens=md.num_accepted_tokens, query_start_loc=md.query_start_loc_d,
            max_query_len=sidx.size(-1))
        x_d, B_d, C_d = self.split_hidden_states_B_C_fn(hbc_d)
        idx = (sidx[:, 0] if sidx.dim() == 2 else sidx)[:nd]
        if idx.dtype != torch.int32 or not idx.is_contiguous():
            idx = idx.to(torch.int32).contiguous()
        lay.step(views, idx, x_d.view(nd, H, P), dt_d, B_d.view(nd, G, N), C_d.view(nd, G, N),
                 out_d.view(nd, H, P), self.A, self.D, self.dt_bias, dt_softplus=True,
                 dt_limit=(0.0, float('inf')))
