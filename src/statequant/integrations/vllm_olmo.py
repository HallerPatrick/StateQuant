"""vLLM plugin, OLMo-Hybrid part: keep every ``OlmoHybridGatedDeltaNetAttention`` layer's recurrent state packed
(statequant's int4 / int8 / bf16 slot layout, ``kernels.slots.SlotLayout``) and decode it with the GDN slot kernels.

Enabled by ``vllm_plugin.register`` together with the Qwen GDN layers (STATEQUANT_VLLM set, ``gdn`` in
STATEQUANT_VLLM_LAYERS). Restrictions (asserted): no speculative decoding, no prefix caching
(``mamba_cache_mode='none'``); TP is not tested.

How OLMo-Hybrid's GDN layer differs from Qwen's, and what that means here:

  * head dims d_k 96, d_v 192 (not powers of two): ``SlotLayout`` pads the key tiles inside the kernels and stores
    exactly d_k x d_v; PACE-4's key groups of 64 are ragged on d_k 96 (64 + 32: two fp16 scales per value row,
    4.33 b/value instead of 4.25).
  * separate projections (q, k, v, gate fused into ``in_proj_qkvg``; ``a_proj``, ``b_proj``), a conv without bias,
    no GQA (30 key and 30 value heads), the gate ``g_proj`` only enters the output norm: all outside the recurrence,
    untouched.
  * negative eigenvalues: beta = 2 sigmoid(b) (``allow_neg_eigval``). The decode rows take g and beta from vLLM's own
    ``fused_olmo_hybrid_gdn_gating`` (the same values the stock recurrent kernel reads) instead of the slot kernel's
    in-kernel gating (which is Qwen's beta = sigmoid(b)).
  * prefill: vLLM's ``chunk_gated_delta_rule`` with q, k normalised in the kernel, as stock, started from the
    dequantized slot (+ pending buffer) and stored back packed.

What is patched (installed vLLM files untouched): ``OlmoHybridGatedDeltaNetAttention.__init__`` (the asserts, the
layout) and ``._forward_core`` (below). The state shape / dtype calculators are ``vllm_plugin``'s, which answer
packed for OLMo-Hybrid's model hook and layer.
"""
from __future__ import annotations

import torch

_STATE = {}
LAYER = ('vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn', 'OlmoHybridGatedDeltaNetAttention')
MODEL_MODULE = 'vllm.model_executor.models.olmo_hybrid'


def patch(layout):
    """Patch the layer class; ``layout(hv, hk, dk, dv)`` is ``vllm_plugin.layout``. Idempotent."""
    from vllm.model_executor.layers.mamba.gdn import olmo_gdn_linear_attn as O
    cls = O.OlmoHybridGatedDeltaNetAttention
    if getattr(cls, '_statequant_patched', False):
        return
    init0 = cls.__init__
    _STATE['saved'] = {n: cls.__dict__.get(n) for n in ('__init__', '_forward_core')}

    def __init__(self, config, vllm_config, *args, **kw):
        init0(self, config, vllm_config, *args, **kw)
        assert vllm_config.speculative_config is None, 'statequant vLLM plugin: speculative decoding unsupported'
        assert not vllm_config.cache_config.enable_prefix_caching, \
            'statequant vLLM plugin: prefix caching is unsupported (pass enable_prefix_caching=False)'
        assert vllm_config.cache_config.mamba_cache_mode == 'none', vllm_config.cache_config.mamba_cache_mode
        self._sq_layout = layout(self.num_v_heads // self.tp_size, self.num_k_heads // self.tp_size,
                                 self.head_k_dim, self.head_v_dim)
        self._sq_views = None
        if not _STATE.get('logged'):
            _STATE['logged'] = True
            lay = self._sq_layout
            import logging
            logging.getLogger('vllm').info(
                'statequant: OLMo-Hybrid GDN recurrent state stored as %s, %d B/slot per layer (stock bf16 state %d '
                'B); decode via statequant Triton slot kernels', lay, lay.nbytes, lay.HV * lay.DV * lay.DK * 2)

    cls.__init__ = __init__
    cls._forward_core = _forward_core
    cls._statequant_patched = True
    _STATE['O'] = O


def unpatch():
    """Undo ``patch`` (tests)."""
    O = _STATE.get('O')
    if O is None or 'saved' not in _STATE:
        _STATE.clear()
        return
    cls = O.OlmoHybridGatedDeltaNetAttention
    for n, f in _STATE['saved'].items():
        if f is None:
            if n in cls.__dict__:
                delattr(cls, n)
        else:
            setattr(cls, n, f)
    cls._statequant_patched = False
    _STATE.clear()


def _views(self):
    blob = self.kv_cache[1]
    key = (blob.data_ptr(), tuple(blob.shape), tuple(blob.stride()))
    if self._sq_views is None or self._sq_views[0] != key:
        self._sq_views = (key, self._sq_layout.views(blob))
    return self._sq_views[1]


def _forward_core(self, mixed_qkv, b, a, core_attn_out):
    """Replacement of ``OlmoHybridGatedDeltaNetAttention._forward_core`` (non-speculative path only)."""
    O = _STATE['O']
    md = O.get_forward_context().attn_metadata
    md = md.get(self.prefix) if isinstance(md, dict) else None
    if md is None:                                           # profiling / warmup dummy runs
        return
    assert md.spec_sequence_masks is None, 'statequant vLLM plugin: speculative decoding is not supported'
    lay = self._sq_layout
    views = _views(self)
    conv_state = self.kv_cache[0] if O.is_conv_state_dim_first() else self.kv_cache[0].transpose(-1, -2)
    N = md.num_actual_tokens
    mixed_qkv, b, a = mixed_qkv[:N], b[:N], a[:N]
    w = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
    nd, npf = md.num_decodes, md.num_prefills
    idx_all = md.non_spec_state_indices_tensor

    # 1. causal conv (as vLLM; OLMo's conv has no bias)
    if npf > 0:
        x = O.causal_conv1d_fn(mixed_qkv.transpose(0, 1), w, None, activation=self.activation,
                               conv_states=conv_state, has_initial_state=md.has_initial_state,
                               cache_indices=idx_all, query_start_loc=md.non_spec_query_start_loc,
                               metadata=md).transpose(0, 1)
    elif nd > 0:
        x = O.causal_conv1d_update(mixed_qkv, conv_state, w, None, self.activation,
                                   conv_state_indices=idx_all[:N], validate_data=False)
    else:
        return

    # 2. gates as vLLM: g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b) (x 2 with negative eigenvalues)
    g, beta = O.fused_olmo_hybrid_gdn_gating(self.A_log, a, b, self.dt_bias, self.allow_neg_eigval)   # [1, N, HV]
    kd = lay.HK * lay.DK

    # 3a. decode rows: the packed step on their slots
    if nd > 0:
        rows = N if npf == 0 else nd                         # decode-only: N (vLLM's padded batch), padding -> slot 0
        xd = x[:rows]
        if xd.stride(-1) != 1:
            xd = xd.contiguous()
        idx = idx_all[:rows]
        if not idx.is_contiguous() or idx.dtype != torch.int32:
            idx = idx.to(torch.int32).contiguous()
        q = xd[:, :kd].unflatten(1, (lay.HK, lay.DK))
        k = xd[:, kd:2 * kd].unflatten(1, (lay.HK, lay.DK))
        v = xd[:, 2 * kd:].unflatten(1, (lay.HV, lay.DV))
        lay.step(views, idx, q, k, v, core_attn_out[:rows], self.head_k_dim ** -0.5, g=g[0, :rows],
                 beta=beta[0, :rows])

    # 3b. prefill rows: initial state from the packed slot (+ pending buffer), vLLM's chunk kernel, store packed
    if npf > 0:
        q, k, v = self.rearrange_mixed_qkv(x[nd:])
        pidx = md.prefill_state_indices
        init = lay.read(views, pidx)
        init = torch.where(md.prefill_has_initial_state[:, None, None, None], init, 0.).contiguous()
        o, last = O.chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g[:, nd:], beta=beta[:, nd:], initial_state=init, output_final_state=True,
            cu_seqlens=md.prefill_query_start_loc, chunk_indices=md.chunk_indices, chunk_offsets=md.chunk_offsets,
            use_qk_l2norm_in_kernel=True)
        lay.write(views, pidx, last)
        o = o.squeeze(0)
        core_attn_out[nd:nd + o.shape[0]] = o
