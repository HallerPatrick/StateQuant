"""vLLM plugin, KDA part: keep every Ling-3.0 KDA layer's recurrent state packed (statequant's int4 / int8 / bf16 slot
layout with a decode buffer and per-key-channel decays, ``kernels.kda.KDASlotLayout``) and decode it with the KDA
Triton step.

Enabled by ``vllm_plugin.register`` when STATEQUANT_VLLM is set and STATEQUANT_VLLM_LAYERS includes ``kda`` (default:
all supported layer types). Layer: ``BailingMoeV3KimiDeltaAttention`` (vLLM's ``bailing_moe_v3``: Ling-3.0,
``BailingMoeV3ForCausalLM``). vLLM's other KDA layers (Kimi Linear's ``KimiGatedDeltaNetAttention``, GLM-5-Next,
Kimi-K3) are not packed: the state calculators answer them as stock vLLM does (and with ``gdn`` also enabled, the GDN
part refuses ``KimiGatedDeltaNetAttention`` at construction). Restrictions (asserted): tensor parallel 1, no
speculative decoding / MTP, no prefix caching (``mamba_cache_mode='none'``).

What is patched (installed vLLM files untouched):

  * ``MambaStateShapeCalculator.kda_state_shape`` / ``MambaStateDtypeCalculator.kda_state_dtype``: the recurrent
    state (stock: [H, d, d] in fp32, or bf16 with ``mamba_ssm_cache_dtype='bfloat16'``) becomes one uint8 blob of
    ``KDASlotLayout.nbytes`` per slot; the conv state is unchanged. Scoped: only calls from
    ``vllm.model_executor.models.bailing_moe_v3`` (its model-level hooks, used for the hybrid page-size alignment, and
    the layer's ``get_state_shape`` / ``get_state_dtype``) get the packed answer; every other caller gets vLLM's.
  * ``BailingMoeV3KimiDeltaAttention._forward`` (the body of the ``bailing_v3_kda_attention`` custom op): the q / k / v
    short convolutions as vLLM; decode rows run ``KDASlotLayout.step`` on their slots (q, k l2-normalised in the
    kernel, the gate from vLLM's ``fused_kda_gate`` (safe gate / lower bound applied there), padding rows with
    slot <= 0 skipped, per-sequence buffer counters, folds of exactly the sequences whose buffer filled; capturable
    in FULL CUDA graphs); prefill rows read their slot's state (dequantize + pending buffer, fp32), run vLLM's
    ``chunk_kda`` from it, and store the final state packed with an empty buffer. In a mixed batch the decode rows
    take the packed step (stock vLLM runs them through ``chunk_kda`` as length-1 sequences).
  * ``BailingMoeV3KimiDeltaAttention.__init__``: the asserts above.
"""
from __future__ import annotations

import sys

import torch

_CALLERS = frozenset({'vllm.model_executor.models.bailing_moe_v3'})
_STATE = {}


def _scoped(depth=2):
    """True if the calculator was called from vLLM's bailing_moe_v3 module (model hooks or the KDA layer)."""
    return sys._getframe(depth).f_globals.get('__name__') in _CALLERS


def layout(cfg, heads, head_dim):
    from ..kernels.kda import KDASlotLayout
    fmt, m, bb, _ = cfg
    key = (heads, head_dim)
    lays = _STATE.setdefault('layouts', {})
    if key not in lays:
        lays[key] = KDASlotLayout(fmt, heads, head_dim, head_dim, m=m, buffer_bits=bb)
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
    shape0 = mu.MambaStateShapeCalculator.kda_state_shape
    dtype0 = mu.MambaStateDtypeCalculator.kda_state_dtype

    def kda_state_shape(cls, tp_world_size, num_heads, head_dim, num_k_heads=None, head_k_dim=None,
                        conv_kernel_size=4, num_spec=0):
        conv_shape, rec_shape = shape0(tp_world_size, num_heads, head_dim, num_k_heads=num_k_heads,
                                       head_k_dim=head_k_dim, conv_kernel_size=conv_kernel_size, num_spec=num_spec)
        if not _scoped():
            return conv_shape, rec_shape
        assert tp_world_size == 1, 'statequant vLLM plugin (KDA): tensor parallel > 1 is not supported'
        assert num_spec == 0, 'statequant vLLM plugin (KDA): speculative decoding / MTP is not supported'
        assert num_k_heads in (None, num_heads) and head_k_dim in (None, head_dim), (num_k_heads, head_k_dim)
        H, dv, dk = rec_shape
        assert dv == dk == head_dim, rec_shape
        return conv_shape, (layout(cfg, H, head_dim).nbytes,)

    def kda_state_dtype(cls, model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype='auto'):
        conv_dtype, rec_dtype = dtype0(model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype)
        if not _scoped():
            return conv_dtype, rec_dtype
        return conv_dtype, torch.uint8

    mu.MambaStateShapeCalculator.kda_state_shape = classmethod(kda_state_shape)
    mu.MambaStateDtypeCalculator.kda_state_dtype = classmethod(kda_state_dtype)
    _STATE['calculators'] = (shape0, dtype0)


def _patch_layer(cfg):
    from vllm.model_executor.models import bailing_moe_v3 as M
    cls = M.BailingMoeV3KimiDeltaAttention
    if getattr(cls, '_statequant_patched', False):
        return
    init0 = cls.__init__

    def __init__(self, config, *args, **kw):
        init0(self, config, *args, **kw)
        vc = M.get_current_vllm_config()
        assert self.tp_size == 1, 'statequant vLLM plugin (KDA): tensor parallel > 1 is not supported'
        assert self.num_speculative_tokens == 0 and vc.speculative_config is None, \
            'statequant vLLM plugin (KDA): speculative decoding / MTP is not supported'
        assert not vc.cache_config.enable_prefix_caching, \
            'statequant vLLM plugin (KDA): prefix caching is unsupported (pass enable_prefix_caching=False)'
        assert vc.cache_config.mamba_cache_mode == 'none', vc.cache_config.mamba_cache_mode
        self._sq_layout = layout(cfg, self.local_num_heads, self.head_dim)
        self._sq_views = None
        if not _STATE.get('logged'):
            import statequant
            from vllm.logger import init_logger
            _STATE['logged'] = True
            lay = self._sq_layout
            init_logger('vllm.statequant').info(
                'statequant (%s): KDA recurrent state stored as %s; stock state %d B/slot per layer in fp32; '
                'decode via statequant Triton slot kernel', statequant.__file__, lay, lay.HV * lay.DV * lay.DK * 4)

    cls.__init__ = __init__
    cls._forward = _forward
    cls._statequant_patched = True
    _STATE['M'] = M


def _views(self):
    blob = self.kv_cache[1]
    key = (blob.data_ptr(), tuple(blob.shape), tuple(blob.stride()))
    if self._sq_views is None or self._sq_views[0] != key:
        self._sq_views = (key, self._sq_layout.views(blob))
    return self._sq_views[1]


def _forward(self, q_proj_states, k_proj_states, v_proj_states, g1, beta, core_attn_out):
    """Replacement of ``BailingMoeV3KimiDeltaAttention._forward`` (non-speculative path only)."""
    M = _STATE['M']
    md = M.get_forward_context().attn_metadata
    if md is None:
        return
    md = md.get(self.prefix) if isinstance(md, dict) else None
    if md is None:                                           # profile / warmup dummy runs (as vLLM)
        return
    assert md.spec_sequence_masks is None, 'statequant vLLM plugin (KDA): speculative decoding is not supported'
    lay = self._sq_layout
    views = _views(self)
    conv_state = self.kv_cache[0]
    if not M.is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    conv_q, conv_k, conv_v = conv_state.chunk(3, dim=-2)
    N = md.num_actual_tokens
    H, D = lay.HV, lay.DK
    qs, ks, vs = q_proj_states[:N], k_proj_states[:N], v_proj_states[:N]
    g1, beta = g1[0, :N], beta[0, :N]                         # [N, H, D] fp32, [N, H] fp32
    w = [c.weight.view(c.weight.size(0), c.weight.size(2)) for c in (self.q_conv1d, self.k_conv1d, self.v_conv1d)]
    bias = [c.bias for c in (self.q_conv1d, self.k_conv1d, self.v_conv1d)]
    nd, npf = md.num_decodes, md.num_prefills
    idx_all = md.non_spec_state_indices_tensor

    # 1. causal convs (as vLLM: one varlen conv over every row when there are prefills, else the update)
    if npf > 0:
        q, k, v = [M.causal_conv1d_fn(x.transpose(0, 1), wi, bi, activation='silu', conv_states=cs,
                                      has_initial_state=md.has_initial_state, cache_indices=idx_all,
                                      query_start_loc=md.non_spec_query_start_loc, metadata=md).transpose(0, 1)
                   for x, wi, bi, cs in zip((qs, ks, vs), w, bias, (conv_q, conv_k, conv_v))]
    elif nd > 0:
        q, k, v = [M.causal_conv1d_update(x, cs, wi, bi, activation='silu', conv_state_indices=idx_all[:N],
                                          validate_data=False)
                   for x, wi, bi, cs in zip((qs, ks, vs), w, bias, (conv_q, conv_k, conv_v))]
    else:
        return

    # 2a. decode rows: the packed step on their slots
    if nd > 0:
        rows = N if npf == 0 else nd                         # decode-only: N (vLLM's padded batch), padding -> slot 0
        idx = idx_all[:rows]
        if not idx.is_contiguous() or idx.dtype != torch.int32:
            idx = idx.to(torch.int32).contiguous()
        qd, kd, vd = [x[:rows] if x.stride(-1) == 1 else x[:rows].contiguous() for x in (q, k, v)]
        lay.step(views, idx, qd.unflatten(1, (H, D)), kd.unflatten(1, (H, D)), vd.unflatten(1, (H, D)),
                 g1[:rows], beta[:rows], core_attn_out[0, :rows], self.head_dim ** -0.5)

    # 2b. prefill rows: initial state from the packed slot (+ pending buffer), vLLM's chunk kernel, store packed
    if npf > 0:
        T = N - nd
        qp, kp, vp = [x[nd:N].reshape(1, T, H, D) for x in (q, k, v)]
        pidx = md.prefill_state_indices
        init = lay.read(views, pidx)
        init = torch.where(md.prefill_has_initial_state[:, None, None, None], init, 0.).contiguous()
        o, last = M.chunk_kda(q=qp, k=kp, v=vp, g=g1[nd:N].unsqueeze(0).contiguous(),
                              beta=beta[nd:N].unsqueeze(0).contiguous(), initial_state=init,
                              output_final_state=True, use_qk_l2norm_in_kernel=True, safe_gate=self.safe_gate,
                              lower_bound=self.lower_bound, cu_seqlens=md.prefill_query_start_loc)
        lay.write(views, pidx, last)
        core_attn_out[0, nd:N] = o[0]
