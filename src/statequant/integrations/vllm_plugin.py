"""vLLM plugin: store every Qwen3.5 / Qwen3-Next (/ Qwen4-Exp) and OLMo-Hybrid GDN layer's recurrent state packed
(statequant's int4 / int8 / bf16 slot layout with a decode buffer) and decode it with statequant's Triton step / fold
kernels; OLMo-Hybrid's layer: ``vllm_olmo``; Mamba2 / SSD layers (``MambaMixer2``): ``vllm_mamba2``; KDA layers
(Ling-3.0): ``vllm_kda``; all enabled from here.

Activation (every engine / worker process loads ``vllm.general_plugins``; this one does nothing unless asked):

    STATEQUANT_VLLM=pace4|int4v|int8k|bf16   format of the stored state (or int<bits><k|v>g<group>, e.g. int8vg64)
                                             unset / off: stock vLLM, nothing is patched
    STATEQUANT_VLLM_M=8                      decode buffer length m (power of two)
    STATEQUANT_VLLM_BUFFER_BITS=8            buffered keys / delta values: 8 (int8 + fp16 scale) or 16 (bf16)
    STATEQUANT_VLLM_LAYERS=gdn,mamba2,kda    layer types to pack (default: all): ``gdn`` = Qwen3.5 / Qwen3-Next GDN
                                             (this module) and OLMo-Hybrid GDN (``vllm_olmo``), ``mamba2`` = every
                                             ``MambaMixer2`` (NemotronH, Falcon-H1, Granite-4.0-H, Zamba2, Mamba2;
                                             ``vllm_mamba2``), ``kda`` = Ling-3.0's ``BailingMoeV3KimiDeltaAttention``
                                             (``vllm_kda``)
    STATEQUANT_VLLM_FUSED=auto               decode step: 1 = one fused launch per layer, 0 = the five-kernel step,
                                             auto (default) = ``kernels.fold.use_fused``: fused, except value-axis
                                             formats above FUSED_MAX_BH_AXIS_V decode rows x value heads (decided
                                             per captured batch size)

What is patched (from here; the installed vLLM files are untouched):

  * ``MambaStateShapeCalculator.gated_delta_net_state_shape`` / ``MambaStateDtypeCalculator.gated_delta_net_state_dtype``:
    the layer's second state (stock: the ssm state [HV, d_v, d_k] in bf16) becomes one uint8 blob of
    ``SlotLayout.nbytes`` per slot. The model-level hooks (``get_mamba_state_shape_from_config``) and the layer's
    ``get_state_shape`` both go through these calculators, so the hybrid page-size alignment, the KV-cache spec, the
    memory planner and ``bind_kv_cache`` all see the packed size. The conv state is unchanged.
    These calculators are shared by every GDN model in vLLM, but only ``QwenGatedDeltaNetAttention`` and
    ``OlmoHybridGatedDeltaNetAttention`` get a packed forward: the patched calculators look at their caller and
    answer packed only for the Qwen3-Next / Qwen3.5 / Qwen4-Exp / OLMo-Hybrid model hooks and for those two layer
    classes; any other caller (MiniCPM-V 4.6, ...) raises ``NotImplementedError`` while the plugin is active, at
    model-config time, instead of handing it a uint8 blob it cannot use.
  * ``QwenGatedDeltaNetAttention._forward_core``: the conv step as in vLLM; decode rows run
    ``SlotLayout.step`` on their slots (one fused launch or the five-kernel step, see STATEQUANT_VLLM_FUSED;
    gating in the kernel, padding rows with slot <= 0 skipped, per-sequence buffer counters, fold of exactly the
    sequences whose buffer filled); prefill rows read their slot's state in
    torch (dequantize + pending buffer), run vLLM's chunk kernel from it, and write the final state back packed with
    an empty buffer. Decode-only steps are CUDA-graph capturable (FULL graphs).
  * ``QwenGatedDeltaNetAttention.__init__``: asserts no speculative decoding and no prefix caching.
  * ``OlmoHybridGatedDeltaNetAttention.__init__`` / ``._forward_core``: the same for OLMo-Hybrid (d_k 96, d_v 192,
    beta in (0, 2)); see ``vllm_olmo``.
  * ``GatedDeltaNetAttention.__init__`` (vLLM's GDN base class): raises ``NotImplementedError`` for every subclass other
    than these two, so an unsupported model fails when its layers are built, before any weights are loaded.
  * ``_warmup_prefill_kernels``: run with an fp32 dummy state (the stock warmup builds one in the state dtype).
"""
from __future__ import annotations

import os
import re

import torch

ENV = 'STATEQUANT_VLLM'
_PRESETS = {'pace4': (4, 'k', 64), 'int4k': (4, 'k', 64), 'int4v': (4, 'v', 32), 'int8k': (8, 'k', 32),
            'int8v': (8, 'v', 32)}
_STATE = {}


def config():
    """(fmt, m, buffer_bits, fused) from the environment, or None when the plugin is off."""
    mode = os.environ.get(ENV, '').strip().lower()
    if mode in ('', '0', 'off', 'none', 'stock'):
        return None
    from ..quantizer import Quantizer
    if mode == 'bf16':
        fmt = 'bf16'
    elif mode in _PRESETS:
        fmt = Quantizer(*_PRESETS[mode])
    else:
        m_ = re.fullmatch(r'int([48])([kv])g(\d+)', mode)
        assert m_, f'{ENV}={mode!r}: use pace4, int4v, int8k, int8v, bf16 or int<4|8><k|v>g<group>'
        fmt = Quantizer(int(m_.group(1)), m_.group(2), int(m_.group(3)))
    m = int(os.environ.get('STATEQUANT_VLLM_M', '8'))
    bb = int(os.environ.get('STATEQUANT_VLLM_BUFFER_BITS', '8'))
    fz = os.environ.get('STATEQUANT_VLLM_FUSED', 'auto').strip().lower()
    assert fz in ('auto', '', '0', '1'), f'STATEQUANT_VLLM_FUSED={fz!r}: use 0, 1 or auto'
    fused = None if fz in ('auto', '') else fz == '1'
    return fmt, m, bb, fused


def layout(hv, hk, dk, dv):
    from ..kernels.slots import SlotLayout
    fmt, m, bb, fused = _STATE['cfg']
    key = (hv, hk, dk, dv)
    if key not in _STATE.setdefault('layouts', {}):
        _STATE['layouts'][key] = SlotLayout(fmt, hv, hk, dk, dv, m=m, buffer_bits=bb, fused=fused)
    return _STATE['layouts'][key]


def register():
    """``vllm.general_plugins`` entry point. Idempotent; a no-op unless STATEQUANT_VLLM is set."""
    cfg = config()
    if cfg is None or _STATE.get('registered'):
        return
    _STATE['cfg'] = cfg
    kinds = layer_kinds()
    if 'gdn' in kinds:
        _patch_calculators()
        _patch_layer()
        from . import vllm_olmo
        vllm_olmo.patch(layout)
    if 'mamba2' in kinds:
        from . import vllm_mamba2
        vllm_mamba2.patch(cfg)
    if 'kda' in kinds:
        from . import vllm_kda
        vllm_kda.patch(cfg)
    _STATE['registered'] = True


LAYER_KINDS = ('gdn', 'mamba2', 'kda')


def layer_kinds():
    """The layer types to pack, from STATEQUANT_VLLM_LAYERS (comma separated; default: all of LAYER_KINDS)."""
    v = os.environ.get('STATEQUANT_VLLM_LAYERS', '').strip().lower()
    kinds = tuple(k.strip() for k in v.split(',') if k.strip()) if v else LAYER_KINDS
    bad = [k for k in kinds if k not in LAYER_KINDS]
    assert not bad, f'STATEQUANT_VLLM_LAYERS: unknown {bad}; use a subset of {",".join(LAYER_KINDS)}'
    return kinds


# ---------------------------------------------------------------------------------------------------------------
# state shape / dtype
# ---------------------------------------------------------------------------------------------------------------
# callers of the GDN state calculators that get the packed state: the model-level hooks of these modules ...
PACKED_MODEL_MODULES = ('vllm.model_executor.models.qwen3_next', 'vllm.model_executor.models.qwen3_5',
                        'vllm.models.qwen4_exp', 'vllm.model_executor.models.olmo_hybrid')
# ... and layers of these classes (``get_state_shape`` / the base class's ``get_state_dtype``)
PACKED_LAYER = ('vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn', 'QwenGatedDeltaNetAttention')
PACKED_LAYERS = (PACKED_LAYER, ('vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn',
                                'OlmoHybridGatedDeltaNetAttention'))


def _caller_supported(frame) -> tuple[bool, str]:
    """Is the caller of a GDN state calculator one whose layers run the packed forward? -> (ok, description)"""
    mod = frame.f_globals.get('__name__', '?')
    self = frame.f_locals.get('self')
    cls = frame.f_locals.get('cls')
    owner = type(self) if self is not None else cls
    who = f'{getattr(owner, "__qualname__", "?")} ({mod})' if owner is not None else mod
    if self is not None:                                  # a layer's get_state_shape / get_state_dtype
        ok = any((c.__module__, c.__name__) in PACKED_LAYERS for c in type(self).__mro__)
        return ok, who
    return any(mod == m or mod.startswith(m + '.') for m in PACKED_MODEL_MODULES), who


def _refuse(who):
    raise NotImplementedError(
        f'statequant vLLM plugin ({ENV}={os.environ.get(ENV)!r}) stores packed GDN states only for Qwen3-Next / '
        f'Qwen3.5 / Qwen4-Exp (QwenGatedDeltaNetAttention) and OLMo-Hybrid (OlmoHybridGatedDeltaNetAttention), not '
        f'for {who}. Unset {ENV} to run this model with stock vLLM.')


def _patch_calculators():
    import sys

    from vllm.model_executor.layers.mamba import mamba_utils as mu
    shape0 = mu.MambaStateShapeCalculator.gated_delta_net_state_shape
    dtype0 = mu.MambaStateDtypeCalculator.gated_delta_net_state_dtype
    _STATE['calculators'] = (shape0, dtype0)

    def gated_delta_net_state_shape(cls, tp_world_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                                    conv_kernel_size, num_spec=0):
        ok, who = _caller_supported(sys._getframe(1))
        if not ok:
            _refuse(who)
        assert num_spec == 0, 'statequant vLLM plugin: speculative decoding is not supported'
        conv_shape, _ = shape0(tp_world_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel_size,
                               num_spec)
        lay = layout(num_v_heads // tp_world_size, num_k_heads // tp_world_size, head_k_dim, head_v_dim)
        return conv_shape, (lay.nbytes,)

    def gated_delta_net_state_dtype(cls, model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype='auto'):
        ok, who = _caller_supported(sys._getframe(1))
        if not ok:
            _refuse(who)
        conv_dtype, _ = dtype0(model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype)
        return conv_dtype, torch.uint8

    mu.MambaStateShapeCalculator.gated_delta_net_state_shape = classmethod(gated_delta_net_state_shape)
    mu.MambaStateDtypeCalculator.gated_delta_net_state_dtype = classmethod(gated_delta_net_state_dtype)


# ---------------------------------------------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------------------------------------------
def _patch_layer():
    from vllm.model_executor.layers.mamba.gdn import base as gdn_base
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as Q
    cls = Q.QwenGatedDeltaNetAttention
    if getattr(cls, '_statequant_patched', False):
        return
    from vllm.model_executor.layers.mamba.gdn import olmo_gdn_linear_attn as O
    base = gdn_base.GatedDeltaNetAttention
    base_init0 = base.__init__
    _STATE['base_init'] = (base, base_init0)
    supported = (cls, O.OlmoHybridGatedDeltaNetAttention)

    def base_init(self, *args, **kw):                     # refuse other GDN layers at construction, before weights
        if not isinstance(self, supported):
            _refuse(f'{type(self).__qualname__} ({type(self).__module__})')
        base_init0(self, *args, **kw)
    base.__init__ = base_init
    init0, warm0 = cls.__init__, cls._warmup_prefill_kernels

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
            stock = lay.HV * lay.DV * lay.DK * 2
            Q.logger.info('statequant: GDN recurrent state stored as %s, %d B/slot per layer (stock bf16 state '
                          '%d B); decode via statequant Triton slot kernels', lay, lay.nbytes, stock)

    def _warmup_prefill_kernels(self, qkv_or_qkvz, v_dim):
        dt = type(self).get_state_dtype(self)
        self.get_state_dtype = lambda: (dt[0], torch.float32)
        try:
            return warm0(self, qkv_or_qkvz, v_dim)
        finally:
            del self.get_state_dtype

    cls.__init__ = __init__
    cls._warmup_prefill_kernels = _warmup_prefill_kernels
    cls._forward_core = _forward_core
    cls._statequant_patched = True
    _STATE['Q'] = Q


def _views(self):
    blob = self.kv_cache[1]
    key = (blob.data_ptr(), tuple(blob.shape), tuple(blob.stride()))
    if self._sq_views is None or self._sq_views[0] != key:
        self._sq_views = (key, self._sq_layout.views(blob))
    return self._sq_views[1]


def _forward_core(self, mixed_qkv, b, a, core_attn_out):
    """Replacement of ``QwenGatedDeltaNetAttention._forward_core`` (non-speculative path only)."""
    Q = _STATE['Q']
    md = Q.get_forward_context().attn_metadata
    md = md.get(self.prefix) if isinstance(md, dict) else None
    if md is None:                                           # profiling run
        self._warmup_prefill_kernels(mixed_qkv, 0)
        return
    assert md.spec_sequence_masks is None, 'statequant vLLM plugin: speculative decoding is not supported'
    lay = self._sq_layout
    views = _views(self)
    conv_state = self.kv_cache[0] if Q.is_conv_state_dim_first() else self.kv_cache[0].transpose(-1, -2)
    N = md.num_actual_tokens
    mixed_qkv, b, a = mixed_qkv[:N], b[:N], a[:N]
    w = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
    nd, npf = md.num_decodes, md.num_prefills
    idx_all = md.non_spec_state_indices_tensor

    # 1. causal conv (as vLLM)
    if npf > 0:
        x = Q.causal_conv1d_fn(mixed_qkv.transpose(0, 1), w, self.conv1d.bias, activation=self.activation,
                               conv_states=conv_state, has_initial_state=md.has_initial_state,
                               cache_indices=idx_all, query_start_loc=md.non_spec_query_start_loc,
                               metadata=md).transpose(0, 1)
    elif nd > 0:
        x = Q.causal_conv1d_update(mixed_qkv, conv_state, w, self.conv1d.bias, self.activation,
                                   conv_state_indices=idx_all[:N], validate_data=False)
    else:
        return

    kd = lay.HK * lay.DK
    # 2a. decode rows: the packed step on their slots
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
        lay.step(views, idx, q, k, v, core_attn_out[:rows], self.head_k_dim ** -0.5,
                 gate=(a[:rows], b[:rows], self.A_log, self.dt_bias))

    # 2b. prefill rows: initial state from the packed slot (+ pending buffer), vLLM's chunk kernel, store packed
    if npf > 0:
        q, k, v, g, beta = Q.fused_post_conv_prep(
            conv_output=x[nd:], a=a[nd:], b=b[nd:], A_log=self.A_log, dt_bias=self.dt_bias,
            num_k_heads=lay.HK, head_k_dim=self.head_k_dim, head_v_dim=self.head_v_dim,
            apply_l2norm=True, output_g_exp=False)
        pidx = md.prefill_state_indices
        init = lay.read(views, pidx)
        init = torch.where(md.prefill_has_initial_state[:, None, None, None], init, 0.).contiguous()
        o, last = self.chunk_gated_delta_rule(
            q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0), g=g.unsqueeze(0), beta=beta.unsqueeze(0),
            initial_state=init, output_final_state=True, cu_seqlens=md.prefill_query_start_loc,
            chunk_indices=md.chunk_indices, chunk_offsets=md.chunk_offsets, use_qk_l2norm_in_kernel=False,
            aiter_prefill_metadata=md.aiter_prefill_metadata)
        lay.write(views, pidx, last)
        o = o.squeeze(0)
        core_attn_out[nd:nd + o.shape[0]] = o
