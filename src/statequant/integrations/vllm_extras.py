"""Small vLLM conveniences that are not state quantization (separate ``vllm.general_plugins`` entry point).

STATEQUANT_VLLM_QEMBED=1   build Qwen3.5 / Qwen3-Next ``embed_tokens`` with the checkpoint's quant config, so a
                           compressed-tensors checkpoint with int-quantized embeddings (WNA16) loads; vLLM 0.32
                           constructs these embeddings without a quant config.
STATEQUANT_VLLM_OLMO_CONV=1 (default; 0 = off) let OLMo-Hybrid's fused conv1d weight loader read the shard id that
                           vLLM 0.32's ``AutoWeightsLoader`` attaches to the tensor (``weight.shard_id``) instead of
                           passing it as an argument: without it, loading allenai/Olmo-Hybrid-7B fails in stock vLLM
                           0.32 (``TypeError: list indices must be integers or slices, not NoneType`` in
                           ``olmo_gdn_linear_attn._make_fused_conv1d_weight_loader``). Only acts where vLLM would
                           raise; stock and plugin runs both need it.
"""
import os


def register():
    if os.environ.get('STATEQUANT_VLLM_OLMO_CONV', '1') != '0':
        _olmo_conv_shard_id()
    if os.environ.get('STATEQUANT_VLLM_QEMBED', '0') != '1':
        return
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from vllm.model_executor.models import qwen3_5, qwen3_next
    for cls in (qwen3_5.Qwen3_5Model, qwen3_next.Qwen3NextModel):
        if getattr(cls, '_sq_qembed', False):
            continue
        init0 = cls.__init__

        def __init__(self, *args, _init0=init0, **kw):
            _init0(self, *args, **kw)
            if self.quant_config is not None:
                prefix = kw.get('prefix', '')
                self.embed_tokens = VocabParallelEmbedding(
                    self.vocab_size, self.config.hidden_size, quant_config=self.quant_config,
                    prefix=f'{prefix}.embed_tokens' if prefix else 'embed_tokens')
        cls.__init__ = __init__
        cls._sq_qembed = True


def _olmo_conv_shard_id():
    try:
        from vllm.model_executor.layers.mamba.gdn import olmo_gdn_linear_attn as O
    except ImportError:
        return
    make0 = O._make_fused_conv1d_weight_loader
    if getattr(make0, '_sq_shard_id', False):
        return

    def _make_fused_conv1d_weight_loader(dims, tp_size, tp_rank):
        load = make0(dims, tp_size, tp_rank)

        def weight_loader(param, loaded_weight, loaded_shard_id=None):
            if loaded_shard_id is None:
                loaded_shard_id = getattr(loaded_weight, 'shard_id', None)
            return load(param, loaded_weight, loaded_shard_id)
        return weight_loader
    _make_fused_conv1d_weight_loader._sq_shard_id = True
    O._make_fused_conv1d_weight_loader = _make_fused_conv1d_weight_loader
