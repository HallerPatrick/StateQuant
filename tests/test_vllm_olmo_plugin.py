"""OLMo-Hybrid's GDN layer in the vLLM plugin, without an engine: the plugin's ``_forward_core`` (packed slot state,
statequant kernels) against vLLM's own ``OlmoHybridGatedDeltaNetAttention._forward_core`` (fp32 ssm state, fla
kernels) on the same layer weights, the same conv states and hand-built GDN metadata: a prefill of three
sequences, a mixed step (two decode rows + a continued prefill), then decode-only steps (with a padding row).
d_k 96, d_v 192, beta in (0, 2). Needs vLLM and a GPU; the end-to-end check is ``tests/vllm_packed_check.py
--model allenai/Olmo-Hybrid-7B``."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('vllm')
if not torch.cuda.is_available():
    pytest.skip('needs a GPU', allow_module_level=True)

import statequant as sq  # noqa: E402
from statequant.integrations import vllm_olmo  # noqa: E402
from statequant.kernels import SlotLayout  # noqa: E402

DEV = 'cuda'
H, DK, DV, KC = 4, 96, 192, 4
CONV = 2 * H * DK + H * DV
NSLOT = 8


def _layer(O, state, conv, w, A_log, dt_bias, lay=None):
    cls = O.OlmoHybridGatedDeltaNetAttention
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    obj.prefix, obj.tp_size, obj.activation, obj.allow_neg_eigval = 'model.layers.0.linear_attn', 1, 'silu', True
    obj.num_k_heads = obj.num_v_heads = H
    obj.head_k_dim, obj.head_v_dim, obj.key_dim, obj.value_dim = DK, DV, H * DK, H * DV
    obj.conv1d = SimpleNamespace(weight=w, bias=None)
    obj.A_log, obj.dt_bias = A_log, dt_bias
    obj.kv_cache = (conv, state)
    obj._sq_layout, obj._sq_views = lay, None
    return obj


def _md(dec_slots, pre_slots, pre_lens, pre_init):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    nd, npf = len(dec_slots), len(pre_slots)
    lens = [1] * nd + list(pre_lens)
    qsl = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    kw = dict(num_prefills=npf, num_prefill_tokens=sum(pre_lens), num_decodes=nd, num_decode_tokens=nd,
              num_spec_decodes=0, num_spec_decode_tokens=0, num_actual_tokens=int(qsl[-1]),
              non_spec_query_start_loc=qsl.to(DEV),
              non_spec_state_indices_tensor=torch.tensor(dec_slots + pre_slots, dtype=torch.int32, device=DEV))
    if npf:
        pq = qsl[nd:] - nd
        hist = torch.tensor([True] * nd + list(pre_init), device=DEV)          # decode rows have history
        nums, bp, tp = compute_causal_conv1d_metadata(qsl, device=DEV)
        kw.update(has_initial_state=hist, prefill_query_start_loc=pq.to(DEV),
                  prefill_state_indices=torch.tensor(pre_slots, dtype=torch.int32, device=DEV),
                  prefill_has_initial_state=hist[nd:], nums_dict=nums, batch_ptr=bp, token_chunk_offset_ptr=tp)
    return GDNAttentionMetadata(**kw)


@pytest.mark.parametrize('fmt,tol', [('bf16', 2e-2), (sq.Quantizer(8, 'k', 32), 3e-2), (sq.Quantizer(4, 'k', 64), 0.25)],
                         ids=['bf16', 'int8k', 'pace4'])
def test_plugin_forward_core_matches_stock(monkeypatch, fmt, tol):
    from vllm.model_executor.layers.mamba.gdn import olmo_gdn_linear_attn as O
    stock_core = O.OlmoHybridGatedDeltaNetAttention.__dict__['_forward_core']
    if getattr(O.OlmoHybridGatedDeltaNetAttention, '_statequant_patched', False):
        stock_core = vllm_olmo._STATE['saved']['_forward_core']
    monkeypatch.setattr(O, 'is_conv_state_dim_first', lambda: True)
    monkeypatch.setitem(vllm_olmo._STATE, 'O', O)
    g = torch.Generator(device=DEV).manual_seed(0)
    w = (torch.randn(CONV, 1, KC, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    A_log = (torch.rand(H, device=DEV, generator=g) * 2 - 3)
    dt_bias = torch.randn(H, device=DEV, generator=g) * 0.5
    lay = SlotLayout(fmt, H, H, DK, DV, m=8, buffer_bits=8)
    conv_s = torch.zeros(NSLOT, CONV, KC - 1, device=DEV, dtype=torch.bfloat16)
    conv_p = conv_s.clone()
    ssm = torch.zeros(NSLOT, H, DV, DK, device=DEV, dtype=torch.float32)
    blob = lay.alloc(NSLOT)
    ref = _layer(O, ssm, conv_s, w, A_log, dt_bias)
    pk = _layer(O, blob, conv_p, w, A_log, dt_bias, lay)
    md_box = {}
    monkeypatch.setattr(O, 'get_forward_context', lambda: SimpleNamespace(attn_metadata=md_box))

    def step(md, T):
        md_box.clear()
        md_box[ref.prefix] = md
        x = torch.randn(T, CONV, device=DEV, generator=g).to(torch.bfloat16)
        a = torch.randn(T, H, device=DEV, generator=g).to(torch.bfloat16)
        b = torch.randn(T, H, device=DEV, generator=g).to(torch.bfloat16)
        o_s = torch.zeros(T, H, DV, device=DEV, dtype=torch.bfloat16)
        o_p = torch.zeros(T, H, DV, device=DEV, dtype=torch.bfloat16)
        stock_core(ref, x.clone(), b, a, o_s)                # (the decode conv writes its output into x)
        vllm_olmo._forward_core(pk, x.clone(), b, a, o_p)
        return o_s.float(), o_p.float()

    errs = []
    # prefill of three sequences (no history), slots 3, 1, 6
    o_s, o_p = step(_md([], [3, 1, 6], [5, 17, 33], [False, False, False]), 55)
    errs.append(((o_s - o_p).norm() / o_s.norm()).item())
    # mixed: decode rows for slots 3 and 1, slot 6 continues its prompt with 9 tokens
    o_s, o_p = step(_md([3, 1], [6], [9], [True]), 11)
    errs.append(((o_s - o_p).norm() / o_s.norm()).item())
    # decode-only steps across a fold (m = 8)
    for t in range(12):
        o_s, o_p = step(_md([3, 1, 6], [], [], []), 3)
        errs.append(((o_s - o_p).norm() / o_s.norm()).item())
    print(f'\n[{lay}] rel L2 |o_plugin - o_stock| per step: ' + ' '.join(f'{e:.1e}' for e in errs))
    assert max(errs) < tol, errs
    assert torch.equal(conv_s, conv_p)                       # the conv state is vLLM's, untouched
    # the slot state is the stock fp32 state up to the format
    S = lay.read(lay.views(blob), torch.tensor([3, 1, 6], device=DEV))
    rel = ((S - ssm[[3, 1, 6]]).norm() / ssm[[3, 1, 6]].norm()).item()
    print(f'stored state vs stock: rel L2 {rel:.1e}')
    assert rel < tol


def test_decode_padding_row(monkeypatch):
    """vLLM pads a captured decode batch with slot 0 (NULL block) / -1: no state touched, output row zero."""
    from vllm.model_executor.layers.mamba.gdn import olmo_gdn_linear_attn as O
    monkeypatch.setattr(O, 'is_conv_state_dim_first', lambda: True)
    monkeypatch.setitem(vllm_olmo._STATE, 'O', O)
    lay = SlotLayout(sq.Quantizer(4, 'k', 64), H, H, DK, DV, m=8)
    blob = lay.alloc(NSLOT)
    blob[0] = 0
    w = torch.randn(CONV, 1, KC, device=DEV).to(torch.bfloat16)
    pk = _layer(O, blob, torch.zeros(NSLOT, CONV, KC - 1, device=DEV, dtype=torch.bfloat16), w,
                torch.zeros(H, device=DEV), torch.zeros(H, device=DEV), lay)
    md = _md([2, 0], [], [], [])
    monkeypatch.setattr(O, 'get_forward_context', lambda: SimpleNamespace(attn_metadata={pk.prefix: md}))
    before = blob.clone()
    out = torch.full((2, H, DV), 7., device=DEV, dtype=torch.bfloat16)
    vllm_olmo._forward_core(pk, torch.randn(2, CONV, device=DEV).to(torch.bfloat16),
                            torch.randn(2, H, device=DEV).to(torch.bfloat16),
                            torch.randn(2, H, device=DEV).to(torch.bfloat16), out)
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.equal(blob[0], before[0]) and not torch.equal(blob[2], before[2])
