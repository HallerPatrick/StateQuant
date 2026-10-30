"""The patch on fla's GatedDeltaNet (state_v_first layout, gate fused in the kernel) with a small random model: an
identity carry reproduces the unpatched model, prompt and cached decoding; quantizing changes it; calibration fits a
head allocation; positions count across calls. Needs a GPU and fla."""
import pytest
import torch

fla = pytest.importorskip('fla')
if not torch.cuda.is_available():
    pytest.skip('needs a GPU', allow_module_level=True)

import statequant as sq  # noqa: E402


@pytest.fixture(scope='module')
def model():
    from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    torch.manual_seed(0)
    cfg = GatedDeltaNetConfig(hidden_size=256, num_hidden_layers=2, num_heads=2, head_dim=128, expand_v=1,
                              vocab_size=512, fuse_cross_entropy=False)
    return GatedDeltaNetForCausalLM(cfg).cuda().eval()


def greedy(model, ids, n):
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        logits, pkv = [out.logits], out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        for _ in range(n):
            out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            logits.append(out.logits)
            nxt = out.logits[:, -1:].argmax(-1)
    return torch.cat(logits, 1)


def ids(T=600, B=2, seed=1):
    return torch.randint(0, 512, (B, T), generator=torch.Generator().manual_seed(seed)).cuda()


def test_layers_found(model):
    assert len(sq.find_layers(model)) == 2


def test_identity_carry_is_exact(model):
    x = ids()
    ref = greedy(model, x, 12)
    with sq.apply(model, sq.Identity(), sq.Schedule(prefill=256, decode=4, buffer_bits=0)):
        out = greedy(model, x, 12)
    assert (out - ref).abs().max().item() < 2e-2 * ref.abs().max().item()     # chunked vs one call: kernel rounding


def test_quantized_differs_and_restores(model):
    x = ids()
    ref = greedy(model, x, 12)
    with sq.apply(model, sq.rtn(2, prefill=64, decode=1)):
        q = greedy(model, x, 12)
    assert (q - ref).abs().max().item() > 1e-2
    assert torch.equal(greedy(model, x, 12), ref)                                 # patch removed on exit


def test_rounds_at_sequence_positions(model):
    """A prompt split into two calls rounds at the same places as one call."""
    x = ids(T=512, B=1)
    m = sq.Uniform(sq.Quantizer(3, 'k', 32))
    sch = sq.Schedule(prefill=128, decode=128, buffer_bits=0)
    with sq.apply(model, m, sch), torch.no_grad():
        one = model(input_ids=x, use_cache=True).logits
        out = model(input_ids=x[:, :200], use_cache=True)
        two = torch.cat([out.logits, model(input_ids=x[:, 200:], past_key_values=out.past_key_values,
                                           use_cache=True).logits], 1)
    assert (one - two).abs().max().item() < 2e-2 * one.abs().max().item()


def test_calibrate_headmix(model):
    q = sq.calibrate(model, sq.pace48(paper=False), ids(T=1024, B=2), period=256)
    assert set(q.masks) == {0, 1} and all(int(m.sum()) == 1 for m in q.masks.values())  # 2 heads, a quarter -> 1
    with sq.apply(model, q):
        greedy(model, ids(), 4)
    r = sq.report(model, q)
    assert r['layers'] == 2 and r['d_k'] == 128 and r['payload_bits_per_value'] == q.bits_per_value(128, 128)
