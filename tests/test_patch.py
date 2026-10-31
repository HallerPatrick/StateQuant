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


class _Recorder(sq.StateQuantizer):
    """Keeps the state at every rounding boundary and returns it unchanged (no rounding flips to amplify kernel noise)."""

    def __init__(self):
        self.seen = []

    def quantize(self, S, layer):
        self.seen.append((layer, S.clone()))
        return S

    def bits_per_value(self, d_k, d_v):
        return 32.


def test_rounds_at_sequence_positions(model):
    """A prompt split into two calls rounds at the same places as one call: the same boundaries see the same states,
    and a quantized run gives the same logits up to kernel rounding."""
    x = ids(T=512, B=1)
    sch = sq.Schedule(prefill=128, decode=128, buffer_bits=0)
    one, two = _Recorder(), _Recorder()
    with sq.apply(model, one, sch), torch.no_grad():
        model(input_ids=x, use_cache=True)
    with sq.apply(model, two, sch), torch.no_grad():
        out = model(input_ids=x[:, :200], use_cache=True)
        model(input_ids=x[:, 200:], past_key_values=out.past_key_values, use_cache=True)
    for layer in (0, 1):                                   # per layer: boundaries 128, 256, 384, 512 in order
        a_s = [S for l, S in one.seen if l == layer]
        b_s = [S for l, S in two.seen if l == layer]
        assert len(a_s) == len(b_s) == 4
        for a, b in zip(a_s, b_s):
            assert (a - b).norm() < 1e-2 * a.norm()
    # int8: rounding flips from chunk-boundary kernel noise stay small (int3 flips exceeded 2e-2 on H100 / fla 0.5)
    m = sq.Uniform(sq.Quantizer(8, 'k', 32))
    with sq.apply(model, m, sch), torch.no_grad():
        lo = model(input_ids=x, use_cache=True).logits
        out = model(input_ids=x[:, :200], use_cache=True)
        lt = torch.cat([out.logits, model(input_ids=x[:, 200:], past_key_values=out.past_key_values,
                                          use_cache=True).logits], 1)
    assert (lo - lt).abs().max().item() < 2e-2 * lo.abs().max().item()


def test_calibrate_headmix(model):
    q = sq.calibrate(model, sq.pace48(paper=False), ids(T=1024, B=2), period=256)
    assert set(q.masks) == {0, 1} and all(int(m.sum()) == 1 for m in q.masks.values())  # 2 heads, a quarter -> 1
    with sq.apply(model, q):
        greedy(model, ids(), 4)
    r = sq.report(model, q)
    assert r['layers'] == 2 and r['d_k'] == 128 and r['payload_bits_per_value'] == q.bits_per_value(128, 128)


def _merge(caches):
    """Stack per-row fla caches (pure recurrent model: no padding needed) into one batch cache."""
    base = caches[0]
    for i in range(len(base)):
        st = base[i]
        st['recurrent_state'] = torch.cat([c[i]['recurrent_state'] for c in caches])
        st['conv_state'] = tuple(torch.cat(xs) for xs in zip(*[c[i]['conv_state'] for c in caches]))
    return base


def test_set_positions_after_merged_prefill(model):
    """Rows prefilled in separate calls and merged into one batch cache: with set_positions the batch decodes as each
    row alone does (rounding every 8 tokens at each row's own positions, int8 buffer); without, it does not."""
    lens, n = (300, 203), 20
    rows = [ids(T=t, B=1, seed=s) for t, s in zip(lens, (3, 4))]
    forced = ids(T=n, B=2, seed=5)
    sch = sq.Schedule(prefill=256, decode=8, buffer_bits=8)
    m = sq.Uniform(sq.Quantizer(4, 'k', 64))

    def decode(pkv, toks, setpos):
        out = []
        for t in range(n):
            if setpos and t == 0:
                sq.set_positions(model, setpos)
            o = model(input_ids=toks[:, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            out.append(o.logits)
        return torch.cat(out, 1)
    with sq.apply(model, m, sch), torch.no_grad():
        alone = torch.cat([decode(model(input_ids=r, use_cache=True).past_key_values, forced[b:b + 1], None)
                           for b, r in enumerate(rows)])
        merged = {}
        for setpos in (lens, None):
            pkv = _merge([model(input_ids=r, use_cache=True).past_key_values for r in rows])
            merged[setpos is not None] = decode(pkv, forced, setpos)
    tol = 1e-3 * alone.abs().max().item()
    assert (merged[True] - alone).abs().max().item() < tol
    assert (merged[False] - alone).abs().max().item() > tol
