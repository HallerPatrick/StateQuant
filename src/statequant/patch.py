"""Attach a state quantizer and schedule to a model (fake-quant reference path).

Every supported mixer belongs to a recurrent-state family (``statequant.families``: Gated DeltaNet, KDA, Mamba2,
MiniMax lightning, RWKV-7, RWKV-6) whose adapter names the ops the mixer calls, e.g. transformers'
``torch_chunk_gated_delta_rule`` / ``mamba2_chunk_scan`` / fla's ``chunk_kda``, of the form
    op(..., initial_state=S0, output_final_state=True) -> (o, S_T)          ('scan' ops)
    op(state=S, ...) -> o, S updated in place                               ('step' ops: Mamba2's decode)
We wrap those ops: a scan is split at the schedule's rounding positions, the state is carried across the pieces at full
precision and replaced by its quantized copy at every boundary; a step re-rounds the state in place when its position
hits a boundary. A forward pre-hook on each layer tells the op which layer is calling, so methods with a per-layer
allocation (HeadMix, ChannelProtect) work on models that share one op across layers.

The state handed to the method is always the family's canonical [B, H, d_k, d_v] float32 view (fla's
``state_v_first`` and Mamba2's [B, H, head_dim, d_state] are transposed in and out), with a ``families.Piece`` that
exposes the op arguments, the state before the piece, the per-token log decay and the canonical per-token inputs.
"""
from __future__ import annotations

import contextlib
import contextvars

import torch

from .families import BY_NAME, Piece, family_of, gdn_log_decay
from .methods import StateQuantizer, as_method
from .schedule import Schedule

_GDN = BY_NAME['gdn']
# backwards-compatible names (the Gated DeltaNet adapter's)
OP_NAMES = tuple(o.name for o in _GDN.ops)
OP_ATTRS = _GDN.op_attrs
LAYER_SUFFIXES = _GDN.layer_suffixes
_POSITIONAL = _GDN.ops[0].positional
log_decay = gdn_log_decay

_LAYER = contextvars.ContextVar('statequant_layer', default=None)


def find_layers(model) -> list[torch.nn.Module]:
    """The model's recurrent-state mixers (every supported family), in forward order."""
    return [m for m in model.modules() if family_of(m) is not None]


def layer_families(model) -> list[str]:
    """Family name of every mixer of ``find_layers(model)``."""
    return [family_of(m).name for m in find_layers(model)]


def _round_activation(x, bits):
    """Symmetric absmax rounding with one scale per (batch, position, head) over the head dim (the decode buffer);
    16 bits = a bf16 buffer."""
    if bits == 16:
        return x.to(torch.bfloat16).float()
    qmax = 2 ** (bits - 1) - 1
    sc = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    return torch.round(x / sc).clamp(-qmax, qmax) * sc


class _Carry:
    """The wrapper around one op. ``boundary(S, layer, piece)`` returns the state to carry on (canonical layout).
    With ``stateful`` (a method's ``piece(S, layer, piece)``, ``statequant.methods_ext``) EVERY piece is handed over,
    with ``piece.hit`` / ``.start`` / ``.prompt_end`` set, and its return value is carried."""

    def __init__(self, orig, schedule: Schedule, boundary, positions: dict, family=_GDN, op=None, stateful=None):
        self.orig, self.sch, self.boundary, self.pos = orig, schedule, boundary, positions
        self.fam, self.op = family, op or family.ops[0]
        self.stateful = stateful

    def _kwargs(self, args, kw):
        if args:
            assert len(args) <= len(self.op.positional), f'unexpected positional arguments: {len(args)}'
            kw = {**dict(zip(self.op.positional, args)), **kw}
        return kw

    def _round_buffer(self, piece, bits):
        for c in self.sch.buffer:
            n = self.fam.buffer_args[c]
            if torch.is_tensor(piece.get(n)):
                piece[n] = _round_activation(piece[n].float(), bits).to(piece[n].dtype)

    def __call__(self, *args, **kw):
        npos = len(args)
        kw = self._kwargs(args, kw)
        layer = _LAYER.get()
        assert layer is not None, f'{self.fam.name} op called outside a patched layer'
        assert kw.get('cu_seqlens') is None, 'variable-length packing (cu_seqlens) is not supported'
        if self.op.kind == 'step':
            return self._step(kw, npos, layer)
        fam, op = self.fam, self.op
        T, B = fam.seq_len(op, kw), fam.batch(op, kw)
        S = kw.pop(op.state_arg, None)
        asked = kw.pop(op.final_arg, None) if op.final_arg else None
        if S is None:                                    # a new sequence starts
            self.pos[layer] = torch.zeros(B, dtype=torch.long)
        pos = self.pos.get(layer)
        if pos is None or pos.numel() != B:
            pos = torch.zeros(B, dtype=torch.long)
        period = self.sch.decode if T == 1 else self.sch.prefill
        bb = self.sch.buffer_bits if (T == 1 or self.sch.buffer_in_prefill) else 0
        names = fam.seq_args(op, kw, T)
        seq = {n: kw[n] for n in names}
        keep = {n: x for n, x in kw.items() if n not in seq}
        outs, off, start = [], 0, S is None
        while off < T:
            step = min(T - off, int((period - pos % period).min()))
            piece = {n: x[:, off:off + step] for n, x in seq.items()}
            raw = None
            if bb:
                raw = dict(piece)
                self._round_buffer(piece, bb)
            S0 = S
            o, S = fam.call(op, self.orig, piece, keep, S, npos, raw)
            outs.append(o)
            off += step
            pos = pos + step
            hit = (pos % period == 0) & (pos > 0)
            end = bool(self.sch.prompt_end and T > 1 and off == T)
            if end:                                      # Schedule(prompt_end=True): also round at the prompt end
                hit = torch.ones_like(hit)
            if self.stateful is not None:                # statequant.methods_ext: the method sees every piece
                full = {**keep, **piece}
                pc = Piece(full, fam, layer, S0, pos, hit=hit, start=start, prompt_end=end)
                S = fam.state_out(self.stateful(fam.state_in(S, full), layer, pc), S, full)
                start = False
                continue
            start = False
            if bool(hit.any()):
                full = {**keep, **piece}
                Sk = fam.state_in(S, full)
                Sq = self.boundary(Sk, layer, Piece(full, fam, layer, S0, pos))
                Sq = fam.state_out(Sq, S, full)
                S = Sq if bool(hit.all()) else torch.where(hit.to(S.device).view(-1, 1, 1, 1), Sq, S)
        self.pos[layer] = pos
        return fam.result(op, torch.cat(outs, dim=1), S, asked)

    def _step(self, kw, npos, layer):
        """One decoded token through an op that updates the state in place."""
        fam, op = self.fam, self.op
        S = kw.pop(op.state_arg)
        B = S.shape[0]
        pos = self.pos.get(layer)
        if pos is None or pos.numel() != B:
            pos = torch.zeros(B, dtype=torch.long)
        period, bb = self.sch.decode, self.sch.buffer_bits
        raw = None
        if bb:
            raw = dict(kw)
            self._round_buffer(kw, bb)
        pos = pos + 1
        hit = (pos % period == 0) & (pos > 0)
        S0 = S.clone() if (bool(hit.any()) or self.stateful is not None) else None
        o, S = fam.call(op, self.orig, kw, {}, S, npos, raw)
        if self.stateful is not None:                    # statequant.methods_ext: the method sees every step
            full = {**kw, op.state_arg: S}
            pc = Piece(full, fam, layer, S0, pos, hit=hit, start=False, prompt_end=False)
            S.copy_(fam.state_out(self.stateful(fam.state_in(S, full), layer, pc), S, full))
            self.pos[layer] = pos
            return o
        if bool(hit.any()):
            full = {**kw, op.state_arg: S}
            Sq = fam.state_out(self.boundary(fam.state_in(S, full), layer, Piece(full, fam, layer, S0, pos)), S, full)
            S.copy_(Sq if bool(hit.all()) else torch.where(hit.to(S.device).view(-1, 1, 1, 1), Sq, S))
        self.pos[layer] = pos
        return o


class _PackedCarry(_Carry):
    """Deployment path: the prompt runs the reference carry (rounded every ``prefill`` tokens) and its final state is
    packed (so it is rounded at the end of the prompt too); every decode step then runs the Triton kernels of
    ``statequant.kernels.PackedGDNState`` on the packed state and its buffer of ``decode`` tokens. The model's own
    cache keeps an unused copy of the state; memory savings are realised by a serving engine, not here."""

    def __init__(self, orig, schedule, boundary, positions, fmt, states, family=_GDN, op=None):
        super().__init__(orig, schedule, boundary, positions, family, op)
        self.fmt, self.states = fmt, states

    def __call__(self, *args, **kw):
        from .kernels import PackedGDNState
        npos = len(args)
        kw = self._kwargs(args, kw)
        layer = _LAYER.get()
        q = kw['q']
        if q.shape[1] > 1 or kw.get('initial_state') is None:
            rest = dict(kw)
            o, S = super().__call__(*[rest.pop(n) for n in self.op.positional[:npos]], **rest)
            v_first = bool(kw.get('state_v_first', False))
            Sk = (S.float().transpose(-1, -2) if v_first else S.float()).contiguous()
            B, H = Sk.shape[:2]
            self.states[layer] = st = PackedGDNState(
                Sk, self.fmt, m=self.sch.decode, hv=H, hk=kw['k'].shape[2],
                buffer_bits=self.sch.buffer_bits or 16, out_dtype=kw['v'].dtype)
            Sd = st.dequantize().view(B, H, *Sk.shape[2:])
            return o, (Sd.transpose(-1, -2) if v_first else Sd).contiguous().to(S.dtype)
        assert kw.get('use_qk_l2norm_in_kernel', False), 'the packed kernels l2-normalise q and k'
        st = self.states[layer]
        k, v = kw['k'], kw['v']
        B, H, DV = v.shape[0], v.shape[2], v.shape[3]
        beta = kw['beta'].float()
        if kw.get('use_beta_sigmoid_in_kernel'):
            beta = beta.sigmoid()
        if kw.get('allow_neg_eigval'):
            beta = beta * 2
        sc = kw.get('scale') or q.shape[-1] ** -0.5
        o = st.step(q.reshape(-1, q.shape[-1]), k.reshape(-1, k.shape[-1]), v.reshape(B * H, DV),
                    log_decay(kw).reshape(B * H), beta.reshape(B * H), sc)
        self.pos[layer] = self.pos[layer] + 1
        return o.view(B, 1, H, DV).to(v.dtype), kw['initial_state']


@contextlib.contextmanager
def _patched(model, schedule: Schedule, boundary, carry=None):
    layers = find_layers(model)
    assert layers, f'no recurrent-state layers found in {type(model).__name__}'
    fams = [family_of(m) for m in layers]
    positions, saved, handles, undo = {}, [], [], []
    make = carry or _Carry
    try:
        bound = set()
        for i, (m, fam) in enumerate(zip(layers, fams)):
            for n in fam.op_attrs:                       # older transformers (<= 5.9) bind the ops per instance
                if callable(m.__dict__.get(n)):
                    saved.append((m, n, m.__dict__[n]))
                    setattr(m, n, make(m.__dict__[n], schedule, boundary, positions, fam, fam.op(n)))
                    bound.add(i)
        for m, fam in zip(layers, fams):
            u = fam.patch_layer(m)
            if u is not None:
                undo.append(u)
        mods = {}
        for i, (m, fam) in enumerate(zip(layers, fams)):
            if i not in bound:
                mod = fam.op_module(m)
                mods.setdefault((fam.name, mod.__name__), (fam, mod))
        for fam, mod in mods.values():
            ops = [op for op in fam.ops if callable(getattr(mod, op.name, None))]
            assert ops, f'{mod.__name__} defines none of {tuple(o.name for o in fam.ops)}'
            for op in ops:
                orig = getattr(mod, op.name)
                saved.append((mod, op.name, orig))
                setattr(mod, op.name, make(orig, schedule, boundary, positions, fam, op))

        def enter(i):
            def hook(_m, _a):
                _LAYER.set(i)                            # hooks must return None, or torch replaces the args
            return hook

        def leave(_m, _a, _o):
            _LAYER.set(None)
        for i, m in enumerate(layers):
            handles.append(m.register_forward_pre_hook(enter(i)))
            handles.append(m.register_forward_hook(leave))
        yield layers
    finally:
        for mod, n, orig in reversed(saved):
            setattr(mod, n, orig)
        for h in handles:
            h.remove()
        for u in undo:
            u()


@contextlib.contextmanager
def apply(model, method, schedule: Schedule | None = None, backend: str = 'reference'):
    """Run ``model`` with its recurrent state stored by ``method`` on ``schedule`` (default: the method's own).

        with statequant.apply(model, statequant.pace4()):
            model.generate(...)

    backend 'reference': fake quantization at the schedule's rounding positions (any method, every family; the
            evaluation path).
            'packed': the deployment path for ``Uniform`` symmetric int4/int8 formats on Gated DeltaNet layers - the
            decode runs the packed Triton kernels with a buffer of ``schedule.decode`` (key, delta value) pairs at
            ``buffer_bits``; the state is also rounded at the end of the prompt, since it has to be stored packed
            there.
    """
    method = as_method(method)
    if method.needs_calibration:
        assert getattr(method, 'masks', None), f'{method!r} needs statequant.calibrate(model, method, data) first'
    schedule = schedule or getattr(method, 'schedule', None) or Schedule()
    boundary = lambda S, layer, kw: method.quantize(S, layer)  # noqa: E731
    carry = None
    if getattr(method, 'stateful', False):              # statequant.methods_ext: the method sees every piece
        assert backend == 'reference', 'stateful methods run on the reference backend'
        if hasattr(method, 'reset'):
            method.reset()

        def carry(orig, sch, bnd, pos, fam, op):
            return _Carry(orig, sch, bnd, pos, fam, op, stateful=method.piece)
    elif backend == 'packed':
        from .methods import Uniform
        assert isinstance(method, Uniform), 'the packed backend supports Uniform formats only'
        assert schedule.buffer == 'kv' and not schedule.buffer_in_prefill, \
            "the packed buffer rounds keys and delta values of decoded tokens: use buffer='kv', buffer_in_prefill=False"
        other = sorted(set(layer_families(model)) - {'gdn'})
        if other:
            raise NotImplementedError(f'the packed backend supports Gated DeltaNet layers only, not {other}')
        states = {}

        def carry(orig, sch, bnd, pos, fam, op):
            return _PackedCarry(orig, sch, bnd, pos, method.fmt, states, fam, op)
    else:
        assert backend == 'reference', backend
    with _patched(model, schedule, boundary, carry) as layers:
        yield layers


@torch.no_grad()
def calibrate(model, method: StateQuantizer, input_ids: torch.Tensor, period: int | None = None, batch_size: int = 1):
    """Fit a method's allocation: run ``input_ids`` [N, T] with a full-precision state and hand the state at every
    ``period``-token boundary, the piece's decay and the piece (``families.Piece``: op inputs, ``.inputs()``) to
    ``method.observe``; then ``method.finalize()``. ``period`` defaults to the method's ``calibration_period``
    (else 256)."""
    if not method.needs_calibration:
        return method
    period = period or getattr(method, 'calibration_period', 256)
    sch = Schedule(prefill=period, decode=period, buffer_bits=0)

    def boundary(S, layer, kw):
        method.observe(layer, S, kw.log_decay(), kw)
        return S
    dev = next(model.parameters()).device
    with _patched(model, sch, boundary):
        for i in range(0, len(input_ids), batch_size):
            model(input_ids=input_ids[i:i + batch_size].to(dev), use_cache=False)
    method.finalize()
    return method
