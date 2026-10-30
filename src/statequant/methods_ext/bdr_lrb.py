"""BDR and LRB: "Preserving Updates in Low-Bit Recurrent State Quantization" (Li, Dong, Xiao, Xing; arXiv 2610.15709).

Base format (App. A.1): asymmetric n-bit (n = 4) integer grid per group of 128 contiguous key-dimension elements of
one value column, Delta = (max - min) / (2^n - 1), integer zero point z = round(-min / Delta), codes
q = clamp(round(x / Delta) + z, 0, 2^n - 1), Q(x) = Delta (q - z); Delta and z stored in fp32 (4.5 bits/element).

BDR (Sec. 3.2, Eq. 9): at a rounding of s + d (s: the decayed stored state, d: the update written since then),
an element whose effective update  d~ = Q(s + d) - s  has the opposite sign of d and |d| > eps_B * Delta is moved one
grid step toward d:  Q(s + d) + sign(d) Delta  (if that grid point lies in the grid's range, App. A.3; otherwise it is
left unchanged). ``BDR(fmt, eps_B)`` is a method usable with ANY schedule and any integer format
(``AffineGrid`` or ``statequant.Quantizer``): s is the state stored at the previous rounding times the decay
accumulated since, d = S_t - s.

LRB (Sec. 3.1, Eq. 6-7, App. A.4): the quantized state is only decayed between merges (stored codes fixed,
accumulated decay applied on read); every update k_t u_t^T is appended to an fp16 low-rank bypass L = K V^T whose key
factors are decayed in place. Every m steps: Delta_L = grid spacing of A S^ + L, M = 1[|L| < eps_L Delta_L],
L' = T_k(L . M) (randomised SVD, k + 8 projections, two power iterations), D = L - L',
S^ = Q[_BDR](A S^ + D), and the bypass restarts from the k columns of L' (at most k + m columns).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..quantizer import Format, Identity, Quantizer
from ..schedule import Schedule, per_token
from .hooks import StatefulMethod, decay_shape, terms, written_value

# Table 6 of the paper (k: bypass rank kept after a merge, m: merge interval, eps_L: bypass threshold, eps_B: BDR
# threshold), with the paper's average / peak bits for its 1.3B models.
LRB_CONFIGS = {
    'gdn': dict(k=1, m=15, eps_L=0.50, eps_B=0.30),      # avg 5.25, peak 6.00 (d_k 256, d_v 512)
    'gla': dict(k=1, m=15, eps_L=0.50, eps_B=0.30),      # avg 5.31, peak 6.07 (d_k 256, d_v 512, per-row decay)
    'mamba2': dict(k=1, m=5, eps_L=0.10, eps_B=0.25),    # avg 5.64, peak 6.78 (d_state 128, headdim 64)
    'retnet': dict(k=1, m=15, eps_L=0.70, eps_B=0.30),   # avg 5.25, peak 6.00 (d_k 256, d_v 512)
}
# statequant family -> the paper's architecture whose configuration it uses (``LRB(arch='auto')``): KDA has GDN's
# delta rule and GLA's per-row decay (both configurations are identical); lightning's constant per-head decay with an
# additive update is RetNet's recurrence; RWKV-6 (per-row decay, additive) is GLA's
FAMILY_ARCH = {'gdn': 'gdn', 'kda': 'gdn', 'mamba2': 'mamba2', 'lightning': 'retnet', 'rwkv6': 'gla'}


# ---------------------------------------------------------------------------------------------------------------
# the paper's base quantizer, and a generic code/step view of integer formats for BDR
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AffineGrid(Format):
    """App. A.1 / Eq. 10-11: asymmetric ``bits``-bit grid per ``group`` contiguous key-dimension elements of a value
    column, fp32 step Delta and integer zero point z (both stored as fp32: ``bits + 64 / group`` bits/element)."""
    bits: int = 4
    group: int = 128
    meta_bits: int = 64                      # fp32 Delta + fp32 z per group

    def grid(self, x):
        """x [..., d_k, d_v] -> (codes, Delta, z, qmin, qmax) in the grouped layout [..., d_v, d_k/g, g]."""
        X = _key_groups(x.float(), self.group)
        lo, hi = X.amin(-1, keepdim=True), X.amax(-1, keepdim=True)
        step = ((hi - lo) / (2 ** self.bits - 1)).clamp_min(1e-12)
        z = torch.round(-lo / step)
        q = (torch.round(X / step) + z).clamp(0, 2 ** self.bits - 1)
        return q, step, z, 0, 2 ** self.bits - 1

    @staticmethod
    def decode(q, step, z):
        return step * (q - z)

    def fake(self, S):
        q, step, z, _, _ = self.grid(S)
        return _key_ungroup(self.decode(q, step, z), S.shape).to(S.dtype)

    def step_size(self, S):
        """Delta of every element's group, in the shape of S."""
        _, step, _, _, _ = self.grid(S)
        X = _key_groups(S.float(), self.group)
        return _key_ungroup(step.expand_as(X), S.shape)

    def bits_per_value(self, d_k, d_v):
        return self.bits + self.meta_bits / min(self.group, d_k)


def _key_groups(x, g):
    x = x.transpose(-1, -2)                                     # [..., d_v, d_k]
    g = min(g, x.shape[-1])                                     # d_k < group (tiny models): one group per column
    assert x.shape[-1] % g == 0, f'group {g} does not divide d_k {x.shape[-1]}'
    return x.reshape(*x.shape[:-1], x.shape[-1] // g, g)


def _key_ungroup(X, shape):
    return X.reshape(*shape[:-2], shape[-1], shape[-2]).transpose(-1, -2)


class _QuantizerGrid:
    """The code / step / offset view of a ``statequant.Quantizer`` (rtn), matching ``Quantizer.fake`` exactly."""

    def __init__(self, fmt: Quantizer):
        assert fmt.rounding == 'rtn', 'BDR modifies nearest rounding; use rounding="rtn"'
        self.fmt = fmt

    def grid(self, X):                                           # X already grouped [..., n_groups, group]
        f = self.fmt
        if not f.affine:
            qmax = 2 ** (f.bits - 1) - 1
            scale = f._cast_scale(X.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax)
            return torch.round(X / scale).clamp(-qmax, qmax), scale, torch.zeros_like(scale), -qmax, qmax
        qmax = 2 ** f.bits - 1
        lo, hi = X.amin(-1, keepdim=True), X.amax(-1, keepdim=True)
        scale = f._cast_scale(((hi - lo) / qmax).clamp_min(1e-12))
        if f.zero_point == 'int':
            lo = torch.round(lo / scale) * scale
        elif f.scale_bits == 8:
            lo = lo.to(torch.float8_e4m3fn).float()
        elif f.scale_dtype == 'fp16':
            lo = lo.to(torch.float16).float()
        return torch.round((X - lo) / scale).clamp(0, qmax), scale, lo, 0, qmax


def step_size(fmt, S):
    """Grid step of every element's group, in the shape of S (``AffineGrid``, rtn ``Quantizer``; ``Identity``: 0)."""
    if isinstance(fmt, AffineGrid):
        return fmt.step_size(S)
    if isinstance(fmt, Identity):
        return torch.zeros_like(S.float())
    assert isinstance(fmt, Quantizer) and fmt.rotation is None, f'no element grid step for {fmt!r}'
    X = fmt._groups(S.float())
    return fmt._ungroup(_QuantizerGrid(fmt).grid(X)[1].expand_as(X), S.shape)


def bdr_round(fmt, S, s, eps_B: float, return_stats: bool = False):
    """Round S = s + d with Bounded Directional Rounding (Eq. 9). fmt: ``AffineGrid`` or a rtn ``Quantizer`` (any
    axis, symmetric/affine, optional Hadamard rotation: BDR then acts in the rotated basis, where s and d are rotated
    alike). Returns the dequantized state (and {'reversed', 'corrected'} fractions)."""
    S, s = S.float(), s.float()
    if isinstance(fmt, Identity):                                # no grid: nothing is rounded or reversed
        return (S, {'reversed': 0., 'corrected': 0.}) if return_stats else S
    if isinstance(fmt, AffineGrid):
        X, sg = _key_groups(S, fmt.group), _key_groups(s, fmt.group)
        q, step, z, qmin, qmax = fmt.grid(S)
        deq = lambda c: fmt.decode(c, step, z)                   # noqa: E731
        ungroup = lambda Y: _key_ungroup(Y, S.shape)             # noqa: E731
    else:
        assert isinstance(fmt, Quantizer), f'BDR needs an integer grid, got {fmt!r}'
        Sr, sr = (fmt._rot(S), fmt._rot(s)) if fmt.rotation else (S, s)
        X, sg = fmt._groups(Sr), fmt._groups(sr)
        q, step, off, qmin, qmax = _QuantizerGrid(fmt).grid(X)
        deq = lambda c: c * step + off                           # noqa: E731

        def ungroup(Y):
            out = fmt._ungroup(Y, Sr.shape)
            return fmt._rot(out, inverse=True) if fmt.rotation else out
    d = X - sg
    rev = (deq(q) - sg) * d < 0                                  # effective update opposite to the update
    fix = rev & (d.abs() > eps_B * step)
    nq = q + torch.sign(d)
    fix = fix & (nq >= qmin) & (nq <= qmax)                      # the corrected point must lie on the grid
    out = ungroup(deq(torch.where(fix, nq, q))).to(S.dtype)
    if return_stats:
        return out, {'reversed': rev.float().mean().item(), 'corrected': fix.float().mean().item()}
    return out


# ---------------------------------------------------------------------------------------------------------------
# decay bookkeeping shared by BDR and LRB
# ---------------------------------------------------------------------------------------------------------------
def _piece_decay(lg, S):
    """Total decay factor of a piece, broadcastable against S: exp(sum_t log a_t)."""
    return decay_shape(lg.sum(1), S).exp()


class BDR(StatefulMethod):
    """Bounded Directional Rounding as a rounding rule on any schedule: at every rounding point of the schedule the
    state S_t is rounded with s = A_{t0:t} S^_{t0} (the state stored at the previous rounding, decayed since) and
    d = S_t - s. ``fmt``: ``AffineGrid`` (the paper's INT4, default) or any rtn ``Quantizer``.

    Per-token rounding (``per_token()``) is the paper's "BDR alone" ablation (Table 7); with LRB, BDR is applied at
    the merges (``LRB(eps_B=...)``)."""

    def __init__(self, fmt: Format | None = None, eps_B: float = 0.3, name: str | None = None):
        self.fmt = fmt or AffineGrid(4, 128)
        self.eps_B = eps_B
        self.name = name or f'BDR(eps_B={eps_B:g}) {self.fmt!r}'
        self.layers, self.stats = {}, {}

    def step(self, ctx):
        st = self._layer_state(ctx, lambda c: {'Sq': torch.zeros_like(c.S_out), 'acc': None})
        lg = terms(ctx.pc)[0]
        a = _piece_decay(lg, ctx.S_out)
        st['acc'] = a if st['acc'] is None else st['acc'] * a
        hit = ctx.hit
        if not bool(hit.any()):
            return ctx.S_out
        s = st['acc'] * st['Sq']
        Sq, self.stats = bdr_round(self.fmt, ctx.S_out, s, self.eps_B, return_stats=True)
        if not bool(hit.all()):
            m = hit.to(Sq.device).view(-1, 1, 1, 1)
            Sq = torch.where(m, Sq, ctx.S_out)
            st['Sq'] = torch.where(m, Sq, st['Sq'])
            st['acc'] = torch.where(m, torch.ones_like(st['acc']), st['acc'])
        else:
            st['Sq'], st['acc'] = Sq, torch.ones_like(st['acc'])
        return Sq

    def bits_per_value(self, d_k, d_v):
        return self.fmt.bits_per_value(d_k, d_v)


# ---------------------------------------------------------------------------------------------------------------
# LRB
# ---------------------------------------------------------------------------------------------------------------
def randomized_svd(A, k, oversample=8, power_iters=2, generator=None):
    """Top-k factors of A [..., n, p] by Halko-Martinsson-Tropp (Alg. 4.4: subspace iteration with QR after every
    product): A ~= P Q^T with P = U sqrt(S), Q = V sqrt(S). Returns P [..., n, k], Q [..., p, k] (fp32)."""
    n, p = A.shape[-2:]
    ell = min(k + oversample, n, p)
    Om = torch.randn(*A.shape[:-2], p, ell, device=A.device, dtype=A.dtype, generator=generator)
    Y = A @ Om
    for _ in range(power_iters):
        Qy = torch.linalg.qr(Y).Q
        Qz = torch.linalg.qr(A.transpose(-1, -2) @ Qy).Q
        Y = A @ Qz
    Qm = torch.linalg.qr(Y).Q                                    # [..., n, ell]
    Ub, Sv, Vh = torch.linalg.svd(Qm.transpose(-1, -2) @ A, full_matrices=False)
    U = Qm @ Ub[..., :k]
    r = Sv[..., :k].sqrt()
    return U * r[..., None, :], Vh[..., :k, :].transpose(-1, -2) * r[..., None, :]


class LRB(StatefulMethod):
    """Low-Rank Bypass (+ BDR at merges unless ``eps_B=None``).

    k, m, eps_L, eps_B   rank kept after a merge, merge interval, bypass threshold, BDR threshold (``LRB_CONFIGS``)
    fmt                  the stored-state format (``AffineGrid(4, 128)``, the paper's INT4)
    records              'fp16': the bypass factors are kept in fp16 exactly as Eq. 6 (keys decayed in place, one
                         column appended per token); needs one-token pieces (``per_token()`` schedule, the paper's
                         token-by-token evaluation incl. the context). 'fp32': the bypass is reconstructed in fp32 at
                         the merges as S_t - A S^ (period-m schedule, much faster); only the k kept columns are fp16.
    arch                 'auto': k, m, eps_L, eps_B from ``LRB_CONFIGS[FAMILY_ARCH[family]]`` per layer family
                         (explicit k / m / eps_L / eps_B override it); or one of LRB_CONFIGS' keys for every layer
    decay_scalars        fp32 accumulated-decay scalars per head; None: 1 for a per-head decay, d_k for a per-row
                         decay (GLA, KDA, RWKV-6)
    oversample, power_iters, seed   randomised SVD (App. A.4: k + 8 projections, two power iterations)
    """

    _UNSET = object()

    def __init__(self, k: int | None = None, m: int | None = None, eps_L: float | None = None, eps_B=_UNSET,
                 fmt: AffineGrid | None = None, records: str = 'fp16', arch: str = 'gdn',
                 decay_scalars: int | None = None, oversample: int = 8, power_iters: int = 2, seed: int = 0,
                 name: str | None = None):
        assert records in ('fp16', 'fp32') and (arch == 'auto' or arch in LRB_CONFIGS)
        self.arch, self.override = arch, {n: x for n, x in dict(k=k, m=m, eps_L=eps_L).items() if x is not None}
        if eps_B is not LRB._UNSET:
            self.override['eps_B'] = eps_B
        cfg = self.config('gdn' if arch == 'auto' else None)
        self.k, self.m, self.eps_L, self.eps_B = cfg['k'], cfg['m'], cfg['eps_L'], cfg['eps_B']
        self.fmt = fmt or AffineGrid(4, 128)
        self.records, self.decay_scalars = records, decay_scalars
        self.oversample, self.power_iters, self.seed = oversample, power_iters, seed
        self.layers, self.stats, self._gen, self.families = {}, {}, {}, set()
        e = self.eps_B
        bdr = f'+BDR({e:g})' if e is not None else ''
        self.name = name or (f'LRB({arch}: k={self.k}, m={self.m}, eps_L={self.eps_L:g}){bdr} {self.fmt!r} '
                             f'records {records}')
        if records == 'fp32':
            assert arch != 'auto' or 'm' in self.override, 'records="fp32" runs on period-m pieces: fix arch or m'
        self.schedule = per_token() if records == 'fp16' else Schedule(prefill=self.m, decode=self.m, buffer_bits=0)

    def config(self, family=None):
        """k, m, eps_L, eps_B for a layer of ``family`` (a family name)."""
        arch = self.arch
        if arch == 'auto':
            if family not in FAMILY_ARCH:
                raise NotImplementedError(f'LRB has no configuration for the {family} family')
            arch = FAMILY_ARCH[family]
        return {**LRB_CONFIGS[arch], **self.override}

    def reset(self):
        self.layers, self._gen = {}, {}

    def _generator(self, device):
        g = self._gen.get(device)
        if g is None:
            g = self._gen[device] = torch.Generator(device=device).manual_seed(self.seed)
        return g

    def _init(self, ctx):
        S = ctx.S_out
        B, H, dk, dv = S.shape
        return {'Sq': torch.zeros_like(S), 'acc': None,
                'K': S.new_zeros(B, H, dk, 0, dtype=torch.float16), 'V': S.new_zeros(B, H, dv, 0, dtype=torch.float16)}

    def step(self, ctx):
        st = self._layer_state(ctx, self._init)
        lg, key, _, v, rule = terms(ctx.pc)
        cfg = self.config(ctx.family.name)
        self.families.add(ctx.family.name)
        S_out = ctx.S_out
        T = lg.shape[1]
        a = _piece_decay(lg, S_out)
        st['acc'] = a if st['acc'] is None else st['acc'] * a
        if self.records == 'fp16':
            assert T == 1, 'records="fp16" needs one-token pieces (schedule per_token())'
            u = written_value(S_out, ctx.S_in, lg, key, v, rule)   # this token's update k_t u_t^T
            st['K'] = torch.cat([(st['K'].float() * a).half(), key[:, 0, :, :, None].half()], -1)
            st['V'] = torch.cat([st['V'], u[..., None].half()], -1)
            L = st['K'].float() @ st['V'].float().transpose(-1, -2)
        else:
            L = S_out - st['acc'] * st['Sq']
        merge = (ctx.pos % cfg['m'] == 0) & (ctx.pos > 0)
        if not bool(merge.any()):
            return st['acc'] * st['Sq'] + L
        assert bool(merge.all()), 'LRB: all sequences of a batch must be at the same position'
        return self._merge(st, L, cfg)

    def _merge(self, st, L, cfg=None):
        cfg = cfg or self.config('gdn' if self.arch == 'auto' else None)
        k, eps_L, eps_B = cfg['k'], cfg['eps_L'], cfg['eps_B']
        s = st['acc'] * st['Sq']
        mask = L.abs() < eps_L * step_size(self.fmt, s + L)             # stays in the bypass
        if k > 0:
            P, Q = randomized_svd(L * mask, k, self.oversample, self.power_iters, self._generator(L.device))
            K, V = P.half(), Q.half()
            Lp = K.float() @ V.float().transpose(-1, -2)
        else:
            K, V = L.new_zeros(*L.shape[:-1], 0).half(), L.new_zeros(*L.shape[:-2], L.shape[-1], 0).half()
            Lp = torch.zeros_like(L)
        D = L - Lp
        if eps_B is None:
            Sq = self.fmt.fake(s + D)
            self.stats = {}
        else:
            Sq, self.stats = bdr_round(self.fmt, s + D, s, eps_B, return_stats=True)
        self.stats['kept'] = mask.float().mean().item()
        st.update(Sq=Sq, acc=torch.ones_like(st['acc']), K=K, V=V)
        return Sq + Lp

    def buffer_bits_per_value(self, d_k, d_v, heads_per_key=1):
        return 0.                                   # the bypass is counted in bits_per_value

    def bypass_bits_per_value(self, d_k, d_v, columns):
        return 16 * columns * (d_k + d_v) / (d_k * d_v)

    def _accounting(self, d_k, d_v):
        fams = sorted(self.families)
        fam = fams[0] if fams else ('gdn' if self.arch == 'auto' else None)
        cfg = self.config(fam) if (self.arch != 'auto' or fam) else self.config('gdn')
        ds = self.decay_scalars
        if ds is None:
            from ..families import BY_NAME
            ds = d_k if (fam in BY_NAME and BY_NAME[fam].decay == 'channel') else 1
        return self.fmt.bits_per_value(d_k, d_v) + 32 * ds / (d_k * d_v), cfg

    def bits_per_value(self, d_k, d_v):
        """Average storage over decoding (Table 6 'Avg. bits'): state format + accumulated decay + an fp16 bypass
        holding k + j columns after the j-th step since a merge, j = 0..m-1 (mean k + (m-1)/2). With
        arch='auto', the configuration of the (first) family seen in ``apply``."""
        base, cfg = self._accounting(d_k, d_v)
        return base + self.bypass_bits_per_value(d_k, d_v, cfg['k'] + (cfg['m'] - 1) / 2)

    def peak_bits_per_value(self, d_k, d_v):
        """Table 6 'Peak bits': k + m bypass columns just before a merge."""
        base, cfg = self._accounting(d_k, d_v)
        return base + self.bypass_bits_per_value(d_k, d_v, cfg['k'] + cfg['m'])
