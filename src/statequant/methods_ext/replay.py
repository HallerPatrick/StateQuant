"""ReplaySSM (arXiv 2610.13778; code: github.com/Johnny-Liou/ReplaySSM, ``emulation/``): a recurrence checkpoint and a
read copy of the state, both cast from the FP32 state at every flush, with a replay buffer of the tokens since.

Per layer and request:
    HP      the recurrence checkpoint, FP32 / FP16 / FP8 / NVFP4, stochastic rounding (SR) when not FP32
    LP      the read copy, FP8 / NVFP4 with round-to-nearest-even (RTN); or no separate copy ('shared' modes: HP is read)
    ring    L records of the decoded tokens since the last flush. Delta rule (GDN, KDA): key BF16 (once per key head),
            value BF16, the token's update u = beta (v - S^T k) FP16, gate and beta FP32. Additive (Mamba2): x BF16,
            B BF16 (once per group), dt FP32.

Decode, window position w = 0 .. L-1 (emulation/gdn_replay_unf_vf.py, mamba2_replay.py):
    w < L-1   the output reads LP (HP in shared modes) plus the ring: the delta-rule recurrence started from LP at the
              window start with every earlier token's u rounded to FP16 (GDN: LP contracted as BF16 operands). The
              token's u is cached. Nothing is written to HP / LP.
    w = L-1   flush: the window is re-solved from HP and the raw ring (k, v, g, beta; the cached u do not enter), with
              BF16 operands and FP32 accumulation as the emulation kernel; the token's output comes from that solution,
              which is then cast separately into HP (SR) and LP (RTN), never one from the other.
Prefill is full precision. The first decoded token reads the FP32 post-prefill state; after it that state is cast into
HP and LP (the 'seed'), and the window continues from LP.

Formats (emulation/quant_tl.py): FP16 unscaled; FP8 E4M3 with one FP32 scale amax / 448 per 128 consecutive key
entries of one value column (the trailing dim of the kernels' state); NVFP4 E2M1 with one E4M3 scale per 16 key
entries and one FP32 outer scale amax / (6 * 448) per request and layer over all heads, a block whose E4M3 scale rounds
to zero decoding to zero. SR: floor(|x| / ulp + u) on the target grid. The emulation draws u from Philox keyed by
(seed 1234 + window * 1103515245, element offset); here from a torch generator seeded the same per window, so the SR
draws are not bit-identical to the kernels'. Every request in a batch decodes in lockstep (same window position).

Memory: payload = HP bits + LP bits; the ring is reported as the buffer: L x (key / heads_per_key + 2 values + 64 bits)
for the delta rule. No configuration of the paper is at or below PACE-4's 5.03 resident bits on Qwen3.5-4B.
"""
from __future__ import annotations

import torch

from ..schedule import Schedule
from .hooks import StatefulMethod, decay_shape, terms, written_value

FORMATS = ('fp32', 'fp16', 'fp8', 'nvfp4')
# preset name -> (HP format, HP stochastic, LP format or None = shared); emulation/modes.py
MODES = {
    's32_rs': ('fp32', False, None),
    'f16_sr_rs': ('fp16', True, None),
    'f8_sr_rs': ('fp8', True, None),
    'f4_sr_rs': ('nvfp4', True, None),
    'tp8_hi32': ('fp32', False, 'fp8'),
    'tp8_hi16_sr': ('fp16', True, 'fp8'),
    'tp4_hi32': ('fp32', False, 'nvfp4'),
    'tp4_hi16_sr': ('fp16', True, 'nvfp4'),
    'tp4_hi8_sr': ('fp8', True, 'nvfp4'),
}
E4M3 = (3, -6, 448.)                                   # mantissa bits, min normal exponent, max
E2M1 = (1, 0, 6.)
FP16 = (10, -14, 65504.)
SEED = 1234


def _bf16(x):
    return x.to(torch.bfloat16).float()


def grid_ulp(x, mant, min_exp):
    """Spacing of the float grid (``mant`` mantissa bits, minimum normal exponent ``min_exp``) at |x|."""
    a = x.abs()
    e = torch.frexp(a.clamp_min(2. ** min_exp))[1].float() - 1          # floor(log2), exactly
    e = e.clamp_min(min_exp)
    return torch.where(a < 2. ** min_exp, torch.full_like(a, 2. ** (min_exp - mant)), torch.exp2(e - mant))


def cast_grid(x, mant, min_exp, vmax):
    """RTN, ties to even, onto the grid, saturating at +-vmax."""
    xc = x.clamp(-vmax, vmax)
    ulp = grid_ulp(xc, mant, min_exp)
    return torch.round(xc / ulp) * ulp


def sr_grid(x, u, mant, min_exp, vmax):
    ulp = grid_ulp(x, mant, min_exp)
    return cast_grid(torch.sign(x) * torch.floor(x.abs() / ulp + u) * ulp, mant, min_exp, vmax)


def _blocks(S, n):
    """[B, H, d_k, d_v] -> [B, H, d_v, d_k / n, n]: blocks of n consecutive key entries of one value column."""
    B, H, dk, dv = S.shape
    assert dk % n == 0, f'block {n} does not divide d_k = {dk}'
    return S.transpose(-1, -2).reshape(B, H, dv, dk // n, n)


def _unblocks(X, shape):
    B, H, dk, dv = shape
    return X.reshape(B, H, dv, dk).transpose(-1, -2)


def cast_state(S, fmt, u=None):
    """Round S [B, H, d_k, d_v] (fp32) to ``fmt``; ``u`` (uniforms like S) selects stochastic rounding."""
    x = S.float()
    if fmt == 'fp32':
        return x
    if fmt == 'fp16':
        if u is None:
            return x.half().float()
        ulp = grid_ulp(x, *FP16[:2])
        return (torch.sign(x) * torch.floor(x.abs() / ulp + u) * ulp).half().float()
    rnd = (lambda v, uu, g: cast_grid(v, *g)) if u is None else (lambda v, uu, g: sr_grid(v, uu, *g))
    if fmt == 'fp8':
        n = min(128, x.shape[-2])
        X = _blocks(x, n)
        U = None if u is None else _blocks(u, n)
        scale = X.abs().amax(-1, keepdim=True) / E4M3[2]
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return _unblocks(rnd(X / scale, U, E4M3) * scale, x.shape)
    assert fmt == 'nvfp4', fmt
    X = _blocks(x, 16)
    U = None if u is None else _blocks(u, 16)
    s_g = x.abs().flatten(1).amax(-1) / (E2M1[2] * E4M3[2])             # per request, over all heads of the layer
    s_g = torch.where(s_g > 0, s_g, torch.ones_like(s_g)).view(-1, 1, 1, 1, 1)
    s_b = cast_grid(X.abs().amax(-1, keepdim=True) / (E2M1[2] * s_g), *E4M3)
    dead = s_b <= 0
    scale = torch.where(dead, torch.ones_like(s_b), s_b) * s_g
    Q = torch.where(dead, torch.zeros_like(X), rnd(X / scale, U, E2M1) * scale)
    return _unblocks(Q, x.shape)


def format_bits(fmt, d_k, d_v, heads):
    if fmt == 'fp32':
        return 32.
    if fmt == 'fp16':
        return 16.
    if fmt == 'fp8':
        return 8. + 32. / min(128, d_k)
    return 4. + 8. / 16 + 32. / (heads * d_k * d_v)


class ReplaySSM(StatefulMethod):
    """mode     a preset of ``MODES`` ('tp4_hi16_sr': NVFP4 read copy, FP16 + SR checkpoint; 'f4_sr_rs': one shared
                NVFP4 + SR copy; ...), or None with ``hp`` / ``hp_sr`` / ``lp`` given
    L        ring length = flush period (16 in the paper)
    """
    rules = ('delta', 'additive')

    def __init__(self, mode: str | None = 'tp4_hi16_sr', L: int = 16, hp: str | None = None, hp_sr: bool = True,
                 lp: str | None = None, name: str | None = None):
        if mode is not None:
            hp, hp_sr, lp = MODES[mode]
        assert hp in FORMATS and (lp is None or lp in FORMATS) and L >= 1
        self.mode, self.L, self.hp, self.hp_sr, self.lp = mode, L, hp, hp_sr and hp != 'fp32', lp
        self.layers, self.heads = {}, None
        self.name = name or (f'ReplaySSM {mode} L={L}' if mode else
                             f'ReplaySSM HP {hp}{"+SR" if self.hp_sr else ""}, read {lp or "shared"}, L={L}')
        # prefill in full precision; prompt_end marks the last piece of the prompt; decode pieces are single tokens
        self.schedule = Schedule(prefill=1 << 30, decode=1, buffer_bits=0, prompt_end=True)

    # -- the two copies ----------------------------------------------------------------------------------------------
    def _cast(self, S, window):
        u = None
        if self.hp_sr:
            g = torch.Generator(device=S.device).manual_seed(SEED + window * 1103515245)
            u = torch.rand(S.shape, generator=g, device=S.device)
        hp = cast_state(S, self.hp, u)
        lp = hp if self.lp is None else cast_state(S, self.lp)
        return hp, lp

    def _read(self, S, rule):
        return _bf16(S) if rule == 'delta' else S             # GDN reads its checkpoint as BF16 operands

    # -- the flush solve ---------------------------------------------------------------------------------------------
    @staticmethod
    def solve(S0, recs, rule, read=False):
        """State after the ring's records from S0, re-solved from the raw records (emulation/gdn_replay_unf_vf.py,
        flush branch): BF16 keys and checkpoint operands, FP32 accumulation, BF16 residuals and weighted updates.
        ``read``: the state the flush token's output reads instead (delta rule: the kernel's output contracts the
        checkpoint as BF16 and adds the unrounded u)."""
        if not recs:
            return S0
        lg = torch.stack([r['lg'] for r in recs], 2)                       # [B, H, n] or [B, H, n, d_k]
        if lg.dim() == 3:
            lg = lg[..., None]
        k = torch.stack([r['k'] for r in recs], 2)                          # [B, H, n, d_k] (BF16 values)
        v = torch.stack([r['v'] for r in recs], 2)                          # [B, H, n, d_v]
        G = lg.cumsum(2)                                                    # [B, H, n, 1 or d_k]
        Gt = G[:, :, -1:]
        if rule == 'additive':
            kw = _bf16((Gt - G).exp() * k)
            return Gt[:, :, 0, :, None].exp() * S0 + kw.transpose(-1, -2) @ v
        beta = torch.stack([r['beta'] for r in recs], 2)                    # [B, H, n]
        S0b = _bf16(S0)
        r = _bf16(v - ((G.exp() * k) @ S0b))                                # v_j - (A_{0:j} S0)^T k_j
        n = k.shape[2]
        u = torch.zeros_like(v)
        for j in range(n):
            if j:
                coef = (((G[:, :, j:j + 1] - G[:, :, :j]).exp() * k[:, :, :j]) * k[:, :, j:j + 1]).sum(-1)  # [B,H,j]
                acc = (coef[..., None] * u[:, :, :j]).sum(2)
            else:
                acc = 0.
            u[:, :, j] = beta[:, :, j, None] * (r[:, :, j] - acc)
        W = (Gt - G).exp()                                                  # [B, H, n, 1 or d_k]
        rnd = (lambda x: x) if read else _bf16
        if W.shape[-1] == 1:
            upd = k.transpose(-1, -2) @ rnd(W * u)
        else:
            upd = rnd(W * k).transpose(-1, -2) @ rnd(u)
        return Gt[:, :, 0, :, None].exp() * (S0b if read else S0) + upd

    # -- the piece hooks ---------------------------------------------------------------------------------------------
    def _init(self, ctx):
        return {'hp': None, 'lp': None, 'seed': None, 'recs': [], 'w': 0, 'window': 0}

    def _st(self, layer):
        return (self.layers or {}).get(layer)

    def prepare(self, S, layer, pc):
        """Before a piece: the state the op should start from."""
        st = self._st(layer)
        if st is None:
            return S
        T = pc.inputs()['k'].shape[1]
        rule = pc.family.rule
        if T > 1:                                            # a multi-token call after decoding: continue from the
            if st['hp'] is None:                             # recurrence (HP + ring), in full precision
                return S
            S = self.solve(st['hp'], st['recs'], rule)
            self.layers.pop(layer)
            return S
        if st['w'] == self.L - 1:                            # flush: re-solve the window from HP
            base = st['seed'] if st['hp'] is None else st['hp']
            return self.solve(base, st['recs'], rule, read=True)
        if st['hp'] is None:                                 # the seed token reads the FP32 post-prefill state
            return self._read(st['seed'], rule)
        return S

    def step(self, ctx):
        pc, layer, S = ctx.pc, ctx.layer, ctx.S_out
        self.heads = S.shape[1]
        if pc.inputs()['k'].shape[1] > 1:                    # prefill piece: full precision
            if ctx.prompt_end:
                self.layers = self.layers or {}
                st = self._init(ctx)
                st['seed'], st['B'] = S.float().clone(), S.shape[0]
                self.layers[layer] = st
            return S
        st = self._st(layer)
        if st is None:                                       # decoding without a prompt call: seed from zeros
            st = self._init(ctx)
            st['seed'], st['B'] = ctx.S_in.float().clone(), S.shape[0]
            self.layers = self.layers or {}
            self.layers[layer] = st
        rule = self._rule = pc.family.rule
        lg, k, beta, v, _ = terms(pc)
        rec = {'lg': lg[:, 0].float(), 'k': _bf16(k[:, 0]), 'v': _bf16(v[:, 0])}
        if rule == 'delta':
            rec['beta'] = _bf16(beta[:, 0].float())
        if st['w'] == self.L - 1:                            # flush: the window re-solved from HP with this token (as
            base = st['seed'] if st['hp'] is None else st['hp']          # the kernel; the op's output came from the
            S_new = self.solve(base, st['recs'] + [rec], rule)           # same solve without it), cast into both
            st['hp'], st['lp'] = self._cast(S_new, st['window'])         # copies
            st['seed'], st['recs'], st['w'] = None, [], 0
            st['window'] += 1
            return self._read(st['lp'], rule)
        S_in = ctx.S_in
        u = written_value(S, S_in, lg, k, v, rule)           # this token's update, from the read path
        if rule == 'delta':
            u = u.half().float()                             # the ring's FP16 u
        if st['hp'] is None:                                 # seed: cast the FP32 post-prefill state, read LP from here
            st['hp'], st['lp'] = self._cast(st['seed'], st['window'])
            S_in = self._read(st['lp'], rule)
        a = decay_shape(lg[:, 0], S).exp()
        R = a * S_in + k[:, 0, :, :, None] * (u if rule == 'delta' else v[:, 0])[:, :, None, :]
        st['recs'].append(rec)
        st['w'] += 1
        return R

    # -- accounting --------------------------------------------------------------------------------------------------
    def bits_per_value(self, d_k, d_v):
        h = self.heads or 1
        b = format_bits(self.hp, d_k, d_v, h)
        return b + (format_bits(self.lp, d_k, d_v, h) if self.lp else 0.)

    def buffer_bits_per_value(self, d_k, d_v, heads_per_key=1):
        """The ring: L records. Delta rule: BF16 key (shared by ``heads_per_key`` value heads), BF16 value, FP16 u,
        FP32 gate and beta. Additive (Mamba2, read d_k = d_state, d_v = head_dim): BF16 x, BF16 B (per group), FP32
        dt; priced as the delta rule when the family is not known yet."""
        per_tok = 16 * d_k / heads_per_key + 16 * d_v + 16 * d_v + 64
        if getattr(self, '_rule', 'delta') == 'additive':
            per_tok = 16 * d_k / heads_per_key + 16 * d_v + 32
        return self.L * per_tok / (d_k * d_v)
