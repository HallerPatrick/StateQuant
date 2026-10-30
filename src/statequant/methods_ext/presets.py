"""Presets for the methods in ``statequant.methods_ext`` (re-exported by ``statequant.presets`` / ``statequant``)."""
from __future__ import annotations

from ..methods import Uniform
from ..schedule import Schedule, per_token
from .bdr_lrb import BDR, LRB, AffineGrid
from .damp_paper import DampPaper
from .gramian import GramianAlloc
from .leapquant import LeapQuant


def _with(method, schedule):
    method.schedule = schedule
    return method


class _Priced(Uniform):
    """A Uniform format whose metadata is priced explicitly (the paper's INT4 stores fp32 Delta and zero point)."""

    def bits_per_value(self, d_k, d_v):
        return self.fmt.bits_per_value(d_k, d_v)


def int4_g128():
    """arXiv 2610.15709's INT4 baseline: AffineGrid(4, 128) (fp32 step + zero point, 4.5 b/value), every token."""
    return _with(_Priced(AffineGrid(4, 128), name='INT4 g128 (2610.15709)'), per_token())


def bdr_int4(eps_B: float = 0.3):
    """BDR alone on the INT4 baseline, rounding every token (Table 7 'BDR')."""
    return _with(BDR(AffineGrid(4, 128), eps_B, name=f'BDR INT4 g128 eps_B={eps_B:g}'), per_token())


def lrb_bdr(arch: str = 'auto', records: str = 'fp16', bdr: bool = True):
    """LRB + BDR with the paper's per-architecture configuration (Table 6; 'auto': per layer family, see
    ``bdr_lrb.FAMILY_ARCH``); ``bdr=False``: LRB alone (Table 7)."""
    kw = {} if bdr else {'eps_B': None}
    return LRB(arch=arch, records=records, name=f'LRB{"+BDR" if bdr else ""} ({arch}, {records})', **kw)


def leapquant(bits: int = 8, r: int | None = None, p: int = 16, records: str = 'fp16'):
    """LeapQuant: window p = 16, r = 4 fp16 compensators (8 at 4 bits), fp32 smoothing, symmetric per-value-channel
    residual at ``bits`` (8, 6, 4)."""
    return LeapQuant(bits, r, p, records)


def gramian(bits: int = 4, C: int = 1, allocation: str = 'gramian'):
    """Calibration-free decay-weighted allocation (arXiv 2609.30950) for GDN: per-head widths, 32-key-channel fp16
    block scales, write-back every token (C=1) or per 64-token prefill chunk + every decode token (C=64)."""
    return GramianAlloc(bits, C, allocation)


def damp_paper_int8():
    """DAMP INT8 as specified in arXiv 2608.27513 v2: 16 fp16 key channels / head, INT8 symmetric g32 fp32 scale,
    calibrated SR; every decode token requantized, fp32 prefill inside each chunk (9.875 b/value)."""
    return DampPaper(8)


def damp_paper_int4():
    """DAMP with the paper's INT4 low tier (asymmetric g32, fp32 scale + zero point), same selector and SR."""
    return DampPaper(4)


def int8k():
    """Plain int8, key-axis groups of 32, on PACE's schedule (prompt every 256, decode every 8 with an int8 buffer):
    the vLLM plugin's ``int8k`` as a reference row."""
    from ..presets import rtn
    return rtn(8, 'k', 32, prefill=256, decode=8, buffer_bits=8)


EXT_PRESETS = {'int8k': int8k,
    'int4_g128': int4_g128, 'bdr_int4': bdr_int4,
    'lrb_bdr': lrb_bdr, 'lrb': lambda: lrb_bdr(bdr=False), 'lrb_bdr_fp32': lambda: lrb_bdr('gdn', records='fp32'),
    'leapquant8': lambda: leapquant(8), 'leapquant6': lambda: leapquant(6), 'leapquant4': lambda: leapquant(4),
    'leapquant8_fp32': lambda: leapquant(8, records='fp32'),
    'gramian4': lambda: gramian(4, 1), 'gramian4_c64': lambda: gramian(4, 64), 'gramian6': lambda: gramian(6, 1),
    'uniform4_blk32': lambda: gramian(4, 1, 'uniform'), 'uniform4_blk32_c64': lambda: gramian(4, 64, 'uniform'),
    'damp_paper_int8': damp_paper_int8, 'damp_paper_int4': damp_paper_int4,
}

__all__ = ['int4_g128', 'bdr_int4', 'lrb_bdr', 'leapquant', 'gramian', 'damp_paper_int8', 'damp_paper_int4',
           'EXT_PRESETS', 'Schedule']
