"""Published methods as ready-made (method, schedule) recipes.

Each preset returns a method object with its schedule attached as ``.schedule``, so

    q = statequant.pace48()
    statequant.calibrate(model, q, calib_ids)
    with statequant.apply(model, q):
        ...

uses the method as published. Pass ``schedule=`` to ``apply`` to run the same format on another schedule.

``paper=True`` reproduces the PACE paper's evaluation harness exactly (nonlinear_lin_rrn/src/eval_matrix.py cells),
including that harness's choices that a deployment would not make: the decode buffer's rounding is also applied to the
prompt's keys/values, and calibration statistics are taken with the harness's calibration format.
"""
from __future__ import annotations

from .methods import DSQ, ChannelProtect, HeadMix, Uniform
from .quantizer import FloatCast, Identity, Quantizer
from .schedule import Schedule, per_token
from .step_quant import StepQuant


def _with(method, schedule):
    method.schedule = schedule
    return method


def full_precision():
    return _with(Uniform(Identity(32), name='fp32'), Schedule(prefill=256, decode=1, buffer_bits=0))


def rtn(bits: int = 4, axis: str = 'k', group: int = 32, affine: bool = False, prefill: int = 1, decode: int = 1,
        buffer_bits: int = 0, **fmt):
    """Plain round-to-nearest state, per token by default (the naive baseline)."""
    return _with(Uniform(Quantizer(bits, axis, group, affine, **fmt)), Schedule(prefill, decode, buffer_bits))


def pace4(paper: bool = True):
    """PACE-4: symmetric int4, scale per 64 key rows of a value column, prompt rounded every 256 tokens, generation
    every 8 with an int8 key/value buffer. 4.25 b/value payload (4.33 on OLMo-Hybrid's d_k 96: ragged groups of 64 +
    32). Paper cell ``uniform:4:256:kg64:dP8:buf8``."""
    return _with(Uniform(Quantizer(4, 'k', 64), name='PACE-4'),
                 Schedule(prefill=256, decode=8, buffer_bits=8, buffer='kv', buffer_in_prefill=paper))


def pace48(paper: bool = True, frac: float = 0.25):
    """PACE-4/8: affine int4 (integer zero point) in 32-row key-axis groups; per layer the quarter of heads with the
    largest retained rounding error at int8 with a Hadamard rotation. Buffer: int4 keys, full-precision values.
    Paper cell ``mix:4:8:256:g32:int:product:0.25:full:bufk4:dP8:axk``.

    paper=True ranks heads exactly as that cell: error energy of a value-axis, float-zero-point int4 quantizer, with
    the amplifier evaluated at the prefill period (256, so it is close to 1 for most heads). paper=False ranks by the
    error of the deployed low-precision format, amplified at the decode period, as the paper's text describes."""
    lo = Quantizer(4, 'k', 32, affine=True, zero_point='int')
    hi = Quantizer(8, 'k', 32, affine=True, zero_point='int', rotation='hadamard')
    if paper:
        m = HeadMix(lo, hi, frac, score='retained', score_period=256,
                    cal_format=Quantizer(4, 'v', 32, affine=True, zero_point='fp'), name='PACE-4/8')
    else:
        m = HeadMix(lo, hi, frac, score='retained', score_period=8, name='PACE-4/8')
    return _with(m, Schedule(prefill=256, decode=8, buffer_bits=4, buffer='k', buffer_in_prefill=paper))


def damp_int8():
    """DAMP INT8 as the PACE evaluation harness implements it (cell ``damp:8:256:g32:16:int:rtn:full`` at carry scope
    ``damp``): 16 of 128 key channels per head in fp16, the rest affine int8 in 32-wide value-axis blocks with a
    Hadamard rotation; prompt rounded per 256-token chunk, every generated token rounded. NOTE: this is not the DAMP
    paper's recipe (arXiv 2608.27513 v2, App. A.1): there INT8 is symmetric with one fp32 scale and no zero point per
    32 values, no rotation, plus calibrated value-dimension scaling and reordering ("SR") folded into the V/O
    projections."""
    lo = Quantizer(8, 'v', 32, affine=True, zero_point='int', rotation='hadamard')
    m = ChannelProtect(lo, FloatCast('fp16'), k=16, cal_format=Quantizer(8, 'v', 32, affine=True, zero_point='fp'),
                       name='DAMP INT8')
    return _with(m, Schedule(prefill=256, decode=1, buffer_bits=0))


def damp_int4(paper: bool = True):
    """DAMP's construction at four bits on PACE's schedule (the like-for-like row of the paper): 32 key channels per
    head at int8, rest int4. Paper cell ``damp:4:256:g32:32:int:rtn:full:hi8:buf8:dP8``."""
    lo = Quantizer(4, 'v', 32, affine=True, zero_point='int', rotation='hadamard')
    hi = Quantizer(8, 'v', 32, affine=True, zero_point='int', rotation='hadamard')
    m = ChannelProtect(lo, hi, k=32, cal_format=Quantizer(4, 'v', 32, affine=True, zero_point='fp'), name='DAMP INT4')
    return _with(m, Schedule(prefill=256, decode=8, buffer_bits=8, buffer='kv', buffer_in_prefill=paper))


def qmamba_dsq(paper: bool = True, decode: int = 8):
    """Q-Mamba's DSQ quantizer (without its fine-tuning). Q-Mamba rounds every write (decode=1); the paper row runs it
    on PACE's schedule. Paper cell ``dsq:4:256:dP8:buf8``."""
    return _with(DSQ(4), Schedule(prefill=256, decode=decode, buffer_bits=8 if decode > 1 else 0,
                                  buffer_in_prefill=paper and decode > 1))


def nemotron():
    """Nemotron 3 Super's SSM-cache recipe (arXiv 2604.12374): fp16 with stochastic rounding at every write."""
    return _with(Uniform(FloatCast('fp16', stochastic=True), name='Nemotron fp16+SR'),
                 Schedule(prefill=1, decode=1, buffer_bits=0))


def stepquant(bits: int = 4, pivots: int | None = None):
    """STEPQuant (arXiv 2609.38169) at a nominal 4 or 6 bits, rounding at every state write as published."""
    return _with(StepQuant(bits, pivots), per_token())


PRESETS = {'fp32': full_precision, 'rtn': rtn, 'pace4': pace4, 'pace48': pace48, 'damp_int8': damp_int8,
           'damp_int4': damp_int4, 'qmamba_dsq': qmamba_dsq, 'nemotron': nemotron,
           'stepquant4': lambda: stepquant(4), 'stepquant6': lambda: stepquant(6)}


# -- methods in statequant.methods_ext (BDR/LRB, LeapQuant, Gramian allocation, DAMP paper recipe) ----------------
from .methods_ext.presets import (EXT_PRESETS, bdr_int4, damp_paper_int4, damp_paper_int8, gramian,  # noqa: E402
                                  int4_g128, leapquant, lrb_bdr)

PRESETS.update(EXT_PRESETS)
