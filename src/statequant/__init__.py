"""statequant: recurrent-state quantization for gated-delta (linear attention / hybrid) language models.

    import statequant as sq
    q = sq.pace48()                          # or sq.pace4(), sq.damp_int8(), sq.qmamba_dsq(), sq.nemotron(), sq.rtn(4),
                                             # sq.stepquant(4)
    sq.calibrate(model, q, calib_ids)        # only methods with a fitted allocation need this
    with sq.apply(model, q):                 # the method's own schedule; or sq.apply(model, q, sq.Schedule(...))
        out = model.generate(ids, max_new_tokens=256)
    print(sq.report(model, q))

Building blocks: formats (Quantizer, FloatCast, Identity), methods (Uniform, HeadMix, ChannelProtect, DSQ),
schedules (Schedule, per_token, periodic).
"""
from .methods import DSQ, ChannelProtect, HeadMix, StateQuantizer, Uniform, as_method
from .patch import apply, calibrate, find_layers, layer_families
from .presets import (PRESETS, damp_int4, damp_int8, full_precision, nemotron, pace4, pace48, qmamba_dsq, rtn,
                      stepquant)
from .step_quant import StepQuant
from . import methods_ext  # noqa: F401  (BDR/LRB, LeapQuant, GramianAlloc, DampPaper)
from .methods_ext import BDR, LRB, DampPaper, GramianAlloc, LeapQuant
from .presets import bdr_int4, damp_paper_int4, damp_paper_int8, gramian, int4_g128, leapquant, lrb_bdr
from .quantizer import FloatCast, Format, Identity, Quantizer, hadamard
from .report import report
from .schedule import Schedule, per_token, periodic

__version__ = '0.1.1'
