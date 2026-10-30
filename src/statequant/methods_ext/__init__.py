"""Published recurrent-state compression methods beyond the core set (see ``hooks.py`` for what each needs from a
layer patch):

    BDR, LRB          "Preserving Updates in Low-Bit Recurrent State Quantization" (arXiv 2610.15709)
    LeapQuant         arXiv 2609.38166
    GramianAlloc      "Low-Bit Recurrent States in Hybrid LMs" (arXiv 2609.30950)
    DampPaper         DAMP's own recipe (arXiv 2608.27513 v2, App. A.1-A.3)
"""
from .bdr_lrb import BDR, LRB, LRB_CONFIGS, AffineGrid, bdr_round, randomized_svd
from .damp_paper import DampPaper, sr_candidates, sr_forward, sr_inverse
from .gramian import GramianAlloc, allocate, clip_quantize
from .hooks import StatefulMethod, check_family, terms
from .leapquant import LeapQuant, compensators
from .presets import (EXT_PRESETS, bdr_int4, damp_paper_int4, damp_paper_int8, gramian, int4_g128, leapquant,
                      lrb_bdr)
