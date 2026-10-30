"""Packed (real-storage) decode path: int4 / int8 / bf16 recurrent state with a delta-value decode buffer.

    from statequant.kernels import PackedGDNState, ReferenceGDNState, SlotLayout, pack, unpack

``pack`` / ``unpack`` are pure torch (no Triton needed); ``PackedGDNState`` (sequences in lockstep) and
``SlotLayout.step`` (slot-addressed sequences, each at its own position: the serving layout) need Triton and a GPU. Mamba2 / SSD layers:
``PackedMamba2State``, ``Mamba2SlotLayout`` (``mamba2.py``). KDA layers (per-key-channel decay): ``PackedKDAState``,
``KDASlotLayout`` (``kda.py``).
"""
from .fold import PackedGDNState, ReferenceGDNState, pack, pack_vk, packed_format, unpack, unpack_vk
from .slots import SlotLayout
from .mamba2 import Mamba2SlotLayout, PackedMamba2State, ReferenceMamba2State, mamba2_recurrence
from .kda import KDASlotLayout, PackedKDAState, ReferenceKDAState, kda_recurrence

__all__ = ['PackedGDNState', 'ReferenceGDNState', 'SlotLayout', 'pack', 'unpack', 'pack_vk', 'unpack_vk',
           'packed_format',
           'PackedMamba2State', 'ReferenceMamba2State', 'Mamba2SlotLayout', 'mamba2_recurrence',
           'PackedKDAState', 'ReferenceKDAState', 'KDASlotLayout', 'kda_recurrence']
