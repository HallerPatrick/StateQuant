"""Memory accounting: bits per state value and bytes per sequence for a model under a method and schedule."""
from __future__ import annotations

from .methods import as_method
from .patch import find_layers
from .schedule import Schedule


def _dims(layer):
    from .families import family_of
    return family_of(layer).dims(layer)


def report(model, method, schedule: Schedule | None = None) -> dict:
    """Resident recurrent-state memory per sequence: payload (state incl. scales) + decode buffer."""
    method = as_method(method)
    schedule = schedule or getattr(method, 'schedule', None) or Schedule()
    layers = find_layers(model)
    dims = {_dims(m) for m in layers}
    assert len(dims) == 1, f'layers with different state shapes: {sorted(dims)}'
    d_k, d_v, h_v, h_k = dims.pop()
    payload = method.bits_per_value(d_k, d_v)
    buf = (method.buffer_bits_per_value(d_k, d_v, heads_per_key=h_v // h_k) if hasattr(method, 'buffer_bits_per_value')
           else schedule.buffer_bits_per_value(d_k, d_v, heads_per_key=h_v // h_k))
    n_vals = len(layers) * h_v * d_k * d_v
    return {'method': repr(method), 'schedule': schedule, 'layers': len(layers), 'heads': h_v, 'd_k': d_k, 'd_v': d_v,
            'payload_bits_per_value': payload, 'buffer_bits_per_value': buf,
            'resident_bits_per_value': payload + buf,
            'peak_bits_per_value': (method.peak_bits_per_value(d_k, d_v) if hasattr(method, 'peak_bits_per_value')
                                    else payload) + buf,
            'state_MiB_per_seq': n_vals * (payload + buf) / 8 / 2 ** 20,
            'bf16_state_MiB_per_seq': n_vals * 16 / 8 / 2 ** 20}
