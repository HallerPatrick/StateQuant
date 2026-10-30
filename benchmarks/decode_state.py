"""Per-layer decode-step latency of the recurrent state: bf16 state stepped per token by fla's fused recurrent kernel
vs the packed state with a delta-value buffer (statequant.kernels.PackedGDNState).

Shape: one Qwen3.5-4B GDN layer, HV = 32 value heads, HK = 16 key heads, d_k = d_v = 128. Every arm is captured in a
CUDA graph of ``m`` consecutive steps (so a packed arm holds exactly one fold) and timed with CUDA events; the
reported figure is the minimum over repeats of (graph time / steps), in microseconds per layer per step.

    PYTHONPATH=src python benchmarks/decode_state.py [--batches 1,2,4,...] [--m 8] [--repeats 20] [--slots]

Arms
    fla bf16 in place    fla's fused_recurrent kernel launched with ht = h0 = S (bf16): stock vLLM's per-token
                         update of a bf16 state in place
    fla fp32 in place    the same with an fp32 state
    bf16 buf8 / buf16    PackedGDNState('bf16', buffer_bits=8 / 16): bf16 state, m-token buffer (int8 / bf16)
    int4 k g64           PackedGDNState(Quantizer(4, 'k', 64)) (PACE-4's format), int8 buffer
    int4 v g32, int8 k g32
    ... 5-launch / fused every packed arm twice: the five-kernel step (fused=False) and the single-launch fused step
                         (fused=True, ``kernels/_fused_triton.py``)
    slot int4 k g64      (--slots) SlotLayout.step on B slots with in-kernel gating (the vLLM plugin's call)

Mamba2 arm (``--mamba2 nemotron-h-4b,nemotron-3-nano-4b``; run in the vLLM venv): one MambaMixer2 layer's decode
step, (heads, groups, head_dim, d_state) from MAMBA2_SHAPES, slot-indexed as in vLLM (B slots of a larger cache)
    vllm fp32 / bf16     vLLM's Triton ``selective_state_update`` in place on an fp32 / bf16 ssm state (the stock
                         decode call of ``MambaMixer2``: dt expanded over head_dim, dt_bias + softplus, D)
    bf16 buf8 / buf16, int4 k g64, int8 k g32, int4 v g32 (group halved to divide head_dim, e.g. 16 for 80)
                         ``Mamba2SlotLayout.step`` (the plugin's call), m-token buffer
"""
from __future__ import annotations

import argparse

import torch

import statequant as sq
from statequant.kernels import PackedGDNState, SlotLayout

HV, HK, DK, DV = 32, 16, 128, 128
GDN_SHAPES = {'qwen3.5-4b': (32, 16, 128, 128), 'olmo-hybrid-7b': (30, 30, 96, 192)}   # (HV, HK, d_k, d_v)


def time_graph(fn, steps, repeats, inner=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for _ in range(steps):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(steps):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    best = float('inf')
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(inner):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1e3 / (inner * steps))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', default='1,2,4,8,16,32,64,128,512')
    ap.add_argument('--m', type=int, default=8)
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--slots', action='store_true', help='also time SlotLayout.step (the vLLM plugin call)')
    ap.add_argument('--mamba2', default='', help='Mamba2 arm instead (vLLM venv): comma list of shapes from '
                    'MAMBA2_SHAPES, e.g. nemotron-h-4b,nemotron-3-nano-4b')
    ap.add_argument('--gdn-shape', default='qwen3.5-4b', choices=list(GDN_SHAPES),
                    help='GDN layer shape (olmo-hybrid-7b: d_k 96, d_v 192, padded key tiles)')
    a = ap.parse_args()
    if a.mamba2:
        return main_mamba2(a)
    global HV, HK, DK, DV
    HV, HK, DK, DV = GDN_SHAPES[a.gdn_shape]
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel as frec_kernel

    dev, m, scale = 'cuda', a.m, DK ** -0.5
    packed = {'bf16 buf8': ('bf16', 8), 'bf16 buf16': ('bf16', 16), 'int4 k g64': (sq.Quantizer(4, 'k', 64), 8),
              'int4 v g32': (sq.Quantizer(4, 'v', 32), 8), 'int8 k g32': (sq.Quantizer(8, 'k', 32), 8)}
    fmts = {f'{n} {"fused" if fz else "5-launch"}': (f, bb, fz) for n, (f, bb) in packed.items()
            for fz in (False, True)}
    slot_arms = {f'slot int4 k g64 {"fused" if fz else "5-launch"}': fz for fz in (False, True)} if a.slots else {}
    arms = ['fla bf16 in place', 'fla fp32 in place'] + list(fmts) + list(slot_arms)
    rows, per_seq = [], {}
    print(f'GPU: {torch.cuda.get_device_name()}  (shared GPU: timings are indicative only); GDN shape {a.gdn_shape} '
          f'(HV {HV}, HK {HK}, d_k {DK}, d_v {DV})')
    x = torch.randn(4096, 4096, device=dev)                    # bring the clocks up
    for _ in range(300):
        x @ x
    del x
    for B in [int(x) for x in a.batches.split(',')]:
        torch.manual_seed(0)
        BH = B * HV
        q = torch.randn(B, 1, HK, DK, device=dev, dtype=torch.bfloat16)
        k = torch.randn(B, 1, HK, DK, device=dev, dtype=torch.bfloat16)
        v = torch.randn(B, 1, HV, DV, device=dev, dtype=torch.bfloat16)
        g = -torch.rand(B, 1, HV, device=dev) * 0.08
        beta = torch.rand(B, 1, HV, device=dev, dtype=torch.bfloat16).sigmoid()
        S0 = torch.randn(B, HV, DK, DV, device=dev) * 0.1
        o_buf = torch.empty_like(v)
        res = {}
        for name, sdt in (('fla bf16 in place', torch.bfloat16), ('fla fp32 in place', torch.float32)):
            S_ = S0.to(sdt)

            def inplace():
                frec_kernel[(DV // 8 * B * HV,)](
                    q=q, k=k, v=v, g=g, gk=None, gv=None, beta=beta, A_log=None, dt_bias=None, o=o_buf, h0=S_,
                    ht=S_, cu_seqlens=None, scale=scale, T=1, H=HK, HV=HV, K=DK, V=DV,
                    BK=1 << (DK - 1).bit_length(), BV=8,
                    USE_G=True, USE_GK=False, USE_GV=False, IS_BETA_HEADWISE=True, USE_QK_L2NORM_IN_KERNEL=True,
                    APPLY_BETA_SIGMOID=False, ALLOW_NEG_EIGVAL=False, STATE_V_FIRST=False, IS_VARLEN=False,
                    USE_GATE_IN_KERNEL=False, HAS_DT_BIAS=False, USE_INITIAL_STATE=True, STORE_FINAL_STATE=True,
                    num_warps=1, num_stages=3)
            res[name] = time_graph(inplace, m, a.repeats)
            per_seq[name] = S_[0].numel() * S_.element_size()
            del S_

        qf, kf = q.reshape(B * HK, DK), k.reshape(B * HK, DK)
        vf, gf, bf = v.reshape(BH, DV), g.reshape(BH), beta.reshape(BH)
        for name, (fmt, bb, fz) in fmts.items():
            st = PackedGDNState(S0, fmt, m=m, hv=HV, hk=HK, buffer_bits=bb, fused=fz)
            res[name] = time_graph(lambda: st.step(qf, kf, vf, gf, bf, scale), m, a.repeats)
            if not fz:
                per_seq[name.rsplit(' ', 1)[0]] = st.stored_bytes()
            del st
        if slot_arms:
            lay = SlotLayout(sq.Quantizer(4, 'k', 64), HV, HK, DK, DV, m=m, buffer_bits=8)
            blob = lay.alloc(B + 1)
            views = lay.views(blob)
            idx = torch.arange(1, B + 1, device=dev, dtype=torch.int32)
            lay.write(views, idx, S0.transpose(-1, -2))
            qs, ks, vs = q.reshape(B, HK, DK), k.reshape(B, HK, DK), v.reshape(B, HV, DV)
            ga = torch.randn(B, HV, device=dev, dtype=torch.bfloat16)
            gb_ = torch.randn(B, HV, device=dev, dtype=torch.bfloat16)
            A_log, dt_bias = torch.randn(HV, device=dev) * 0.5, torch.randn(HV, device=dev, dtype=torch.bfloat16)
            o_s = torch.empty(B, HV, DV, device=dev, dtype=torch.bfloat16)
            for name, fz in slot_arms.items():
                res[name] = time_graph(lambda: lay.step(views, idx, qs, ks, vs, o_s, scale,
                                                        gate=(ga, gb_, A_log, dt_bias), fused=fz), m, a.repeats)
            del blob, views
        del S0
        torch.cuda.empty_cache()
        rows.append((B, res))
        print(f'B={B:<4d} ' + '  '.join(f'{n}: {t:7.1f}' for n, t in res.items()), flush=True)

    print(f'\nper-layer decode step, us, amortised over the fold (min over {a.repeats} repeats of a CUDA graph of '
          f'{m} steps = one fold per packed arm; shared GPU, indicative only)')
    w = 11
    hdr = [n.replace(' in place', '').replace('5-launch', '5L').replace(' fused', ' F') for n in arms]
    print(f"{'batch':>6} " + ' '.join(f'{h:>{max(w, len(h))}}' for h in hdr))
    for B, res in rows:
        print(f'{B:>6} ' + ' '.join(f'{res[n]:>{max(w, len(h))}.1f}' for n, h in zip(arms, hdr)))
    print('\nratios to fla bf16 in place (< 1: faster)')
    print(f"{'batch':>6} " + ' '.join(f'{h:>{max(w, len(h))}}' for h in hdr[1:]))
    for B, res in rows:
        print(f'{B:>6} ' + ' '.join(f'{res[n] / res[arms[0]]:>{max(w, len(h))}.2f}' for n, h in zip(arms[1:], hdr[1:])))

    print('\nresident bytes per sequence per layer (state + buffer + decay; B = 1 normalisation)')
    print(f"{'arm':<26} {'state':>10} {'buffer':>10} {'decay':>8} {'total':>10} {'KiB':>8} {'bits/value':>11}")
    for nm in ('fla bf16 in place', 'fla fp32 in place'):
        bf = per_seq.pop(nm)
        print(f"{nm:<26} {bf:>10d} {0:>10d} {0:>8d} {bf:>10d} {bf / 1024:>8.1f} {bf * 8 / (HV * DK * DV):>11.3f}")
    nB = rows[-1][0]
    for name, sb in per_seq.items():
        d = {kk: vv / nB for kk, vv in sb.items()}
        print(f"{name:<26} {d['state']:>10.0f} {d['buffer']:>10.0f} {d['decay']:>8.0f} {d['total']:>10.0f} "
              f"{d['total'] / 1024:>8.1f} {d['total'] * 8 / (HV * DK * DV):>11.3f}")


MAMBA2_SHAPES = {'nemotron-h-4b': (112, 8, 64, 128), 'nemotron-3-nano-4b': (96, 8, 80, 128),
                 'granite-4.0-h-1b': (48, 1, 64, 128), 'falcon-h1-7b': (24, 1, 128, 256)}


def main_mamba2(a):
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update

    from statequant.kernels import Mamba2SlotLayout
    dev, m = 'cuda', a.m
    print(f'GPU: {torch.cuda.get_device_name()}  (shared GPU: timings are indicative only); GDN shape {a.gdn_shape} '
          f'(HV {HV}, HK {HK}, d_k {DK}, d_v {DV})')
    x_ = torch.randn(4096, 4096, device=dev)
    for _ in range(300):
        x_ @ x_
    del x_
    fmts = {'bf16 buf8': ('bf16', 8), 'bf16 buf16': ('bf16', 16), 'int4 k g64': (sq.Quantizer(4, 'k', 64), 8),
            'int8 k g32': (sq.Quantizer(8, 'k', 32), 8), 'int4 v g32': (sq.Quantizer(4, 'v', 32), 8)}
    for shape in a.mamba2.split(','):
        H, G, P, N = MAMBA2_SHAPES[shape]
        arms = ['vllm fp32', 'vllm bf16'] + list(fmts)
        rows, nbytes = [], {}
        print(f'\n== {shape}: heads {H}, groups {G}, head_dim {P}, d_state {N}, m = {m} ==', flush=True)
        for B in [int(b) for b in a.batches.split(',')]:
            torch.manual_seed(0)
            nslot = B + 8
            idx = torch.randperm(nslot - 1, device=dev)[:B].to(torch.int32) + 1          # slots 1.., shuffled
            conv = torch.randn(B, H * P + 2 * G * N, device=dev, dtype=torch.bfloat16)
            x = conv[:, :H * P].view(B, H, P)
            Bm = conv[:, H * P:H * P + G * N].view(B, G, N)
            Cm = conv[:, H * P + G * N:].view(B, G, N)
            dt = torch.randn(B, H, device=dev, dtype=torch.bfloat16) - 2
            A = -torch.rand(H, device=dev) * 4
            D = torch.randn(H, device=dev, dtype=torch.bfloat16)
            dtb = torch.randn(H, device=dev, dtype=torch.bfloat16)
            out = torch.empty(B, H, P, device=dev, dtype=torch.bfloat16)
            res = {}
            for name, sdt in (('vllm fp32', torch.float32), ('vllm bf16', torch.bfloat16)):
                S = (torch.randn(nslot, H, P, N, device=dev) * 0.1).to(sdt)
                A_d = A[:, None, None].expand(-1, P, N)
                dt_d = dt[:, :, None].expand(-1, -1, P)
                dtb_d = dtb[:, None].expand(-1, P)
                D_d = D[:, None].expand(-1, P)

                def stock():
                    selective_state_update(S, x, dt_d, A_d, Bm, Cm, D_d, dtb_d, dt_softplus=True,
                                           state_batch_indices=idx, dst_state_batch_indices=idx, out=out)
                res[name] = time_graph(stock, m, a.repeats)
                del S
            for name, (fmt, bb) in fmts.items():
                f = fmt
                if f != 'bf16' and f.axis == 'v' and P % f.group:
                    g = f.group
                    while P % g:
                        g //= 2
                    f = sq.Quantizer(f.bits, 'v', g)
                lay = Mamba2SlotLayout(f, H, G, N, P, m=m, buffer_bits=bb)
                nbytes[name] = lay.nbytes
                blob = lay.alloc(nslot)
                views = lay.views(blob)
                lay.write(views, torch.arange(nslot, device=dev), torch.randn(nslot, H, P, N, device=dev) * 0.1)
                res[name] = time_graph(lambda: lay.step(views, idx, x, dt, Bm, Cm, out, A, D, dtb), m, a.repeats)
                del blob, views
            torch.cuda.empty_cache()
            rows.append((B, res))
            print(f'B={B:<4d} ' + '  '.join(f'{n}: {t:7.1f}' for n, t in res.items()), flush=True)
        w = 11
        print(f'\nper-layer decode step, us ({shape}; min over {a.repeats} repeats of a CUDA graph of {m} steps)')
        print(f"{'batch':>6} " + ' '.join(f'{h:>{max(w, len(h))}}' for h in arms))
        for B, res in rows:
            print(f'{B:>6} ' + ' '.join(f'{res[n]:>{max(w, len(h))}.1f}' for n, h in zip(arms, arms)))
        print('bytes per sequence per layer: ' + ', '.join(
            [f'vllm fp32 {H * P * N * 4}', f'vllm bf16 {H * P * N * 2}'] + [f'{n} {b}' for n, b in nbytes.items()]))

if __name__ == '__main__':
    main()
