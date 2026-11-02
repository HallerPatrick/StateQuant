"""Render the README / docs figures from the CSVs in docs/figures/data (python docs/figures/make_figures.py)."""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
})
MODELS = ['Qwen3.5-0.8B', 'Nemotron-3-Nano-4B', 'Olmo-Hybrid-7B', 'Ling-3.0-tiny']
FAMILY = {'Qwen3.5-0.8B': 'GDN', 'Qwen3.5-4B': 'GDN', 'Nemotron-3-Nano-4B': 'Mamba2', 'Olmo-Hybrid-7B': 'GDN',
          'Ling-3.0-tiny': 'KDA'}
LABEL = {
    'damp_paper_int8': 'DAMP INT8', 'leapquant8': 'LeapQuant-8', 'int8k': 'int8 k g32',
    'leapquant6': 'LeapQuant-6', 'damp_paper_int4': 'DAMP INT4', 'leapquant4': 'LeapQuant-4',
    'pace48': 'PACE-4/8', 'lrb_bdr': 'LRB + BDR', 'gramian6': 'Gramian-6',
    'stepquant6': 'STEPQuant-6', 'nemotron': 'fp16 + SR (Nemotron)', 'pace4': 'PACE-4', 'qmamba_dsq': 'Q-Mamba DSQ',
    'stepquant4': 'STEPQuant-4', 'gramian4_c64': 'Gramian-4, C=64', 'gramian4': 'Gramian-4, C=1',
    'bdr_int4': 'BDR int4', 'int4_g128': 'int4 g128, every token', 'rtn': 'RTN int4, every token',
}
ORDER = ['nemotron', 'damp_paper_int8', 'int8k', 'leapquant8', 'leapquant6', 'damp_paper_int4',
         'leapquant4', 'pace48', 'lrb_bdr', 'gramian6', 'stepquant6', 'pace4', 'qmamba_dsq',
         'stepquant4', 'gramian4_c64', 'gramian4', 'bdr_int4', 'int4_g128', 'rtn']
LINTHRESH, NOISE = 0.01, 0.005


def read(name):
    with open(os.path.join(HERE, 'data', name)) as f:
        return list(csv.DictReader(f))


def symlog_ticks(ax, axis='x'):
    ticks = [-0.05, 0, 0.01, 0.1, 1]
    fmt = FuncFormatter(lambda v, _: '0' if v == 0 else f'{v:+g}')
    if axis == 'x':
        ax.set_xticks(ticks), ax.xaxis.set_major_formatter(fmt), ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_yticks(ticks), ax.yaxis.set_major_formatter(fmt), ax.yaxis.set_minor_formatter(NullFormatter())


def fig_presets():
    rows = [r for r in read('hf_presets.csv') if r['preset'] in LABEL]
    by = {(r['model'], r['preset']): r for r in rows}
    presets = [p for p in ORDER if any((m, p) in by for m in MODELS)]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(12, 0.32 * len(presets) + 1.6), sharey=True)
    for ax, model in zip(axes, MODELS):
        ax.axvspan(-NOISE, NOISE, color=GRID, alpha=0.8, lw=0, zorder=0)
        for y, p in enumerate(presets):
            r = by.get((model, p))
            if r is None:
                ax.text(0.003, y, 'not run', color=MUTED, ha='left', va='center', fontsize=7)
                continue
            d = float(r['d_decode'])
            ax.barh(y, d, height=0.7, color=BLUE if p == 'pace4' else INK2, zorder=2)
            ax.annotate(f'{float(r["resident_bits"]):.1f} b', (d, y), xytext=(4 if d >= 0 else -4, 0),
                        textcoords='offset points', ha='left' if d >= 0 else 'right', va='center', fontsize=7,
                        color=MUTED)
        ax.axvline(0, color=INK2, lw=0.8, zorder=3)
        ax.set_xscale('symlog', linthresh=LINTHRESH, linscale=1.0)
        ax.set_xlim(-0.1, 8)
        symlog_ticks(ax)
        ax.grid(axis='y', visible=False)
        ax.set_title(f'{model}  ({FAMILY[model]})', loc='left')
        ax.tick_params(axis='x', labelsize=7.5)
    axes[0].set_yticks(range(len(presets)), [LABEL[p] for p in presets])
    axes[0].invert_yaxis()
    fig.supxlabel('Δ decode loss vs unquantized state (nats/token, symlog; grey band = ±0.005 noise)  ·  '
                  'labels: resident bits per state value', fontsize=8.5, color=INK2)
    fig.suptitle('Every state-quantization method on four hybrid models (HF reference path, teacher-forced decode)',
                 x=0.01, ha='left', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'hf_presets.png'), dpi=160)
    plt.close(fig)


def grouped(ax, groups, series, value, colors, width=0.8):
    """Vertical grouped bars: one group per entry of groups, one bar per series (missing values skipped)."""
    n = len(series)
    for j, s in enumerate(series):
        xs, ys = [], []
        for i, g in enumerate(groups):
            v = value(g, s)
            if v is not None:
                xs.append(i + (j - (n - 1) / 2) * width / n), ys.append(v)
        ax.bar(xs, ys, width=width / n * 0.92, color=colors[j], label=s, zorder=2)
    ax.set_xticks(range(len(groups)))
    ax.grid(axis='x', visible=False)


def fig_vllm():
    rows = read('vllm_e2e.csv')
    models = ['Qwen3.5-4B', 'Nemotron-3-Nano-4B', 'Olmo-Hybrid-7B', 'Ling-3.0-tiny']
    engines = ['stock bf16', 'plugin bf16', 'int8k', 'pace4', 'int4v']
    names = ['stock\nbf16 state', 'plugin bf16\n(control)', 'int8 k g32', 'int4 k g64\n(PACE-4)', 'int4 v g32']
    by = {(r['model'], r['engine']): r for r in rows}

    def ratio(e, m):
        r, ref = by.get((m, e)), by[(m, 'stock fp32')]
        return None if r is None else float(r['max_concurrency_4232']) / float(ref['max_concurrency_4232'])

    def dloss(e, m):
        r, ref = by.get((m, e)), by[(m, 'stock fp32')]
        return None if r is None else float(r['decode_loss']) - float(ref['decode_loss'])
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.9))
    grouped(a1, engines, models, ratio, colors)
    a1.axhline(1, color=INK2, lw=0.8, zorder=3)
    a1.set_ylim(0.8, 2.05)
    a1.set_ylabel('max concurrency, × stock fp32 state')
    a1.set_title('Concurrent 4232-token requests at a fixed KV budget', loc='left')
    a2.axhspan(-NOISE, NOISE, color=GRID, alpha=0.8, lw=0, zorder=0)
    grouped(a2, engines, models, dloss, colors)
    a2.axhline(0, color=INK2, lw=0.8, zorder=3)
    a2.set_ylabel('Δ decode loss vs stock fp32 state')
    a2.set_title('Teacher-forced decode loss (grey band = ±0.005)', loc='left')
    for ax in (a1, a2):
        ax.set_xticklabels(names, fontsize=8)
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle('vLLM 0.32 end to end: packed recurrent state (statequant plugin) vs stock', x=0.01, ha='left',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(HERE, 'vllm_e2e.png'), dpi=160)
    plt.close(fig)


# -- README results figures: own, lighter theme ------------------------------------------------------------------
CLEAN = {
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'font.size': 10,
    'axes.spines.left': False, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#c9c8c3',
    'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.8, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.labelcolor': INK2, 'xtick.major.size': 0, 'ytick.major.size': 0, 'ytick.minor.size': 0,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'axes.titlepad': 22,
}
STOCK = '#9b9a95'
VIOLET = '#8457d6'


def _heading(ax, title, sub):
    ax.set_title(title, loc='left')
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=MUTED, va='bottom')


def fig_latency():
    rows = read('decode_latency_a100.csv')
    batches = [int(r['batch']) for r in rows]
    series = [('fla fp32 in place', 'stock fp32', STOCK, (0, (4, 2))), ('fla bf16 in place', 'stock bf16', STOCK, '-'),
              ('int4 v g32 fused', 'int4 v g32', ORANGE, '-'), ('int4 k g64 fused', 'PACE-4', BLUE, '-')]
    with plt.rc_context(CLEAN):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ends = []
        for col, name, c, ls in series:
            ys = [float(r[col]) for r in rows]
            ax.plot(batches, ys, color=c, lw=2.4 if c != STOCK else 1.8, ls=ls, marker='o', ms=4.5, zorder=3,
                    markeredgecolor='white', markeredgewidth=1)
            ends.append([ys[-1], name, c])
        ends.sort(reverse=True)                                        # end labels, spread apart on the log axis
        for i in range(1, len(ends)):
            ends[i][0] = min(ends[i][0], ends[i - 1][0] / 1.3)
        for yl, name, c in ends:
            ax.annotate(name, (batches[-1] * 1.12, yl), va='center', fontsize=9, annotation_clip=False,
                        color=c if c != STOCK else INK2, fontweight='bold' if c != STOCK else 'normal')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(batches)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        ax.grid(axis='x', visible=False)
        ax.spines['bottom'].set_visible(True)
        ax.set_xlim(batches[0] / 1.3, batches[-1] * 2.6)
        ax.set_xlabel('batch size')
        ax.set_ylabel('µs per GDN layer per step')
        _heading(ax, 'Decode step latency', 'Qwen3.5-4B shapes, A100 80GB, CUDA graph, lower is better')
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, 'decode_latency.png'), dpi=180)
        plt.close(fig)


def fig_memory():
    rows = read('state_memory_qwen35_4b.csv')
    names = [r['state'] for r in rows]
    mib = [int(r['bytes_per_layer']) * int(r['gdn_layers']) / 2 ** 20 for r in rows]
    ref = mib[0]
    hue = {'int8 k g32': AQUA, 'GGA': VIOLET, 'int4 v g32': ORANGE, 'PACE-4': BLUE}   # as in the latency plot
    colors = [STOCK if n.startswith('stock') else next(c for k, c in hue.items() if n.startswith(k)) for n in names]
    with plt.rc_context(CLEAN):
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        y = range(len(rows))
        ax.barh(y, mib, height=0.58, color=colors, zorder=2)
        for i, v in enumerate(mib):
            ax.annotate(f'{v:.1f} MiB' + (f'  ({ref / v:.1f}x smaller)' if i else ''), (v, i), xytext=(6, 0),
                        textcoords='offset points', va='center', fontsize=9, color=INK2)
        ax.set_yticks(list(y))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0, ref * 1.25)
        ax.grid(visible=False)
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
        _heading(ax, 'Recurrent state memory per sequence', 'Qwen3.5-4B in vLLM (24 GDN layers), decode buffer included, lower is better')
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, 'state_memory.png'), dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    fig_presets(), fig_vllm(), fig_latency(), fig_memory()
