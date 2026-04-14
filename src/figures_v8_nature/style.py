"""Nature/Cell-inspired figure style for PathOmicDRP v8.

Palette: Okabe-Ito colorblind-safe + one accent (Nature house style ~2021+).
Typography: Arial 7pt ticks / 8pt labels / 9pt titles / 10pt bold panel letters.
Spines: bottom+left only; tick length 2.5; axis lw 0.8.
Figure widths: 89mm (single), 120mm (1.5), 183mm (double column).
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito + curated accents
PAL = {
    'genomic':        '#E69F00',   # amber
    'transcriptomic': '#56B4E9',   # sky
    'proteomic':      '#009E73',   # green
    'histology':      '#CC79A7',   # pink
    'fusion':         '#0072B2',   # blue
    'hero':           '#D62828',   # signature red
    'hero_light':     '#F4A9A9',
    'gray_dark':      '#4A4A4A',
    'gray':           '#9A9A9A',
    'gray_light':     '#D9D9D9',
    'pos':            '#4A7C59',
    'neg':            '#C44536',
    'text':           '#222222',
}

# Subtype palette (Lumi/HER2/TNBC/Other)
SUB = {'Luminal':'#4C72B0','HER2+':'#DD8452','TNBC':'#C44E52','Triple-Negative':'#C44E52','Other':'#8D8D8D'}

# MOA palette
MOA = {
    'DNA damage':'#D62828','DNA-damaging':'#D62828',
    'Tubulin':'#0072B2','Microtubule':'#0072B2',
    'Hormone':'#E69F00',
    'Kinase':'#CC79A7','Targeted':'#CC79A7',
    'mTOR/NAD':'#56B4E9',
    'Apoptosis':'#009E73',
}

MM = 1/25.4  # mm -> inch
W_SINGLE = 89 * MM
W_ONEHALF = 120 * MM
W_DOUBLE = 183 * MM


def apply_rc():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'axes.labelweight': 'regular',
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#222222',
        'axes.labelcolor': '#222222',
        'axes.titlepad': 4.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.color': '#222222',
        'ytick.color': '#222222',
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.fontsize': 7,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'mathtext.default': 'regular',
    })


def panel(ax, letter, x=-0.18, y=1.06, size=10):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=size, fontweight='bold', va='top', ha='left', color='#111111')


def despine(ax, left=True, bottom=True):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)


def annot_p(ax, x1, x2, y, p, fs=6.5, bar_h=0.02, color='#333'):
    """Significance bracket with numeric p."""
    ax.plot([x1, x1, x2, x2], [y, y+bar_h, y+bar_h, y], lw=0.7, color=color, clip_on=False)
    if p < 1e-4:
        s = f'p={p:.0e}'.replace('e-0','e-').replace('e+0','e')
    else:
        s = f'p={p:.3f}'
    ax.text((x1+x2)/2, y+bar_h*1.4, s, ha='center', va='bottom', fontsize=fs, color=color)


def sig_stars(p):
    return '***' if p<1e-3 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
