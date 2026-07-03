"""
=====================================================================
 fig_style.py — Unified figure design system for the ICAE2026 paper
=====================================================================
 ONE style for every figure (Part 1/2/3). Blue + white primary,
 Q1-journal quality. All palettes were machine-validated for
 colorblind safety (CVD dE >= 12) and lightness/chroma bands on a
 white surface — do NOT swap hues casually; re-validate if you do.

 Usage in the notebook / any figure script:
     from scripts.analysis.fig_style import (apply_style, MODEL_COLORS,
         LCA_COLORS, seq_cmap, div_cmap, EMPHASIS, INK, savefig)
     apply_style()
=====================================================================
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------ tokens
# Ink & chrome (recessive: the DATA is the darkest thing on the page)
INK = {
    'primary':  '#0b0b0b',   # titles, values
    'secondary': '#52514e',  # axis labels
    'muted':    '#898781',   # tick labels, captions
    'grid':     '#e1e0d9',   # hairline solid gridlines (never dashed)
    'baseline': '#c3c2b7',   # axis spine
    'surface':  '#ffffff',   # figure/axes background (paper = white)
}

# Categorical — the 4 surrogate models, FIXED order & hue (never recycle).
# Validated: CVD worst adjacent dE 21.6 (tritan), all PASS on white.
# Aqua/yellow are sub-3:1 contrast => relief rule: always direct-label values.
MODEL_COLORS = {
    'PI-HGAT':    '#2a78d6',  # blue   — the hero model
    'XGBoost':    '#1baf7a',  # aqua
    'ANN (MLP)':  '#eda100',  # yellow
    'Linear Reg': '#4a3aa7',  # violet
}
EMPHASIS = {'focus': '#2a78d6', 'context': '#c3c2b7'}  # highlight-one form

# Ordinal — 5 embodied LCA modules (A1-A3 -> C1-C4 life-cycle order),
# single-hue blue light->dark. Validated --ordinal: ALL PASS.
# B6 (operational) is plotted separately/contrast — never inside this ramp.
LCA_COLORS = {
    'A1-A3': '#86b6ef',
    'A4-A5': '#5598e7',
    'B2-B3': '#2a78d6',
    'B4':    '#1c5cab',
    'C1-C4': '#0d366b',
}
B6_COLOR = '#1baf7a'   # operational carbon — aqua, distinct from embodied blues

# Sequential ramp (continuous magnitude: TOPSIS closeness, heatmaps)
_SEQ_STEPS = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
# Diverging (SHAP +/-, deltas): blue <-> red with neutral gray midpoint
_DIV_STEPS = ['#0d366b', '#2a78d6', '#86b6ef', '#f0efec', '#f0a4a4', '#d03b3b', '#7a1f1f']


def seq_cmap():
    return LinearSegmentedColormap.from_list('icae_seq', _SEQ_STEPS)


def div_cmap():
    return LinearSegmentedColormap.from_list('icae_div', _DIV_STEPS)


# ------------------------------------------------------------------ rcParams
def apply_style():
    """Apply the unified style. Call ONCE before any figure."""
    mpl.rcParams.update({
        # typography — one sans everywhere (E&B/Q1 figure convention)
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.titleweight': 'bold',
        # ink hierarchy
        'text.color': INK['primary'],
        'axes.labelcolor': INK['secondary'],
        'xtick.color': INK['muted'], 'ytick.color': INK['muted'],
        # recessive chrome: hairline solid grid, no top/right spines
        'axes.grid': True, 'grid.color': INK['grid'],
        'grid.linewidth': 0.6, 'grid.linestyle': '-',
        'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': INK['baseline'], 'axes.linewidth': 0.8,
        # marks: thin, breathing room
        'lines.linewidth': 1.6, 'lines.markersize': 5,
        'patch.linewidth': 0,
        # surfaces & export
        'figure.facecolor': INK['surface'], 'axes.facecolor': INK['surface'],
        'savefig.facecolor': INK['surface'],
        'figure.dpi': 120, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'legend.frameon': False,
    })


def savefig(fig, name, base_dir=None):
    """Save into results/figures/ as both PNG (600 dpi) and PDF (vector)."""
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'results', 'figures')
    os.makedirs(base_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(base_dir, f'{name}.{ext}'))
    print(f'Saved: results/figures/{name}.png|.pdf')
