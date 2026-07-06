# -*- coding: utf-8 -*-
"""Patch NZEB_PIPELINE_ICAE2026_v3.ipynb per the 2026-07-06 Q1 review plan.

Edits (cell indices as of the v3 copy):
  1   seed_all: also seed python `random` (old review minor #B4 leftover)
  10  S2: fixed 6/2/1 scenario split from pi_hgat.data_split (shared w/ step3)
  35  S8: external-test cell reuses data_split.row_to_params
  39  S9 figure-slot markdown: renumber to FIGURE_PLAN (Fig5/Fig6/Fig7)
  40  S9: rename Fig3->Fig5_PredictionPerf, Fig4_BenchmarkBar->FigS2, FigS1 log-y
  42  S10 markdown: renumber Fig4/Fig5 -> Fig6/Fig7, document meta check
  44  S10: full rewrite — meta assert, Fig6 2x2 + S8 star overlay,
      Fig7 learning curve (4 models, 3-seed band, clipped y-axis), tables
  49  S11 figure-slot markdown: renumber to Fig3/Fig4
  50  S11: rename Fig6_Lifespan->Fig3_Lifespan, Fig7_LCEDistribution->Fig4_...
  64  S13: Fig10 two-panel (all modules + embodied zoom w/o B6)
  78  S16: Fig14 horizontal colorbar (was colliding with panel-b ylabel)
"""
import json
import sys

NB = 'NZEB_PIPELINE_ICAE2026_v3.ipynb'

nb = json.load(open(NB, encoding='utf-8'))
cells = nb['cells']


def set_source(i, text, expect=None):
    cur = ''.join(cells[i]['source'])
    if expect is not None and expect not in cur:
        raise SystemExit(f'cell {i}: expected marker {expect!r} not found — aborting')
    cells[i]['source'] = text.splitlines(keepends=True)


def sub(i, old, new, count=1):
    cur = ''.join(cells[i]['source'])
    if old not in cur:
        raise SystemExit(f'cell {i}: substring not found:\n{old}')
    cells[i]['source'] = cur.replace(old, new, count).splitlines(keepends=True)


# --- cell 1: seed python `random` too -------------------------------------- #
sub(1, """def seed_all(s=42):
    np.random.seed(s); torch.manual_seed(s)""",
    """import random

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)""")

# --- cell 10: S2 data load + fixed 6/2/1 split ------------------------------ #
set_source(10, '''print('Loading LHS samples from aggregated_LHS_results.csv...')
from pi_hgat.data_split import (FEATURE_NAMES, SCENARIO_SPLIT, load_lhs_arrays,
                                split_indices, describe_split, row_to_params)

# Surrogate learns GROSS site EUI = f(P1..P7, climate). PV (P8) / BESS (P9) are
# excluded here by design — they enter only via objectives.net_energy() in S11.
# Data loading, feature mapping, combo hashing AND the scenario split all live
# in pi_hgat.data_split — the same module scripts/analysis/step3_robustness.py
# imports, so S10's offline studies can never diverge from this notebook.
df_lhs, samples, X_flat, Y_eui, groups, combo_id = load_lhs_arrays()
Y_eui = Y_eui.reshape(-1, 1)

print(f'X: {X_flat.shape}, Y: {Y_eui.shape}')
print(f'Gross EUI range: {Y_eui.min():.1f} - {Y_eui.max():.1f} kWh/m2/yr')
print(f'Unique parameter combos: {len(set(combo_id))} (paired across {len(set(groups))} scenarios; '
      f'250 LHS rows/scenario, one pair collides after round(4))')

# Fixed 6/2/1 scenario split (deliberate — see pi_hgat/data_split.py docstring):
#   train spans the FULL ΔT hull [0, 4.472] incl. both extremes -> the surrogate
#   never extrapolates ΔT when the MOO queries it at 0 / +2.03 / +4.47 °C;
#   val = 2 scenarios (stable early stopping); test = scenario 3 (ΔT 2.665, a
#   genuine ±0.5 °C interpolation gap). Climate EXTRApolation evidence = LOSO
#   9-fold (S10); parameter generalization = external LHS seed 2810 (S8).
idx_train, idx_val, idx_test = split_indices(groups)
print('\\nScenario split (fixed 6 train / 2 val / 1 test):')
print(describe_split(groups, X_flat).to_string(index=False))
print(f'Split: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test')
''', expect='GroupShuffleSplit')

# --- cell 35: S8 external test uses shared row_to_params -------------------- #
sub(35, """ext_dataset, ext_X, ext_scen = [], [], []
for _, row in ext_df.iterrows():
    s = {
        'P1_Wall_U': 1.0 / row['@@P1_Wall_R@@'],
        'P2_Roof_U': 1.0 / row['@@P2_Roof_R@@'],
        'P3_Roof_Reflectance': 1.0 - row['@@P3_Roof_Abs@@'],
        'P4_Win_U': row['@@P4_U@@'],
        'P4_Win_SHGC': row['@@P4_SHGC@@'],
        'P5_COP': row['@@P5_COP@@'],
        'P6_Cool_SP': row['@@P6_ClgSetp@@'],
        'P7_LPD': row['@@P7_LPD@@'],
        'Climate_DeltaT': row['Climate_DeltaT'],
    }
    x_vec = [s[k] for k in FEATURE_NAMES]""",
    """ext_dataset, ext_X, ext_scen = [], [], []
for _, row in ext_df.iterrows():
    s = row_to_params(row)   # same mapping as training data (pi_hgat.data_split)
    x_vec = [s[k] for k in FEATURE_NAMES]""")

# --- cell 39: S9 figure-slot markdown renumber ------------------------------ #
set_source(39, '''> ### 📊 FIGURE SLOT — **Fig. 5 · Predicted vs Actual (4 panels)** *(PART 1)*
> 4 panel PI-HGAT ★ / XGBoost / ANN / LR on the test set (scenario 3); identity line + R² annotation; màu theo `MODEL_COLORS`. **Data:** `hgat_t/hgat_p`, `Yte`, `xgb_p`, `ann_p`, `lr_p`. **Status:** ✅ `Fig5_PredictionPerf`
>
> ### 📊 FIGURE SLOT — **Fig. 6 · Benchmark & robustness (2×2)** *(PART 1, produced in S10)*
> (a) test R² mean±σ (10 seeds) + ★ giá trị single-run của notebook; (b) seed boxplot; (c) train-vs-test R² (overfitting check); (d) train time (log s). **Status:** ✅ `Fig6_BenchmarkRobustness` (S10)
>
> ### 📊 FIGURE SLOT — **Fig. 7 · Learning curve — sample-size sufficiency** *(PART 1, produced in S10)*
> Test R² theo n combos/scenario (25→249) × 4 model × 3 seeds (band = min–max); chứng minh 250 LHS/climate là đủ (249/62,500 = 0.4% full factorial). **Status:** ✅ `Fig7_LearningCurve` (S10)
''', expect='FIGURE SLOT')

# --- cell 40: S9 figure renames + FigS1 log-y ------------------------------- #
sub(40, "# FIGURE 5, 6 + supplementary training-loss curve",
        "# FIG 5 (pred-vs-actual) + supplementary FigS1/FigS2 — numbering per FIGURE_PLAN")
sub(40, "# --- FIG 3: Predicted vs Actual (4-panel) ---",
        "# --- FIG 5: Predicted vs Actual (4-panel) ---")
sub(40, "savefig(fig, 'Fig3_PredictionPerf')", "savefig(fig, 'Fig5_PredictionPerf')")
sub(40, "# --- FIG 4: Benchmark Bar Chart (R2) — placeholder for the planned 2x2 (multi-seed) panel ---",
        "# --- FIG S2 (supplementary): single-run benchmark bar — the paper's Fig. 6 is the\n# multi-seed 2x2 produced in S10; this quick bar stays as a supplement only. ---")
sub(40, "savefig(fig, 'Fig4_BenchmarkBar')", "savefig(fig, 'FigS2_BenchmarkBar')")
sub(40, """ax.set_ylabel('Loss (MSE + bound penalty)', fontname='Arial')""",
        """ax.set_yscale('log')   # initial loss ~1.4e4 vs final ~1 — linear axis hides convergence
ax.set_ylabel('Loss (MSE + bound penalty, log)', fontname='Arial')""")

# --- cell 42: S10 markdown -------------------------------------------------- #
set_source(42, '''## PART 1 · Section 10 — Robustness & Validation Studies (Fig. 6, Fig. 7, Tables)
Consumes artifacts from `scripts/analysis/step3_robustness.py` (run offline:
`python scripts/analysis/step3_robustness.py --study all`). The script imports the
**same** `pi_hgat.data_split` (fixed 6/2/1 scenario split) and `TRAIN_PARAMS` as this
notebook; the cell below first **verifies `results/step3_meta.json` against the live
config** and refuses to plot stale artifacts. The notebook's own S6/S8 single-run
result is overlaid (★) so both evidence sources are directly comparable.
- **Fig. 6 (2×2):** test R² mean±σ (10 seeds) · seed boxplot · train-vs-test R² (overfitting) · training time.
- **Fig. 7:** sample-size learning curve — 4 models × 3 seeds (band = min–max), 25→249 combos/scenario, tested on held-out scenario 3 — evidence for the "is 250 LHS/scenario enough?" question.
- **Tables:** LOSO 9-fold per-scenario MAE (climate generalization, incl. SSP585-2080s ΔT=4.47) · held-out-combo split (parameter generalization) · physics monotonicity-loss ablation.
''', expect='Section 10')

# --- cell 44: S10 full rewrite ---------------------------------------------- #
set_source(44, '''# S10 — robustness & validation studies. Input chain: pi_hgat.data_split (split
# shared with step3 script) + df_results from S8 (in-notebook single run) +
# step3 CSV artifacts (verified against live config via step3_meta.json).
from scripts.analysis.fig_style import apply_style, savefig, MODEL_COLORS
from pi_hgat.config import TRAIN_PARAMS, GNN_PARAMS
from pi_hgat.data_split import FEATURE_NAMES, SCENARIO_SPLIT
import os, json
apply_style()

ORDER = ['PI-HGAT', 'XGBoost', 'ANN (MLP)', 'Linear Reg']

# --- provenance check: refuse to plot artifacts that do not match this notebook ---
META_PATH = 'results/step3_meta.json'
assert os.path.exists(META_PATH), (
    'step3 artifacts not found — run: python scripts/analysis/step3_robustness.py --study all')
meta = json.load(open(META_PATH, encoding='utf-8'))
_live = {'scenario_split': SCENARIO_SPLIT, 'train_params': dict(TRAIN_PARAMS),
         'gnn_params': dict(GNN_PARAMS), 'feature_names': list(FEATURE_NAMES)}
_stale = [k for k, v in _live.items() if meta.get(k) != v]
if meta.get('smoke'):
    _stale.append('smoke-run artifacts')
assert not _stale, ('step3 artifacts are STALE (mismatch: ' + ', '.join(_stale) +
                    ') — re-run: python scripts/analysis/step3_robustness.py --study all')
print(f"step3 artifacts verified against notebook config "
      f"(generated {meta['timestamp']}, study={meta['study']})")

ms = pd.read_csv('results/step3_multiseed.csv')

# --- FIG 6 (2x2): benchmark & robustness ---
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# (a) test R2, mean +/- sigma over seeds; star = this notebook's S6/S8 single run
ax = axes[0, 0]
agg = ms.groupby('model')['r2_test'].agg(['mean', 'std']).reindex(ORDER)
bars = ax.bar(agg.index, agg['mean'], yerr=agg['std'], capsize=3,
              color=[MODEL_COLORS[m] for m in agg.index], alpha=0.9)
for b, v in zip(bars, agg['mean']):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015, f'{v:.3f}',
            ha='center', fontsize=8, fontname='Arial')
star_lbl = False
for k, m in enumerate(ORDER):
    if m in df_results.index:
        ax.scatter(k, df_results.loc[m, 'R²'], marker='*', s=110, color='#7a1f1f',
                   zorder=5, label=None if star_lbl else 'this notebook (S8 single run)')
        star_lbl = True
ax.set_ylabel('Test R² (mean ± σ, 10 seeds)')
ax.set_ylim(0.5, 1.08)
ax.legend(prop={'size': 7}, loc='lower right')
ax.set_title('(a) Accuracy across seeds', fontweight='bold')

# (b) boxplot of test R2 across seeds
ax = axes[0, 1]
data = [ms.loc[ms.model == m, 'r2_test'].values for m in ORDER]
bp = ax.boxplot(data, tick_labels=ORDER, patch_artist=True, widths=0.55)
for patch, m in zip(bp['boxes'], ORDER):
    patch.set_facecolor(MODEL_COLORS[m]); patch.set_alpha(0.75)
ax.set_ylabel('Test R² (10 seeds)')
ax.set_title('(b) Seed robustness (XGB/LR deterministic at fixed split)', fontweight='bold')

# (c) train vs test R2 (overfitting check)
ax = axes[1, 0]
for m in ORDER:
    sub = ms[ms.model == m]
    ax.scatter(sub['r2_train'], sub['r2_test'], s=25, alpha=0.8,
               color=MODEL_COLORS[m], label=m)
lims = [0.75, 1.005]
ax.plot(lims, lims, '--', lw=1, color='#c3c2b7')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Train R²'); ax.set_ylabel('Test R²')
ax.set_title('(c) Overfitting check', fontweight='bold')
ax.legend(prop={'size': 7}, loc='lower right')

# (d) training time (log scale)
ax = axes[1, 1]
tim = ms.groupby('model')['fit_seconds'].mean().reindex(ORDER)
bars = ax.bar(tim.index, tim.values, color=[MODEL_COLORS[m] for m in tim.index], alpha=0.9)
for b, v in zip(bars, tim.values):
    ax.text(b.get_x() + b.get_width()/2, v * 1.15, f'{v:.3g}s',
            ha='center', fontsize=8, fontname='Arial')
ax.set_yscale('log')
ax.set_ylabel('Mean training time (s, log)')
ax.set_title('(d) Training cost', fontweight='bold')

for ax in axes.flatten():
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontname('Arial'); t.set_fontsize(7)
plt.tight_layout()
savefig(fig, 'Fig6_BenchmarkRobustness')
plt.show()

# --- FIG 7: sample-size learning curve (is 250 LHS/scenario enough?) ---
lc = pd.read_csv('results/step3_learncurve.csv')
Y_FLOOR = -0.05                       # display floor; values below are clipped & flagged
fig, ax = plt.subplots(figsize=(6.5, 4))
for m in ORDER:
    sub = lc[lc.model == m]
    if not len(sub):
        continue
    agg = sub.groupby('n_per_scenario')['r2_test'].agg(['mean', 'min', 'max']).sort_index()
    ax.plot(agg.index, agg['mean'].clip(lower=Y_FLOOR), marker='o', ms=4, lw=1.8,
            color=MODEL_COLORS[m], label=m)
    ax.fill_between(agg.index, agg['min'].clip(lower=Y_FLOOR),
                    agg['max'].clip(lower=Y_FLOOR), color=MODEL_COLORS[m], alpha=0.15, lw=0)
    for n_val, mu in agg['mean'].items():   # flag clipped points with their true value
        if mu < Y_FLOOR:
            ax.annotate(f'{mu:.1f}', (n_val, Y_FLOOR), xytext=(0, -11),
                        textcoords='offset points', ha='center', fontsize=6,
                        color=MODEL_COLORS[m])
# this notebook's S6 model (full 249 combos) for cross-reference
if 'PI-HGAT' in df_results.index:
    ax.scatter([249], [df_results.loc['PI-HGAT', 'R²']], marker='*', s=140,
               color='#7a1f1f', zorder=6, label='this notebook (S6/S8)')
ax.axvline(249, color='#c3c2b7', ls='--', lw=1.2)
ax.set_ylim(Y_FLOOR, 1.02)
ax.text(0.985, 0.05,
        'current design: 249 unique combos/scenario\\n'
        '(250 LHS runs; 0.4% of the 5⁶×4 = 62,500 full factorial)\\n'
        f'values < {Y_FLOOR} clipped (printed below markers)',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5,
        fontname='Arial', color='#6b6a63')
ax.set_xlabel('LHS combos per climate scenario (train, 6 scenarios)')
ax.set_ylabel('Test R² (held-out scenario 3, mean of 3 seeds)')
ax.set_title('Sample-size sufficiency', fontweight='bold')
ax.legend(prop={'size': 7}, loc='center right')
savefig(fig, 'Fig7_LearningCurve')
plt.show()

# --- Table: LOSO per-scenario MAE (climate generalization) ---
loso = pd.read_csv('results/step3_loso.csv')
t_loso = loso.pivot_table(index=['fold_scenario', 'delta_t'],
                          columns='model', values='mae')[ORDER].round(2)
print('LOSO — test MAE (kWh/m²/yr) per held-out climate scenario:')
display(t_loso)
t_loso.to_csv('results/step3_table_loso_mae.csv')

# --- Table: parameter generalization (held-out combos, all scenarios) ---
combo = pd.read_csv('results/step3_combosplit.csv').set_index('model').reindex(ORDER)
print('\\nParameter generalization — unseen combos (GroupSplit by combo_id):')
display(combo.round(4))

# --- Table: physics-loss ablation ---
abl = pd.read_csv('results/step3_ablation.csv')
t_abl = abl.groupby('variant')[['r2_test', 'rmse', 'viol_wallU',
                                'viol_cop', 'viol_deltaT']].mean().round(4)
print('\\nPhysics monotonicity-loss ablation (mean over 3 seeds; viol = finite-diff violation rate):')
display(t_abl)
''', expect='step3_robustness')

# --- cell 49: S11 figure-slot markdown renumber ------------------------------ #
sub(49, 'Fig. 6 · Component lifespans vs 20-yr study period',
        'Fig. 3 · Component lifespans vs 20-yr study period')
sub(49, 'Fig. 7 · Embodied LCE by module @ max renovation',
        'Fig. 4 · Embodied LCE by module @ max renovation')

# --- cell 50: rename lifespans/embodied figures ------------------------------ #
sub(50, '# FIGURE 3 & 4', '# FIG 3 (lifespans) & FIG 4 (embodied LCE @ max) — numbering per FIGURE_PLAN')
sub(50, "savefig(fig, 'Fig6_Lifespan')", "savefig(fig, 'Fig3_Lifespan')")
sub(50, "savefig(fig, 'Fig7_LCEDistribution')", "savefig(fig, 'Fig4_LCEDistribution')")

# --- cell 64: Fig10 two-panel (B6 dwarfs embodied on a single axis) ---------- #
sub(64, """x = np.arange(len(modules)); w = 0.26
shades = ['#c3c2b7', '#2a78d6', '#0d366b']
fig, ax = plt.subplots(figsize=(8.5, 4.5))
for j, (label, v) in enumerate(lce.items()):
    bars = ax.bar(x + (j - 1) * w, v, width=w, label=label, color=shades[j])
ax.set_xticks(x); ax.set_xticklabels(modules)
ax.set_ylabel('LCE (tCO2eq over 20 yr)')
ax.set_title('Embodied vs operational carbon across retrofit depth (NZEB paradox)')
ax.legend(prop={'size': 8})
savefig(fig, 'Fig10_LCEComparison')
plt.show()""",
    """w = 0.26
shades = ['#c3c2b7', '#2a78d6', '#0d366b']
emb_modules = [m for m in modules if m != 'B6']   # B6 (~10^3 t) dwarfs embodied (~10^1 t)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 4.3),
                               gridspec_kw={'width_ratios': [1.15, 1]})
x = np.arange(len(modules))
for j, (label, v) in enumerate(lce.items()):
    axa.bar(x + (j - 1) * w, v, width=w, label=label, color=shades[j])
axa.set_xticks(x); axa.set_xticklabels(modules)
axa.set_ylabel('LCE (tCO2eq over 20 yr)')
axa.set_title('(a) All modules — B6 operational dominates', fontweight='bold')
axa.legend(prop={'size': 8})
xe = np.arange(len(emb_modules))
for j, (label, v) in enumerate(lce.items()):
    ve = [v[modules.index(m)] for m in emb_modules]
    axb.bar(xe + (j - 1) * w, ve, width=w, color=shades[j])
axb.set_xticks(xe); axb.set_xticklabels(emb_modules)
axb.set_ylabel('LCE (tCO2eq over 20 yr)')
axb.set_title('(b) Embodied modules only (B6 excluded)', fontweight='bold')
fig.suptitle('Embodied vs operational carbon across retrofit depth (NZEB paradox)',
             fontweight='bold', fontsize=10)
plt.tight_layout()
savefig(fig, 'Fig10_LCEComparison')
plt.show()""")

# --- cell 78: Fig14 colorbar layout fix -------------------------------------- #
sub(78, """sm = ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, ax=list(axes[:3]), fraction=0.02, pad=0.01, label='Zone importance (GNNExplainer)')

ax = axes[3]
ax.scatter(deg, zone_imp, s=32, color='#2a78d6')""",
    """sm = ScalarMappable(norm=norm, cmap=cmap)
# horizontal colorbar under the floor plans — a vertical one stole width from
# panel (b) and its label collided with the scatter's y-axis label
fig.colorbar(sm, ax=list(axes[:3]), orientation='horizontal', fraction=0.05,
             pad=0.06, label='Zone importance (GNNExplainer)')

ax = axes[3]
ax.scatter(deg, zone_imp, s=32, color='#2a78d6')""")
sub(78, "ax.set_ylabel('Mean importance')", "ax.set_ylabel('Mean zone importance')")

json.dump(nb, open(NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('patched', NB, f'({len(cells)} cells)')
