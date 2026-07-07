# -*- coding: utf-8 -*-
"""Patch #10 (user feedback 2026-07-07): scenario renaming + display cleanups.

1+2. Remove the red-star overlays in Fig6(a) and Fig7 (single-run cross-check
     was confusing/redundant for paper figures; consistency statement moves
     to the manuscript text instead).
3.   Fig5 XGBoost panel: annotate the systematic offset (diagnosed tree
     DeltaT-interpolation failure) so readers don't mistake it for a bug.
4.   Canonical scenario IDs everywhere: 1_Baseline/2..9 -> Baseline/S1..S8
     with GCM+SSP+horizon labels from pi_hgat.data_split.SCENARIO_LABELS
     (folders + CSVs already renamed/regenerated).
5.   Per-section wall-time recording (SECTION_TIMES) + new final section S17:
     section-time table/figure + per-model computational cost comparison
     (train time, per-eval latency, projected MOO cost).
"""
import json
import re
import uuid

NB = 'NZEB_PIPELINE_ICAE2026_v3.ipynb'
nb = json.load(open(NB, encoding='utf-8'))
cells = nb['cells']


def sub(i, old, new, count=1):
    cur = ''.join(cells[i]['source'])
    if old not in cur:
        raise SystemExit(f'cell {i}: substring not found:\n{old[:200]}')
    cells[i]['source'] = cur.replace(old, new, count).splitlines(keepends=True)


def code_cell(src):
    return {'cell_type': 'code', 'execution_count': None, 'id': uuid.uuid4().hex[:8],
            'metadata': {}, 'outputs': [], 'source': src.splitlines(keepends=True)}


def md_cell(src):
    return {'cell_type': 'markdown', 'id': uuid.uuid4().hex[:8], 'metadata': {},
            'source': src.splitlines(keepends=True)}


# ---------------- cell 1: SECTION_TIMES init -------------------------------- #
sub(1, "def seed_all(s=42):",
       "SECTION_TIMES = {}   # wall-clock seconds per section, summarized in S17\n\ndef seed_all(s=42):")

# ---------------- cell 8 markdown: folder names ----------------------------- #
sub(8, "`data/jEPlus-LHS/{1_Baseline, 2..9}/LHS-*/eplustbl.csv`",
       "`data/jEPlus-LHS/{Baseline, S1..S8}/LHS-*/eplustbl.csv`")

# ---------------- cell 10: comment scenario IDs ----------------------------- #
sub(10, "#   (stable early stopping); test = scenario 3 (ΔT 2.665, a genuine ±0.5 °C\n"
        "#   interpolation gap). Scenario 5 (SSP585-2080s, ΔT 4.472) is held out of",
        "#   (stable early stopping); test = S2 (ΔT 2.665, a genuine ±0.5 °C\n"
        "#   interpolation gap). S4 (SSP5-8.5 2080s, ΔT 4.472) is held out of")
sub(10, "#   external LHS seed 2810 at scenario 5, merged with its held-out MAIN rows (S8b).",
        "#   external LHS seed 2810 at S4, merged with its held-out MAIN rows (S8b).")
sub(10, "f'(scenario 5, n=250, held out entirely -> S8 extrapolation test)')",
        "f'(S4, n=250, held out entirely -> S8 extrapolation test)')")

# ---------------- cell 14: S3 figures — canonical labels -------------------- #
sub(14, "from scripts.analysis.fig_style import apply_style, savefig",
        "from scripts.analysis.fig_style import apply_style, savefig\nfrom pi_hgat.data_split import scenario_label")
sub(14, "ax.set_xticklabels([f'{s}\\n(+{scen_dt[s]:.1f}°C)' for s in scen_order], fontsize=7)",
        "ax.set_xticklabels([scenario_label(s) for s in scen_order], fontsize=6)")
sub(14, "base = d[d.Scenario == '1_Baseline'].drop_duplicates('combo')",
        "base = d[d.Scenario == 'Baseline'].drop_duplicates('combo')")
sub(14, """for s in scen_order:
    if s == '1_Baseline':
        continue
    sub = d[d.Scenario == s]""",
    """for s in scen_order:
    if s == 'Baseline':
        continue
    sub = d[d.Scenario == s]""")
sub(14, "label_offset = {'7': (-9, 9), '2': (9, -15)}",
        "label_offset = {'S6': (-11, 9), 'S1': (11, -15)}")
sub(14, """for s in scen_order:
    if s == '1_Baseline':
        continue
    dx, dy = label_offset.get(s, (0, 9))""",
    """for s in scen_order:
    if s == 'Baseline':
        continue
    dx, dy = label_offset.get(s, (0, 9))""")
sub(14, "futures = scen_dt.drop('1_Baseline')",
        "futures = scen_dt.drop('Baseline')")
sub(14, "pick = ['1_Baseline', (futures - futures.median()).abs().idxmin(), futures.idxmax()]",
        "pick = ['Baseline', (futures - futures.median()).abs().idxmin(), futures.idxmax()]")
sub(14, """labels3 = [f'Baseline (Sc. {pick[0]})\\n(+0.0°C)',
           f'Median future (Sc. {pick[1]})\\n(+{scen_dt[pick[1]]:.1f}°C)',
           f'Worst 2080s (Sc. {pick[2]})\\n(+{scen_dt[pick[2]]:.1f}°C)']""",
    """labels3 = [f'{role}\\n{scenario_label(p)}' for role, p in
           zip(['— Baseline —', '— Median future —', '— Worst 2080s —'], pick)]""")

# ---------------- cell 32 markdown: tier scenario IDs ------------------------ #
sub(32, "- **(a) Combo generalization at seen climate** (scenarios 1_Baseline, 2) — the model",
        "- **(a) Combo generalization at seen climate** (scenarios Baseline, S1) — the model")
sub(32, "- **(b) Combo + climate extrapolation** (scenario 5, SSP585-2080s, ΔT=4.472) — this",
        "- **(b) Combo + climate extrapolation** (S4, ACCESS-CM2 SSP5-8.5 2080s, ΔT=4.472) — this")

# ---------------- cell 35: S8 external tiers ------------------------------- #
sub(35, "seen_df = ext_df[ext_df.Scenario.astype(str).isin(['1_Baseline', '2'])].reset_index(drop=True)",
        "seen_df = ext_df[ext_df.Scenario.astype(str).isin(['Baseline', 'S1'])].reset_index(drop=True)")
sub(35, "# at scenario 5, which is excluded from train/val entirely (pi_hgat/data_split.py).",
        "# at S4, which is excluded from train/val entirely (pi_hgat/data_split.py).")
sub(35, "# since scenario 5 was ALSO in train at the time, a good score there only proved",
        "# since that scenario was ALSO in train at the time, a good score there only proved")
sub(35, "# ---- (b) COMBO + CLIMATE EXTRAPOLATION: scenario 5 (SSP585-2080s) ----",
        "# ---- (b) COMBO + CLIMATE EXTRAPOLATION: S4 (ACCESS-CM2 SSP5-8.5 2080s) ----")
sub(35, "print(f'\\n===== (b) COMBO + CLIMATE EXTRAPOLATION: scenario 5 / SSP585-2080s, ΔT=4.472 '",
        "print(f'\\n===== (b) COMBO + CLIMATE EXTRAPOLATION: S4 / ACCESS-CM2 SSP5-8.5 2080s, ΔT=4.472 '")
sub(35, "print('\\nPI-HGAT external MAE per scenario (kWh/m2/yr) — scenario 5 row is external-combo-only (n=150):')",
        "print('\\nPI-HGAT external MAE per scenario (kWh/m2/yr) — S4 row is external-combo-only (n=150):')")

# ---------------- cell 40: Fig5 XGBoost annotation --------------------------- #
sub(40, """    lims = [min(yt.min(), yp.min()) - 5, max(yt.max(), yp.max()) + 5]
    ax.plot(lims, lims, '--', lw=1.5, color='#c3c2b7')
    r2 = df_results.loc[name, 'R²']
    ax.set_title(f'{name} (R² = {r2:.4f})', fontname='Arial', fontweight='bold')""",
    """    lims = [min(yt.min(), yp.min()) - 5, max(yt.max(), yp.max()) + 5]
    ax.plot(lims, lims, '--', lw=1.5, color='#c3c2b7')
    r2 = df_results.loc[name, 'R²']
    ax.set_title(f'{name} (R² = {r2:.4f})', fontname='Arial', fontweight='bold')
    if name == 'XGBoost':
        # Diagnosed, expected behavior — NOT a bug: trees are piecewise-constant
        # in Climate_DeltaT and the test scenario S2 (ΔT=2.67°C) falls inside the
        # 2.18–3.14°C training gap, so predictions snap to the nearest trained
        # ΔT branch → a systematic offset band (verified by counterfactual ΔT
        # swap in review + the staircase PDP in S15b).
        bias = float(np.mean(np.asarray(yp) - np.asarray(yt)))
        ax.text(0.03, 0.97,
                f'systematic offset {bias:+.1f} kWh/m²/yr:\\ntrees cannot interpolate ΔT inside the\\n2.18–3.14°C training gap (test = S2, ΔT 2.67°C)',
                transform=ax.transAxes, ha='left', va='top', fontsize=7,
                color='#6b6a63', style='italic')""")

# ---------------- cell 42 markdown: scenario IDs ----------------------------- #
sub(42, "25→249 combos/scenario, tested on held-out scenario 3",
        "25→249 combos/scenario, tested on held-out scenario S2")
sub(42, "(fixed 5/2/1 scenario split + dedicated scenario-5 extrapolation holdout)",
        "(fixed 5/2/1 scenario split + dedicated S4 extrapolation holdout)")

# ---------------- cell 44: remove stars + label updates ---------------------- #
sub(44, """for b, v in zip(bars, agg['mean']):
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
ax.set_title('(a) Accuracy across seeds', fontweight='bold')""",
    """for b, v in zip(bars, agg['mean']):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015, f'{v:.3f}',
            ha='center', fontsize=8, fontname='Arial')
ax.set_ylabel('Test R² (mean ± σ, 10 seeds)')
ax.set_ylim(0.5, 1.08)
ax.set_title('(a) Accuracy across seeds', fontweight='bold')""")

sub(44, """# this notebook's S6 model (full 249 combos) for cross-reference
if 'PI-HGAT' in df_results.index:
    ax.scatter([249], [df_results.loc['PI-HGAT', 'R²']], marker='*', s=140,
               color='#7a1f1f', zorder=6, label='this notebook (S6/S8)')
ax.axvline(249, color='#c3c2b7', ls='--', lw=1.2)""",
    """ax.axvline(249, color='#c3c2b7', ls='--', lw=1.2)""")

sub(44, "ax.set_ylabel('Test R² (held-out scenario 3, mean of 3 seeds)')",
        "ax.set_ylabel('Test R² (held-out scenario S2, mean of 3 seeds)')")
sub(44, "    # (a) combined R2 vs dT per model. Folds 1_Baseline (dT=0) and 5 (dT=4.47)",
        "    # (a) combined R2 vs dT per model. Folds Baseline (dT=0) and S4 (dT=4.47)")

# ---------------- all timer cells: record SECTION_TIMES ---------------------- #
pat = re.compile(
    r"print\(f'\\n\[Section (\w+)\] Execution time: \{time\.time\(\) - section_(\w+)_start:\.2f\} seconds'\)")
n_timers = 0
for c in cells:
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    m = pat.search(src)
    if m:
        n, var = m.group(1), m.group(2)
        new = (f"SECTION_TIMES['S{n}'] = time.time() - section_{var}_start\n"
               f"print(f'\\n[Section {n}] Execution time: {{SECTION_TIMES[\"S{n}\"]:.2f}} seconds')")
        c['source'] = src.replace(m.group(0), new).splitlines(keepends=True)
        n_timers += 1
if n_timers != 16:
    raise SystemExit(f'expected 16 timer cells, patched {n_timers}')

# ---------------- new final section S17: computational cost ------------------ #
md17 = '''## Section 17 — Computational Cost Summary
Wall-clock time per pipeline section (this run) and the per-model cost comparison for the
paper's results section: training time (multi-seed mean from S10 artifacts), single-sample
inference latency (as used inside the NSGA-III loop), and the projected MOO cost if the
pipeline ran on each surrogate. PI-HGAT pays a graph-build + message-passing premium per
evaluation — the table quantifies exactly how much accuracy-vs-cost trade-off the paper
is claiming.
'''

code17 = '''# S17 — computational cost summary
from scripts.analysis.fig_style import apply_style, savefig, MODEL_COLORS
import time as _t
apply_style()

ORDER = ['PI-HGAT', 'XGBoost', 'ANN (MLP)', 'Linear Reg']
SECTION_DESC = {
    'S1': 'KG input (Neo4j→PyG)', 'S2': 'LHS load + split', 'S3': 'Climate fingerprint',
    'S4': 'HeteroData build', 'S5': 'Model init', 'S6': 'PI-HGAT training',
    'S7': 'Baseline training', 'S8': 'Benchmark + external', 'S9': 'Prediction figures',
    'S10': 'Robustness (render)', 'S11': 'MOO setup', 'S12': 'NSGA-III (baseline)',
    'S13': 'TOPSIS + NZE', 'S14': 'Sensitivity + climate MOO', 'S15': 'GNNExplainer + PDP',
    'S16': 'Spatial explanation',
}

# ---- (a) per-section wall time ----
sec_df = pd.DataFrame([{'section': k, 'description': SECTION_DESC.get(k, ''),
                        'seconds': round(v, 2)} for k, v in SECTION_TIMES.items()])
sec_df['pct'] = (100 * sec_df['seconds'] / sec_df['seconds'].sum()).round(1)
print(f'Total pipeline wall time: {sec_df.seconds.sum():.1f}s '
      f'({sec_df.seconds.sum()/60:.1f} min)')
display(sec_df)

# ---- (b) per-model computational cost ----
ms_t = pd.read_csv('results/step3_multiseed.csv').groupby('model')['fit_seconds'].agg(['mean', 'std'])

def _lat(fn, n=100):
    fn()                                   # warm-up
    t0 = _t.perf_counter()
    for _ in range(n):
        fn()
    return (_t.perf_counter() - t0) / n * 1000   # ms/eval

p_probe = levels_to_params([0] * 9)
xv = np.array([[p_probe[k] for k in FEATURE_NAMES]])
xs_probe = scaler.transform(xv)
def _ann_pred():
    with torch.no_grad():
        ann(torch.tensor(xs_probe, dtype=torch.float).to(device))
lat = {'PI-HGAT': _lat(lambda: problem.predict_gross_eui(p_probe)),
       'XGBoost': _lat(lambda: xgb_model.predict(xs_probe)),
       'ANN (MLP)': _lat(_ann_pred),
       'Linear Reg': _lat(lambda: lr_model.predict(xs_probe))}

N_EVALS = 156 * 200 * 3   # pop x gens x (S12 + 2 climate-aware runs in S14)
rows = []
for m in ORDER:
    moo_min = N_EVALS * lat[m] / 1000 / 60
    rows.append({'model': m,
                 'train_s (multiseed mean±σ)': f"{ms_t.loc[m, 'mean']:.1f} ± {ms_t.loc[m, 'std']:.1f}",
                 'inference (ms/eval)': round(lat[m], 3),
                 f'projected MOO ({N_EVALS:,} evals, min)': round(moo_min, 1)})
cost_df = pd.DataFrame(rows).set_index('model')
moo_col = cost_df.columns[-1]
cost_df['Δ vs PI-HGAT (min)'] = (cost_df[moo_col] - cost_df.loc['PI-HGAT', moo_col]).round(1)
print(f"\\nMeasured PI-HGAT MOO sections this run: S12+S14 = "
      f"{SECTION_TIMES.get('S12', float('nan')) + SECTION_TIMES.get('S14', float('nan')):.0f}s "
      f"(cross-check for the projection; overhead beyond raw evals = NSGA-III bookkeeping)")
display(cost_df)
cost_df.to_csv('results/computational_cost.csv')
sec_df.to_csv('results/section_times.csv', index=False)

# ---- figure: section times + projected MOO cost ----
fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.5, 4.4),
                               gridspec_kw={'width_ratios': [1.3, 1]})
order_sec = sec_df.sort_values('seconds')
axa.barh(order_sec['section'], order_sec['seconds'], color='#2a78d6', alpha=0.9)
for y, (s, d) in enumerate(zip(order_sec['seconds'], order_sec['description'])):
    axa.text(s * 1.02 + 0.3, y, f'{s:.1f}s — {d}', va='center', fontsize=6.5,
             fontname='Arial', color='#0b0b0b')
axa.set_xscale('log')
axa.set_xlabel('Wall time (s, log)')
axa.set_title('(a) Pipeline wall time per section (this run)', fontweight='bold')

vals = [N_EVALS * lat[m] / 1000 / 60 for m in ORDER]
bars = axb.bar(ORDER, vals, color=[MODEL_COLORS[m] for m in ORDER], alpha=0.9)
for b, v in zip(bars, vals):
    axb.text(b.get_x() + b.get_width() / 2, v * 1.15, f'{v:.1f} min',
             ha='center', fontsize=8, fontname='Arial')
axb.set_yscale('log')
axb.set_ylabel(f'Projected MOO time (min, log) — {N_EVALS:,} evals')
axb.set_title('(b) Surrogate cost inside NSGA-III', fontweight='bold')
for t in axb.get_xticklabels():
    t.set_fontsize(7)
plt.tight_layout()
savefig(fig, 'FigS3_ComputeCost')
plt.show()
'''

cells.append(md_cell(md17))
cells.append(code_cell(code17))

json.dump(nb, open(NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'patched {NB}: {n_timers} timer cells + S17 appended ({len(cells)} total cells)')
