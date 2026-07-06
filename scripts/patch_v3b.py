# -*- coding: utf-8 -*-
"""Patch #2 for NZEB_PIPELINE_ICAE2026_v3.ipynb (user feedback 2026-07-06 pm).

  12  S3 markdown: state data provenance of Fig C1-C3 (raw E+ results, no model)
  14  Fig C2: paired design forces dEUI(0)=0 -> fit a*dT + b*dT^2 THROUGH ORIGIN
      (old linear fit had a spurious -2.5 intercept => negative penalty < 0.37 degC);
      Fig C3: title notes only 3 of 9 scenarios are shown (by design)
  44  Fig 6: drop the overfitting panel (user request) -> 1x3 layout
  54  NSGA-III: denser front (ref dirs 12->16 partitions, pop 124->156, gen 100->200)
  60  Fig 8a: overlay ALL evaluated candidates (grey) so the front's sparseness is
      readable as selectivity, not missing data
  62  Fig 9: same grey overlay per pairwise panel; fix suptitle star glyph (tofu box)
  68  S14 climate MOO: same NSGA-III settings as S12
"""
import json

NB = 'NZEB_PIPELINE_ICAE2026_v3.ipynb'
nb = json.load(open(NB, encoding='utf-8'))
cells = nb['cells']


def sub(i, old, new, count=1):
    cur = ''.join(cells[i]['source'])
    if old not in cur:
        raise SystemExit(f'cell {i}: substring not found:\n{old[:200]}')
    cells[i]['source'] = cur.replace(old, new, count).splitlines(keepends=True)


# --- cell 12: provenance note -------------------------------------------------- #
sub(12, """- **Fig. C3:** end-use decomposition (baseline vs median future vs worst 2080s) — cooling drives the increase.""",
    """- **Fig. C3:** end-use decomposition (baseline vs median future vs worst 2080s) — cooling drives the increase.

**Data provenance:** all three figures are computed **directly from `data/aggregated_LHS_results.csv`**
(2,250 EnergyPlus runs = 249 paired combos × 9 climate scenarios, aggregated by jEPlus) — no surrogate
model is involved anywhere in this section. C3 shows 3 *representative* scenarios of the 9 by design
(baseline / median future / worst 2080s); C1 shows all 9.""")

# --- cell 14: C2 fit through origin ------------------------------------------- #
sub(14, """slope, intercept = np.polyfit(xs, ys, 1)
xr = np.linspace(0, max(xs) * 1.05, 20)
ax.plot(xr, slope * xr + intercept, '--', color='#7a1f1f', lw=1.6,
        label=f'mean cooling penalty: +{slope:.2f} kWh/m²/yr per °C')""",
    """# The paired design forces ΔEUI(ΔT=0) = 0 exactly, so the fit must pass through
# the origin; a plain linear fit gave a spurious −2.5 intercept (implying a
# *negative* penalty below +0.37°C). The response is mildly convex, so fit
# ΔEUI = a·ΔT + b·ΔT² (least squares, no intercept).
xs_a, ys_a = np.asarray(xs), np.asarray(ys)
(a_lin, b_quad), *_ = np.linalg.lstsq(np.column_stack([xs_a, xs_a**2]), ys_a, rcond=None)
xr = np.linspace(0, max(xs) * 1.05, 40)
ax.plot(xr, a_lin * xr + b_quad * xr**2, '--', color='#7a1f1f', lw=1.6,
        label=f'ΔEUI = {a_lin:.2f}·ΔT + {b_quad:.2f}·ΔT² (through origin)')""")

sub(14, "ax.set_title('End-use shift: cooling drives the climate-change penalty')",
        "ax.set_title('End-use shift: cooling drives the climate-change penalty\\n(3 representative of 9 scenarios)')")

# --- cell 44: Fig 6 -> 1x3, drop overfitting panel ----------------------------- #
sub(44, """# --- FIG 6 (2x2): benchmark & robustness ---
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# (a) test R2, mean +/- sigma over seeds; star = this notebook's S6/S8 single run
ax = axes[0, 0]""",
    """# --- FIG 6 (1x3): benchmark & robustness (overfitting panel dropped per review) ---
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))

# (a) test R2, mean +/- sigma over seeds; star = this notebook's S6/S8 single run
ax = axes[0]""")
sub(44, """# (b) boxplot of test R2 across seeds
ax = axes[0, 1]""",
    """# (b) boxplot of test R2 across seeds
ax = axes[1]""")
sub(44, """# (c) train vs test R2 (overfitting check)
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
ax = axes[1, 1]""",
    """# (c) training time (log scale)
ax = axes[2]""")
sub(44, "ax.set_title('(d) Training cost', fontweight='bold')",
        "ax.set_title('(c) Training cost', fontweight='bold')")

# --- cell 54: denser NSGA-III -------------------------------------------------- #
sub(54, """ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
algorithm = NSGA3(pop_size=124, ref_dirs=ref_dirs,""",
    """# 16 partitions -> 153 reference directions; pop/gen raised so the integer
# front is as dense as the (partly EUI-LCE-degenerate) problem allows.
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=16)
algorithm = NSGA3(pop_size=156, ref_dirs=ref_dirs,""")
sub(54, "termination=('n_gen', 100),", "termination=('n_gen', 200),")

# --- cell 60: Fig 8a grey overlay of all evaluated candidates ------------------ #
sub(60, """F = res.F
best_obj = F[best_idx]

# --- FIG 8a: 3D Pareto ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(F[:, 0], F[:, 1]/1e6, F[:, 2]/1e6, c=closeness, cmap=seq_cmap(), s=40, alpha=0.8)""",
    """F = res.F
best_obj = F[best_idx]

# All evaluated candidates (every generation) — context that makes the small
# non-dominated set readable as *selectivity*, not missing results. The front
# is intrinsically thin: LCE ≈ grid-factor × net-import energy + small embodied
# terms, so objectives f1 and f3 are nearly collinear and the true trade-off
# surface is close to a 1-D curve (EUI vs LCC).
F_pop = np.unique(np.vstack([a.pop.get('F') for a in res.history]), axis=0)

# --- FIG 8a: 3D Pareto ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(F_pop[:, 0], F_pop[:, 1]/1e6, F_pop[:, 2]/1e6, color='#c3c2b7',
           s=6, alpha=0.18, label=f'evaluated candidates (n={len(F_pop)})')
sc = ax.scatter(F[:, 0], F[:, 1]/1e6, F[:, 2]/1e6, c=closeness, cmap=seq_cmap(), s=40, alpha=0.9,
                label=f'Pareto front (n={len(F)})')""")

# --- cell 62: Fig 9 grey overlay + star glyph fix ------------------------------ #
_old62 = ('    y_data = F[:, idx_y] if idx_y == 0 else F[:, idx_y] / 1e6\n'
          '    \n'
          '    sc = ax.scatter(x_data, y_data, c=closeness, cmap=seq_cmap(), s=25, alpha=0.8)')
_new62 = ('    y_data = F[:, idx_y] if idx_y == 0 else F[:, idx_y] / 1e6\n'
          '\n'
          '    xp = F_pop[:, idx_x] if idx_x == 0 else F_pop[:, idx_x] / 1e6\n'
          '    yp = F_pop[:, idx_y] if idx_y == 0 else F_pop[:, idx_y] / 1e6\n'
          '    ax.scatter(xp, yp, color=\'#c3c2b7\', s=5, alpha=0.18)\n'
          '    sc = ax.scatter(x_data, y_data, c=closeness, cmap=seq_cmap(), s=28, alpha=0.9)')
sub(62, _old62, _new62)
sub(62, "fig.suptitle('Pairwise Pareto Solutions (★ = best compromise)')",
        "fig.suptitle('Pairwise Pareto solutions over all evaluated candidates (red star = TOPSIS best)')")

# --- cell 68: S14 same NSGA settings as S12 ------------------------------------ #
sub(68, """    alg = NSGA3(pop_size=124, ref_dirs=ref_dirs,""",
    """    alg = NSGA3(pop_size=156, ref_dirs=ref_dirs,""")
sub(68, "r = minimize(prob_dt, alg, seed=42, termination=('n_gen', 100), verbose=False)",
        "r = minimize(prob_dt, alg, seed=42, termination=('n_gen', 200), verbose=False)")

json.dump(nb, open(NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('patched', NB)
