# -*- coding: utf-8 -*-
"""Patch #7: render the generalized loso_ext study in S10 (FigC4 + tables).

Adds to cell 44 a block reading results/step3_loso_ext.csv:
  - FigC4 (2 panels): (a) combined R2 vs dT per model, extrapolation folds
    flagged; (b) PI-HGAT MAE main (seen combos) vs external (unseen combos)
    per scenario -- the parameter-generalization cost isolated per climate.
  - Table: combined R2 and MAE pivots (scenario x model).
Updates the S10 markdown (cell 42) to document the study.
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


# --- cell 42: document the new study ---------------------------------------- #
sub(42, "- **Tables:** LOSO 9-fold per-scenario MAE",
        "- **Fig. C4 + table (loso_ext):** systematic per-climate generalization — every scenario "
        "held out of train/val in turn and evaluated on its 250 MAIN rows (combos seen in training) "
        "AND its 150-row seed-2810 external replicate (combos unseen); the main-vs-external gap "
        "isolates the parameter-generalization cost from the climate cost, for all 9 climates "
        "(no hand-picked extrapolation scenario).\n"
        "- **Tables:** LOSO 9-fold per-scenario MAE")

# --- cell 44: append loso_ext rendering -------------------------------------- #
sub(44, """print('\\nPhysics monotonicity-loss ablation (mean over 3 seeds; viol = finite-diff violation rate):')
display(t_abl)""",
    """print('\\nPhysics monotonicity-loss ablation (mean over 3 seeds; viol = finite-diff violation rate):')
display(t_abl)

# --- FIG C4 + tables: loso_ext (per-climate combo+climate generalization) ---
if os.path.exists('results/step3_loso_ext.csv'):
    lx = pd.read_csv('results/step3_loso_ext.csv')
    lx['fold_scenario'] = lx['fold_scenario'].astype(str)
    order_dt = lx[['fold_scenario', 'delta_t']].drop_duplicates().sort_values('delta_t')

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.5, 4.3))

    # (a) combined R2 vs dT per model. Folds 1_Baseline (dT=0) and 5 (dT=4.47)
    # are EXTRApolation folds (dT outside the remaining-train hull); the rest
    # are interpolation folds.
    Y_FLOOR = -0.05
    for m in ORDER:
        sub_m = (lx[(lx.model == m) & (lx.eval_set == 'combined')]
                 .set_index('fold_scenario').reindex(order_dt.fold_scenario))
        axa.plot(sub_m['delta_t'], sub_m['r2'].clip(lower=Y_FLOOR), marker='o',
                 ms=4, lw=1.8, color=MODEL_COLORS[m], label=m)
        for dt_v, r2_v in zip(sub_m['delta_t'], sub_m['r2']):
            if r2_v < Y_FLOOR:
                axa.annotate(f'{r2_v:.1f}', (dt_v, Y_FLOOR), xytext=(0, -11),
                             textcoords='offset points', ha='center', fontsize=6,
                             color=MODEL_COLORS[m])
    for dt_v, lbl in [(0.0, 'extrapolation\\n(below hull)'), (4.472, 'extrapolation\\n(above hull)')]:
        axa.axvline(dt_v, color='#c3c2b7', ls=':', lw=1)
        axa.text(dt_v, 1.06, lbl, ha='center', fontsize=6, color='#6b6a63')
    for _, r in order_dt.iterrows():
        axa.text(r['delta_t'], Y_FLOOR + 0.02, r['fold_scenario'], ha='center',
                 fontsize=6, color='#6b6a63')
    axa.set_ylim(Y_FLOOR, 1.12)
    axa.set_xlabel('Held-out scenario ΔT (°C)')
    axa.set_ylabel('R² (combined: 250 MAIN + 150 external)')
    axa.set_title('(a) Per-climate generalization (scenario fully held out)',
                  fontweight='bold')
    axa.legend(prop={'size': 7}, loc='lower center')

    # (b) PI-HGAT MAE: main (seen combos) vs external (unseen combos) per fold
    ph = lx[lx.model == 'PI-HGAT'].pivot_table(index=['fold_scenario', 'delta_t'],
                                               columns='eval_set', values='mae').reset_index()
    ph = ph.sort_values('delta_t')
    xpos = np.arange(len(ph))
    axb.bar(xpos - 0.19, ph['main'], width=0.38, color='#2a78d6',
            label='MAIN rows (combos seen in training)')
    axb.bar(xpos + 0.19, ph['external'], width=0.38, color='#86b6ef',
            label='External replicate (unseen combos)')
    axb.set_xticks(xpos)
    axb.set_xticklabels([f"{s}\\n(+{d:.1f}°C)" for s, d in
                         zip(ph['fold_scenario'], ph['delta_t'])], fontsize=6.5)
    axb.set_ylabel('PI-HGAT MAE (kWh/m²/yr)')
    axb.set_title('(b) Parameter-generalization cost per climate (PI-HGAT)',
                  fontweight='bold')
    axb.legend(prop={'size': 7})
    plt.tight_layout()
    savefig(fig, 'FigC4_LOSOExternal')
    plt.show()

    t_lx_r2 = lx[lx.eval_set == 'combined'].pivot_table(
        index=['fold_scenario', 'delta_t'], columns='model', values='r2')[ORDER].round(4)
    print('loso_ext — combined R² (scenario fully held out; 250 MAIN + 150 unseen-combo external):')
    display(t_lx_r2)
    t_lx_mae = lx[lx.eval_set == 'combined'].pivot_table(
        index=['fold_scenario', 'delta_t'], columns='model', values='mae')[ORDER].round(2)
    print('loso_ext — combined MAE (kWh/m²/yr):')
    display(t_lx_mae)
    t_lx_r2.to_csv('results/step3_table_loso_ext_r2.csv')""")

json.dump(nb, open(NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('patched', NB)
