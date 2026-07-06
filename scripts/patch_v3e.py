# -*- coding: utf-8 -*-
"""Patch #5: fix FigC3 end-use shift readability (user feedback 2026-07-06 evening).

Issues (verified against data, not bugs but legibility gaps):
  1. Only the Cooling segment got a value label; Lighting/Equipment/Fans/Pumps
     had no numbers even though they're plotted.
  2. x-axis labels ("Median future" / "Worst 2080s") didn't show which of the
     9 scenario IDs they correspond to (= scenario 2 and scenario 5).
  3. Pumps = 0.008 kWh/m2/yr (~0.01% of total) is a real value but invisibly
     thin at this scale -- needs an explicit note, not a forced label.
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


sub(14, """futures = scen_dt.drop('1_Baseline')
pick = ['1_Baseline', (futures - futures.median()).abs().idxmin(), futures.idxmax()]
labels3 = [f'Baseline\\n(+0.0°C)', f'Median future\\n(+{scen_dt[pick[1]]:.1f}°C)',
           f'Worst 2080s\\n(+{scen_dt[pick[2]]:.1f}°C)']
GJ_TO_KWH_M2 = 1000.0 / 3.6 / 4982.0
colors5 = ['#0d366b', '#1c5cab', '#2a78d6', '#5598e7', '#86b6ef']
fig, ax = plt.subplots(figsize=(6, 4.2))
bottoms = np.zeros(3)
for (name, col), c in zip(eu_cols.items(), colors5):
    v = np.array([d[d.Scenario == s][col].mean() * GJ_TO_KWH_M2 for s in pick])
    ax.bar(labels3, v, bottom=bottoms, label=name, color=c, width=0.6)
    if name == 'Cooling':
        for xi, vi in enumerate(v):
            ax.text(xi, vi / 2, f'{vi:.0f}', ha='center', color='white',
                    fontsize=8, fontweight='bold')
    bottoms += v
ax.set_ylabel('Mean end-use intensity (kWh/m²/yr)')
ax.set_title('End-use shift: cooling drives the climate-change penalty\\n(3 representative of 9 scenarios)')
ax.legend(prop={'size': 7}, loc='upper left')""",
    """futures = scen_dt.drop('1_Baseline')
pick = ['1_Baseline', (futures - futures.median()).abs().idxmin(), futures.idxmax()]
# x-labels include the scenario ID (matches the 9-scenario numbering used
# everywhere else) alongside the ΔT and the "role" (baseline/median/worst).
labels3 = [f'Baseline (Sc. {pick[0]})\\n(+0.0°C)',
           f'Median future (Sc. {pick[1]})\\n(+{scen_dt[pick[1]]:.1f}°C)',
           f'Worst 2080s (Sc. {pick[2]})\\n(+{scen_dt[pick[2]]:.1f}°C)']
GJ_TO_KWH_M2 = 1000.0 / 3.6 / 4982.0
colors5 = ['#0d366b', '#1c5cab', '#2a78d6', '#5598e7', '#86b6ef']
text_colors5 = ['white', 'white', 'white', '#0b0b0b', '#0b0b0b']
fig, ax = plt.subplots(figsize=(6.5, 4.4))
bottoms = np.zeros(3)
values5 = {}
for (name, col), c, tc in zip(eu_cols.items(), colors5, text_colors5):
    v = np.array([d[d.Scenario == s][col].mean() * GJ_TO_KWH_M2 for s in pick])
    ax.bar(labels3, v, bottom=bottoms, label=name, color=c, width=0.6)
    values5[name] = v
    # Value label per segment; skip if the segment is too thin to hold text
    # (Pumps ≈ 0.01 kWh/m² here -- flagged separately below instead).
    for xi, vi in enumerate(v):
        if vi >= 2.0:
            ax.text(xi, bottoms[xi] + vi / 2, f'{vi:.1f}', ha='center', va='center',
                    color=tc, fontsize=7.5, fontweight='bold')
    bottoms += v
ax.text(0.02, 0.98, f"Pumps ≈ {values5['Pumps'].mean():.3f} kWh/m²/yr in all 3 "
        "columns (negligible, not visible at this scale)",
        transform=ax.transAxes, ha='left', va='top', fontsize=6.5,
        color='#6b6a63', style='italic')
ax.set_ylabel('Mean end-use intensity (kWh/m²/yr)')
ax.set_title('End-use shift: cooling drives the climate-change penalty\\n(3 representative of 9 scenarios)')
ax.legend(prop={'size': 7}, loc='upper left')""")

json.dump(nb, open(NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('patched', NB)
