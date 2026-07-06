# -*- coding: utf-8 -*-
"""Patch #4: add scenario-name labels to FigC2 (user feedback 2026-07-06 evening).

FigC2 was correct (8 non-baseline scenarios, all present) but had no scenario
identifiers on the plot, and scenarios 7 (ΔT=1.875) / 2 (ΔT=1.879) overlap
almost exactly (0.004°C apart) so their clusters look like one blob — making
it look like data was missing. This adds a labelled marker per scenario with
offsets so the 7/2 pair doesn't collide, turning the near-duplicate ΔT into a
visible, explained feature instead of a confusing gap.
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


sub(14, """base = d[d.Scenario == '1_Baseline'].drop_duplicates('combo').set_index('combo')['EUI_kWh_m2']
fig, ax = plt.subplots(figsize=(6.5, 4))
xs, ys = [], []
rng = np.random.RandomState(0)
for s in scen_order:
    if s == '1_Baseline':
        continue
    sub = d[d.Scenario == s]
    delta = (sub['EUI_kWh_m2'] - sub['combo'].map(base)).dropna().values
    x = np.full(len(delta), scen_dt[s])
    ax.scatter(x + rng.uniform(-0.06, 0.06, len(delta)), delta, s=6, alpha=0.2, color='#5598e7')
    ax.scatter([scen_dt[s]], [delta.mean()], s=45, color='#0d366b', zorder=5)
    xs.extend(x); ys.extend(delta)""",
    """base = d[d.Scenario == '1_Baseline'].drop_duplicates('combo').set_index('combo')['EUI_kWh_m2']
fig, ax = plt.subplots(figsize=(6.5, 4))
xs, ys, means = [], [], {}
rng = np.random.RandomState(0)
for s in scen_order:
    if s == '1_Baseline':
        continue
    sub = d[d.Scenario == s]
    delta = (sub['EUI_kWh_m2'] - sub['combo'].map(base)).dropna().values
    x = np.full(len(delta), scen_dt[s])
    ax.scatter(x + rng.uniform(-0.06, 0.06, len(delta)), delta, s=6, alpha=0.2, color='#5598e7')
    ax.scatter([scen_dt[s]], [delta.mean()], s=45, color='#0d366b', zorder=5)
    xs.extend(x); ys.extend(delta); means[s] = delta.mean()

# Scenario-name labels: 7 (ΔT=1.875) and 2 (ΔT=1.879) are two distinct CMIP6
# GCM/SSP combinations whose mean warming coincidentally differs by only
# 0.004°C, so their clusters overlap almost exactly — offset their labels to
# opposite sides instead of letting them stack illegibly on top of each other.
label_offset = {'7': (-9, 9), '2': (9, -15)}
for s in scen_order:
    if s == '1_Baseline':
        continue
    dx, dy = label_offset.get(s, (0, 9))
    ax.annotate(s, (scen_dt[s], means[s]), xytext=(dx, dy), textcoords='offset points',
                fontsize=7, ha='center', color='#0d366b')""")

json.dump(nb, open(NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('patched', NB)
