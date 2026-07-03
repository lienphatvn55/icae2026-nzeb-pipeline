import json
import re
import shutil

# 1. Copy backup to main
shutil.copyfile('NZEB_PIPELINE_ICAE2026.backup-2026-07-03.ipynb', 'NZEB_PIPELINE_ICAE2026.ipynb')

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 2. Fix Linear Reg in Fig 7
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        for i, line in enumerate(cell['source']):
            if "['PI-HGAT', 'XGBoost', 'ANN (MLP)']" in line:
                cell['source'][i] = line.replace("['PI-HGAT', 'XGBoost', 'ANN (MLP)']", "['PI-HGAT', 'XGBoost', 'ANN (MLP)', 'Linear Reg']")

# 3. Rename Sections
section_replacements = {
    '## PART 0 · Section 1 —': '## PART 0 · Section 1 —',
    '## PART 0 · Sections 2–3 —': '## PART 0 · Section 2 —',
    '## PART 1 · Section 4 —': '## PART 1 · Section 3 —',
    '## PART 1 · Section 5 —': '## PART 1 · Section 4 —',
    '## PART 1 · Sections 6–7 —': '## PART 1 · Section 5 —',
    '## PART 1 · Section 8 —': '## PART 1 · Section 6 —',
    '## PART 1 · Section 9 —': '## PART 1 · Section 7 —',
    '## PART 1 · Section 10 —': '## PART 1 · Section 8 —',
    '## PART 2 · Section 11 —': '## PART 2 · Section 9 —',
    '## PART 2 · Section 12 —': '## PART 2 · Section 10 —',
    '## PART 2 · Section 13 —': '## PART 2 · Section 11 —',
    '## PART 3 · Section 14 —': '## PART 3 · Section 12 —'
}

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'markdown':
        for i in range(len(cell['source'])):
            for k, v in section_replacements.items():
                if cell['source'][i].startswith(k):
                    cell['source'][i] = cell['source'][i].replace(k, v)

# 4. Rename Figures
figure_replacements = {
    'Fig. 2 · KG schema': 'Fig. 2 · KG schema',
    'Fig. 5 · Predicted vs Actual': 'Fig. 3 · Predicted vs Actual',
    'Fig. 6 · Benchmark & robustness': 'Fig. 4 · Benchmark & robustness',
    'Fig. 7 · Learning curve': 'Fig. 5 · Learning curve',
    'Fig. 3 · Component lifespans': 'Fig. 6 · Component lifespans',
    'Fig. 4 · Embodied LCE': 'Fig. 7 · Embodied LCE',
    'Fig. 8 · Pareto front': 'Fig. 8 · Pareto front',
    'Fig. 9 · Pairwise Pareto': 'Fig. 9 · Pairwise Pareto',
    'Fig. 10 · LCE by module': 'Fig. 10 · LCE by module',
    'Fig. 11 · Renovation-level heatmap': 'Fig. 11 · Renovation-level heatmap',
    'Fig. 12 · Global feature importance': 'Fig. 12 · Global feature importance',
    'Fig. 13 · Edge-type': 'Fig. 13 · Edge-type',
    'Fig. 14 · Spatial explanation map': 'Fig. 14 · Spatial explanation map'
}

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'markdown':
        for i in range(len(cell['source'])):
            for k, v in figure_replacements.items():
                if k in cell['source'][i]:
                    cell['source'][i] = cell['source'][i].replace(k, v)

# 5. Rename savefig
savefig_replacements = {
    'Fig5_PredictionPerf': 'Fig3_PredictionPerf',
    'Fig6_BenchmarkBar': 'Fig4_BenchmarkBar',
    'Fig6_BenchmarkRobustness': 'Fig4_BenchmarkRobustness',
    'Fig7_LearningCurve': 'Fig5_LearningCurve',
    'Fig3_Lifespan': 'Fig6_Lifespan',
    'Fig4_LCEDistribution': 'Fig7_LCEDistribution'
}

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        for i in range(len(cell['source'])):
            for k, v in savefig_replacements.items():
                if k in cell['source'][i]:
                    cell['source'][i] = cell['source'][i].replace(k, v)

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')

print('Notebook successfully recreated to the exact state before the prompt.')
