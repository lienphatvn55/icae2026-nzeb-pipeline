import json

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        for i, line in enumerate(cell['source']):
            if "for m in ['PI-HGAT', 'XGBoost', 'ANN (MLP)']:" in line:
                cell['source'][i] = line.replace(
                    "['PI-HGAT', 'XGBoost', 'ANN (MLP)']",
                    "['PI-HGAT', 'XGBoost', 'ANN (MLP)', 'Linear Reg']"
                )

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')
