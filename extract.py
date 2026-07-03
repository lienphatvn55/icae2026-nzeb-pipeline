import json

with open('d:/1. Research/0. CONFERENCE PAPER/2026.09_ICAE2026/3. DATA_CODE/CODE/NZEB_PIPELINE_ICAE2026.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

output = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if 'PART' in source or 'Section' in source or 'FIGURE SLOT' in source or 'TABLE SLOT' in source or 'EXTERNAL TEST SET PREDICTION' in source:
            output.append(f"Cell {i}:\n{source}\n---")

with open('d:/1. Research/0. CONFERENCE PAPER/2026.09_ICAE2026/3. DATA_CODE/CODE/extracted_sections.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))
