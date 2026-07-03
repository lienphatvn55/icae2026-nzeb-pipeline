import json

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'r', encoding='utf-8') as f:
    nb1 = json.load(f)

with open('cells.txt', 'w', encoding='utf-8') as out:
    for i, c in enumerate(nb1['cells']):
        src = ''.join(c.get('source', []))
        out.write(f'CELL {i} ({c.get("cell_type")}):\n{src[:100]}...\n-----------------\n')
