import json

nb_path = 'NZEB_PIPELINE_ICAE2026.ipynb'
out_path = 'nb_source.md'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(out_path, 'w', encoding='utf-8') as fout:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = "".join(cell.get('source', []))
            fout.write(f"### [Markdown Cell {i}]\n{source}\n\n")
        elif cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            fout.write(f"### [Code Cell {i}]\n```python\n{source}\n```\n\n")
