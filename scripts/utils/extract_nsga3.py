import json

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        # Section 11/12/13
        if i >= len(nb['cells']) - 5: 
            for o in cell.get('outputs', []):
                data = o.get('data', {})
                if 'text/plain' in data:
                    print(''.join(data['text/plain']))
                if 'text/html' in data:
                    print(''.join(data['text/html']))
                if o.get('output_type') == 'stream':
                    print(''.join(o.get('text', [])))
