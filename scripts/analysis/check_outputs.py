import json

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    ct = c['cell_type']
    outputs = c.get('outputs', [])
    print(f"Cell {i} ({ct}): {len(outputs)} outputs")
    for o in outputs:
        ot = o.get('output_type', '?')
        if ot == 'stream':
            text = ''.join(o.get('text', []))
            print(f"  [stream] {text[:500]}")
        elif ot == 'execute_result':
            data = o.get('data', {})
            if 'text/plain' in data:
                text = ''.join(data['text/plain'])
                print(f"  [result] {text[:500]}")
            if 'text/html' in data:
                text = ''.join(data['text/html'])
                print(f"  [html] {text[:500]}")
        elif ot == 'error':
            print(f"  [ERROR] {o.get('ename','')}: {o.get('evalue','')}")
