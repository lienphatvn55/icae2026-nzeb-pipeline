import json

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 16 is the benchmark results cell
c16 = nb['cells'][16]
for o in c16.get('outputs', []):
    ot = o.get('output_type', '?')
    data = o.get('data', {})
    if 'text/plain' in data:
        print("=== TEXT ===")
        print(''.join(data['text/plain']))
    if 'text/html' in data:
        print("=== HTML ===")
        print(''.join(data['text/html']))
    if ot == 'stream':
        print("=== STREAM ===")
        print(''.join(o.get('text', [])))
