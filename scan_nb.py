import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open('NZEB_PIPELINE_ICAE2026_v3.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']

# Print cells 32-45 (evaluation, visualization, robustness)
for i in range(32, 46):
    c = cells[i]
    src = ''.join(c['source'])
    print(f"\n{'='*80}")
    print(f"Cell {i} [{c['cell_type']}]:")
    print(src[:2000])
    if len(src) > 2000:
        print(f"\n... (truncated, total {len(src)} chars)")
