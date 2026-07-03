import sys, json

sys.stdout.reconfigure(encoding='utf-8')

nb_path = r'd:\1. Research\0. CONFERENCE PAPER\2026.09_ICAE2026\3. DATA_CODE\CODE\NZEB_PIPELINE_ICAE2026.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))
cells = nb['cells']

for i, c in enumerate(cells):
    ctype = c['cell_type']
    source = ''.join(c['source'])
    print(f"{'='*80}")
    print(f"CELL {i} [{ctype.upper()}]")
    print(f"{'='*80}")
    print(source)
    print()
