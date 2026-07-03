import json

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'r', encoding='utf-8') as f:
    nb1 = json.load(f)

with open('NZEB_PIPELINE_ICAE2026.backup-2026-07-03.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

with open('nb_diff.txt', 'w', encoding='utf-8') as out:
    out.write(f'nb1 cells: {len(nb1.get("cells", []))}\n')
    out.write(f'nb2 cells: {len(nb2.get("cells", []))}\n')
    
    cells1 = nb1.get('cells', [])
    cells2 = nb2.get('cells', [])
    
    for i in range(min(len(cells1), len(cells2))):
        c1 = ''.join(cells1[i].get('source', []))
        c2 = ''.join(cells2[i].get('source', []))
        if c1 != c2:
            out.write(f'Difference at cell {i}\n')
            out.write('--- CURRENT ---\n')
            out.write(c1 + '\n')
            out.write('--- BACKUP ---\n')
            out.write(c2 + '\n')
            out.write('====================\n')
