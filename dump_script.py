import json
with open('NZEB_PIPELINE_ICAE2026.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('first5.txt', 'w', encoding='utf-8') as f:
    for i in range(10):
        t = nb['cells'][i]['cell_type']
        f.write('Cell ' + str(i) + ' type: ' + t + '\n')
        f.write(''.join(nb['cells'][i].get('source', []))[:200] + '\n\n')
