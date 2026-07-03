import json
with open('NZEB_PIPELINE_ICAE2026.backup-2026-07-03.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('backup_info.txt', 'w', encoding='utf-8') as out:
    out.write('--- Sections ---\n')
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            if len(cell['source']) > 0 and '## PART' in cell['source'][0]:
                out.write(cell['source'][0].strip() + '\n')

    out.write('\n--- Figures ---\n')
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            for line in cell['source']:
                if 'FIG' in line.upper() and 'SLOT' in line.upper():
                    out.write(line.strip() + '\n')
