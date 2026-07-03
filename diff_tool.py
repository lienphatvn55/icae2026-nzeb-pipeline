import difflib
with open('backup_code.py', 'r', encoding='utf-8') as f1, open('current_code.py', 'r', encoding='utf-8') as f2:
    d = difflib.unified_diff(f1.readlines(), f2.readlines(), fromfile='backup', tofile='current')
with open('diff.txt', 'w', encoding='utf-8') as out:
    out.writelines(d)

