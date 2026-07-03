import nbformat

nb_path = 'NZEB_PIPELINE_ICAE2026.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for c in nb.cells:
    if c.cell_type == 'code':
        source = "".join(c.get('source', []))
        if '# FIGURE 8' in source and 'import numpy as np' not in source:
            # We inject the TOPSIS calculation
            insert = """
import numpy as np
F = res.F
norm_F = F / np.sqrt((F**2).sum(axis=0))
P = norm_F / norm_F.sum(axis=0)
E = -np.nansum(P * np.log(P), axis=0) / np.log(len(F))
W = (1 - E) / (1 - E).sum()
V = norm_F * W
ideal = V.min(axis=0)
anti_ideal = V.max(axis=0)
d_ideal = np.sqrt(((V - ideal)**2).sum(axis=1))
d_anti = np.sqrt(((V - anti_ideal)**2).sum(axis=1))
closeness = d_anti / (d_ideal + d_anti)
best_idx = np.argmax(closeness)
best_obj = res.F[best_idx]
"""
            new_source = source.replace('apply_style()', 'apply_style()\n' + insert)
            c.source = new_source
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Fig 8 patched.")
