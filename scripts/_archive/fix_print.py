import nbformat

nb_path = 'NZEB_PIPELINE_ICAE2026.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for c in nb.cells:
    if c.cell_type == 'code':
        source = "".join(c.get('source', []))
        if "print('Building 5000 HeteroData graphs" in source:
            new_source = source.replace(
                "print('Building 5000 HeteroData graphs with type-specific features...')", 
                "print(f'Building {len(samples)} HeteroData graphs with type-specific features...')")
            new_source = new_source.replace(
                "print(f'{i+1}/5000 ...')",
                "print(f'{i+1}/{len(samples)} ...')")
            c.source = new_source
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Notebook print statements fixed.")
