import json
import os
import pandas as pd

nb_path = 'NZEB_PIPELINE_ICAE2026.ipynb'
out_path = 'notebook_evaluation.txt'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(out_path, 'w', encoding='utf-8') as fout:
    fout.write("=== NOTEBOOK OUTPUTS ===\n")
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if 'FIGURE SLOT' in source:
                fout.write(f"--- FIGURE SLOT found in cell {i} ---\n")
                
            outputs = cell.get('outputs', [])
            for out in outputs:
                if out['output_type'] == 'stream':
                    text = "".join(out.get('text', []))
                    fout.write(f"Cell {i} stream:\n")
                    fout.write(text[:500] + ('...' if len(text) > 500 else '') + "\n")
                elif out['output_type'] in ('execute_result', 'display_data'):
                    data = out.get('data', {})
                    if 'text/plain' in data:
                        text = "".join(data['text/plain'])
                        fout.write(f"Cell {i} text/plain:\n")
                        fout.write(text[:500] + ('...' if len(text) > 500 else '') + "\n")

    fout.write("\n=== pareto_solutions.csv ===\n")
    csv_path = 'results/pareto_solutions.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        fout.write(f"Exists! Shape: {df.shape}\n")
        fout.write(f"Columns: {df.columns.tolist()}\n")
        if 'NZE_class' in df.columns:
            fout.write(df['NZE_class'].value_counts().to_string() + "\n")
    else:
        fout.write("Does NOT exist.\n")
