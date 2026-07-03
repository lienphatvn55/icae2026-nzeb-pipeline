import json
import os
import pandas as pd

nb_path = 'NZEB_PIPELINE_ICAE2026.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== NOTEBOOK OUTPUTS ===")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if 'FIGURE SLOT' in source:
            print(f"--- FIGURE SLOT found in cell {i} ---")
            
        outputs = cell.get('outputs', [])
        for out in outputs:
            if out['output_type'] == 'stream':
                text = "".join(out.get('text', []))
                print(f"Cell {i} stream:")
                print(text[:500] + ('...' if len(text) > 500 else ''))
            elif out['output_type'] == 'execute_result' or out['output_type'] == 'display_data':
                data = out.get('data', {})
                if 'text/plain' in data:
                    text = "".join(data['text/plain'])
                    print(f"Cell {i} text/plain:")
                    print(text[:500] + ('...' if len(text) > 500 else ''))

print("\n=== pareto_solutions.csv ===")
csv_path = 'results/pareto_solutions.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Exists! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    if 'NZE_class' in df.columns:
        print(df['NZE_class'].value_counts())
else:
    print("Does NOT exist.")

