import json
import re
import time

def main():
    with open('NZEB_PIPELINE_ICAE2026.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    replacements = {
        'Học theo Fig.4 BCGS của paper GAT-BEM 2025.': 'Inspired by Fig. 4 (BCGS) of the GAT-BEM 2025 paper.',
        'sơ đồ hetero-KG': 'hetero-KG schema',
        'Nội dung:': 'Content:',
        'số lượng in ra ở cell trên': 'count outputted above',
        'BẮT BUỘC:': 'REQUIRED:',
        'lưu res.X, res.F, closeness': 'save res.X, res.F, closeness',
        'để mọi figure tái lập được': 'so all figures are reproducible',
        'trên test set': 'on the test set',
        'màu theo MODEL_COLORS': 'colors by MODEL_COLORS',
        'Thay thế Fig5_PredictionPerf.png': 'Replaces Fig5_PredictionPerf.png',
        'chờ retrain data thật': 'pending retraining on real data'
    }

    # First pass: translate Vietnamese
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'markdown':
            for i in range(len(cell['source'])):
                for k, v in replacements.items():
                    cell['source'][i] = cell['source'][i].replace(k, v)

    # Second pass: group cells into sections and wrap with time tracker
    new_cells = []
    
    section_counter = 0
    in_section = False
    
    # Find indices of markdown cells starting with "## PART"
    section_indices = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            if len(cell['source']) > 0 and '## PART' in cell['source'][0] and 'Section' in cell['source'][0]:
                section_indices.append(i)
                
    section_indices.append(len(nb['cells'])) # dummy end
    
    for s in range(len(section_indices)-1):
        start_idx = section_indices[s]
        end_idx = section_indices[s+1]
        
        # Add the markdown header
        new_cells.append(nb['cells'][start_idx])
        
        # Add a start timer cell
        start_code = f"import time\nsection_{s+1}_start = time.time()"
        new_cells.append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [start_code]
        })
        
        # See if this is Section 8
        header_text = nb['cells'][start_idx]['source'][0]
        is_section_8 = "Section 8" in header_text
        
        # Add all cells in this section
        for j in range(start_idx+1, end_idx):
            new_cells.append(nb['cells'][j])
            
        if is_section_8:
            # Append external test code block
            ext_test_code = (
                "# --- EXTERNAL TEST SET PREDICTION ---\n"
                "print('\\n--- Evaluating on External Test Set (Seed 2810) ---')\n"
                "import pandas as pd\n"
                "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error\n"
                "ext_df = pd.read_csv('data/external_test_results.csv')\n"
                "print(f'Loaded {len(ext_df)} external test samples.')\n\n"
                "from scripts.models.build_graphs import build_graph_dataset\n"
                "from torch_geometric.loader import DataLoader\n"
                "import torch\n\n"
                "ext_dataset = build_graph_dataset(ext_df)\n"
                "ext_loader = DataLoader(ext_dataset, batch_size=64, shuffle=False)\n\n"
                "best_model = PI_HGAT(hidden_channels=config.HIDDEN_CHANNELS, out_channels=1, heads=config.GAT_HEADS)\n"
                "best_model.load_state_dict(torch.load('results/models/best_hgat_v2.pt'))\n"
                "best_model.eval()\n\n"
                "ext_preds, ext_trues = [], []\n"
                "with torch.no_grad():\n"
                "    for batch in ext_loader:\n"
                "        pred = best_model(batch.x_dict, batch.edge_index_dict)\n"
                "        ext_preds.append(pred.squeeze())\n"
                "        ext_trues.append(batch['Zone'].y)\n"
                "ext_preds = torch.cat(ext_preds).numpy()\n"
                "ext_trues = torch.cat(ext_trues).numpy()\n\n"
                "ext_r2 = r2_score(ext_trues, ext_preds)\n"
                "ext_rmse = mean_squared_error(ext_trues, ext_preds, squared=False)\n"
                "ext_mae = mean_absolute_error(ext_trues, ext_preds)\n"
                "print(f'External Test - R2: {ext_r2:.3f}, RMSE: {ext_rmse:.2f}, MAE: {ext_mae:.2f}')\n\n"
                "# Append to benchmark table\n"
                "bench_df.loc[len(bench_df)] = ['PI-HGAT (External Test)', ext_r2, ext_rmse, ext_mae]\n"
                "display(bench_df.tail(2))\n"
            )
            new_cells.append({
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [line + '\n' for line in ext_test_code.split('\n')][:-1]
            })
            
        # Add end timer cell
        end_code = f"print(f'\\n[Section {s+1}] Execution time: {{time.time() - section_{s+1}_start:.2f}} seconds')"
        new_cells.append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [end_code]
        })

    nb['cells'] = new_cells
    
    with open('NZEB_PIPELINE_ICAE2026.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        f.write('\n')
        
    print('Notebook updated successfully.')

if __name__ == '__main__':
    main()
