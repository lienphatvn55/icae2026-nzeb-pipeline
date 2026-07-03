import json

with open('NZEB_PIPELINE_ICAE2026.backup-2026-07-03.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update the first markdown cell
new_intro = '''# PI-HGAT NZEB Retrofit Pipeline — ICAE 2026

**Integrated decision-support framework for hot-humid NZEB office retrofits, Ho Chi Minh City.** Case study: DOE Medium Office prototype (4,982 m2), baseline EUI 122.1 kWh/m2/yr, Koppen Aw.

This notebook is the single end-to-end runner; it mirrors the framework figure (`results/figures/0. FRAMEWORK DEMO.png`) part by part:

| Framework block | Notebook sections | Key inputs | Outputs |
| :--- | :--- | :--- | :--- |
| **PART 0 — Data Input (Building/Urban/Graph)** | S1 | `data/registry/neo4j_query_table_data_2026-6-2.json` | PyG-ready KG topology, Fig. 1–2 |
| **PART 0 — Simulation & Calibration (offline)** | S2–S3 | `data/aggregated_LHS_results.csv` (9 climate x 250 LHS, built by `scripts/data/aggregate_lhs_results.py`) | X (9 features: P1–P7 + ΔT), Y (**GROSS** site EUI), Fig. C1–C3 |
| **PART 1 — AI-based Prediction** | S4–S10 | HeteroData graphs | trained PI-HGAT + baselines, Fig. 3–5 |
| **PART 2 — Multi-Objective Optimization** | S11–S14 | surrogate f1 + `pi_hgat/objectives.py` (f2 LCC, f3 LCE) | Pareto set, TOPSIS ranking, Fig. 6–11, C5 |
| **PART 3 — XAI & Recommendations** | S15–S16 | trained model + Pareto optimum | GNNExplainer subgraph, trade-off report, Fig. 12–14 |

**Energy convention (review fix B1, 2026-07-03):** the surrogate predicts **gross demand-side EUI** (P1–P7 + climate only). PV (P8) / BESS (P9) enter EXACTLY ONCE, through the single `objectives.net_energy()` model (zero-export self-consumption) — used consistently by f1, LCC-OC, LCA-B6 and `assess_nze()`. MOO decision variables are **integer level indices** on the true simulated jEPlus ladders (`config.P*_LEVELS`) — no out-of-distribution queries (review fix B2/B3).

**Economic/LCA basis (locked 2026-07-02):** 20-yr study period, 8% real discount, elec 0.137 USD/kWh (EVN before-VAT), grid EF 0.6592 kgCO2e/kWh (MONRE 1726/2024), PV yield 1,420 kWh/kWp/yr. All constants live in `pi_hgat/config.py`; sourced registry: `data/jEPlus-LHS/ICAE2026_DataRegistry_P1-P9.xlsx`.'''

nb['cells'][0]['source'] = [line + '\n' for line in new_intro.split('\n')]
nb['cells'][0]['source'][-1] = nb['cells'][0]['source'][-1].rstrip('\n')

# 2. English translation
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

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'markdown':
        for i in range(len(cell['source'])):
            for k, v in replacements.items():
                cell['source'][i] = cell['source'][i].replace(k, v)

# 3. Add timing and external test
new_cells = []
section_indices = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        if len(cell['source']) > 0 and '## PART' in cell['source'][0] and 'Section' in cell['source'][0]:
            section_indices.append(i)
section_indices.append(len(nb['cells']))

# Include all pre-section cells (like the intro markdown and global imports!)
new_cells.extend(nb['cells'][:section_indices[0]])

for s in range(len(section_indices)-1):
    start_idx = section_indices[s]
    end_idx = section_indices[s+1]
    
    # Header
    new_cells.append(nb['cells'][start_idx])
    
    # Start timer
    start_code = f"import time\nsection_{s+1}_start = time.time()"
    new_cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [start_code]
    })
    
    # Check if Section 8
    header_text = nb['cells'][start_idx]['source'][0]
    is_section_8 = "Section 8" in header_text
    
    # Contents
    for j in range(start_idx+1, end_idx):
        new_cells.append(nb['cells'][j])
        
    if is_section_8:
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
        
    # End timer
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

print("Rebuild success.")
