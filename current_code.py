import os, sys, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader as TorchDL
from torch_geometric.loader import DataLoader as PyGDL
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb

from pi_hgat.config import *
from pi_hgat.graph_builder import GraphBuilder
from pi_hgat.synthetic_data import SyntheticDataGenerator
from pi_hgat.models import PI_HGAT, BaselineANN
from pi_hgat.physics_loss import PhysicsLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

def seed_all(s=42):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
seed_all(42)

# ===============
import time
section_1_start = time.time()

# ===============
print('Loading KG from Neo4j JSON...')
builder = GraphBuilder(NEO4J_JSON_PATH)
baseline_data = builder.create_heterodata()
print(baseline_data)

n_nodes = sum(baseline_data[nt].num_nodes for nt in baseline_data.node_types)
n_edges = sum(baseline_data[et].num_edges for et in baseline_data.edge_types)
print(f'\nNodes: {n_nodes}, Edges: {n_edges}')
print('Node feature dims:')
for nt in baseline_data.node_types:
    print(f'  {nt}: {baseline_data[nt].x.shape}')

# ===============
# FIG 2 (Mock/Setup - requires external image layout, but we can print stats)
print('Node Counts:', n_nodes)
print('Edge Counts:', n_edges)
# Actual drawing of the KG schema is done outside in a graphics tool.

# ===============
print(f'\n[Section 1] Execution time: {time.time() - section_1_start:.2f} seconds')

# ===============
import time
section_2_start = time.time()

# ===============
print('Loading LHS samples from aggregated_LHS_results.csv...')
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, GroupShuffleSplit

df = pd.read_csv(r'data/aggregated_LHS_results.csv')

samples = []
X = []
Y = []
groups = []

climate_delta_map = {'1_Baseline': 0.0, '2': 1.879, '3': 2.665, '4': 2.179, '5': 4.472, '6': 1.27, '7': 1.875, '8': 1.611, '9': 3.144}

P8_LEVELS = [0, 30, 60, 90, 120, 150]
P9_LEVELS = [0, 30, 60, 90, 120, 150]

for _, row in df.iterrows():
    s = {
        'P1_Wall_U': 1.0 / row['@@P1_Wall_R@@'],
        'P2_Roof_U': 1.0 / row['@@P2_Roof_R@@'],
        'P3_Roof_Reflectance': 1.0 - row['@@P3_Roof_Abs@@'],
        'P4_Win_U': row['@@P4_U@@'],
        'P4_Win_SHGC': row['@@P4_SHGC@@'],
        'P5_COP': row['@@P5_COP@@'],
        'P6_Cool_SP': row['@@P6_ClgSetp@@'],
        'P7_LPD': row['@@P7_LPD@@'],
        'P8_PV_kW': random.choice(P8_LEVELS),
        'P9_BESS_kWh': random.choice(P9_LEVELS),
        'Climate_DeltaT': climate_delta_map.get(str(row['Scenario']), 0.0)
    }
    samples.append(s)
    X.append([s['P1_Wall_U'], s['P2_Roof_U'], s['P3_Roof_Reflectance'],
              s['P4_Win_U'], s['P4_Win_SHGC'], s['P5_COP'],
              s['P6_Cool_SP'], s['P7_LPD'], s['P8_PV_kW'], s['P9_BESS_kWh'],
              s['Climate_DeltaT']])
    
    # Calculate Net EUI (PV + BESS self-consumption logic)
    gross_eui_kwh_m2 = row['EUI_MJ_m2'] / 3.6
    pv_yield = 1420.0
    pv_gen_kwh = s['P8_PV_kW'] * pv_yield
    
    # BESS increases self-consumption
    daily_pv = pv_gen_kwh / 365.0
    bess_ratio = s['P9_BESS_kWh'] / daily_pv if daily_pv > 0 else 0
    sc_factor = min(1.0, 0.6 + 0.4 * min(1.0, bess_ratio))
    
    net_eui_kwh_m2 = gross_eui_kwh_m2 - (pv_gen_kwh * sc_factor / 4982.0)
    Y.append([max(0.0, net_eui_kwh_m2)])
    groups.append(row['Scenario'])

X_flat = np.array(X)
Y_eui = np.array(Y)
groups = np.array(groups)

print(f'X: {X_flat.shape}, Y: {Y_eui.shape}')
print(f'EUI range: {Y_eui.min():.1f} – {Y_eui.max():.1f} kWh/m²/yr')
print(f'Baseline check (first sample): {Y_eui[0,0]:.1f}')

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
idx_train, idx_temp = next(gss1.split(X_flat, Y_eui, groups=groups))
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
idx_val, idx_test = next(gss2.split(X_flat[idx_temp], Y_eui[idx_temp], groups=groups[idx_temp]))
idx_val, idx_test = idx_temp[idx_val], idx_temp[idx_test]
print(f'Group split by Job_ID (Scenario)')
print(f'Split: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test')

# ===============
print(f'\n[Section 2] Execution time: {time.time() - section_2_start:.2f} seconds')

# ===============
import time
section_3_start = time.time()

# ===============
print('Building {len(samples)} HeteroData graphs with type-specific features...')
t0 = time.time()
dataset = []
for i, s in enumerate(samples):
    data = builder.create_sample_graph(s)
    data.y = torch.tensor([[Y_eui[i, 0]]], dtype=torch.float)
    # Store flat params for global skip connection
    data.global_params = torch.tensor([X_flat[i]], dtype=torch.float)
    dataset.append(data)
    if (i+1) % 1000 == 0:
        print(f'  {i+1}/{len(samples)} ...')

train_loader = PyGDL([dataset[i] for i in idx_train], batch_size=TRAIN_PARAMS['batch_size'], shuffle=True)
val_loader   = PyGDL([dataset[i] for i in idx_val],   batch_size=TRAIN_PARAMS['batch_size'])
test_loader  = PyGDL([dataset[i] for i in idx_test],  batch_size=TRAIN_PARAMS['batch_size'])
print(f'DataLoaders ready ({time.time()-t0:.1f}s)')

# ===============
print(f'\n[Section 3] Execution time: {time.time() - section_3_start:.2f} seconds')

# ===============
import time
section_4_start = time.time()

# ===============
metadata = baseline_data.metadata()
print('Edge types:', [f'{s}-[{r}]->{t}' for s,r,t in metadata[1]])

model = PI_HGAT(
    metadata=metadata,
    hidden_channels=GNN_PARAMS['hidden_channels'],
    out_channels=1,
    num_layers=GNN_PARAMS['num_layers'],
    heads=GNN_PARAMS['heads'],
    dropout=GNN_PARAMS['dropout'],
    global_dim=11, 
).to(device)

# Init lazy modules with dummy forward
dummy = next(iter(train_loader)).to(device)
model(dummy.x_dict, dummy.edge_index_dict, dummy.batch_dict, dummy.global_params)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nModel parameters: {n_params:,}')

# ===============
print(f'\n[Section 4] Execution time: {time.time() - section_4_start:.2f} seconds')

# ===============
import time
section_5_start = time.time()

# ===============
optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_PARAMS['lr'],
                             weight_decay=TRAIN_PARAMS['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_PARAMS['epochs'], eta_min=1e-6)
criterion = PhysicsLoss(lambda_bound=0.1, lambda_mono=0.0)

def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total = 0.
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict,
                        batch.batch_dict, batch.global_params)
            loss, _, _, _ = criterion(out, batch.y)
            if is_train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)

hist = {'train': [], 'val': []}
best_val, patience_cnt = 1e9, 0
print('Training PI-HGAT...')
t0 = time.time()

for ep in range(1, TRAIN_PARAMS['epochs']+1):
    tl = run_epoch(model, train_loader, optimizer)
    vl = run_epoch(model, val_loader)
    scheduler.step(vl)
    hist['train'].append(tl); hist['val'].append(vl)

    if vl < best_val:
        best_val = vl; patience_cnt = 0
        torch.save(model.state_dict(), 'best_hgat_v2.pt')
    else:
        patience_cnt += 1
        if patience_cnt >= TRAIN_PARAMS['patience']:
            print(f'Early stop @ epoch {ep}'); break

    if ep % 20 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f'  Ep {ep:3d}: train={tl:.2f}  val={vl:.2f}  lr={lr:.1e}')

print(f'Done in {time.time()-t0:.1f}s (best val={best_val:.4f})')
model.load_state_dict(torch.load('best_hgat_v2.pt', weights_only=True))

# ===============
print(f'\n[Section 5] Execution time: {time.time() - section_5_start:.2f} seconds')

# ===============
import time
section_6_start = time.time()

# ===============
from sklearn.preprocessing import StandardScaler

Xtr, Ytr = X_flat[idx_train], Y_eui[idx_train].ravel()
Xv, Yv   = X_flat[idx_val],   Y_eui[idx_val].ravel()
Xte, Yte = X_flat[idx_test],  Y_eui[idx_test].ravel()

scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xv_s  = scaler.transform(Xv)
Xte_s = scaler.transform(Xte)
print('Features standardized (NORM_STATS computed).')

# 1. Linear Regression
print('Training Linear Regression...')
lr_model = LinearRegression().fit(Xtr_s, Ytr)

# 2. XGBoost
print('Training XGBoost...')
xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                              random_state=42, tree_method='exact')
xgb_model.fit(Xtr_s, Ytr)

# 3. ANN
print('Training ANN...')
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader as TorchDL
import copy

ann = BaselineANN(Xtr_s.shape[1], 1).to(device)
ann_opt = torch.optim.Adam(ann.parameters(), lr=1e-3, weight_decay=1e-5)
ann_crit = nn.MSELoss()
ds = TensorDataset(torch.tensor(Xtr_s, dtype=torch.float),
                    torch.tensor(Ytr, dtype=torch.float).unsqueeze(1))
dl = TorchDL(ds, batch_size=64, shuffle=True)

best_ann_loss = 1e9
patience = 20
patience_counter = 0
best_ann_state = None

for ep in range(300):
    ann.train()
    for bx, by in dl:
        bx, by = bx.to(device), by.to(device)
        ann_opt.zero_grad()
        ann_crit(ann(bx), by).backward()
        ann_opt.step()
        
    ann.eval()
    with torch.no_grad():
        val_loss = ann_crit(ann(torch.tensor(Xv_s, dtype=torch.float).to(device)), 
                            torch.tensor(Yv, dtype=torch.float).unsqueeze(1).to(device)).item()
        if val_loss < best_ann_loss:
            best_ann_loss = val_loss
            best_ann_state = copy.deepcopy(ann.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
if best_ann_state is not None:
    ann.load_state_dict(best_ann_state)

print('All baselines trained.')

# ===============
print(f'\n[Section 6] Execution time: {time.time() - section_6_start:.2f} seconds')

# ===============
import time
section_7_start = time.time()

# ===============
def metrics(yt, yp):
    return dict(R2=r2_score(yt, yp),
                RMSE=mean_squared_error(yt, yp)**0.5,
                MAE=mean_absolute_error(yt, yp),
                MAPE=mean_absolute_percentage_error(yt, yp)*100)

# PI-HGAT
model.eval()
hgat_p, hgat_t = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict,
                    batch.batch_dict, batch.global_params)
        hgat_p.extend(out.cpu().numpy().flatten())
        hgat_t.extend(batch.y.cpu().numpy().flatten())
hgat_p, hgat_t = np.array(hgat_p), np.array(hgat_t)

# Baselines (using standardized features)
lr_p  = lr_model.predict(Xte_s)
xgb_p = xgb_model.predict(Xte_s)
ann.eval()
with torch.no_grad():
    ann_p = ann(torch.tensor(Xte_s, dtype=torch.float).to(device)).cpu().numpy().flatten()

results = {
    'PI-HGAT':    metrics(hgat_t, hgat_p),
    'XGBoost':    metrics(Yte, xgb_p),
    'ANN (MLP)':  metrics(Yte, ann_p),
    'Linear Reg': metrics(Yte, lr_p),
}
df = pd.DataFrame(results).T
df.columns = ['R²', 'RMSE', 'MAE', 'MAPE (%)']
print('\n===== BENCHMARK RESULTS =====')
display(df.round(4))

# ===============
print(f'\n[Section 7] Execution time: {time.time() - section_7_start:.2f} seconds')

# ===============
import time
section_8_start = time.time()

# ===============
# FIGURE 5, 6, 7
from scripts.analysis.fig_style import apply_style, savefig, MODEL_COLORS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Gọi lại apply_style trước mỗi plot để chống Jupyter override
apply_style()

# --- FIG 5: Predicted vs Actual (4-panel) ---
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
targets = [(hgat_t, hgat_p, 'PI-HGAT'), (Yte, xgb_p, 'XGBoost'), 
           (Yte, ann_p, 'ANN (MLP)'), (Yte, lr_p, 'Linear Reg')]

for i, ax in enumerate(axes.flatten()):
    yt, yp, name = targets[i]
    color = MODEL_COLORS.get(name, '#000000')
    ax.scatter(yt, yp, alpha=0.5, s=15, color=color)
    
    lims = [min(yt.min(), yp.min()) - 5, max(yt.max(), yp.max()) + 5]
    ax.plot(lims, lims, '--', lw=1.5, color='#c3c2b7')
    r2 = df.loc[name, 'R²']
    ax.set_title(f'{name} (R² = {r2:.4f})', fontname='Arial', fontweight='bold')
    ax.set_xlabel('Actual Net EUI (kWh/m²/yr)', fontname='Arial')
    ax.set_ylabel('Predicted Net EUI', fontname='Arial')

plt.tight_layout()
savefig(fig, 'Fig5_PredictionPerf')
plt.show()

# --- FIG 6: Benchmark Bar Chart (R2) ---
apply_style()
fig, ax = plt.subplots(figsize=(6, 4))
colors = [MODEL_COLORS[m] for m in df.index]
bars = ax.bar(df.index, df['R²'], color=colors, alpha=0.9)

for bar, val in zip(bars, df['R²']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', fontsize=9, fontname='Arial', color='#0b0b0b')

ax.set_title('Model Comparison (R² Score)', fontname='Arial', fontweight='bold')
ax.set_ylabel('R² Score', fontname='Arial')
ax.set_ylim(0, 1.1)
ax.grid(axis='y') # Tắt lưới dọc, chỉ giữ lưới ngang cho bar chart
# Ép tick labels dùng Arial
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontname('Arial')
    
savefig(fig, 'Fig6_BenchmarkBar')
plt.show()

# --- FIG 7: Learning Curve (PI-HGAT) ---
apply_style()
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hist['train'], label='Train Loss', color=MODEL_COLORS['PI-HGAT'], lw=2)
ax.plot(hist['val'], label='Val Loss', color='#eda100', lw=2) 
ax.set_title('PI-HGAT Learning Curve', fontname='Arial', fontweight='bold')
ax.set_xlabel('Epoch', fontname='Arial')
ax.set_ylabel('Loss (MSE)', fontname='Arial')

# Ép tick labels dùng Arial
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontname('Arial')
    
ax.legend(prop={'family': 'Arial', 'size': 8})
savefig(fig, 'Fig7_LearningCurve')
plt.show()

# ===============
# --- EXTERNAL TEST SET PREDICTION ---
print('\n--- Evaluating on External Test Set (Seed 2810) ---')
import pandas as pd
import numpy as np
import random
import torch
from torch_geometric.loader import DataLoader as PyGDL
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

ext_df = pd.read_csv('data/external_test_results.csv')
print(f'Loaded {len(ext_df)} external test samples.')

climate_delta_map = {'1_Baseline': 0.0, '2': 1.879, '3': 2.665, '4': 2.179, '5': 4.472, '6': 1.27, '7': 1.875, '8': 1.611, '9': 3.144}
P8_LEVELS = [0, 30, 60, 90, 120, 150]
P9_LEVELS = [0, 30, 60, 90, 120, 150]

ext_samples = []
ext_dataset = []
ext_Y = []

for _, row in ext_df.iterrows():
    s = {
        'P1_Wall_U': 1.0 / row['@@P1_Wall_R@@'],
        'P2_Roof_U': 1.0 / row['@@P2_Roof_R@@'],
        'P3_Roof_Reflectance': 1.0 - row['@@P3_Roof_Abs@@'],
        'P4_Win_U': row['@@P4_U@@'],
        'P4_Win_SHGC': row['@@P4_SHGC@@'],
        'P5_COP': row['@@P5_COP@@'],
        'P6_Cool_SP': row['@@P6_ClgSetp@@'],
        'P7_LPD': row['@@P7_LPD@@'],
        'P8_PV_kW': random.choice(P8_LEVELS),
        'P9_BESS_kWh': random.choice(P9_LEVELS),
        'Climate_DeltaT': climate_delta_map.get(str(row.get('Scenario', '1_Baseline')), 0.0)
    }
    
    gross_eui_kwh_m2 = row['EUI_MJ_m2'] / 3.6
    pv_yield = 1420.0
    pv_gen_kwh = s['P8_PV_kW'] * pv_yield
    
    daily_pv = pv_gen_kwh / 365.0
    bess_ratio = s['P9_BESS_kWh'] / daily_pv if daily_pv > 0 else 0
    sc_factor = min(1.0, 0.6 + 0.4 * min(1.0, bess_ratio))
    
    net_eui_kwh_m2 = max(0.0, gross_eui_kwh_m2 - (pv_gen_kwh * sc_factor / 4982.0))
    
    data = builder.create_sample_graph(s)
    data.y = torch.tensor([[net_eui_kwh_m2]], dtype=torch.float)
    x_flat = [s['P1_Wall_U'], s['P2_Roof_U'], s['P3_Roof_Reflectance'],
              s['P4_Win_U'], s['P4_Win_SHGC'], s['P5_COP'],
              s['P6_Cool_SP'], s['P7_LPD'], s['P8_PV_kW'], s['P9_BESS_kWh'],
              s['Climate_DeltaT']]
    data.global_params = torch.tensor([x_flat], dtype=torch.float)
    ext_dataset.append(data)
    ext_Y.append(net_eui_kwh_m2)

ext_loader = PyGDL(ext_dataset, batch_size=TRAIN_PARAMS['batch_size'], shuffle=False)

model.eval()
ext_preds, ext_trues = [], []
with torch.no_grad():
    for batch in ext_loader:
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict, batch.global_params)
        ext_preds.extend(out.cpu().numpy().flatten())
        ext_trues.extend(batch.y.cpu().numpy().flatten())

ext_r2 = r2_score(ext_trues, ext_preds)
ext_rmse = mean_squared_error(ext_trues, ext_preds)**0.5
ext_mae = mean_absolute_error(ext_trues, ext_preds)
ext_mape = mean_absolute_percentage_error(ext_trues, ext_preds) * 100
print(f'External Test - R2: {ext_r2:.3f}, RMSE: {ext_rmse:.2f}, MAE: {ext_mae:.2f}, MAPE: {ext_mape:.2f}%')

df.loc['PI-HGAT (Ext. Test)'] = [ext_r2, ext_rmse, ext_mae, ext_mape]
display(df.round(4))

# ===============
print(f'\n[Section 8] Execution time: {time.time() - section_8_start:.2f} seconds')

# ===============
import time
section_9_start = time.time()

# ===============
from pymoo.core.problem import ElementwiseProblem
from pi_hgat.objectives import ObjectiveCalculator
import torch
import pandas as pd

# Initialize objectives calculator
obj_calc = ObjectiveCalculator(builder)

class NZEBRetrofitProblem(ElementwiseProblem):
    def __init__(self, model_surrogate, builder):
        # 10 variables (P1..P9 + Climate). Climate is fixed to HCMC mean for MOO.
        self.var_names = ['P1_Wall_U', 'P2_Roof_U', 'P3_Roof_Reflectance', 'P4_Win_U', 'P4_Win_SHGC', 'P5_COP', 'P6_Cool_SP', 'P7_LPD', 'P8_PV_kW', 'P9_BESS_kWh']
        super().__init__(n_var=10, n_obj=3, n_ieq_constr=0,
                         xl=np.array([0.29, 0.18, 0.30, 1.00, 0.15, 2.96, 24.0, 2.50, 0.0, 0.0]),
                         xu=np.array([1.07, 0.45, 0.85, 2.87, 0.22, 5.00, 27.0, 6.66, 150.0, 150.0]))
        self.model = model_surrogate
        self.model.eval()
        self.builder = builder

    def _evaluate(self, x, out, *args, **kwargs):
        # 1. Map decision variables
        params = {
            'P1_Wall_U': x[0], 'P2_Roof_U': x[1], 'P3_Roof_Reflectance': x[2],
            'P4_Win_U': x[3], 'P4_Win_SHGC': x[4], 'P5_COP': x[5],
            'P6_Cool_SP': x[6], 'P7_LPD': x[7], 'P8_PV_kW': x[8], 'P9_BESS_kWh': x[9],
            'Climate_DeltaT': 0.0  # Assume mean climate for design optimization
        }
        
        # 2. Predict Gross EUI using PI-HGAT
        data = self.builder.create_sample_graph(params)
        
        x_flat = [params['P1_Wall_U'], params['P2_Roof_U'], params['P3_Roof_Reflectance'],
                  params['P4_Win_U'], params['P4_Win_SHGC'], params['P5_COP'],
                  params['P6_Cool_SP'], params['P7_LPD'], params['P8_PV_kW'], params['P9_BESS_kWh'], params['Climate_DeltaT']]
        data.global_params = torch.tensor([x_flat], dtype=torch.float)
        
        with torch.no_grad():
            batch_dict = {nt: torch.zeros(data[nt].x.size(0), dtype=torch.long, device=device) 
                          for nt in data.node_types}
            out_eui = self.model(
                {nt: data[nt].x.to(device) for nt in data.node_types},
                {et: data[et].edge_index.to(device) for et in data.edge_types},
                batch_dict,
                data.global_params.to(device)
            )
            gross_eui = out_eui.item()
            
        # Post-process Net EUI
        pv_yield = 1420.0
        pv_gen_kwh = params['P8_PV_kW'] * pv_yield
        daily_pv = pv_gen_kwh / 365.0
        bess_ratio = params['P9_BESS_kWh'] / daily_pv if daily_pv > 0 else 0
        sc_factor = min(1.0, 0.6 + 0.4 * min(1.0, bess_ratio))
        net_eui = max(0.0, gross_eui - (pv_gen_kwh * sc_factor / 4982.0))
            
        # 3. Calculate LCC and LCA (using NET EUI for operational costs/emissions)
        lcc = obj_calc.calculate_lcc(params, net_eui)
        lca = obj_calc.calculate_lca(params, net_eui)
        
        # 4. Assign objectives (Minimize all)
        out["F"] = [net_eui, lcc, lca]

problem = NZEBRetrofitProblem(model, builder)
print("MOO Problem Defined: 10 Variables, 3 Objectives (EUI, LCC, LCA)")
bounds_df = pd.DataFrame({'Lower Bound': problem.xl, 'Upper Bound': problem.xu}, index=problem.var_names)
print('\nDecision Variable Bounds (jEPlus scale):')
display(bounds_df)

# ===============
# FIGURE 3 & 4
from scripts.analysis.fig_style import apply_style, savefig, LCA_COLORS
import matplotlib.pyplot as plt
import numpy as np
from pi_hgat.config import LIFESPANS_SHORT
apply_style()

# --- FIG 3: Lifespans ---
fig, ax = plt.subplots(figsize=(8, 4))
components = [f'{k}: {v} yrs' for k, v in LIFESPANS_SHORT.items()]
lifespans = list(LIFESPANS_SHORT.values())
y_pos = np.arange(len(components))
ax.barh(y_pos, lifespans, color='#2a78d6', edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(components)
ax.invert_yaxis()
ax.set_xlabel('Lifespan (Years)')
ax.set_title('Building Components Lifespan vs Project Life')
ax.axvline(x=20, color='#f0a4a4', linestyle='--', linewidth=2, label='Study Period (20 yrs)')
ax.legend(loc='lower right')
savefig(fig, 'Fig3_Lifespan')
plt.show()

# --- FIG 4: LCA Modules at Max Level ---
params_max = {
    'P1_Wall_U': 0.29, 'P2_Roof_U': 0.18, 'P3_Roof_Reflectance': 0.85,
    'P4_Win_U': 1.00, 'P4_Win_SHGC': 0.15, 'P5_COP': 5.00,
    'P6_Cool_SP': 27.0, 'P7_LPD': 2.50, 'P8_PV_kW': 150.0, 'P9_BESS_kWh': 150.0
}
lca_breakdown = obj_calc.calculate_lca_breakdown(params_max, 80.0) 
modules = ['A1-A3', 'A4-A5', 'B2-B3', 'B4', 'C1-C4']
vals = [lca_breakdown[m] for m in modules]
colors = [LCA_COLORS[m] for m in modules]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(modules, vals, color=colors, edgecolor='black', alpha=0.9)
ax.set_ylabel('Life Cycle Emissions (kgCO2eq)')
ax.set_title('LCE Distribution by LCA Modules (Max Retrofit)')
savefig(fig, 'Fig4_LCEDistribution')
plt.show()

# ===============
print(f'\n[Section 9] Execution time: {time.time() - section_9_start:.2f} seconds')

# ===============
import time
section_10_start = time.time()

# ===============
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

# Generate reference directions for NSGA-III (3 objectives)
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

algorithm = NSGA3(pop_size=92, ref_dirs=ref_dirs)  # pop_size 92 is better for n_partitions=12

print("Running NSGA-III optimization... (This may take a minute)")
import time
t0 = time.time()
res = minimize(problem,
               algorithm,
               seed=42,
               termination=('n_gen', 50),
               save_history=True,
               verbose=True)

print(f"Optimization finished in {time.time()-t0:.1f}s")
print(f"Found {len(res.F)} Pareto optimal solutions.")

# ===============
# FIGURE 8
from scripts.analysis.fig_style import apply_style, savefig, seq_cmap
import matplotlib.pyplot as plt
apply_style()

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


# --- FIG 8a: 3D Pareto ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(F[:, 0], F[:, 1]/1e6, F[:, 2]/1e6, c=closeness, cmap=seq_cmap(), s=40, alpha=0.8)
ax.scatter(best_obj[0], best_obj[1]/1e6, best_obj[2]/1e6, color='red', s=150, marker='*', label='TOPSIS Best')

ax.set_xlabel('Net EUI (kWh/m2/yr)')
ax.set_ylabel('LCC (Million $)')
ax.set_zlabel('LCA (Million kgCO2eq)')
ax.set_title('NSGA-III Pareto Front')
plt.colorbar(sc, label='TOPSIS Closeness')
plt.legend()
savefig(fig, 'Fig8_Pareto3D')
plt.show()

# --- FIG 8b: Convergence (Hypervolume) ---
if res.history is not None:
    from pymoo.indicators.hv import Hypervolume
    ref_point = np.array([200.0, 3e6, 1e7])
    hv = Hypervolume(ref_point=ref_point)
    hv_history = []
    for algo in res.history:
        opt = algo.opt
        if len(opt) > 0:
            hv_history.append(hv.do(opt.get("F")))
        else:
            hv_history.append(0.0)
            
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hv_history, color='#2a78d6', marker='o', markersize=4)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Hypervolume')
    ax.set_title('NSGA-III Convergence')
    savefig(fig, 'Fig8b_Convergence')
    plt.show()

# ===============
print(f'\n[Section 10] Execution time: {time.time() - section_10_start:.2f} seconds')

# ===============
import time
section_11_start = time.time()

# ===============
from scipy.stats import entropy
import pandas as pd
import os

F = res.F

# --- Entropy-TOPSIS ---
# 1. Normalize Decision Matrix
norm_F = F / np.sqrt((F**2).sum(axis=0))

# 2. Calculate Entropy Weights
P = norm_F / norm_F.sum(axis=0)
E = -np.nansum(P * np.log(P), axis=0) / np.log(len(F))
W = (1 - E) / (1 - E).sum()
print(f"Objective Weights (Entropy): EUI={W[0]:.3f}, LCC={W[1]:.3f}, LCA={W[2]:.3f}")

# 3. Weighted Normalized Matrix
V = norm_F * W

# 4. Ideal and Anti-Ideal Solutions (Cost criteria: min is ideal)
ideal = V.min(axis=0)
anti_ideal = V.max(axis=0)

# 5. Distances & Closeness
d_ideal = np.sqrt(((V - ideal)**2).sum(axis=1))
d_anti = np.sqrt(((V - anti_ideal)**2).sum(axis=1))
closeness = d_anti / (d_ideal + d_anti)

best_idx = np.argmax(closeness)
best_solution = res.X[best_idx]
best_obj = res.F[best_idx]

print("\n===== OPTIMAL COMPROMISE SOLUTION (TOPSIS) =====")
print(f"EUI: {best_obj[0]:.2f} kWh/m2/yr")
print(f"LCC: ${best_obj[1]:,.2f}")
print(f"LCA: {best_obj[2]:,.2f} kgCO2eq")
print(f"Parameters: P1={best_solution[0]:.2f}, P2={best_solution[1]:.2f}, P3={best_solution[2]:.2f}, "
      f"P4_U={best_solution[3]:.2f}, P4_SHGC={best_solution[4]:.2f}, P5={best_solution[5]:.2f}, "
      f"P6={best_solution[6]:.1f}, P7={best_solution[7]:.2f}, P8={best_solution[8]:.1f}kW, "
      f"P9={best_solution[9]:.1f}kWh")

# Save pareto_solutions.csv
df_pareto = pd.DataFrame(res.X, columns=['P1_Wall_U', 'P2_Roof_U', 'P3_Roof_Reflectance', 'P4_Win_U', 'P4_Win_SHGC', 'P5_COP', 'P6_Cool_SP', 'P7_LPD', 'P8_PV_kW', 'P9_BESS_kWh'])
df_pareto['EUI'] = res.F[:, 0]
df_pareto['LCC'] = res.F[:, 1]
df_pareto['LCA'] = res.F[:, 2]
df_pareto['Closeness'] = closeness
df_pareto['NZE_class'] = np.where(df_pareto['EUI'] <= 0.0, 'NZE', 'Near-NZE')

os.makedirs('results', exist_ok=True)
df_pareto.to_csv('results/pareto_solutions.csv', index=False)
print(f"\nSaved {len(df_pareto)} solutions to results/pareto_solutions.csv")
print(df_pareto['NZE_class'].value_counts())

# ===============
# FIGURE 9, 10, 11
from scripts.analysis.fig_style import apply_style, savefig, seq_cmap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
apply_style()

# --- FIG 9: Pairwise Pareto ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
labels = ['Net EUI (kWh/m²/yr)', 'LCC (Million $)', 'LCA (Million kgCO2eq)']
pairs = [(0, 1), (0, 2), (1, 2)]

for i, ax in enumerate(axes):
    idx_x, idx_y = pairs[i]
    x_data = F[:, idx_x] if idx_x == 0 else F[:, idx_x] / 1e6
    y_data = F[:, idx_y] if idx_y == 0 else F[:, idx_y] / 1e6
    
    sc = ax.scatter(x_data, y_data, c=closeness, cmap=seq_cmap(), s=25, alpha=0.8)
    ax.set_xlabel(labels[idx_x])
    ax.set_ylabel(labels[idx_y])

cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('TOPSIS Closeness Coefficient')
fig.suptitle('Pairwise Pareto Solutions')
savefig(fig, 'Fig9_Pairwise')
plt.show()

# --- FIG 11: Heatmap ---
df_pareto = pd.read_csv('results/pareto_solutions.csv')
df_pareto_sorted = df_pareto.sort_values(by='Closeness', ascending=False)
X_pareto = df_pareto_sorted[['P1_Wall_U', 'P2_Roof_U', 'P3_Roof_Reflectance', 'P4_Win_U', 'P4_Win_SHGC', 'P5_COP', 'P6_Cool_SP', 'P7_LPD', 'P8_PV_kW', 'P9_BESS_kWh']]

X_min = X_pareto.min()
X_max = X_pareto.max()
X_norm = (X_pareto - X_min) / (X_max - X_min + 1e-9)

plt.figure(figsize=(10, 6))
sns.heatmap(X_norm.T, cmap=seq_cmap(), cbar_kws={'label': 'Normalized Level'})
plt.title('Renovation Levels for Pareto Solutions (Sorted by TOPSIS)')
plt.xlabel('Pareto Rank')
plt.ylabel('Decision Variables')
plt.tight_layout()
savefig(plt.gcf(), 'Fig11_Heatmap')
plt.show()

# ===============
print(f'\n[Section 11] Execution time: {time.time() - section_11_start:.2f} seconds')

# ===============
import time
section_12_start = time.time()

# ===============
from torch_geometric.explain import Explainer, GNNExplainer
import networkx as nx

# Prepare data using Best Compromise Solution from TOPSIS
best_params = {
    'P1_Wall_U': best_solution[0], 'P2_Roof_U': best_solution[1], 'P3_Roof_Reflectance': best_solution[2],
    'P4_Win_U': best_solution[3], 'P4_Win_SHGC': best_solution[4], 'P5_COP': best_solution[5],
    'P6_Cool_SP': best_solution[6], 'P7_LPD': best_solution[7], 'P8_PV_kW': best_solution[8], 
    'P9_BESS_kWh': best_solution[9],
    'Climate_DeltaT': 0.0
}

data = builder.create_sample_graph(best_params)
data.global_params = torch.tensor([[best_solution[0], best_solution[1], best_solution[2],
                                    best_solution[3], best_solution[4], best_solution[5],
                                    best_solution[6], best_solution[7], best_solution[8], 
                                    best_solution[9], 0.0]], dtype=torch.float)


batch_dict = {nt: torch.zeros(data[nt].x.size(0), dtype=torch.long, device=device) for nt in data.node_types}
data = data.to(device)
data.global_params = data.global_params.to(device)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, batch_dict, global_params):
        super().__init__()
        self.model = model
        self.batch_dict = batch_dict
        self.global_params = global_params
    def forward(self, x_dict, edge_index_dict):
        return self.model(x_dict, edge_index_dict, self.batch_dict, self.global_params)

wrapped_model = ModelWrapper(model, batch_dict, data.global_params)

explainer = Explainer(
    model=wrapped_model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(mode='regression', task_level='graph', return_type='raw')
)

print("Running GNNExplainer (learning masks for mutual information)...")
explanation = explainer(data.x_dict, data.edge_index_dict)
print("Explanation completed.")

feat_names = {
    'Zone': ['area', 'volume', 'height', 'LPD', 'PV_share'],
    'Envelope': ['area', 'tilt', 'azimuth', 'is_wall', 'is_roof', 'is_floor', 'is_window', 'U-value', 'Reflectance', 'SHGC', 'ShapeIndex'],
    'System': ['cooling_cap', 'heating_cap', 'COP', 'Cool_SP', 'Heat_SP']
}

print("\n--- Top Node Features by Mask Score ---")
for nt in ['Zone', 'Envelope', 'System']:
    mask = explanation.node_mask_dict[nt].cpu().numpy()
    mean_mask = mask.mean(axis=0)
    top_idx = mean_mask.argsort()[::-1][:3]
    print(f"\n{nt} Features:")
    for i in top_idx:
        print(f"  - {feat_names[nt][i]}: {mean_mask[i]:.4f}")

# ===============
# FIGURE 12, 13, 14
from scripts.analysis.fig_style import apply_style, savefig
import matplotlib.pyplot as plt
import numpy as np
import shap
apply_style()

# --- FIG 12a: Global Feature Importance (GNNExplainer) ---
importances = []
labels = []
for nt in ['Zone', 'Envelope', 'System']:
    mask = explanation.node_mask_dict[nt].cpu().numpy().mean(axis=0)
    for i, score in enumerate(mask):
        importances.append(score)
        labels.append(f"{nt}: {feat_names[nt][i]}")
        
idx = np.argsort(importances)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(np.array(labels)[idx], np.array(importances)[idx], color='#2a78d6')
ax.set_xlabel('GNNExplainer Mask Score')
ax.set_title('Feature Importance per Node Type')
savefig(fig, 'Fig12a_NodeImportance')
plt.show()

# --- FIG 12b: SHAP Beeswarm (XGBoost) ---
explainer_xgb = shap.Explainer(xgb_model)
shap_values = explainer_xgb(Xte_s)

plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, Xte, feature_names=problem.var_names + ['Climate_DeltaT'], show=False)
savefig(plt.gcf(), 'Fig12b_SHAP')
plt.show()

# --- FIG 13: Edge-type Importance ---
edge_importances = []
edge_labels = []
for et in data.edge_types:
    mask = explanation.edge_mask_dict[et].cpu().numpy().mean()
    edge_importances.append(mask)
    edge_labels.append(f"{et[0]}-{et[1]}-{et[2]}")
    
idx_e = np.argsort(edge_importances)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(np.array(edge_labels)[idx_e], np.array(edge_importances)[idx_e], color='#1baf7a')
ax.set_xlabel('Mean Edge Mask Score')
ax.set_title('Edge-type Connection Importance')
savefig(fig, 'Fig13_EdgeImportance')
plt.show()

# ===============
print(f'\n[Section 12] Execution time: {time.time() - section_12_start:.2f} seconds')

# ===============
