import nbformat
import os

nb_path = 'NZEB_PIPELINE_ICAE2026.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# 1. Update Cell 6 (Data loading & GroupShuffleSplit & BESS net energy logic)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "X_flat = np.array(X)" in cell.source:
        new_source = """print('Loading LHS samples from aggregated_LHS_results.csv...')
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
print(f'Split: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test')"""
        nb.cells[i].source = new_source

# 2. Update Cell 14 (Baseline models & StandardScaler)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "Training Linear Regression" in cell.source:
        new_source = """from sklearn.preprocessing import StandardScaler

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
                              random_state=42, tree_method='hist')
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

print('All baselines trained.')"""
        nb.cells[i].source = new_source

# Update Cell 16 (Metrics evaluation - change inputs to Xte_s for baselines)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "lr_p  = lr_model.predict(Xte)" in cell.source:
        new_source = """def metrics(yt, yp):
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
print('\\n===== BENCHMARK RESULTS =====')
display(df.round(4))"""
        nb.cells[i].source = new_source

# 3. Update Cell 21 (MOO evaluate with self-consumption)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "def _evaluate(self, x, out" in cell.source:
        new_source = """from pymoo.core.problem import ElementwiseProblem
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
print('\\nDecision Variable Bounds (jEPlus scale):')
display(bounds_df)
"""
        nb.cells[i].source = new_source


# 4. Update Cell 24 (NSGA-III save_history=True)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "res = minimize(problem," in cell.source:
        new_source = """from pymoo.algorithms.moo.nsga3 import NSGA3
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
print(f"Found {len(res.F)} Pareto optimal solutions.")"""
        nb.cells[i].source = new_source

# 5. Update Cell 27 (Save pareto_solutions.csv)
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "best_idx = np.argmax(closeness)" in cell.source:
        new_source = """from scipy.stats import entropy
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

print("\\n===== OPTIMAL COMPROMISE SOLUTION (TOPSIS) =====")
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
print(f"\\nSaved {len(df_pareto)} solutions to results/pareto_solutions.csv")
print(df_pareto['NZE_class'].value_counts())
"""
        nb.cells[i].source = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Patch completed successfully.")
