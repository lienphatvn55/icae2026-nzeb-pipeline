import nbformat

def update_notebook():
    nb_path = 'NZEB_PIPELINE_ICAE2026.ipynb'
    nb = nbformat.read(nb_path, as_version=4)
    
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and '### Section 2-3: Synthetic Data Generation' in cell.source:
            nb.cells[i].source = '### Section 2-3: Load LHS Sampling Data (from jEPlus E+ simulations)'
            
        if cell.cell_type == 'code' and "gen = SyntheticDataGenerator(builder, num_samples=5000)" in cell.source:
            new_code = """print('Loading LHS samples from aggregated_LHS_results.csv...')
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'data/aggregated_LHS_results.csv')

samples = []
X = []
Y = []

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
              
    # Convert Gross EUI from MJ/m2 to kWh/m2
    gross_eui_kwh_m2 = row['EUI_MJ_m2'] / 3.6
    
    # Calculate Net EUI (Subtract PV generation)
    # PV production: ~1500 kWh/yr per kW in HCMC. Total floor area = 4982 m2
    pv_generation_kwh = s['P8_PV_kW'] * 1500.0
    net_eui_kwh_m2 = gross_eui_kwh_m2 - (pv_generation_kwh / 4982.0)
    
    Y.append([max(10.0, net_eui_kwh_m2)])

X_flat = np.array(X)
Y_eui = np.array(Y)

print(f'X: {X_flat.shape}, Y: {Y_eui.shape}')
print(f'EUI range: {Y_eui.min():.1f} – {Y_eui.max():.1f} kWh/m²/yr')
print(f'Baseline check (first sample): {Y_eui[0,0]:.1f}')

idx = np.arange(len(samples))
idx_train, idx_temp = train_test_split(idx, test_size=0.3, random_state=42)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)
print(f'Split: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test')"""
            nb.cells[i].source = new_code
            nb.cells[i].outputs = []

    nbformat.write(nb, nb_path)
    print("Notebook updated successfully.")

if __name__ == '__main__':
    update_notebook()
