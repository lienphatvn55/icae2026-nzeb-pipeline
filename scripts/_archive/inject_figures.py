import nbformat

def inject():
    nb_path = 'NZEB_PIPELINE_ICAE2026.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    new_cells = []
    
    code_fig2 = """# FIG 2 (Mock/Setup - requires external image layout, but we can print stats)
print('Node Counts:', n_nodes)
print('Edge Counts:', n_edges)
# Actual drawing of the KG schema is done outside in a graphics tool."""

    code_fig567 = """# FIGURE 5, 6, 7
from scripts.analysis.fig_style import apply_style, savefig, MODEL_COLORS
import matplotlib.pyplot as plt
import numpy as np
apply_style()

# --- FIG 5: Predicted vs Actual ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
targets = [(hgat_t, hgat_p, 'PI-HGAT'), (Yte, xgb_p, 'XGBoost'), 
           (Yte, ann_p, 'ANN (MLP)'), (Yte, lr_p, 'Linear Reg')]

for i, ax in enumerate(axes.flatten()):
    yt, yp, name = targets[i]
    color = MODEL_COLORS.get(name, '#000000')
    ax.scatter(yt, yp, alpha=0.5, s=15, color=color)
    
    lims = [min(yt.min(), yp.min()) - 5, max(yt.max(), yp.max()) + 5]
    ax.plot(lims, lims, 'r--', lw=1.5, color='#c3c2b7')
    r2 = df.loc[name, 'R²']
    ax.set_title(f'{name} (R² = {r2:.4f})')
    ax.set_xlabel('Actual Net EUI (kWh/m²/yr)')
    ax.set_ylabel('Predicted Net EUI')

plt.tight_layout()
savefig(fig, 'Fig5_PredictionPerf')
plt.show()

# (Note: Multi-seed learning curve and timing for Fig 6/7 are omitted here for brevity, 
# but would involve wrapping the training loop above and plotting the stats.)
"""

    code_fig34 = """# FIGURE 3 & 4
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
# We use obj_calc to get LCA breakdown for MAX level.
params_max = {
    'P1_Wall_U': 0.29, 'P2_Roof_U': 0.18, 'P3_Roof_Reflectance': 0.85,
    'P4_Win_U': 1.00, 'P4_Win_SHGC': 0.15, 'P5_COP': 5.00,
    'P6_Cool_SP': 27.0, 'P7_LPD': 2.50, 'P8_PV_kW': 150.0, 'P9_BESS_kWh': 150.0
}
lca_breakdown = obj_calc.calculate_lca_breakdown(params_max, 80.0) # Assumed Net EUI = 80
modules = ['A1-A3', 'A4-A5', 'B2-B3', 'B4', 'C1-C4']
vals = [lca_breakdown[m] for m in modules]
colors = [LCA_COLORS[m] for m in modules]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(modules, vals, color=colors, edgecolor='black', alpha=0.9)
ax.set_ylabel('Life Cycle Emissions (kgCO2eq)')
ax.set_title('LCE Distribution by LCA Modules (Max Retrofit)')
savefig(fig, 'Fig4_LCEDistribution')
plt.show()
"""

    code_fig8 = """# FIGURE 8
from scripts.analysis.fig_style import apply_style, savefig, seq_cmap
import matplotlib.pyplot as plt
apply_style()

# --- FIG 8a: 3D Pareto ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(F[:, 0], F[:, 1]/1e6, F[:, 2]/1e6, c=closeness, cmap=seq_cmap(), s=40, alpha=0.8)
ax.scatter(best_obj[0], best_obj[1]/1e6, best_obj[2]/1e6, color='red', s=150, marker='*', label='TOPSIS Best')

ax.set_xlabel('EUI (kWh/m2/yr)')
ax.set_ylabel('LCC (Million $)')
ax.set_zlabel('LCA (Million kgCO2eq)')
ax.set_title('NSGA-III Pareto Front')
plt.colorbar(sc, label='TOPSIS Closeness')
plt.legend()
savefig(fig, 'Fig8_Pareto3D')
plt.show()

# --- FIG 8b: Convergence (Hypervolume) ---
# If save_history=True was passed to minimize, res.history contains the generations.
if res.history is not None:
    from pymoo.indicators.hv import Hypervolume
    ref_point = np.array([200.0, 3e6, 1e7]) # Approx worst values
    hv = Hypervolume(ref_point=ref_point)
    hv_history = []
    for algo in res.history:
        # Extract feasible objective space values
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
"""

    code_fig9_10_11 = """# FIGURE 9, 10, 11
from scripts.analysis.fig_style import apply_style, savefig, seq_cmap, LCA_COLORS
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
apply_style()

# --- FIG 9: Pairwise Pareto ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
labels = ['EUI (kWh/m²/yr)', 'LCC (Million $)', 'LCA (Million kgCO2eq)']
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
import pandas as pd
df_pareto = pd.read_csv('results/pareto_solutions.csv')
df_pareto_sorted = df_pareto.sort_values(by='Closeness', ascending=False)
X_pareto = df_pareto_sorted[['P1_Wall_U', 'P2_Roof_U', 'P3_Roof_Reflectance', 'P4_Win_U', 'P4_Win_SHGC', 'P5_COP', 'P6_Cool_SP', 'P7_LPD', 'P8_PV_kW', 'P9_BESS_kWh']]

# Normalize to 0-1 for visualization
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
"""

    code_fig12_13_14 = """# FIGURE 12, 13, 14
from scripts.analysis.fig_style import apply_style, savefig
import matplotlib.pyplot as plt
import numpy as np
import shap
apply_style()

# --- FIG 12a: Global Feature Importance (GNNExplainer) ---
# Extract node masks
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
"""

    inserted = set()
    for cell in nb.cells:
        new_cells.append(cell)
        if cell.cell_type == 'markdown':
            source = "".join(cell.get('source', []))
            if 'Fig. 2' in source and 'Fig. 2' not in inserted:
                new_cells.append(nbformat.v4.new_code_cell(code_fig2))
                inserted.add('Fig. 2')
            elif 'Fig. 5' in source and 'Fig. 5' not in inserted:
                new_cells.append(nbformat.v4.new_code_cell(code_fig567))
                inserted.add('Fig. 5')
            elif 'Fig. 3' in source and 'Fig. 3' not in inserted:
                new_cells.append(nbformat.v4.new_code_cell(code_fig34))
                inserted.add('Fig. 3')
            elif 'Fig. 8' in source and 'Fig. 8' not in inserted:
                new_cells.append(nbformat.v4.new_code_cell(code_fig8))
                inserted.add('Fig. 8')
            elif 'Fig. 9' in source and 'Fig. 9' not in inserted:
                new_cells.append(nbformat.v4.new_code_cell(code_fig9_10_11))
                inserted.add('Fig. 9')
            elif 'Fig. 12' in source and 'Fig. 12' not in inserted:
                new_cells.append(nbformat.v4.new_code_cell(code_fig12_13_14))
                inserted.add('Fig. 12')

    nb.cells = new_cells
    with open('NZEB_PIPELINE_ICAE2026.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Injected figure code cells into NZEB_PIPELINE_ICAE2026.ipynb")

if __name__ == '__main__':
    inject()
