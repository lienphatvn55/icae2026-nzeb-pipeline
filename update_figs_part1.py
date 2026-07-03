import nbformat
import json

nb_path = 'NZEB_PIPELINE_ICAE2026.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

cells_to_keep = []
for i, c in enumerate(nb.cells):
    if c.cell_type == 'code':
        source = "".join(c.get('source', []))
        
        # 1. Fix XGBoost tree_method
        if 'xgb.XGBRegressor' in source and "tree_method='hist'" in source:
            c.source = source.replace("tree_method='hist'", "tree_method='exact'")
            print("Fixed XGBoost tree_method.")
            
        # 2. Delete the old Cell 19 (starts with fig, axes = plt.subplots(2, 2, figsize=(14, 11)))
        if source.startswith('fig, axes = plt.subplots(2, 2, figsize=(14, 11))'):
            print("Deleted old unstyled 2x2 grid.")
            continue # skip this cell
            
        # 3. Update the injected FIGURE 5, 6, 7 to include all required charts properly styled
        if source.startswith('# FIGURE 5, 6, 7'):
            new_source = """# FIGURE 5, 6, 7
from scripts.analysis.fig_style import apply_style, savefig, MODEL_COLORS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    ax.set_title(f'{name} (R² = {r2:.4f})')
    ax.set_xlabel('Actual Net EUI (kWh/m²/yr)')
    ax.set_ylabel('Predicted Net EUI')

plt.tight_layout()
savefig(fig, 'Fig5_PredictionPerf')
plt.show()

# --- FIG 6: Benchmark Bar Chart (R2) ---
fig, ax = plt.subplots(figsize=(6, 4))
colors = [MODEL_COLORS[m] for m in df.index]
bars = ax.bar(df.index, df['R²'], color=colors, alpha=0.9)
for bar, val in zip(bars, df['R²']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', fontsize=9)
ax.set_title('Model Comparison (R² Score)')
ax.set_ylabel('R² Score')
ax.set_ylim(0, 1.1)
savefig(fig, 'Fig6_BenchmarkBar')
plt.show()

# --- FIG 7: Learning Curve (PI-HGAT) ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hist['train'], label='Train Loss', color=MODEL_COLORS['PI-HGAT'], lw=2)
ax.plot(hist['val'], label='Val Loss', color='#eda100', lw=2) # use yellow for val
ax.set_title('PI-HGAT Learning Curve')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.legend()
savefig(fig, 'Fig7_LearningCurve')
plt.show()
"""
            c.source = new_source
            print("Updated Fig 5, 6, 7 cell with full styled charts.")
            
    cells_to_keep.append(c)

nb.cells = cells_to_keep
with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Notebook updated.")
