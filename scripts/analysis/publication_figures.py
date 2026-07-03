"""
=====================================================================
 !!! MOCK / PLACEHOLDER — DO NOT USE FOR THE PAPER !!!
 Every Fig3-Fig11 here is drawn from generate_mock_pareto_data()
 (random numbers) or hardcoded dummy values. They only exist to lock
 the figure LAYOUT (styled after the 2025 GAT-BEM paper).
 Real sources after the end-to-end notebook run: see docs/CODE_MAP.md §3.
=====================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Define color palette (Purple, Orange, Blue)
PALETTE = ["#6A0DAD", "#FF8C00", "#1E90FF", "#E91E63", "#00CED1", "#FFD700"]
CMAP = LinearSegmentedColormap.from_list("custom_cmap", ["#1E90FF", "#6A0DAD", "#FF8C00"])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

def generate_mock_pareto_data(n=100):
    # Mock Pareto front data (EUI, LCC, LCA)
    eui = np.linspace(40, 100, n) + np.random.normal(0, 2, n)
    lcc = 1e6 * (100 / eui) + np.random.normal(0, 50000, n)
    lca = 5e5 * (100 / eui) + np.random.normal(0, 25000, n)
    
    # Mock decision variables (P1-P9)
    variables = np.random.rand(n, 9)
    topsis = np.random.rand(n)
    
    return np.column_stack((eui, lcc, lca)), variables, topsis

def draw_fig3_lifespan():
    fig, ax = plt.subplots(figsize=(8, 4))
    components = ['Wall Insulation (P1)', 'Roof Insulation (P2)', 'Roof Coating (P3)', 
                  'Glazing (P4)', 'HVAC (P5, P6)', 'LED Lighting (P7)', 'Rooftop PV (P8)', 'BESS (P9)']
    lifespans = [75, 75, 20, 20, 15, 12, 30, 15]
    
    y_pos = np.arange(len(components))
    ax.barh(y_pos, lifespans, color=PALETTE[2], edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(components)
    ax.invert_yaxis()
    ax.set_xlabel('Lifespan (Years)')
    ax.set_title('Fig. 3: Building Components Lifespan vs Project Life')
    
    # Draw study period line (20-yr basis locked 2026-07-02; see pi_hgat/config.py)
    ax.axvline(x=20, color=PALETTE[1], linestyle='--', linewidth=2, label='Study Period (20 yrs)')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('Fig3_Lifespan.png')
    plt.close()

def draw_fig4_lce_distribution():
    fig, ax = plt.subplots(figsize=(8, 5))
    modules = ['A1-A3', 'A4-A5', 'B2-B3', 'B4', 'B6', 'C1-C4']
    values = [400, 40, 50, 120, 800, 60] # Mock data (tons CO2)
    
    ax.bar(modules, values, color=PALETTE[0], edgecolor='black', alpha=0.8)
    ax.set_ylabel('Life Cycle Emissions (tons CO2eq)')
    ax.set_title('Fig. 4: LCE Distribution by LCA Modules (Max Level Scenario)')
    
    for i, v in enumerate(values):
        ax.text(i, v + 10, str(v), ha='center')
        
    plt.tight_layout()
    plt.savefig('Fig4_LCEDistribution.png')
    plt.close()

def draw_fig5_prediction_perf():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    targets = ['EUI (kWh/m²/yr)', 'LCC ($)', 'LCA (kgCO2eq)']
    
    for i, ax in enumerate(axes):
        actual = np.random.uniform(20, 120, 100) if i==0 else np.random.uniform(1e6, 3e6, 100)
        predicted = actual + np.random.normal(0, actual*0.05)
        
        ax.scatter(actual, predicted, color=PALETTE[i%len(PALETTE)], alpha=0.6, s=15)
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=1.5)
        
        ax.set_xlabel(f'Actual {targets[i]}')
        ax.set_ylabel(f'Predicted {targets[i]}')
        ax.set_title(f'PI-HGAT Prediction: {targets[i].split()[0]}')
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.suptitle('Fig. 5: Surrogate Model Performance (Actual vs Predicted)', y=1.05)
    plt.tight_layout()
    plt.savefig('Fig5_PredictionPerf.png')
    plt.close()

def draw_fig6_nsga3_evolution():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    F, _, _ = generate_mock_pareto_data(150)
    
    labels = ['EUI (kWh/m²/yr)', 'LCC (Million $)', 'LCA (Million kgCO2eq)']
    pairs = [(0, 1), (0, 2), (1, 2)]
    
    for i, ax in enumerate(axes):
        idx_x, idx_y = pairs[i]
        x_data = F[:, idx_x] if idx_x == 0 else F[:, idx_x] / 1e6
        y_data = F[:, idx_y] if idx_y == 0 else F[:, idx_y] / 1e6
        
        # Plot mock early generations
        ax.scatter(x_data + np.random.normal(0, np.std(x_data)*0.2, len(x_data)), 
                   y_data + np.random.normal(0, np.std(y_data)*0.2, len(y_data)), 
                   color='gray', alpha=0.3, s=10, label='Early Gen')
        
        # Plot Pareto front
        ax.scatter(x_data, y_data, color=PALETTE[0], s=20, label='Pareto Front')
        
        ax.set_xlabel(labels[idx_x])
        ax.set_ylabel(labels[idx_y])
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend()
            
    plt.suptitle('Fig. 6: NSGA-III Optimization Evolution', y=1.05)
    plt.tight_layout()
    plt.savefig('Fig6_NSGA3Evolution.png')
    plt.close()

def draw_fig7_pairwise_pareto():
    F, _, _ = generate_mock_pareto_data(100)
    df = pd.DataFrame(F, columns=['EUI', 'LCC', 'LCA'])
    df['LCC'] /= 1e6
    df['LCA'] /= 1e6
    
    g = sns.PairGrid(df, diag_sharey=False, corner=True)
    g.map_lower(sns.scatterplot, color=PALETTE[2], alpha=0.7)
    g.map_diag(sns.histplot, kde=True, color=PALETTE[1])
    
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Fig. 7: Pairwise Distribution of Pareto Solutions')
    plt.savefig('Fig7_PairwisePareto.png')
    plt.close()

def draw_fig8_decision_vars():
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()
    var_names = ['P1: Wall U', 'P2: Roof U', 'P3: Reflectance', 'P4: Glazing U', 
                 'P5: COP', 'P6: Cooling SP', 'P7: LPD', 'P8: PV kWp', 'P9: BESS kWh']
    
    _, X, _ = generate_mock_pareto_data(200)
    
    for i in range(9):
        sns.histplot(X[:, i], kde=True, ax=axes[i], color=PALETTE[i%len(PALETTE)], bins=15)
        axes[i].set_title(var_names[i])
        axes[i].set_ylabel('Frequency')
        
    plt.suptitle('Fig. 8: Distribution of Decision Variables in Pareto Set', y=1.02)
    plt.tight_layout()
    plt.savefig('Fig8_DecisionVars.png')
    plt.close()

def draw_fig9_topsis():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    F, _, topsis = generate_mock_pareto_data(200)
    
    labels = ['EUI (kWh/m²/yr)', 'LCC (Million $)', 'LCA (Million kgCO2eq)']
    pairs = [(0, 1), (0, 2), (1, 2)]
    
    for i, ax in enumerate(axes):
        idx_x, idx_y = pairs[i]
        x_data = F[:, idx_x] if idx_x == 0 else F[:, idx_x] / 1e6
        y_data = F[:, idx_y] if idx_y == 0 else F[:, idx_y] / 1e6
        
        sc = ax.scatter(x_data, y_data, c=topsis, cmap=CMAP, s=25, alpha=0.8)
        ax.set_xlabel(labels[idx_x])
        ax.set_ylabel(labels[idx_y])
        ax.grid(True, linestyle=':', alpha=0.6)
        
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('TOPSIS Closeness Coefficient')
    
    plt.suptitle('Fig. 9: Pareto Front Colored by TOPSIS Score', y=1.05)
    plt.savefig('Fig9_TOPSIS.png')
    plt.close()

def draw_fig10_optimal_lce():
    fig, ax = plt.subplots(figsize=(8, 5))
    modules = ['A1-A3', 'A4-A5', 'B2-B3', 'B4', 'B6', 'C1-C4']
    values = [300, 30, 45, 100, 550, 40] # Mock data for optimal solution
    
    ax.bar(modules, values, color=PALETTE[1], edgecolor='black', alpha=0.8)
    ax.set_ylabel('Life Cycle Emissions (tons CO2eq)')
    ax.set_title('Fig. 10: LCE Distribution (Optimal Compromise Solution)')
    
    for i, v in enumerate(values):
        ax.text(i, v + 5, str(v), ha='center')
        
    plt.tight_layout()
    plt.savefig('Fig10_OptimalLCE.png')
    plt.close()

def draw_fig11_heatmap():
    _, X, topsis = generate_mock_pareto_data(30)
    # Sort by TOPSIS score
    sorted_indices = np.argsort(topsis)[::-1]
    X_sorted = X[sorted_indices]
    
    var_names = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']
    df = pd.DataFrame(X_sorted, columns=var_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.T, cmap=CMAP, cbar_kws={'label': 'Normalized Level (0: Baseline, 1: Max Retrofit)'})
    plt.title('Fig. 11: Renovation Levels for Top Pareto Solutions')
    plt.xlabel('Pareto Solutions (Ranked by TOPSIS)')
    plt.ylabel('Decision Variables')
    
    plt.tight_layout()
    plt.savefig('Fig11_Heatmap.png')
    plt.close()

if __name__ == "__main__":
    import os
    print(f"Working directory: {os.getcwd()}")
    print("Generating Figure 3...")
    draw_fig3_lifespan()
    print("Generating Figure 4...")
    draw_fig4_lce_distribution()
    print("Generating Figure 5...")
    draw_fig5_prediction_perf()
    print("Generating Figure 6...")
    draw_fig6_nsga3_evolution()
    print("Generating Figure 7...")
    draw_fig7_pairwise_pareto()
    print("Generating Figure 8...")
    draw_fig8_decision_vars()
    print("Generating Figure 9...")
    draw_fig9_topsis()
    print("Generating Figure 10...")
    draw_fig10_optimal_lce()
    print("Generating Figure 11...")
    draw_fig11_heatmap()
    print("All figures successfully generated!")
