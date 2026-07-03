import json

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# --- CELL 11: Setup NSGA-III ---
cell_11_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Section 11: Define NSGA-III Optimization Problem"]
}

cell_11_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from pymoo.core.problem import ElementwiseProblem\n",
        "from pi_hgat.objectives import ObjectiveCalculator\n",
        "import torch\n",
        "\n",
        "# Initialize objectives calculator\n",
        "obj_calc = ObjectiveCalculator(builder)\n",
        "\n",
        "class NZEBRetrofitProblem(ElementwiseProblem):\n",
        "    def __init__(self, model_surrogate, builder):\n",
        "        # 9 variables (P1..P8 + Climate). Climate is fixed to HCMC mean for MOO.\n",
        "        super().__init__(n_var=9, n_obj=3, n_ieq_constr=0,\n",
        "                         xl=np.array([0.29, 0.18, 0.30, 1.00, 0.15, 2.96, 24.0, 2.50, 0.0]),\n",
        "                         xu=np.array([1.07, 0.45, 0.85, 2.87, 0.22, 5.00, 27.0, 6.66, 50.0]))\n",
        "        self.model = model_surrogate\n",
        "        self.model.eval()\n",
        "        self.builder = builder\n",
        "\n",
        "    def _evaluate(self, x, out, *args, **kwargs):\n",
        "        # 1. Map decision variables\n",
        "        params = {\n",
        "            'P1_Wall_U': x[0], 'P2_Roof_U': x[1], 'P3_Roof_Reflectance': x[2],\n",
        "            'P4_Win_U': x[3], 'P4_Win_SHGC': x[4], 'P5_COP': x[5],\n",
        "            'P6_Cool_SP': x[6], 'P7_LPD': x[7], 'P8_PV_kW': x[8],\n",
        "            'Climate_DeltaT': 0.0  # Assume mean climate for design optimization\n",
        "        }\n",
        "        \n",
        "        # 2. Predict EUI using PI-HGAT\n",
        "        data = self.builder.create_sample_graph(params)\n",
        "        \n",
        "        x_flat = [params['P1_Wall_U'], params['P2_Roof_U'], params['P3_Roof_Reflectance'],\n",
        "                  params['P4_Win_U'], params['P4_Win_SHGC'], params['P5_COP'],\n",
        "                  params['P6_Cool_SP'], params['P7_LPD'], params['P8_PV_kW'], params['Climate_DeltaT']]\n",
        "        data.global_params = torch.tensor([x_flat], dtype=torch.float)\n",
        "        \n",
        "        with torch.no_grad():\n",
        "            # Move to device and add batch dimension using PyG DataLoader logic manually or just view\n",
        "            # For simplicity, we can pass single graph directly if model handles it, or use PyGDL.\n",
        "            # Since our model uses batch_dict, we must construct it:\n",
        "            batch_dict = {nt: torch.zeros(data[nt].x.size(0), dtype=torch.long, device=device) \n",
        "                          for nt in data.node_types}\n",
        "            out_eui = self.model(\n",
        "                {nt: data[nt].x.to(device) for nt in data.node_types},\n",
        "                {et: data[et].edge_index.to(device) for et in data.edge_types},\n",
        "                batch_dict,\n",
        "                data.global_params.to(device)\n",
        "            )\n",
        "            eui = out_eui.item()\n",
        "            \n",
        "        # 3. Calculate LCC and LCA\n",
        "        lcc = obj_calc.calculate_lcc(params, eui)\n",
        "        lca = obj_calc.calculate_lca(params, eui)\n",
        "        \n",
        "        # 4. Assign objectives (Minimize all)\n",
        "        out[\"F\"] = [eui, lcc, lca]\n",
        "\n",
        "problem = NZEBRetrofitProblem(model, builder)\n",
        "print(\"MOO Problem Defined: 9 Variables, 3 Objectives (EUI, LCC, LCA)\")"
    ]
}

# --- CELL 12: Run NSGA-III ---
cell_12_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Section 12: Run NSGA-III"]
}

cell_12_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from pymoo.algorithms.moo.nsga3 import NSGA3\n",
        "from pymoo.optimize import minimize\n",
        "from pymoo.util.ref_dirs import get_reference_directions\n",
        "\n",
        "# Generate reference directions for NSGA-III (3 objectives)\n",
        "ref_dirs = get_reference_directions(\"das-dennis\", 3, n_partitions=12)\n",
        "\n",
        "algorithm = NSGA3(pop_size=91, ref_dirs=ref_dirs)\n",
        "\n",
        "print(\"Running NSGA-III optimization... (This may take a minute)\")\n",
        "t0 = time.time()\n",
        "res = minimize(problem,\n",
        "               algorithm,\n",
        "               seed=42,\n",
        "               termination=('n_gen', 50),\n",
        "               verbose=True)\n",
        "\n",
        "print(f\"Optimization finished in {time.time()-t0:.1f}s\")\n",
        "print(f\"Found {len(res.F)} Pareto optimal solutions.\")"
    ]
}

# --- CELL 13: Entropy-TOPSIS ---
cell_13_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Section 13: Entropy-TOPSIS & Pareto Visualization"]
}

cell_13_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from scipy.stats import entropy\n",
        "\n",
        "F = res.F\n",
        "\n",
        "# --- Entropy-TOPSIS ---\n",
        "# 1. Normalize Decision Matrix\n",
        "norm_F = F / np.sqrt((F**2).sum(axis=0))\n",
        "\n",
        "# 2. Calculate Entropy Weights\n",
        "P = norm_F / norm_F.sum(axis=0)\n",
        "E = -np.nansum(P * np.log(P), axis=0) / np.log(len(F))\n",
        "W = (1 - E) / (1 - E).sum()\n",
        "print(f\"Objective Weights (Entropy): EUI={W[0]:.3f}, LCC={W[1]:.3f}, LCA={W[2]:.3f}\")\n",
        "\n",
        "# 3. Weighted Normalized Matrix\n",
        "V = norm_F * W\n",
        "\n",
        "# 4. Ideal and Anti-Ideal Solutions (Cost criteria: min is ideal)\n",
        "ideal = V.min(axis=0)\n",
        "anti_ideal = V.max(axis=0)\n",
        "\n",
        "# 5. Distances & Closeness\n",
        "d_ideal = np.sqrt(((V - ideal)**2).sum(axis=1))\n",
        "d_anti = np.sqrt(((V - anti_ideal)**2).sum(axis=1))\n",
        "closeness = d_anti / (d_ideal + d_anti)\n",
        "\n",
        "best_idx = np.argmax(closeness)\n",
        "best_solution = res.X[best_idx]\n",
        "best_obj = res.F[best_idx]\n",
        "\n",
        "print(\"\\n===== OPTIMAL COMPROMISE SOLUTION (TOPSIS) =====\")\n",
        "print(f\"EUI: {best_obj[0]:.2f} kWh/m2/yr\")\n",
        "print(f\"LCC: ${best_obj[1]:,.2f}\")\n",
        "print(f\"LCA: {best_obj[2]:,.2f} kgCO2eq\")\n",
        "print(f\"Parameters: P1={best_solution[0]:.2f}, P2={best_solution[1]:.2f}, P3={best_solution[2]:.2f}, \"\n",
        "      f\"P4_U={best_solution[3]:.2f}, P4_SHGC={best_solution[4]:.2f}, P5={best_solution[5]:.2f}, \"\n",
        "      f\"P6={best_solution[6]:.1f}, P7={best_solution[7]:.2f}, P8={best_solution[8]:.1f}kW\")\n",
        "\n",
        "# --- Visualization ---\n",
        "import matplotlib.pyplot as plt\n",
        "from mpl_toolkits.mplot3d import Axes3D\n",
        "\n",
        "fig = plt.figure(figsize=(10, 8))\n",
        "ax = fig.add_subplot(111, projection='3d')\n",
        "sc = ax.scatter(F[:, 0], F[:, 1]/1e6, F[:, 2]/1e6, c=closeness, cmap='viridis', s=40, alpha=0.8)\n",
        "ax.scatter(best_obj[0], best_obj[1]/1e6, best_obj[2]/1e6, color='red', s=150, marker='*', label='TOPSIS Best')\n",
        "\n",
        "ax.set_xlabel('EUI (kWh/m2/yr)')\n",
        "ax.set_ylabel('LCC (Million $)')\n",
        "ax.set_zlabel('LCA (Million kgCO2eq)')\n",
        "ax.set_title('NSGA-III Pareto Front: EUI vs LCC vs LCA', fontsize=14)\n",
        "plt.colorbar(sc, label='TOPSIS Closeness')\n",
        "plt.legend()\n",
        "plt.savefig('pareto_front_3d.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print(\"Saved: pareto_front_3d.png\")"
    ]
}

nb['cells'].extend([
    cell_11_md, cell_11_code,
    cell_12_md, cell_12_code,
    cell_13_md, cell_13_code
])

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
