import json

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# --- CELL 14: GNNExplainer ---
cell_14_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["### Section 14: Spatial Explainability (GNNExplainer)\n",
               "Extract attention/mask weights to explain which components contributed most to the EUI of the Optimal Compromise Solution."]
}

cell_14_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from torch_geometric.explain import Explainer, GNNExplainer\n",
        "import networkx as nx\n",
        "\n",
        "# Prepare data using Best Compromise Solution from TOPSIS\n",
        "best_params = {\n",
        "    'P1_Wall_U': best_solution[0], 'P2_Roof_U': best_solution[1], 'P3_Roof_Reflectance': best_solution[2],\n",
        "    'P4_Win_U': best_solution[3], 'P4_Win_SHGC': best_solution[4], 'P5_COP': best_solution[5],\n",
        "    'P6_Cool_SP': best_solution[6], 'P7_LPD': best_solution[7], 'P8_PV_kW': best_solution[8],\n",
        "    'Climate_DeltaT': 0.0\n",
        "}\n",
        "\n",
        "data = builder.create_sample_graph(best_params)\n",
        "data.global_params = torch.tensor([[best_solution[0], best_solution[1], best_solution[2],\n",
        "                                    best_solution[3], best_solution[4], best_solution[5],\n",
        "                                    best_solution[6], best_solution[7], best_solution[8], 0.0]], dtype=torch.float)\n",
        "\n",
        "batch_dict = {nt: torch.zeros(data[nt].x.size(0), dtype=torch.long, device=device) for nt in data.node_types}\n",
        "data = data.to(device)\n",
        "data.global_params = data.global_params.to(device)\n",
        "\n",
        "class ModelWrapper(torch.nn.Module):\n",
        "    def __init__(self, model, batch_dict, global_params):\n",
        "        super().__init__()\n",
        "        self.model = model\n",
        "        self.batch_dict = batch_dict\n",
        "        self.global_params = global_params\n",
        "    def forward(self, x_dict, edge_index_dict):\n",
        "        return self.model(x_dict, edge_index_dict, self.batch_dict, self.global_params)\n",
        "\n",
        "wrapped_model = ModelWrapper(model, batch_dict, data.global_params)\n",
        "\n",
        "explainer = Explainer(\n",
        "    model=wrapped_model,\n",
        "    algorithm=GNNExplainer(epochs=200),\n",
        "    explanation_type='model',\n",
        "    node_mask_type='attributes',\n",
        "    edge_mask_type='object',\n",
        "    model_config=dict(mode='regression', task_level='graph', return_type='raw')\n",
        ")\n",
        "\n",
        "print(\"Running GNNExplainer (learning masks for mutual information)...\")\n",
        "explanation = explainer(data.x_dict, data.edge_index_dict)\n",
        "print(\"Explanation completed.\")\n",
        "\n",
        "feat_names = {\n",
        "    'Zone': ['area', 'volume', 'height', 'LPD', 'PV_share'],\n",
        "    'Envelope': ['area', 'tilt', 'azimuth', 'is_wall', 'is_roof', 'is_floor', 'is_window', 'U-value', 'Reflectance', 'SHGC'],\n",
        "    'System': ['cooling_cap', 'heating_cap', 'COP', 'Cool_SP', 'Heat_SP']\n",
        "}\n",
        "\n",
        "print(\"\\n--- Top Node Features by Mask Score ---\")\n",
        "for nt in ['Zone', 'Envelope', 'System']:\n",
        "    mask = explanation.node_mask_dict[nt].cpu().numpy()\n",
        "    mean_mask = mask.mean(axis=0)\n",
        "    top_idx = mean_mask.argsort()[::-1][:3]\n",
        "    print(f\"\\n{nt} Features:\")\n",
        "    for i in top_idx:\n",
        "        print(f\"  - {feat_names[nt][i]}: {mean_mask[i]:.4f}\")"
    ]
}

nb['cells'].extend([cell_14_md, cell_14_code])

with open('NZEB_PIPELINE_ICAE2026.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
