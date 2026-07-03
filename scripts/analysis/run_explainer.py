import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.explain import Explainer, GNNExplainer
from pi_hgat.models import PI_HGAT
from pi_hgat.graph_builder import GraphBuilder
from pi_hgat.config import NEO4J_JSON_PATH, GNN_PARAMS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_data():
    builder = GraphBuilder(NEO4J_JSON_PATH)
    baseline_data = builder.create_heterodata()
    metadata = baseline_data.metadata()
    
    model = PI_HGAT(
        metadata=metadata,
        hidden_channels=GNN_PARAMS['hidden_channels'],
        out_channels=1,
        num_layers=GNN_PARAMS['num_layers'],
        heads=GNN_PARAMS['heads'],
        dropout=GNN_PARAMS['dropout'],
        global_dim=10,
    ).to(device)
    
    # Initialize lazy layers
    dummy = baseline_data.clone().to(device)
    batch_dict = {nt: torch.zeros(dummy[nt].x.size(0), dtype=torch.long, device=device) 
                  for nt in dummy.node_types}
    dummy_global = torch.zeros((1, 10), dtype=torch.float, device=device)
    model(dummy.x_dict, dummy.edge_index_dict, batch_dict, dummy_global)
    
    model.load_state_dict(torch.load('best_hgat_v2.pt', weights_only=True))
    model.eval()
    
    return model, builder

def run_explanation():
    model, builder = load_model_and_data()
    
    # Use the best compromise solution from Phase 2
    best_solution = [0.43, 0.40, 0.59, 2.55, 0.19, 5.00, 27.0, 3.16, 50.0]
    
    params = {
        'P1_Wall_U': best_solution[0], 'P2_Roof_U': best_solution[1], 'P3_Roof_Reflectance': best_solution[2],
        'P4_Win_U': best_solution[3], 'P4_Win_SHGC': best_solution[4], 'P5_COP': best_solution[5],
        'P6_Cool_SP': best_solution[6], 'P7_LPD': best_solution[7], 'P8_PV_kW': best_solution[8],
        'Climate_DeltaT': 0.0
    }
    
    data = builder.create_sample_graph(params)
    data.global_params = torch.tensor([[best_solution[0], best_solution[1], best_solution[2],
                                        best_solution[3], best_solution[4], best_solution[5],
                                        best_solution[6], best_solution[7], best_solution[8], 0.0]], dtype=torch.float)
    
    batch_dict = {nt: torch.zeros(data[nt].x.size(0), dtype=torch.long, device=device) 
                  for nt in data.node_types}
    
    data = data.to(device)
    data.global_params = data.global_params.to(device)
    
    print("Initializing GNNExplainer...")
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )
    
    print("Running explanation...")
    # Wrap model to only accept standard args because PyG Explainer can be picky
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, batch_dict, global_params):
            super().__init__()
            self.model = model
            self.batch_dict = batch_dict
            self.global_params = global_params
            
        def forward(self, x_dict, edge_index_dict):
            return self.model(x_dict, edge_index_dict, self.batch_dict, self.global_params)
            
    wrapped_model = ModelWrapper(model, batch_dict, data.global_params)
    explainer.model = wrapped_model
    
    explanation = explainer(data.x_dict, data.edge_index_dict)
    
    print("Explanation completed.")
    return explanation, data, builder

if __name__ == "__main__":
    explanation, data, builder = run_explanation()
    
    print("\n--- Top Node Features ---")
    feat_names = {
        'Zone': ['area', 'volume', 'height', 'LPD', 'PV_share'],
        'Envelope': ['area', 'tilt', 'azimuth', 'is_wall', 'is_roof', 'is_floor', 'is_window', 'U-value', 'Reflectance', 'SHGC'],
        'Material': ['conductance', 'u_mod', 'shgc_mod'],
        'System': ['cooling_cap', 'heating_cap', 'COP', 'Cool_SP', 'Heat_SP'],
        'Climate': ['dbt_mean', 'dbt_max', 'dbt_min', 'rh', 'ghi', 'delta_t']
    }
    
    for nt in explanation.node_mask_dict:
        mask = explanation.node_mask_dict[nt].cpu().numpy()
        mean_mask = mask.mean(axis=0)  # average importance across all nodes of this type
        if len(mean_mask) > 0 and mean_mask.sum() > 0:
            top_idx = mean_mask.argsort()[::-1][:3]
            print(f"\n{nt} Features:")
            for i in top_idx:
                fname = feat_names[nt][i] if i < len(feat_names[nt]) else f"Feat_{i}"
                print(f"  - {fname}: {mean_mask[i]:.4f}")
                
    print("\n--- Generating Explanation Subgraph Visualization ---")
    G = nx.DiGraph()
    node_colors = []
    
    # Node mapping
    global_id = 0
    type_offset = {}
    for nt in data.node_types:
        type_offset[nt] = global_id
        for i in range(data[nt].x.size(0)):
            G.add_node(global_id, type=nt, local_id=i)
            # calculate node score as mean of its feature mask
            score = explanation.node_mask_dict[nt][i].mean().item()
            # Normalize to 0-1 for color mapping (simplistic)
            G.nodes[global_id]['score'] = score
            global_id += 1
            
    # Add edges with scores
    edge_scores_dict = explanation.edge_mask_dict
    for et in data.edge_types:
        if et in edge_scores_dict:
            scores = edge_scores_dict[et].cpu().numpy()
            edges = data.edge_index_dict[et].cpu().numpy()
            src_type, rel, tgt_type = et
            for i in range(edges.shape[1]):
                u = type_offset[src_type] + edges[0, i]
                v = type_offset[tgt_type] + edges[1, i]
                G.add_edge(u, v, weight=scores[i], type=rel)
                
    # Filter graph to show only important parts (Edge weight > 0.5 or Node score > 0.5)
    # Actually, GNNExplainer mask scores can be very small depending on the normalization.
    # Let's take the top 30 edges.
    all_edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
    all_edges.sort(key=lambda x: x[2], reverse=True)
    top_edges = all_edges[:30]
    
    sub_nodes = set()
    for u, v, w in top_edges:
        sub_nodes.add(u)
        sub_nodes.add(v)
        
    H = G.subgraph(list(sub_nodes))
    
    pos = nx.spring_layout(H, seed=42)
    
    plt.figure(figsize=(12, 10))
    # Draw edges
    edge_weights = [H[u][v]['weight'] * 5 for u, v in H.edges()]
    nx.draw_networkx_edges(H, pos, width=edge_weights, alpha=0.7, edge_color='red')
    
    # Draw nodes
    colors = {'Zone': 'lightblue', 'Envelope': 'lightgreen', 'Material': 'orange', 'System': 'lightpink', 'Climate': 'gold'}
    node_c = [colors[H.nodes[n]['type']] for n in H.nodes()]
    node_sizes = [300 + H.nodes[n].get('score', 0) * 5000 for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_color=node_c, node_size=node_sizes, alpha=0.9, edgecolors='black')
    
    # Labels
    labels = {n: f"{H.nodes[n]['type']}_{H.nodes[n]['local_id']}" for n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)
    
    plt.title("GNNExplainer: Most Important Subgraph (Top 30 Edges)", fontsize=16)
    
    # Custom legend
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], color=c, marker='o', linestyle='None', markersize=10, label=t) 
               for t, c in colors.items()]
    plt.legend(handles=handles, loc='upper left')
    
    plt.savefig('gnn_explanation_subgraph.png', dpi=150, bbox_inches='tight')
    print("Saved: gnn_explanation_subgraph.png")
