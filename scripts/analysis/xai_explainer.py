"""
XAI Explainer for PI-HGAT
Implements GNNExplainer for feature and edge importance, and calculates Graph Centrality metrics.
"""
import torch
import networkx as nx
from torch_geometric.explain import Explainer, GNNExplainer
from pi_hgat.models import PI_HGAT
from pi_hgat.graph_builder import GraphBuilder
from pi_hgat.config import NEO4J_JSON_PATH, GNN_PARAMS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_graph_centrality(hetero_data):
    print("--- Graph Centrality Analysis ---")
    # Convert HeteroData to a homogeneous NetworkX graph for topology analysis
    G = nx.Graph()
    
    node_offset = 0
    type_offsets = {}
    
    # Add nodes
    for nt in hetero_data.node_types:
        num_nodes = hetero_data[nt].x.size(0)
        type_offsets[nt] = node_offset
        for i in range(num_nodes):
            G.add_node(node_offset + i, type=nt)
        node_offset += num_nodes
        
    # Add edges
    for et in hetero_data.edge_types:
        src_type, rel_type, dst_type = et
        edge_index = hetero_data[et].edge_index
        for i in range(edge_index.size(1)):
            u = type_offsets[src_type] + edge_index[0, i].item()
            v = type_offsets[dst_type] + edge_index[1, i].item()
            G.add_edge(u, v)
            
    # Calculate Centrality
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    
    # Aggregate by type to find thermal hubs
    avg_degree = {nt: [] for nt in hetero_data.node_types}
    avg_betw = {nt: [] for nt in hetero_data.node_types}
    
    for n, attr in G.nodes(data=True):
        nt = attr['type']
        avg_degree[nt].append(degree_cent[n])
        avg_betw[nt].append(betweenness_cent[n])
        
    for nt in hetero_data.node_types:
        d_val = sum(avg_degree[nt])/len(avg_degree[nt]) if avg_degree[nt] else 0
        b_val = sum(avg_betw[nt])/len(avg_betw[nt]) if avg_betw[nt] else 0
        print(f"Node Type: {nt:10} | Avg Degree Centrality: {d_val:.4f} | Avg Betweenness Centrality: {b_val:.4f}")
        
def run_gnn_explainer():
    print("\n--- GNNExplainer Analysis ---")
    builder = GraphBuilder(NEO4J_JSON_PATH)
    
    # Create a dummy sample to explain
    params = dict(P1_Wall_U=1.07, P2_Roof_U=0.45, P3_Roof_Reflectance=0.30,
                  P4_Win_U=2.87, P4_Win_SHGC=0.22, P5_COP=2.96,
                  P6_Cool_SP=24., P7_LPD=6.66, P8_PV_kW=0., Climate_DeltaT=0.)
                  
    data = builder.create_sample_graph(params)
    # Add batch to data
    batch_dict = {nt: torch.zeros(data[nt].x.size(0), dtype=torch.long) for nt in data.node_types}
    global_params = torch.tensor([[1.07, 0.45, 0.30, 2.87, 0.22, 2.96, 24., 6.66, 0., 0.]])
    
    # Move to device
    data = data.to(device)
    batch_dict = {k: v.to(device) for k,v in batch_dict.items()}
    global_params = global_params.to(device)
    
    # Load Model
    model = PI_HGAT(
        metadata=data.metadata(),
        hidden_channels=GNN_PARAMS['hidden_channels'],
        out_channels=1,
        num_layers=GNN_PARAMS['num_layers'],
        heads=GNN_PARAMS['heads'],
        dropout=GNN_PARAMS['dropout'],
        global_dim=10
    ).to(device)
    
    # Initialize lazy modules
    model(data.x_dict, data.edge_index_dict, batch_dict, global_params)
    
    try:
        model.load_state_dict(torch.load('best_hgat_v2.pt', map_location=device, weights_only=True))
        print("Loaded pre-trained model.")
    except Exception as e:
        print(f"Warning: Could not load 'best_hgat_v2.pt' ({e}), using uninitialized weights for explainer test.")
    
    model.eval()
    
    # Setup Explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )
    
    # Generate Explanation
    kwargs = dict(batch_dict=batch_dict, global_params=global_params)
    
    # Target value (dummy or actual prediction)
    with torch.no_grad():
        target = model(data.x_dict, data.edge_index_dict, **kwargs)
        
    explanation = explainer(data.x_dict, data.edge_index_dict, target=target, **kwargs)
    
    print("\nFeature Importance (Top 3 per Node Type):")
    for nt in data.node_types:
        if nt in explanation.node_mask_dict:
            mask = explanation.node_mask_dict[nt].mean(dim=0)
            topk = torch.topk(mask, k=min(3, mask.size(0)))
            print(f"  {nt}: Indices {topk.indices.cpu().numpy()} with scores {topk.values.cpu().detach().numpy()}")
            if nt == 'Envelope':
                print(f"    (Note: Index 0=Area, 7=U-value, 10=ShapeIndex)")

if __name__ == "__main__":
    builder = GraphBuilder(NEO4J_JSON_PATH)
    dummy_data = builder.create_sample_graph(dict(P1_Wall_U=1.07, P2_Roof_U=0.45, P3_Roof_Reflectance=0.30,
                  P4_Win_U=2.87, P4_Win_SHGC=0.22, P5_COP=2.96,
                  P6_Cool_SP=24., P7_LPD=6.66, P8_PV_kW=0., Climate_DeltaT=0.))
    analyze_graph_centrality(dummy_data)
    run_gnn_explainer()
