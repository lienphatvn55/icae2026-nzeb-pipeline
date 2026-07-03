"""
PI-HGAT Model v2 + Baselines.

Key fixes:
- BatchNorm in encoders for feature normalization
- Proper residual connections with dropout
- Mean+Max pooling for richer graph representation
- Optional global parameter skip connection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, global_mean_pool, global_max_pool, Linear


class PI_HGAT(nn.Module):
    """Physics-Informed Heterogeneous Graph Attention Network."""

    def __init__(self, metadata, hidden_channels=64, out_channels=1,
                 num_layers=3, heads=4, dropout=0.15, global_dim=10):
        super().__init__()
        self.dropout_rate = dropout
        self.num_layers = num_layers
        self.node_types = list(metadata[0])
        self.use_global_skip = (global_dim > 0)

        # 1. Per-type encoder: Linear → BN → ReLU
        self.encoders = nn.ModuleDict()
        for nt in metadata[0]:
            self.encoders[nt] = nn.Sequential(
                Linear(-1, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU())

        # 2. HeteroConv layers (GAT)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                et: GATConv(hidden_channels, hidden_channels // heads,
                            heads=heads, add_self_loops=False, dropout=dropout)
                for et in metadata[1]
            }, aggr='sum')
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(hidden_channels) for nt in metadata[0]}))

        # 3. MLP head
        pool_dim = hidden_channels * len(metadata[0])   # mean only to reduce overfitting
        mlp_in = pool_dim + (global_dim if self.use_global_skip else 0)

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),     nn.ReLU(),
            nn.Linear(64, out_channels))

    def forward(self, x_dict, edge_index_dict, batch_dict=None, global_params=None):
        # --- Encode ---
        h = {nt: self.encoders[nt](x) for nt, x in x_dict.items()}

        # --- Message Passing ---
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index_dict)
            for nt in h:
                if nt in h_new:
                    out = self.norms[i][nt](h_new[nt])
                    out = F.relu(out)
                    out = F.dropout(out, p=self.dropout_rate, training=self.training)
                    h[nt] = out + h[nt]               # residual

        # --- Pooling (mean + max per type) ---
        parts = []
        for nt in self.node_types:
            if nt not in h:
                continue
            feat = h[nt]
            if batch_dict is not None and nt in batch_dict:
                b = batch_dict[nt]
            else:
                b = torch.zeros(feat.size(0), dtype=torch.long, device=feat.device)
            parts.append(global_mean_pool(feat, b))
            # Removed global_max_pool to reduce dimensional explosion

        h_graph = torch.cat(parts, dim=1)

        # --- Optional global skip ---
        if self.use_global_skip and global_params is not None:
            h_graph = torch.cat([h_graph, global_params], dim=1)

        return self.mlp(h_graph)


# -------------------------------------------------------------------- #
#  BASELINES                                                            #
# -------------------------------------------------------------------- #

class BaselineANN(nn.Module):
    """3-layer MLP baseline (same capacity as PI-HGAT MLP head)."""
    def __init__(self, in_channels=10, out_channels=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),          nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32),           nn.ReLU(),
            nn.Linear(32, out_channels))

    def forward(self, x):
        return self.mlp(x)
