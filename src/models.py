"""GraphSAGE and GAT node classifiers, plus an MLP control.

The MLP matters: it consumes the same node features with the graph removed.
Any gap between MLP and GNN is the measured value of topology. Without it,
a good GNN score tells you nothing about whether the graph helped.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.head = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.c1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.c2 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout)
        self.head = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.elu(self.c1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.c2(x, edge_index))
        return self.head(x)


class MLP(nn.Module):
    """Graph-free control. Same features, no message passing."""

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, edge_index=None):
        return self.net(x)


def build(name: str, in_dim: int) -> nn.Module:
    return {"sage": GraphSAGE, "gat": GAT, "mlp": MLP}[name](in_dim)
