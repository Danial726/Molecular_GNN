import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    AttentiveFP,
    GATConv,
    GCNConv,
    GINEConv,
    NNConv,
)
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn import global_mean_pool

from .featurize import ATOM_DIM, BOND_DIM

HIDDEN = 128
LAYERS = 3
DROPOUT = 0.1


class GCN(nn.Module):
    def __init__(self, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.lin0 = nn.Linear(ATOM_DIM, hidden)
        self.convs = nn.ModuleList(GCNConv(hidden, hidden) for _ in range(layers))
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.head = _head(hidden)

    def forward(self, data):
        x = F.relu(self.lin0(data.x))
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index)))
            x = F.dropout(x, DROPOUT, self.training)
        return self.head(global_mean_pool(x, data.batch)).squeeze(-1)


class GIN(nn.Module):
    def __init__(self, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.lin0 = nn.Linear(ATOM_DIM, hidden)
        self.convs = nn.ModuleList(
            GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                     edge_dim=BOND_DIM)
            for _ in range(layers)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.head = _head(hidden)

    def forward(self, data):
        x = F.relu(self.lin0(data.x))
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index, data.edge_attr)))
            x = F.dropout(x, DROPOUT, self.training)
        return self.head(global_mean_pool(x, data.batch)).squeeze(-1)


class GAT(nn.Module):
    def __init__(self, hidden=HIDDEN, layers=LAYERS, heads=4):
        super().__init__()
        self.lin0 = nn.Linear(ATOM_DIM, hidden)
        self.convs = nn.ModuleList(
            GATConv(hidden, hidden // heads, heads=heads, edge_dim=BOND_DIM)
            for _ in range(layers)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.head = _head(hidden)

    def forward(self, data):
        x = F.relu(self.lin0(data.x))
        for conv, bn in zip(self.convs, self.bns):
            x = F.elu(bn(conv(x, data.edge_index, data.edge_attr)))
            x = F.dropout(x, DROPOUT, self.training)
        return self.head(global_mean_pool(x, data.batch)).squeeze(-1)


class MPNN(nn.Module):
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        self.lin0 = nn.Linear(ATOM_DIM, hidden)
        edge_net = nn.Sequential(nn.Linear(BOND_DIM, 32), nn.ReLU(), nn.Linear(32, hidden * hidden))
        self.conv = NNConv(hidden, hidden, edge_net, aggr="mean")
        self.gru = nn.GRU(hidden, hidden)
        self.layers = layers
        self.head = _head(hidden)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        for _ in range(self.layers):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        return self.head(global_mean_pool(out, data.batch)).squeeze(-1)


class AFP(nn.Module):
    def __init__(self, hidden=HIDDEN, layers=2, timesteps=2):
        super().__init__()
        self.net = AttentiveFP(in_channels=ATOM_DIM, hidden_channels=hidden,
                               out_channels=1, edge_dim=BOND_DIM,
                               num_layers=layers, num_timesteps=timesteps, dropout=DROPOUT)

    def forward(self, data):
        return self.net(data.x, data.edge_index, data.edge_attr, data.batch).squeeze(-1)


class GatedGNN(nn.Module):
    """Self-explaining baseline: a learned node gate produces intrinsic importances."""

    def __init__(self, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.lin0 = nn.Linear(ATOM_DIM, hidden)
        self.convs = nn.ModuleList(
            GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                     edge_dim=BOND_DIM)
            for _ in range(layers)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.gate_nn = nn.Linear(hidden, 1)
        self.pool = AttentionalAggregation(self.gate_nn)
        self.head = _head(hidden)

    def node_embeddings(self, data):
        x = F.relu(self.lin0(data.x))
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, data.edge_index, data.edge_attr)))
        return x

    def forward(self, data):
        x = self.node_embeddings(data)
        pooled = self.pool(x, data.batch)
        return self.head(pooled).squeeze(-1)

    def node_importance(self, data):
        x = self.node_embeddings(data)
        return self.gate_nn(x).squeeze(-1)          # raw gate logits per node


def _head(hidden):
    return nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Dropout(DROPOUT), nn.Linear(hidden, 1))


REGISTRY = {
    "gcn": GCN,
    "gin": GIN,
    "gat": GAT,
    "mpnn": MPNN,
    "attentivefp": AFP,
    "gated": GatedGNN,
}


def build(name):
    return REGISTRY[name]()
