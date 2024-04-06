import pdb

import torch
from torch_geometric.nn import GraphConv, GATConv, GINConv

from architecture.mlp import MLP
from architecture.modified_gcn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_channels, out_dim):
        super().__init__()
        self.conv1 = GCNConv()
        self.linear_layer = MLP(1, 200, 1, num_layers=8)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.linear_layer(x).sigmoid()
        return x


class GraphConvNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_channels, out_dim, num_layers):
        super().__init__()
        self.conv1 = GraphConv(in_dim, in_dim)
        self.linear_layer = MLP(in_dim, hidden_channels, out_dim, num_layers=num_layers)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.linear_layer(x).sigmoid()
        return x


class GATNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_channels, out_dim, num_layers):
        super().__init__()
        self.conv1 = GATConv(in_dim, in_dim)
        self.linear_layer = MLP(in_dim, hidden_channels, out_dim, num_layers=num_layers)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.linear_layer(x).sigmoid()
        return x


class GINNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_channels, out_dim, num_layers):
        super().__init__()
        self.linear_layer1 = MLP(in_dim, hidden_channels, in_dim, num_layers=2)
        self.conv1 = GINConv(self.linear_layer1, train_eps=True)
        self.linear_layer = MLP(in_dim, hidden_channels, out_dim, num_layers=num_layers)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.linear_layer(x).sigmoid()
        return x


class ModifiedGraphConvNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_channels, out_dim, num_layers):
        super().__init__()
        self.conv1 = GraphConv(1, 1)
        self.history_linear_layer = MLP(in_dim, hidden_channels, hidden_channels, num_layers=1)
        self.last_state_linear_layer = MLP(1, hidden_channels, hidden_channels, num_layers=1)
        self.last_linear_layer = MLP(hidden_channels, hidden_channels, out_dim, num_layers=num_layers)

    def forward(self, data_x, edge_index):
        last_known_states = data_x[:, 0:1]
        x1 = self.conv1(last_known_states, edge_index)
        x2 = self.last_state_linear_layer(x1)
        x3 = self.history_linear_layer(data_x)
        x = self.last_linear_layer(x2 + x3).sigmoid()
        return x
