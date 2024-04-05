from torch_geometric.nn import GraphConv
import torch

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
    def __init__(self, in_dim, hidden_channels, out_dim):
        super().__init__()
        self.conv1 = GraphConv(1,1)
        self.linear_layer = MLP(1, 200, 1, num_layers=8)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.linear_layer(x).sigmoid()
        return x
