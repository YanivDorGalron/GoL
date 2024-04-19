from typing import Callable, Union

import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn import GraphConv, GATConv, GINConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm

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


class DeepGraphConvNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_channels, conv_hidden_dim, out_dim, num_layers, num_conv_layers, use_activation):
        super().__init__()
        self.convs = nn.Sequential(
            GraphConv(in_dim, conv_hidden_dim),
            *[GraphConv(conv_hidden_dim, conv_hidden_dim) for _ in range(num_conv_layers - 1)]
        )
        for conv in self.convs:
            conv.reset_parameters()
        self.linear_layer = MLP(conv_hidden_dim, hidden_channels, out_dim, num_layers=num_layers)
        self.activation = nn.ReLU() if use_activation else nn.Identity()

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.linear_layer(x).sigmoid()
        return x


class DividedGINConv(MessagePassing):
    def __init__(self, nn1: Callable, nn2: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn1 = nn1
        self.nn2 = nn2
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn1)
        reset(self.nn2)
        self.eps.data.fill_(self.initial_eps)

    def forward(
            self,
            x: Union[Tensor, OptPairTensor],
            edge_index: Adj,
            size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out = self.nn1(out) + (1 + self.eps) * self.nn2(x_r)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class DeepDividedGINConvNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_channels, conv_hidden_dim, out_dim, num_layers, num_conv_layers, use_activation,
                 aggr='mean'):
        super().__init__()
        self.convs = nn.Sequential(
            DividedGINConv(MLP(in_dim, conv_hidden_dim, conv_hidden_dim, num_layers=2),
                           MLP(in_dim, conv_hidden_dim, conv_hidden_dim, num_layers=2), aggr=aggr),
            *[DividedGINConv(MLP(conv_hidden_dim, conv_hidden_dim, conv_hidden_dim, num_layers=2),
                             MLP(conv_hidden_dim, conv_hidden_dim, conv_hidden_dim, num_layers=2), aggr=aggr) for _ in
              range(num_conv_layers - 1)]
        )
        for conv in self.convs:
            conv.reset_parameters()
        self.linear_layer = MLP(conv_hidden_dim, hidden_channels, out_dim, num_layers=num_layers)
        self.activation = nn.ReLU() if use_activation else nn.Identity()

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.linear_layer(x).sigmoid()
        return x


class DeepGINConvNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_channels, conv_hidden_dim, out_dim, num_layers, num_conv_layers, use_activation,
                 aggr='mean'):
        super().__init__()
        self.convs = nn.Sequential(
            GINConv(MLP(in_dim, conv_hidden_dim, conv_hidden_dim, num_layers=2), aggr=aggr),
            *[GINConv(MLP(conv_hidden_dim, conv_hidden_dim, conv_hidden_dim, num_layers=2), aggr=aggr) for _ in
              range(num_conv_layers - 1)]
        )
        for conv in self.convs:
            conv.reset_parameters()
        self.linear_layer = MLP(conv_hidden_dim, hidden_channels, out_dim, num_layers=num_layers)
        self.activation = nn.ReLU() if use_activation else nn.Identity()

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
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
