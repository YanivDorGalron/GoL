from torch_geometric.nn import MessagePassing


class GCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='sum')

    def forward(self, x, edge_index):
        norm = 1
        out = self.propagate(edge_index, x=x, norm=norm)
        out = out

        return out

    def message(self, x_j, norm):
        return x_j
