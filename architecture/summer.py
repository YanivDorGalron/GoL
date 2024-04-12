from torch_geometric.nn import MessagePassing


class SumNeighborsFeatures(MessagePassing):
    def __init__(self):
        super().__init__(aggr='sum')

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out
