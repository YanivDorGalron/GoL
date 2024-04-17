import torch
from torch import nn


class MLP(nn.Module):
    """A simple feed forward neural network"""

    def __init__(self, in_dim, emb_dim, out_dim, num_layers=2):
        super(MLP, self).__init__()
        layer_list = []
        layer_list.append(torch.nn.Linear(in_dim, emb_dim))
        for _ in range(num_layers - 1):
            layer_list.append(torch.nn.BatchNorm1d(emb_dim))
            layer_list.append(torch.nn.LeakyReLU())
            l = torch.nn.Linear(emb_dim, emb_dim)
            l.reset_parameters()
            layer_list.append(l)

        l = torch.nn.Linear(emb_dim, out_dim)
        l.reset_parameters()
        layer_list.append(l)
        self.layers = torch.nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)
