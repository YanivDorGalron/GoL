import torch.nn as nn

class MLP(nn.Module):
    """A simple feed-forward neural network"""

    def __init__(self, in_dim, emb_dim, out_dim, num_layers=2):
        super(MLP, self).__init__()
        self.num_layers = num_layers

        layers = []
        layers.append(nn.Linear(in_dim, emb_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(emb_dim, emb_dim))
            layers.append(nn.BatchNorm1d(emb_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(emb_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)