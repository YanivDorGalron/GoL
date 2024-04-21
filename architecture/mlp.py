import torch.nn as nn


class MLP(nn.Module):
    """A simple feed-forward neural network"""

    def __init__(self, in_dim, emb_dim, out_dim, use_dropout=False, num_layers=2, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        layers = [nn.Linear(in_dim, emb_dim), nn.ReLU(), self.dropout]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(emb_dim, emb_dim))
            layers.append(nn.BatchNorm1d(emb_dim))
            layers.append(nn.ReLU())
            layers.append(self.dropout)  # Add dropout after each hidden layer

        layers.append(nn.Linear(emb_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
