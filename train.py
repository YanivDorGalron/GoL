import numpy as np
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from torch_geometric.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch


class MLP(nn.Module):
    """A simple feed forward neural network"""

    def __init__(self, in_dim, emb_dim, out_dim, num_layers=2):
        super(MLP, self).__init__()
        layer_list = []
        layer_list.append(torch.nn.Linear(in_dim, emb_dim))
        for _ in range(num_layers - 1):
            layer_list.append(torch.nn.BatchNorm1d(emb_dim))
            layer_list.append(torch.nn.ReLU())
            l = torch.nn.Linear(emb_dim, emb_dim)
            l.reset_parameters()
            layer_list.append(l)

        l = torch.nn.Linear(emb_dim, out_dim)
        l.reset_parameters()
        layer_list.append(l)
        self.layers = torch.nn.Sequential(*layer_list)

    def forward(self, x):
        return self.layers(x)


import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class RealGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index,y):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class GCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='sum')

    def forward(self, x, edge_index, y):
        norm = 1
        out = self.propagate(edge_index, x=x, norm=norm)
        out = out

        return out

    def message(self, x_j, norm):
        return x_j


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_channels, out_dim):
        super().__init__()
        self.conv1 = GCNConv()
        # self.conv1 = RealGCNConv(1,1)
        self.linear_layer = MLP(1, 200, 1, num_layers=8)

    def forward(self, x, edge_index, y):
        x = self.conv1(x, edge_index, y)
        x = self.linear_layer(x).sigmoid()
        return x


def train(train_loader, model, optimiser, loss_fn, metric_fn):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    num_graphs = 0
    for data in train_loader:
        optimiser.zero_grad()
        data = data.to(DEVICE)
        y_hat = model(data.x, data.edge_index, data.y)[:, 0]
        loss = loss_fn(y_hat, data.y.to(torch.float32))
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(data.y)
        num_graphs += len(data.y)
    return total_loss / num_graphs


def evaluate(loader, model, metric_fn):
    """Evaluate model on dataset"""
    y_pred, y_true = [], []
    model.eval()
    for data in loader:
        data = data.to(DEVICE)
        y_hat = model(data.x, data.edge_index, data.y)

        y_pred.append(y_hat.detach().cpu())
        y_true.append(data.y.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)
    y_pred = y_pred[:, 0]
    y_pred = (y_pred > 0.5).long()
    y_true = torch.cat(y_true, dim=0)
    return [metric(y_true, y_pred) for metric in metric_fn]


def run(
        model,
        train_loader,
        loaders,
        loss_fn,
        metric_fn,
        use_scheduler=False,
        print_steps=True,
        n_runs=10,
):
    """Train the model for NUM_EPOCHS epochs and run n times"""
    # Instantiate optimiser and scheduler
    optimiser = optim.Adam(model.parameters(), lr=LR)
    # scheduler = (
    #     optim.lr_scheduler.StepLR(optimiser, step_size=DECAY_STEP, gamma=DECAY_RATE)
    #     if use_scheduler
    #     else None
    # )
    scheduler = None
    curves = {name: [] for name in loaders.keys()}
    # pbar = tqdm(range(NUM_EPOCHS))
    for epoch in range(NUM_EPOCHS):
        train_loss = train(
            train_loader, model, optimiser, loss_fn, metric_fn
        )
        if scheduler is not None:
            scheduler.step()
        for name, loader in loaders.items():
            curves[name].append(evaluate(loader, model, metric_fn))
        if print_steps:
            print_str = f"Epoch {epoch}, train loss: {train_loss:.6f}"
            for name, metric in curves.items():
                print_str += f", {name} metric: {metric[-1]:.3f}"
            # pbar.set_postfix(print_str)

    return curves['train'][-1], curves['test'][-1]


def count_neighbors_with_state_1(node, df):
    neighbors_with_state_1 = sum(df[(df['a'] == node)].state_b == 1) + sum(df[(df['b'] == node)].state_a == 1)
    return neighbors_with_state_1


BATCH_SIZE = 32
NUM_EPOCHS = 20
HIDDEN_DIM = 1
NUM_LAYERS = 1
IN_DIM = 1
LR = 1e-3
SEED = 32
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

TRAIN_PORTION = 0.8


def calc_ds(df):
    ds = []
    for i in tqdm(range(df.i.max() - 1)):
        current_df = df[df.i == i]
        next_df = df[df.i == i + 1]

        states = np.concatenate([current_df['state_a'].values, current_df['state_b'].values])
        nodes = np.concatenate([current_df['a'].values, current_df['b'].values])
        concatenated_df = pd.DataFrame({'nodes': nodes, 'states': states}).drop_duplicates().sort_values(by='nodes')
        x = concatenated_df.states.values[:, None]
        x = torch.tensor(x, dtype=torch.float)

        edges = np.stack([current_df['a'].values, current_df['b'].values], axis=1)
        edges = np.concatenate([edges, edges[:, ::-1]])
        edge_index = torch.tensor(edges.transpose(), dtype=torch.long)

        states = np.concatenate([next_df['state_a'].values, next_df['state_b'].values])
        nodes = np.concatenate([next_df['a'].values, next_df['b'].values])
        concatenated_next_df = pd.DataFrame({'nodes': nodes, 'states': states}).drop_duplicates().sort_values(
            by='nodes')

        y = torch.tensor(concatenated_next_df.states.values, dtype=torch.long)
        # y1 = neighbors

        data = Data(x=x, edge_index=edge_index, y=y)  # ,y1=y1)
        data.validate(raise_on_error=True)
        ds.append(data)
    return ds


def diversity(y_true, y_pred):
    return y_pred.float().std(), y_true.float().std()


regulardf = pd.read_csv('../../notebooks/saved/data/RegularGoL.csv')
temporaldf = pd.read_csv('../../notebooks/saved/data/TemporalGoL.csv')
oscilationdf = pd.read_csv('../../notebooks/saved/data/OscilationsGoL.csv')
PD_df = pd.read_csv('../../notebooks/saved/data/PastDependentGoL.csv')
df_list = [regulardf, temporaldf, oscilationdf, PD_df]
name = ['regulardf ', 'temporaldf', 'oscilationdf', 'PD_df']
num_runs = 1


for n, df in zip(name, df_list):
    ds = calc_ds(df)
    train_size = int(len(ds) * TRAIN_PORTION)
    train_dataset = ds[:train_size]
    test_dataset = ds[train_size:]

    train_loader = DataLoader(train_dataset, 5 * BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, 5 * BATCH_SIZE, shuffle=False)
    train_acc_list = []
    test_acc_list = []
    for i in range(num_runs):
        gcn_model = GCN(
            in_dim=IN_DIM,
            hidden_channels=1,
            out_dim=1
        ).to(DEVICE)

        train_acc, test_acc = run(
            gcn_model,
            train_loader,
            {"train": train_loader, "test": test_loader},
            loss_fn=F.binary_cross_entropy,
            metric_fn=[ recall_score, accuracy_score, f1_score, diversity], #precision_score,
            print_steps=False
        )

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    print(n)
    print('[recall,accuracy,f1,diversity_pred,diversity_true]') #precision,
    print('train:', train_acc_list)
    print('test:', test_acc_list)
    print('\n')
