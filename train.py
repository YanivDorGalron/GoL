import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from architecture.models import GraphConvNet
from mesh.utils import create_graphs_from_df, create_unified_graph, get_efficient_eigenvectors


def get_args():
    parser = argparse.ArgumentParser(description='Train a GCN model for the Game of Life')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--hidden_dim', type=int, default=200, help='Dimension of the hidden layer')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of GCN layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=32, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--train_portion', type=float, default=0.8, help='Portion of data to use for training')
    parser.add_argument('--run_name', type=str, default='try', help='name in wandb')
    parser.add_argument('--length_of_past', type=int, default=10, help='How many past states to consider')
    parser.add_argument('--use_pe', action='store_true', help='Wheter to use pe or not')
    args = parser.parse_args()

    # if args.run_name == 'try':
    #     args.run_name = input("Please enter a run name: ")
    return args


def train(train_loader, model, optimiser, loss_fn, metric_fn):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    num_graphs = 0
    for data in train_loader:
        optimiser.zero_grad()
        data = data.to(DEVICE)
        y_hat = model(data.x, data.edge_index)[:, 0]
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
        y_hat = model(data.x, data.edge_index)

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
        metric_name,
        use_scheduler=False,
        print_steps=True,
        n_runs=10,
):
    """Train the model for NUM_EPOCHS epochs and run n times"""
    # Instantiate optimiser and scheduler
    optimiser = optim.Adam(model.parameters(), lr=LR)
    scheduler = (
        optim.lr_scheduler.StepLR(optimiser, step_size=DECAY_STEP, gamma=DECAY_RATE)
        if use_scheduler
        else None
    )
    scheduler = None
    curves = {name: [] for name in loaders.keys()}
    for epoch in tqdm(range(NUM_EPOCHS)):
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
        # Log metrics to wandb
        log_dict = {"train_loss": train_loss}
        for name, metric_values in curves.items():
            for m_name, value in zip(metric_name, metric_values[-1]):
                log_dict[f"{name}_{m_name}"] = value
        wandb.log(log_dict)

    return curves['train'][-1], curves['test'][-1]


def before_pad_array(lst, length_of_past, fill_value=2, ):
    return np.concatenate([[fill_value] * (length_of_past - len(lst)), lst])


def calc_ds(df, length_of_past=1, use_pe=False, file_name='data_list.pt', history_for_pe=10, number_of_eigenvectors=20):
    print('preparing data')
    ds = []
    for i in tqdm(range(df.i.max() - 1)):
        prev_10_ts = df.loc[(df.i > i - length_of_past) & (df.i <= i)]
        if use_pe:
            if history_for_pe == length_of_past:
                partial_df = prev_10_ts
            else:
                partial_df = df.loc[(df.i > i - history_for_pe) & (df.i <= i)]
            graphs = create_graphs_from_df(partial_df)
            unified_graph = create_unified_graph(graphs)
            eigenvectors, eigenvalues = get_efficient_eigenvectors(unified_graph, number_of_eigenvectors)

        states = np.concatenate([prev_10_ts['state_a'].values, prev_10_ts['state_b'].values])
        nodes = np.concatenate([prev_10_ts['a'].values, prev_10_ts['b'].values])
        times = np.concatenate([prev_10_ts['i'].values, prev_10_ts['i'].values])
        concatenated_df = pd.DataFrame(
            {'nodes': nodes, 'states': states, 'times': times}).drop_duplicates().sort_values(by='nodes')

        b = concatenated_df.groupby('nodes').apply(
            lambda g: g.drop_duplicates().sort_values('times').states.values)

        b = b.apply(lambda lst: before_pad_array(lst, length_of_past, fill_value=0)).values

        x = np.stack(b)
        if use_pe:
            x = np.concatenate([x, eigenvectors], axis=1)
        x = torch.tensor(x, dtype=torch.float)

        edges = np.stack([prev_10_ts['a'].values, prev_10_ts['b'].values], axis=1)
        edges = np.concatenate([edges, edges[:, ::-1]])
        edge_index = torch.tensor(edges.transpose(), dtype=torch.long)

        next_df = df[df.i == i + 1]
        states = np.concatenate([next_df['state_a'].values, next_df['state_b'].values])
        nodes = np.concatenate([next_df['a'].values, next_df['b'].values])
        concatenated_next_df = pd.DataFrame({'nodes': nodes, 'states': states}).drop_duplicates().sort_values(
            by='nodes')

        y = torch.tensor(concatenated_next_df.states.values, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.validate(raise_on_error=True)
        ds.append(data)
    print('finished preparing data')
    torch.save(ds, file_name)
    return ds


def diversity(y_true, y_pred):
    return y_pred.float().std().item()


if __name__ == '__main__':
    args = get_args()

    # Update the corresponding variables with the parsed arguments
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    HIDDEN_DIM = args.hidden_dim
    NUM_LAYERS = args.num_layers
    LR = args.lr
    SEED = args.seed
    DEVICE = args.device
    TRAIN_PORTION = args.train_portion
    LENGTH_OF_PAST = args.length_of_past
    USE_PE = True  # args.use_pe
    # START_STATES = args.start_states

    IN_DIM = args.length_of_past
    DECAY_STEP = 0.1
    DECAY_RATE = 5

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    regulardf = pd.read_csv('../../notebooks/saved/data/RegularGoL.csv')
    temporaldf = pd.read_csv('../../notebooks/saved/data/TemporalGoL.csv')
    oscilationdf = pd.read_csv('../../notebooks/saved/data/OscilationsGoL.csv')
    PD_df = pd.read_csv('../../notebooks/saved/data/PastDependentGoL.csv')
    df_list = [regulardf, temporaldf, oscilationdf, PD_df]
    name = ['regulardf', 'temporaldf', 'oscilationdf', 'PD_df']
    num_runs = 1

    for n, df in zip(name[1:], df_list[1:]):
        # Initialize wandb
        wandb.init(project="StaticMPGoL", name=args.run_name + f'_{n}', config=vars(args))

        path = f'./data/{n}_past_{LENGTH_OF_PAST}_use_pe_{USE_PE}.pt'
        if os.path.exists(path):
            print('ds already exist - training starts')
            ds = torch.load(path)
        else:
            print('ds doesnt exist:')
            ds = calc_ds(df, length_of_past=LENGTH_OF_PAST, use_pe=USE_PE,
                         file_name=path)  # ,start_states=START_STATES)
        train_size = int(len(ds) * TRAIN_PORTION)
        train_dataset = ds[:train_size]
        test_dataset = ds[train_size:]

        train_loader = DataLoader(train_dataset, 5 * BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, 5 * BATCH_SIZE, shuffle=False)
        train_acc_list = []
        test_acc_list = []
        for i in range(num_runs):
            gcn_model = GraphConvNet(
                in_dim=IN_DIM,
                hidden_channels=HIDDEN_DIM,
                out_dim=1,
                num_layers=NUM_LAYERS
            ).to(DEVICE)

            train_acc, test_acc = run(
                gcn_model,
                train_loader,
                {"train": train_loader, "test": test_loader},
                loss_fn=F.binary_cross_entropy,
                metric_fn=[recall_score, precision_score, accuracy_score, f1_score, diversity],  # precision_score,
                metric_name=['recall', 'precision', 'accuracy', 'f1', 'diversity_pred'],
                print_steps=False
            )

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
        print(n)
        print('[recall,accuracy,f1,diversity_pred]')  # precision,
        print('train:', train_acc_list)
        print('test:', test_acc_list)
        wandb.finish()
