import argparse
import os
import pdb
import random

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
from architecture.models import GraphConvNet, ModifiedGraphConvNet, GINNet, GATNet, DeepGraphConvNet
from mesh.utils import create_graphs_from_df, create_unified_graph, get_efficient_eigenvectors
from utils import count_consecutive_ones_from_end, concat_multiple_times


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
    parser.add_argument('--use_pe', action='store_true', help='Whether to use pe or not')
    parser.add_argument('--history_for_pe', type=int, default=10,
                        help='number of timestamps to take for calculating the pe')
    parser.add_argument('--number_of_eigenvectors', type=int, default=20,
                        help='number of eigen vector to use for the pe')
    parser.add_argument('--offset', type=int, default=0, help='the offset in time for taking information')

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
        optim.lr_scheduler.StepLR(optimiser, step_size=STEP_SIZE, gamma=GAMMA)
        if use_scheduler
        else None
    )
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


def calc_ds(df, length_of_past=1, use_pe=False, history_for_pe=10, number_of_eigenvectors=20,
            offset=0, n=''):
    if use_pe:
        file_name = (f'./data/{n}_past_{length_of_past}_use_pe_{use_pe}_history_for_pe_{history_for_pe}'
                     f'_number_of_eigenvectors_{number_of_eigenvectors}_offset_{offset}.pt')
    else:
        file_name = f'./data/{n}_past_{length_of_past}_use_pe_{use_pe}_offset_{offset}.pt'

    if os.path.exists(file_name):
        print('ds already exist - training starts')
        ds = torch.load(file_name)
        return ds, file_name
    else:
        print('ds doesnt exist:')

    print('preparing data')
    ds = []
    eigenvectors = None
    for i in tqdm(range(offset, df.i.max() - 1)):
        prev_10_ts = df.loc[(df.i > i - length_of_past - offset) & (df.i <= i - offset)]
        if use_pe:
            # todo: also off set for pe?
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
        # location_of_phenomena = np.apply_along_axis(count_consecutive_ones_from_end, 1, b) >= 3
        # minority_b = b[location_of_phenomena]
        # print(minority_b.shape)
        # x = concat_multiple_times(b, minority_b, times=4)
        if use_pe:
            x = np.concatenate([x, eigenvectors[-x.shape[0]:]], axis=1)
        x = torch.tensor(x, dtype=torch.float)

        edges = np.stack([prev_10_ts['a'].values, prev_10_ts['b'].values], axis=1)
        edges = np.concatenate([edges, edges[:, ::-1]])
        edge_index = torch.tensor(edges.transpose(), dtype=torch.long)

        next_df = df[df.i == i + 1]
        states = np.concatenate([next_df['state_a'].values, next_df['state_b'].values])
        nodes = np.concatenate([next_df['a'].values, next_df['b'].values])
        concatenated_next_df = pd.DataFrame({'nodes': nodes, 'states': states}).drop_duplicates().sort_values(
            by='nodes')

        y_values = concatenated_next_df.states.values
        # orig_values = concatenated_next_df.states.values
        # minority_values = orig_values[location_of_phenomena]
        # y_values = concat_multiple_times(orig_values, minority_values, times=4)
        y = torch.tensor(y_values, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.validate(raise_on_error=True)
        ds.append(data)
    print('finished preparing data')
    torch.save(ds, file_name)
    return ds, file_name


def diversity(y_true, y_pred):
    return y_pred.float().std().item()


if __name__ == '__main__':
    args = get_args()

    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    HIDDEN_DIM = args.hidden_dim
    NUM_LAYERS = args.num_layers
    LR = args.lr
    SEED = args.seed
    DEVICE = args.device
    TRAIN_PORTION = args.train_portion
    LENGTH_OF_PAST = args.length_of_past
    USE_PE = args.use_pe
    HISTORY_FOR_PE = args.history_for_pe
    NUMBER_OF_EIGENVECTORS = args.number_of_eigenvectors if args.use_pe else 0
    OFFSET = args.offset
    IN_DIM = args.length_of_past + NUMBER_OF_EIGENVECTORS
    STEP_SIZE = 5
    GAMMA = 0.9
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    regulardf = pd.read_csv('../../notebooks/saved/data/RegularGoL.csv')
    temporaldf = pd.read_csv('../../notebooks/saved/data/TemporalGoL.csv')
    oscilationdf = pd.read_csv('../../notebooks/saved/data/OscilationsGoL.csv')
    PD_df = pd.read_csv('../../notebooks/saved/data/PastDependentGoL.csv')
    df_list = [regulardf, temporaldf, oscilationdf, PD_df]
    name = ['regulardf', 'temporaldf', 'oscilationdf', 'PD_df']
    num_runs = 1

    for n, df in zip(name[:1], df_list[:1]):
        wandb.init(project="StaticMPGoL", name=args.run_name + f'_{n}', config=vars(args))

        ds, f_name = calc_ds(df, length_of_past=LENGTH_OF_PAST,
                             use_pe=USE_PE, history_for_pe=HISTORY_FOR_PE, n=n,
                             number_of_eigenvectors=NUMBER_OF_EIGENVECTORS, offset=OFFSET)

        random.shuffle(ds)
        train_size = int(len(ds) * TRAIN_PORTION)
        train_dataset = ds[:train_size]
        test_dataset = ds[train_size:]

        train_loader = DataLoader(train_dataset, 5 * BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, 5 * BATCH_SIZE, shuffle=True)
        train_acc_list = []
        test_acc_list = []
        model = DeepGraphConvNet(
            in_dim=IN_DIM,
            hidden_channels=HIDDEN_DIM,
            out_dim=1,
            num_layers=NUM_LAYERS,
            num_conv_layers=3).to(DEVICE)
        for i in range(num_runs):
            train_acc, test_acc = run(
                model,
                train_loader,
                {"train": train_loader, "test": test_loader},
                loss_fn=F.binary_cross_entropy,
                metric_fn=[recall_score, precision_score, accuracy_score, f1_score, diversity],  # precision_score,
                metric_name=['recall', 'precision', 'accuracy', 'f1', 'diversity_pred'],
                print_steps=False,
                use_scheduler=True
            )
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
        print('saving_checkpoint')
        torch.save({'model_state_dict': model.state_dict()}, f'./checkpoints/{args.run_name}_{n}')
        wandb.finish()
