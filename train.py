import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch_geometric.utils
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from tqdm import tqdm

import wandb
from architecture.models import DeepGraphConvNet
from mesh.utils import create_graphs_from_df, create_unified_graph, get_efficient_eigenvectors


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove('tmp')
    return 0  # np.argmin(memory_available)


def get_args():
    parser = argparse.ArgumentParser(description='Train a GCN model for the Game of Life',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, default=f'cuda:{get_freer_gpu()}', help='Device to use for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--hidden_dim', type=int, default=200, help='Dimension of the hidden layer')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of GCN layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=32, help='Random seed')
    parser.add_argument('--train_portion', type=float, default=0.8, help='Portion of data to use for training')
    parser.add_argument('--run_name', type=str, default='try', help='name in wandb')
    parser.add_argument('--length_of_past', type=int, default=10,
                        help='How many past states to consider as node features')
    parser.add_argument('--use_pe', action='store_true', help='Whether to use pe or not')
    # parser.add_argument('--history_for_pe', type=int, default=10,
    #                     help='number of timestamps to take for calculating the pe')
    parser.add_argument('--number_of_eigenvectors', type=int, default=20,
                        help='number of eigen vector to use for the pe')
    parser.add_argument('--offset', type=int, default=0, help='the offset in time for taking information')
    parser.add_argument('--num_conv_layers', type=int, default=1, help='number of conv layers')
    parser.add_argument('--conv_hidden_dim', type=int, default=1, help='conv layers hidden dimension')
    parser.add_argument('--dont_use_scheduler', action='store_true', help='whether to use scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for adam optimizer')
    parser.add_argument('--data_name', type=str,
                        choices=['regular', 'temporal', 'oscillations', 'past-dependent', 'static-oscillations'],
                        default='regular',
                        help='path to dataset')

    args = parser.parse_args()
    return args


def train(train_loader, model, optimiser, loss_fn, metric_fn):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    num_graphs = 0
    for data in train_loader:
        optimiser.zero_grad()
        data = data.to(args.device)
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
        data = data.to(args.device)
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
        weight_decay=0,
        patience=151,
        early_stopping_metric='test_f1',
):
    """Train the model for NUM_EPOCHS epochs and run n times"""
    # Instantiate optimiser and scheduler
    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    scheduler = (
        optim.lr_scheduler.StepLR(optimiser, step_size=STEP_SIZE, gamma=GAMMA)
        if use_scheduler
        else None
    )
    curves = {name: [] for name in loaders.keys()}

    best_metric = -float('inf')
    patience_counter = 0
    for epoch in tqdm(range(args.num_epochs)):
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

        # Early stopping
        current_metric = log_dict[early_stopping_metric]
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return curves['train'][-1], curves['test'][-1]


def before_pad_array(lst, length_of_past, fill_value=2):
    return np.concatenate([[fill_value] * (length_of_past - len(lst)), lst])


def calc_ds(df, length_of_past=1, use_pe=False, history_for_pe=10, number_of_eigenvectors=20,
            offset=0, n=''):
    if use_pe:
        file_name = (f'./data/{n}_past_{length_of_past}_use_pe_{use_pe}_history_for_pe_{history_for_pe}'
                     f'_number_of_eigenvectors_{number_of_eigenvectors}_offset_{offset}.pt')
    else:
        file_name = f'data/{n}_past_{length_of_past}_use_pe_{use_pe}_offset_{offset}.pt'

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
        current_df = df[df.i == i - offset]
        if use_pe:
            partial_df = df.loc[(df.i > i - history_for_pe - offset) & (df.i <= i - offset)]
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
            x = np.concatenate([x, eigenvectors[-x.shape[0]:]], axis=1)
        x = torch.tensor(x, dtype=torch.float)

        edges = np.stack([current_df['a'].values, current_df['b'].values], axis=1)
        edges = np.concatenate([edges, edges[:, ::-1]])
        edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
        # print(torch_geometric.utils.to_dense_adj(edge_index)[0].sum(dim=0))
        next_df = df[df.i == i + 1]
        states = np.concatenate([next_df['state_a'].values, next_df['state_b'].values])
        nodes = np.concatenate([next_df['a'].values, next_df['b'].values])
        concatenated_next_df = pd.DataFrame({'nodes': nodes, 'states': states}).drop_duplicates().sort_values(
            by='nodes')

        y_values = concatenated_next_df.states.values
        y = torch.tensor(y_values, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.validate(raise_on_error=True)
        ds.append(data)
    print('finished preparing data')
    torch.save(ds, file_name)
    return ds, file_name


def diversity(y_true, y_pred):
    return y_pred.float().std().item()


def evaluate_baselines(loaders, loaders_names):
    log_dict = {}
    for _ in tqdm(range(args.num_epochs)):
        for loader, l_name in zip(loaders, loaders_names):
            for data in loader:
                value = run_baseline_on_data(data)
                log_dict[f"{l_name}_f1"] = value
        wandb.log(log_dict)


class SumNeighborsFeatures(MessagePassing):
    def __init__(self):
        super().__init__(aggr='sum')

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out


def run_baseline_on_data(data, use_temporal_condition=False):
    summer = SumNeighborsFeatures()
    adj = torch_geometric.utils.to_dense_adj(edge_index=data.edge_index)[0]
    features_sum = summer(data.x, data.edge_index)
    last_states_sums = features_sum[:, -1]
    number_of_neighbors = adj.sum(dim=1)  # might not work when bs is bigger then 1
    lower_bound = torch.max(2 * number_of_neighbors / 8, torch.tensor(2))
    upper_bound = torch.max(3 * number_of_neighbors / 8, torch.tensor(3))
    # print('lower_bound', lower_bound, 'upper_bound', upper_bound)
    # print('lower_bound', lower_bound, 'upper_bound', upper_bound)
    gol_condition = ((last_states_sums >= lower_bound) & (last_states_sums <= upper_bound))
    if use_temporal_condition:
        last_three_sum = data.x[:, -3:].sum(dim=1)
        total_sum = data.x.sum(dim=1)
        critical_survival_condition = torch.where((last_three_sum == 3), 1, 0)
        must_die = torch.where((total_sum == 11), 0, 1)
        y_pred = (gol_condition | critical_survival_condition) & must_die
    else:
        y_pred = gol_condition
    return f1_score(data.y, y_pred)


if __name__ == '__main__':
    args = get_args()
    NUMBER_OF_EIGENVECTORS = args.number_of_eigenvectors if args.use_pe else 0
    IN_DIM = args.length_of_past + NUMBER_OF_EIGENVECTORS
    USE_SCHEDULER = not args.dont_use_scheduler
    STEP_SIZE = 50
    GAMMA = 0.5
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(f'/home/ygalron/big-storage/notebooks/saved/data/{args.data_name}-GoL.csv')
    wandb.init(project="free_of_bugs", name=args.run_name + f'-{args.data_name}', config=vars(args))

    ds, f_name = calc_ds(df, length_of_past=args.length_of_past,
                         use_pe=args.use_pe, history_for_pe=args.length_of_past, n=args.data_name,
                         number_of_eigenvectors=NUMBER_OF_EIGENVECTORS, offset=args.offset)

    train_size = int(len(ds) * args.train_portion)
    train_dataset = ds[:train_size]
    test_dataset = ds[train_size:]

    train_loader = DataLoader(train_dataset, 1, shuffle=False)
    test_loader = DataLoader(test_dataset, 1, shuffle=False)
    model = DeepGraphConvNet(
        in_dim=IN_DIM,
        hidden_channels=args.hidden_dim,
        conv_hidden_dim=args.conv_hidden_dim,
        out_dim=1,
        num_layers=args.num_layers,
        num_conv_layers=args.num_conv_layers).to(args.device)

    evaluate_baselines([train_loader, test_loader], ['train', 'test'])
    # run(model, train_loader,
    #     {"train": train_loader, "test": test_loader},
    #     loss_fn=F.binary_cross_entropy,
    #     metric_fn=[recall_score, precision_score, accuracy_score, f1_score, diversity],
    #     metric_name=['recall', 'precision', 'accuracy', 'f1', 'diversity_pred'],
    #     print_steps=False,
    #     use_scheduler=USE_SCHEDULER,
    #     weight_decay=args.weight_decay
    #     )
    # print('saving_checkpoint')
    # torch.save({'model_state_dict': model.state_dict()}, f'./checkpoints/{args.run_name}_{args.data_name}.pt')
    wandb.finish()
