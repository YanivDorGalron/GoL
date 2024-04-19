import os
import pdb
from typing import List

import numpy as np
import pandas as pd
import torch
import torch_geometric
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

import wandb
from architecture.positional_encodings import PositionalEncoding
from architecture.summer import SumNeighborsFeatures
from mesh.utils import create_graphs_from_df, create_unified_graph, get_efficient_eigenvectors, \
    get_eigenvectors_of_list_of_graphs


def before_pad_array(lst, length_of_past, fill_value=2):
    return np.concatenate([[fill_value] * (length_of_past - len(lst)), lst])


def diversity(y_true, y_pred):
    return y_pred.float().std().item()


def evaluate_baselines(loaders: List[DataLoader], loaders_names: List[str]):
    log_dict = {}
    for _ in tqdm(range(100)):
        for loader, l_name in zip(loaders, loaders_names):
            assert loader.batch_size == 1, 'baseline works only if each batch is a single graph'
            for data in loader:
                value = run_baseline_on_data(data)
                log_dict[f"{l_name}_f1"] = value
        wandb.log(log_dict)


def run_baseline_on_data(data, use_temporal_condition=False):
    summer = SumNeighborsFeatures()
    adj = torch_geometric.utils.to_dense_adj(edge_index=data.edge_index)[0]
    features_sum = summer(data.x, data.edge_index)
    last_neighbors_state_sum = features_sum[:, -1]
    number_of_neighbors = adj.sum(dim=1)  # might not work when batch size is bigger then 1
    lower_bound = torch.max(2 * number_of_neighbors / 8, torch.tensor(2))
    upper_bound = torch.max(3 * number_of_neighbors / 8, torch.tensor(3))
    gol_condition = ((last_neighbors_state_sum >= lower_bound) & (last_neighbors_state_sum <= upper_bound))
    if use_temporal_condition:
        last_three_sum = data.x[:, -3:].sum(dim=1)
        total_sum = data.x.sum(dim=1)
        critical_survival_condition = torch.where((last_three_sum == 3), 1, 0)
        must_die = torch.where((total_sum == 11), 0, 1)
        y_pred = (gol_condition | critical_survival_condition) & must_die
    else:
        y_pred = gol_condition
    return f1_score(data.y, y_pred)


def calc_ds(df, length_of_past=1, pe_option='none', history_for_pe=10, number_of_eigenvectors=20,
            offset=0, n=''):
    use_pe = False if pe_option == 'none' else True
    history_for_pe = {'supra': history_for_pe, 'temporal': history_for_pe, 'regular': 1}

    if pe_option != 'none':
        file_name = (f'./data/{n}_past_{length_of_past}_{pe_option}_pe_history_for_pe_{history_for_pe[pe_option]}'
                     f'_number_of_eigenvectors_{number_of_eigenvectors}_offset_{offset}.pt')
    else:
        file_name = f'data/{n}_past_{length_of_past}_{pe_option}_pe_offset_{offset}.pt'

    if os.path.exists(file_name):
        print('ds already exist - training starts')
        ds = torch.load(file_name)
        return ds, file_name
    else:
        print('ds doesnt exist:')

    print('preparing data')
    ds = []
    pe_encodings = None
    for i in tqdm(range(offset, df.i.max() - 1)):
        prev_10_ts = df.loc[(df.i > i - length_of_past - offset) & (df.i <= i - offset)]
        current_df = df[df.i == i - offset]
        if use_pe:
            if pe_option == 'temporal':
                graph = create_graphs_from_df(current_df)[0]
                num_nodes = len(list(graph.nodes()))
                pe_encodings = PositionalEncoding(duplicated=num_nodes,
                                                  d_model=number_of_eigenvectors,
                                                  max_len=length_of_past).pe
                pe_encodings = pe_encodings.reshape(num_nodes, -1)
            else:
                partial_df = df.loc[(df.i > i - history_for_pe[pe_option] - offset) & (df.i <= i - offset)]
                graphs = create_graphs_from_df(partial_df)
                num_nodes = len(list(graphs[0].nodes()))
                if pe_option == 'supra':
                    unified_graph = create_unified_graph(graphs)
                    pe_encodings, _ = get_efficient_eigenvectors(unified_graph, number_of_eigenvectors)
                    pe_encodings = pe_encodings.reshape(-1, num_nodes, number_of_eigenvectors)
                    pe_encodings = pe_encodings[-1]  # taking last layer pe
                elif pe_option == 'regular':
                    pe_encodings = get_eigenvectors_of_list_of_graphs(graphs, number_of_eigenvectors)
                    pe_encodings = pe_encodings.reshape(num_nodes, -1)  # N x (T*number_of_eigenvectors)

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
            x = np.concatenate([x, pe_encodings], axis=1)
        x = torch.tensor(x, dtype=torch.float)

        edges = np.stack([current_df['a'].values, current_df['b'].values], axis=1)
        edges = np.concatenate([edges, edges[:, ::-1]])
        edge_index = torch.tensor(edges.transpose(), dtype=torch.long)
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


def seed_all(seed, on_steroids=False):
    if on_steroids:
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    torch_geometric.seed_everything(seed)
