import argparse
import pdb

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from architecture.models import DeepGraphConvNet
from utils import get_freer_gpu, calc_ds, diversity


def get_args():
    parser = argparse.ArgumentParser(description='Train a GCN model for the Game of Life',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, default=f'cuda:{get_freer_gpu()}', help='Device to use for training')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train for')
    parser.add_argument('--hidden_dim', type=int, default=200, help='Dimension of the hidden layer')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of GCN layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=32, help='Random seed')
    parser.add_argument('--train_portion', type=float, default=0.8, help='Portion of data to use for training')
    parser.add_argument('--run_name', type=str, default='try', help='name in wandb')
    parser.add_argument('--length_of_past', type=int, default=11,
                        help='How many past states to consider as node features')
    parser.add_argument('--pe_option', type=str, choices=['supra', 'temporal', 'regular', 'none'], default='none',
                        help='pe type to use, if none will not be used')
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
                        default='regular', help='path to dataset')

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


if __name__ == '__main__':
    args = get_args()
    NUMBER_OF_EIGENVECTORS = args.number_of_eigenvectors if args.pe_option != 'none' else 0
    IN_DIM = args.length_of_past + NUMBER_OF_EIGENVECTORS
    if args.pe_option == 'temporal':
        IN_DIM += (args.length_of_past - 1) * NUMBER_OF_EIGENVECTORS
    USE_SCHEDULER = not args.dont_use_scheduler
    STEP_SIZE = 50
    GAMMA = 0.5
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(f'/home/ygalron/big-storage/notebooks/saved/data/{args.data_name}-GoL.csv')
    wandb.init(project="temporal-pe-oscillations-GoL-sweep", name=args.run_name + f'-{args.data_name}',
               config=vars(args))

    ds, f_name = calc_ds(df, length_of_past=args.length_of_past,
                         pe_option=args.pe_option, history_for_pe=args.length_of_past, n=args.data_name,
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

    # evaluate_baselines([train_loader, test_loader], ['train', 'test'])
    run(model, train_loader,
        {"train": train_loader, "test": test_loader},
        loss_fn=F.binary_cross_entropy,
        metric_fn=[recall_score, precision_score, accuracy_score, f1_score, diversity],
        metric_name=['recall', 'precision', 'accuracy', 'f1', 'diversity_pred'],
        print_steps=False,
        use_scheduler=USE_SCHEDULER,
        weight_decay=args.weight_decay
        )
    print('saving_checkpoint')
    torch.save({'model_state_dict': model.state_dict()}, f'./checkpoints/{args.run_name}_{args.data_name}.pt')
    wandb.finish()
