#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

from parse_args import get_args

args = get_args()
import pandas as pd
import torch
import torch_geometric
from torch.nn import functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
import wandb
import os
from architecture.models import DeepGraphConvNet, DeepGINConvNet, DeepDividedGINConvNet
from utils import calc_ds, diversity, evaluate_baselines, seed_all
from torch.optim.lr_scheduler import CosineAnnealingLR


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
    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)  # ,amsgrad=True)
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

    NUMBER_OF_EIGENVECTORS = args.number_of_eigenvectors if args.pe_option != 'none' else 0
    IN_DIM = args.length_of_past + NUMBER_OF_EIGENVECTORS
    if args.pe_option == 'temporal':
        IN_DIM += (args.length_of_past - 1) * NUMBER_OF_EIGENVECTORS
    USE_SCHEDULER = not args.dont_use_scheduler
    STEP_SIZE = 50
    GAMMA = 0.5

    seed_all(args.seed, on_steroids=False)

    df = pd.read_csv(f'/home/ygalron/big-storage/notebooks/saved/data/{args.data_name}-GoL.csv')
    wandb.init(project="play-ground", name=args.run_name + f'-{args.data_name}', config=vars(args))

    ds, f_name = calc_ds(df, length_of_past=args.length_of_past,
                         pe_option=args.pe_option, history_for_pe=args.length_of_past, n=args.data_name,
                         number_of_eigenvectors=NUMBER_OF_EIGENVECTORS, offset=args.offset)

    train_size = int(len(ds) * args.train_portion)
    train_dataset = ds[:train_size]
    test_dataset = ds[train_size:]

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    model = DeepDividedGINConvNet(
        in_dim=IN_DIM,
        hidden_channels=args.hidden_dim,
        conv_hidden_dim=args.conv_hidden_dim,
        out_dim=1,
        num_layers=args.num_layers,
        num_conv_layers=args.num_conv_layers,
        use_activation=args.use_activation).to(args.device)

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
