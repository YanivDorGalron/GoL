import pandas as pd
from torch_geometric.data import DataLoader

from architecture.models import GraphConvNet
import torch

from train import calc_ds


def evaluate(loader, model):
    """Evaluate model on dataset"""
    y_pred, y_true = [], []
    data_x = []
    model.eval()
    for data in loader:
        data = data.to(DEVICE)
        y_hat = model(data.x, data.edge_index)

        y_pred.append(y_hat.detach().cpu())
        y_true.append(data.y.detach().cpu())
        data_x.append(data.x.detach().cpu())
    y_pred = torch.cat(y_pred, dim=0)
    y_pred = y_pred[:, 0]
    y_pred = (y_pred > 0.5).long()
    y_true = torch.cat(y_true, dim=0)
    data_x = torch.cat(data_x, dim=0)
    return y_pred, y_true, data_x


if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = './checkpoints/11StateHistory5offset5PE_scheduler_500_epochs_1000Dim_num_layers20_temporaldf'
    state_dict = torch.load(path)
    model = GraphConvNet(31, 1000, 1, 20)
    model.load_state_dict(state_dict['model_state_dict'])
    model.to(DEVICE)
    TRAIN_PORTION = 0.1
    BATCH_SIZE = 32
    temporaldf = pd.read_csv('../../notebooks/saved/data/TemporalGoL.csv')
    n = 'temporaldf'
    ds, f_name = calc_ds(temporaldf, length_of_past=11, use_pe=True, n=n,
                         history_for_pe=5, number_of_eigenvectors=20, offset=5)

    train_size = int(len(ds) * TRAIN_PORTION)
    train_dataset = ds[:train_size]
    test_dataset = ds[train_size:]

    train_loader = DataLoader(train_dataset, 5 * BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, 5 * BATCH_SIZE, shuffle=False)
    y_pred, y_true, data_x = evaluate(train_loader, model)
    print("finished")
