import argparse
import argcomplete
import os
import numpy as np


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove('tmp')
    return np.argmin(memory_available)


def get_args():
    parser = argparse.ArgumentParser(description='Train a GCN model for the Game of Life',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, default=f'cuda:{get_freer_gpu()}', help='Device to use for training')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of graph in a batch')
    parser.add_argument('--hidden_dim', type=int, default=200, help='Dimension of the hidden layer')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of GCN layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=32, help='Random seed')
    parser.add_argument('--train_portion', type=float, default=0.8, help='Portion of data to use for training')
    parser.add_argument('--run_name', type=str, default='try', help='name in wandb')
    parser.add_argument('--use_activation', type=bool, default=True, help='whether to use non linearity in conv layers')
    parser.add_argument('--shuffle', type=bool, default=True, help='whether to shuffle when data loading')
    parser.add_argument('--ams_grad', type=bool, default=True, help='optimizer parameter')
    parser.add_argument('--use_dropout', type=bool, default=False, help='Whether to use dropout or not')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate for the dropout layer')
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
    parser.add_argument('--push_all_values_to_device', type=bool, default=True,
                        help='whether to push all values to device')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args
