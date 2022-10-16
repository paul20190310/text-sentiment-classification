# -*- coding: utf-8 -*-

from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import torch
import numpy as np
import argparse
import os

from collections import OrderedDict
from Mylstm import MyLSTM
from save_load_util import save_metrics


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_parser():
    parser = argparse.ArgumentParser(description='set federated learning server argument')
    parser.add_argument('-le', '--local-epoch', type=int, default=2,
                        help='the number times that the learning algorithm work through the entire training dataset, default 5')
    parser.add_argument('-e', '--eval-time', type=int, default=2,
                        help='the number evaluate times in a epoch, default 2')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='batch size of training loader, default 32')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                        help='learning rate, default 0.001')
    parser.add_argument('-r', '--num-round', type=int, default=2,
                        help='num of round of server aggregation, defaults to 2')
    parser.add_argument('-ff', '--fraction-fit', type=float, default=1.0,
                        help='Fraction of clients used during training. Defaults to 1.0.')
    parser.add_argument('-fe', '--fraction-evaluate', type=float, default=1.0,
                        help='Fraction of clients used during validation. Defaults to 1.0.')
    parser.add_argument('-mf', '--min-fit-clients', type=int, default=2,
                        help='Minimum number of clients used during training. Defaults to 2.')
    parser.add_argument('-me', '--min-evaluate-clients', type=int, default=2,
                        help='Minimum number of clients used during validation. Defaults to 2.')
    parser.add_argument('-ma', '--min-available-clients', type=int, default=2,
                        help='Minimum number of total clients in the system. Defaults to 2.')
    parser.add_argument('-s', '--saving-directory', type=str, default='.',
                        help='specify working directory to save model files, default current')
    return parser

local_config = None  # define global server config set by main argument
metrics_dict = {
    'train_loss_list': [],
    'valid_loss_list': []
}

def get_on_fit_config_fn(args):
    '''Return a function which returns training configurations.'''
    def fit_config(server_round: int):
        '''Return a configuration with static batch size and (local) epochs.'''
        config = {
            'current_round': server_round,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'eval_time': args.eval_time,
            'local_epochs': args.local_epoch
        }
        return config
    return fit_config

def get_on_evaluate_config_fn(args):
    '''Return a function which returns testing configurations.'''
    def evaluate_config(server_round: int):
        '''Return a configuration with static batch size and (local) epochs.'''
        config = {
            'current_round': server_round,
            'batch_size': args.batch_size
        }
        return config
    return evaluate_config

def fit_metrics_aggregation_fn(fit_metrics):
    total_data_size = sum(data_size for data_size, _ in fit_metrics)
    factors = [data_size / total_data_size for data_size, _ in fit_metrics]
    
    # metrics[0]為client data size，metrics[1]為client train loss值及valid loss值的dict
    agg_train_loss = sum(metrics[1]['train_loss'] * factor for metrics, factor in zip(fit_metrics, factors))
    agg_valid_loss = sum(metrics[1]['valid_loss'] * factor for metrics, factor in zip(fit_metrics, factors))
    metrics_dict['train_loss_list'].append(agg_train_loss)
    metrics_dict['valid_loss_list'].append(agg_valid_loss)
    return {'train_loss': agg_train_loss, 'valid_loss': agg_valid_loss}

def evaluate_metrics_aggregation_fn(eval_metrics):
    total_data_size = sum(data_size for data_size, _ in eval_metrics)
    factors = [data_size / total_data_size for data_size, _ in eval_metrics]
 
    # metrics[0]為client data size，metrics[1]為client testing loss值及accuracy的dict
    agg_accuracy = sum(metrics[1]['accuracy'] * factor for metrics, factor in zip(eval_metrics, factors))
    return {'accuracy': agg_accuracy}

def evaluate_fn(server_round, parameters, config):
    params_dict = zip(MyLSTM().state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    save_dict = {'model_state_dict': state_dict,
                 'valid_loss': 0.0}
    torch.save(save_dict, local_config['saving_directory'] + '/server_model.pt')
    
    if server_round == local_config['num_round']:
        save_metrics(local_config['saving_directory'] + '/server_metrics.pt',
                     metrics_dict['train_loss_list'],
                     metrics_dict['valid_loss_list'],
                     list(range(1, server_round + 1)))

def main(args):
    global local_config
    local_config = {
        'saving_directory': args.saving_directory,
        'num_round': args.num_round
    }
    
    initial_parameters = None
    if os.path.exists(local_config['saving_directory'] + '/server_model.pt'):
        state_dict = torch.load(local_config['saving_directory'] + '/server_model.pt', map_location=device)
        model = MyLSTM().to(device)
        model.load_state_dict(state_dict['model_state_dict'])
        weights = [val.cpu().numpy() for name, val in model.state_dict().items()]
        initial_parameters = ndarrays_to_parameters(weights)
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        initial_parameters=initial_parameters,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.min_available_clients,
        on_fit_config_fn=get_on_fit_config_fn(args),
        on_evaluate_config_fn=get_on_evaluate_config_fn(args),
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
    )
    
    # Start server
    fl.server.start_server(
        server_address="[::1]:9999",
        config=fl.server.ServerConfig(num_rounds=args.num_round),
        strategy=strategy
    )

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
