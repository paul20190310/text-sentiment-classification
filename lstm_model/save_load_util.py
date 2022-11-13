# -*- coding: utf-8 -*-

import torch
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Save and Load Functions
def save_model(save_path, model, valid_acc, print_prompts=True):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_accuracy': valid_acc}
    torch.save(state_dict, save_path)
    if print_prompts:
        print(f'Model saved to ==> {save_path}')

def load_model(load_path, model, print_prompts=True):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    if print_prompts:
        print(f'Model loaded from <== {load_path}')
    return state_dict['valid_accuracy']

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list,
                 print_prompts=True):
    if save_path == None:
        return
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    torch.save(state_dict, save_path)
    if print_prompts:
        print(f'Metrics saved to ==> {save_path}')

def load_metrics(load_path, print_prompts=True):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    if print_prompts:
        print(f'Metrics loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']
