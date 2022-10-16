# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import argparse

from torchtext.data import TabularDataset, BucketIterator
from collections import OrderedDict
from tqdm.auto import tqdm

from Mylstm import MyLSTM
from save_load_util import save_model, save_metrics
from test_model import test


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_parser():
    parser = argparse.ArgumentParser(description='train lstm model by training and validation dataset')
    parser.add_argument('train_path', type=str,
                        help='specify the path of the training dataset file')
    parser.add_argument('valid_path', type=str,
                        help='specify the path of the validation dataset file')
    parser.add_argument('test_path', type=str,
                        help='specify the path of the testing dataset file')
    parser.add_argument('-s', '--saving-directory', type=str, default='.',
                        help='specify working directory to save model and metrics files, default current')
    parser.add_argument('-tf', '--text-field-path', type=str, default='./field/text_field.pth',
                        help='specify the path of the text field saving file')
    parser.add_argument('-lf', '--label-field-path', type=str, default='./field/label_field.pth',
                        help='specify the path of the label field saving file')
    return parser


model = None  # define global model variable
local_config = None  # define global client config set by main argument

def load_data(data_file_path, text_field_path, label_field_path,
              batch_size, is_shuffle=False):
    ''' load .csv file data and return dataloader '''
    text_field = torch.load(text_field_path)
    label_field = torch.load(label_field_path)
    fields = [('text', text_field), ('label', label_field)]
    
    data = TabularDataset(path=data_file_path, format='CSV',
                          fields=fields, skip_header=True)
    data_iter = BucketIterator(data, batch_size=batch_size,
                               sort_key=lambda x: len(x.text), shuffle=is_shuffle,
                               device=device, sort=True, sort_within_batch=True)
    return data_iter

def train(server_round, model, optimizer, train_loader, valid_loader,
          num_epochs, eval_time_in_epoch, file_path,
          criterion = nn.CrossEntropyLoss(),
          best_valid_loss = float("Inf")):
    # eval every N step
    eval_every = []
    for i in range(eval_time_in_epoch):
        if i == eval_time_in_epoch - 1:
            eval_every.append(len(train_loader) // eval_time_in_epoch +
                              len(train_loader) % eval_time_in_epoch)
        else:
            eval_every.append(len(train_loader) // eval_time_in_epoch)
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    
    # set progress_bar
    progress_bar = tqdm(range(eval_every[0]), leave=True)
    
    # training loop
    model.train()
    for epoch in range(num_epochs):
        eval_time = 0
        for ((text, text_len), labels), _ in train_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to('cpu')
            output = model(text, text_len)
            
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            
            # update running values
            running_loss += loss.item()  # 累加loss值
            global_step += 1
            
            # evaluation step
            if (len(global_steps_list) == 0 and global_step == eval_every[0]) or \
               (len(global_steps_list) > 0 and global_step - global_steps_list[-1] == eval_every[eval_time]):
                progress_bar.close()
                
                model.eval()
                with torch.no_grad():                    
                    # validation loop
                    for ((text, text_len), labels), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)
                        labels = labels.to(device)
                        text = text.to(device)
                        text_len = text_len.to('cpu')
                        output = model(text, text_len)
                        
                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()
                
                # evaluation
                average_train_loss = running_loss / eval_every[eval_time]
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)
                
                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()
                
                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # save model
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_model(file_path + '/model.pt', model, best_valid_loss)
                
                # reset progress_bar
                if global_step < num_epochs * len(train_loader):
                    progress_bar = tqdm(range(eval_every[(eval_time + 1) % eval_time_in_epoch]), leave=True)
                
                eval_time += 1
    
    save_metrics(file_path + '/metrics-' + str(server_round) + '.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    min_loss_index = min(range(len(valid_loss_list)), key=valid_loss_list.__getitem__)
    return train_loss_list[min_loss_index], valid_loss_list[min_loss_index]

class EmotionClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    def set_parameters(self, parameters, config):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        train_iter = load_data(local_config['train_path'],
                               local_config['text_field_path'],
                               local_config['label_field_path'],
                               config['batch_size'], True)
        valid_iter = load_data(local_config['valid_path'],
                               local_config['text_field_path'],
                               local_config['label_field_path'],
                               config['batch_size'])
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        print(f"Round {config['current_round']} training:")
        train_loss, valid_loss = train(config['current_round'],
                                       model, optimizer,
                                       train_loader=train_iter,
                                       valid_loader=valid_iter,
                                       num_epochs=config['local_epochs'],
                                       eval_time_in_epoch=config['eval_time'],
                                       file_path=local_config['saving_directory'])
        print(f"Finished Round {config['current_round']} training!")
        results = {
            'train_loss': float(train_loss),
            'valid_loss': float(valid_loss)
        }
        return self.get_parameters(config), len(train_iter), results
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        test_iter = load_data(local_config['test_path'],
                              local_config['text_field_path'],
                              local_config['label_field_path'],
                              config['batch_size'])
        loss, accuracy = test(model, test_iter)
        results = {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }
        return float(loss), len(test_iter), results

def main(args):
    global model, local_config
    model = MyLSTM().to(device)
    local_config = {
        'train_path': args.train_path,
        'valid_path': args.valid_path,
        'test_path': args.test_path,
        'saving_directory': args.saving_directory,
        'text_field_path': args.text_field_path,
        'label_field_path': args.label_field_path
    }
    fl.client.start_numpy_client(server_address="[::1]:9999", client=EmotionClient())

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
