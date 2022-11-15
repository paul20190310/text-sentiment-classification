# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import TabularDataset, BucketIterator

import argparse
from tqdm.auto import tqdm

from Mylstm import MyLSTM
from save_load_util import save_model, save_metrics, load_model


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_parser():
    parser = argparse.ArgumentParser(description='train lstm model by training and validation dataset')
    parser.add_argument('train_path', type=str,
                        help='specify the path of the training dataset file')
    parser.add_argument('valid_path', type=str,
                        help='specify the path of the validation dataset file')
    parser.add_argument('-mp', '--initial-model-path', type=str, default=None,
                        help='specify the path of initial model saving file. if no specify, use torch default model.')
    parser.add_argument('-s', '--saving-directory', type=str, default='.',
                        help='specify working directory to save model and metrics files, default current')
    parser.add_argument('-tf', '--text-field-path', type=str, default='./field/text_field.pth',
                        help='specify the path of the text field saving file')
    parser.add_argument('-lf', '--label-field-path', type=str, default='./field/label_field.pth',
                        help='specify the path of the label field saving file')
    parser.add_argument('-e', '--epoch', type=int, default=7,
                        help='the number times that the learning algorithm work through the entire training dataset, default 5')
    parser.add_argument('-et', '--eval-time', type=int, default=2,
                        help='the number evaluate times in a epoch, default 2')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='batch size of training loader, default 32')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                        help='learning rate, default 0.001')
    parser.add_argument('-ba', '--best-valid-accuracy', type=float, default=0.0,
                        help='specify the minimum validation accuracy, default 0.0')
    return parser

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

# train
def train(model, optimizer, train_loader, valid_loader,
          num_epochs, eval_time_in_epoch, file_path,
          criterion = nn.CrossEntropyLoss(),
          best_valid_acc = 0.0):
    # eval every N step
    eval_every = []
    for i in range(eval_time_in_epoch):
        if i == eval_time_in_epoch - 1:
            eval_every.append(len(train_loader) // eval_time_in_epoch +
                              len(train_loader) % eval_time_in_epoch)
        else:
            eval_every.append(len(train_loader) // eval_time_in_epoch)
    
    # initialize metrics values
    train_correct = 0
    valid_correct = 0
    train_loss = 0.0
    valid_loss = 0.0
    train_acc_list = []
    valid_acc_list = []
    train_loss_list = []
    valid_loss_list = []
    check_point_list = []
    
    global_step = 0
    global_step_list = []
    
    # set progress_bar
    progress_bar = tqdm(range(eval_every[0]), leave=True)
    
    # training loop
    model.train()
    for epoch in range(num_epochs):
        eval_time = 0
        train_data_size = 0
        for ((text, text_len), labels), _ in train_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to('cpu')
            outputs = model(text, text_len)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update metrics values
            _, predicted_label = torch.max(outputs.data, 1)
            train_data_size += labels.size(0)
            train_correct += (predicted_label == labels).sum().item()
            train_loss += loss.item()  # 累加loss值
            global_step += 1
            progress_bar.update(1)
            
            # evaluation step
            if (len(global_step_list) == 0 and global_step == eval_every[0]) or \
               (len(global_step_list) > 0 and global_step - global_step_list[-1] == eval_every[eval_time]):
                progress_bar.close()
                
                # validation loop
                model.eval()
                with torch.no_grad():
                    valid_data_size = 0
                    for ((text, text_len), labels), _ in tqdm(valid_loader):
                        labels = labels.type(torch.LongTensor)
                        labels = labels.to(device)
                        text = text.to(device)
                        text_len = text_len.to('cpu')
                        outputs = model(text, text_len)
                        
                        loss = criterion(outputs, labels)
                        _, predicted_label = torch.max(outputs.data, 1)
                        valid_data_size += labels.size(0)
                        valid_correct += (predicted_label == labels).sum().item()
                        valid_loss += loss.item()
                
                # evaluation metrics value
                train_acc = train_correct / train_data_size
                valid_acc = valid_correct / valid_data_size
                average_train_loss = train_loss / eval_every[eval_time]
                average_valid_loss = valid_loss / len(valid_loader)
                train_acc_list.append(train_acc)
                valid_acc_list.append(valid_acc)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                check_point_list.append(epoch + (eval_time + 1) / eval_time_in_epoch)
                global_step_list.append(global_step)
                
                # resetting metrics value
                train_correct = 0
                valid_correct = 0
                train_loss = 0.0                
                valid_loss = 0.0
                train_data_size = 0
                model.train()
                
                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Acc: {:.4f}, Valid Acc: {:.4f}, Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              train_acc, valid_acc,
                              average_train_loss, average_valid_loss))
                
                # save model
                if best_valid_acc < valid_acc:
                    best_valid_acc = valid_acc
                    save_model(file_path + '/model.pt', model, best_valid_acc)
                
                # reset progress_bar
                if global_step < num_epochs * len(train_loader):
                    progress_bar = tqdm(range(eval_every[(eval_time + 1) % eval_time_in_epoch]), leave=True)
                
                eval_time += 1
    
    save_metrics(file_path + '/accuracy_metrics.pt', train_acc_list, valid_acc_list, check_point_list)
    save_metrics(file_path + '/loss_metrics.pt', train_loss_list, valid_loss_list, check_point_list)

def main(args):
    model = MyLSTM().to(device)
    if args.initial_model_path:
        load_model(args.initial_model_path, model)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    train_iter = load_data(args.train_path,
                           args.text_field_path,
                           args.label_field_path,
                           args.batch_size, True)
    valid_iter = load_data(args.valid_path,
                           args.text_field_path,
                           args.label_field_path,
                           args.batch_size)
    print('Start Training!')
    train(model, optimizer, train_loader=train_iter, valid_loader=valid_iter,
          num_epochs=args.epoch, eval_time_in_epoch=args.eval_time,
          file_path=args.saving_directory, best_valid_acc=args.best_valid_accuracy)
    print('Finished Training!')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
