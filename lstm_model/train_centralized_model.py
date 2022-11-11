# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import TabularDataset, BucketIterator

import argparse
from tqdm.auto import tqdm

from Mylstm import MyLSTM
from save_load_util import save_model, save_metrics


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_parser():
    parser = argparse.ArgumentParser(description='train lstm model by training and validation dataset')
    parser.add_argument('train_path', type=str,
                        help='specify the path of the training dataset file')
    parser.add_argument('valid_path', type=str,
                        help='specify the path of the validation dataset file')
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
    return parser

# train
def train(model, optimizer, train_loader, valid_loader,
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
                    save_model(file_path + '/best_model.pt', model, best_valid_loss)
                
                # reset progress_bar
                if global_step < num_epochs * len(train_loader):
                    progress_bar = tqdm(range(eval_every[(eval_time + 1) % eval_time_in_epoch]), leave=True)
                
                eval_time += 1
    
    save_model(file_path + '/final_model.pt', model, best_valid_loss)
    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

def main(args):
    text_field = torch.load(args.text_field_path)
    label_field = torch.load(args.label_field_path)
    fields = [('text', text_field), ('label', label_field)]

    # dataset
    train_data = TabularDataset(path=args.train_path,
                                format='CSV', fields=fields, skip_header=True)
    valid_data = TabularDataset(path=args.valid_path,
                                format='CSV', fields=fields, skip_header=True)
    
    # dataloader
    train_iter = BucketIterator(train_data, batch_size=args.batch_size,
                                sort_key=lambda x: len(x.text), shuffle=True,
                                device=device, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid_data, batch_size=args.batch_size,
                                sort_key=lambda x: len(x.text),
                                device=device, sort=True, sort_within_batch=True)
    
    model = MyLSTM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    train(model, optimizer, train_loader=train_iter, valid_loader=valid_iter,
          num_epochs=args.epoch, eval_time_in_epoch=args.eval_time,
          file_path=args.saving_directory)
    print('Finished Training!')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
