# -*- coding: utf-8 -*-

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import Dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler
)

import torch
import argparse
import os

from save_load_util import save_metrics


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_parser():
    parser = argparse.ArgumentParser(description='train bert model by training dataset')
    parser.add_argument('train_path', type=str,
                        help='specify the path of the training dataset file')
    parser.add_argument('valid_path', type=str,
                        help='specify the path of the validation dataset file')
    parser.add_argument('-mds', '--model-saving-directory', type=str, default='./model',
                        help='specify working directory to save model, default ./model')
    parser.add_argument('-tks', '--tokenizer-saving-directory', type=str, default='./tokenizer',
                        help='specify working directory to save tokenizer, default ./tokenizer')
    parser.add_argument('-mts', '--metrics-saving-directory', type=str, default='.',
                        help='specify working directory to save metrics files, default current')
    parser.add_argument('-e', '--epoch', type=int, default=5,
                        help='the number times that the learning algorithm work through the entire training dataset, default 5')
    parser.add_argument('-et', '--eval-time', type=int, default=2,
                        help='the number evaluate times in a epoch, default 2')
    parser.add_argument('-b', '--batch-size', type=int, default=4,
                        help='batch size of training loader, default 4')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.00005,
                        help='learning rate, default 0.00005')
    parser.add_argument('-ba', '--best-valid-accuracy', type=float, default=0.0,
                        help='specify the minimum validation accuracy, default 0.0')
    return parser

def load_data(data_file_path, tokenizer_path, batch_size, is_shuffle=False):
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.save_pretrained(tokenizer_path)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    dataset = Dataset.from_csv(data_file_path)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    return DataLoader(tokenized_dataset, shuffle=is_shuffle, batch_size=batch_size)

def train(model, optimizer, train_loader, valid_loader, num_epochs,
          eval_time_in_epoch, model_save_path, metrics_save_path,
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
    train_loss = 0.0
    valid_loss = 0.0
    train_acc_list = []
    valid_acc_list = []
    train_loss_list = []
    valid_loss_list = []
    check_point_list = []
    
    global_step = 0
    global_step_list = []
    
    # set progress bar
    progress_bar = tqdm(range(eval_every[0]), leave=True)
    
    # training loop
    metric = load_metric("accuracy")
    model.train()
    for epoch in range(num_epochs):
        eval_time = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # update metrics values
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            train_loss += loss.item()
            global_step += 1
            progress_bar.update(1)
            
            # evaluation step
            if (len(global_step_list) == 0 and global_step == eval_every[0]) or \
               (len(global_step_list) > 0 and global_step - global_step_list[-1] == eval_every[eval_time]):
                progress_bar.close()
                train_acc = metric.compute()['accuracy']
                
                # validation loop
                model.eval()
                with torch.no_grad():      
                    for batch in tqdm(valid_loader):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(**batch)
                        
                        logits = outputs.logits
                        predictions = torch.argmax(logits, dim=-1)
                        metric.add_batch(predictions=predictions, references=batch["labels"])
                        loss = outputs.loss
                        valid_loss += loss.item()
                
                # evaluation metrics value
                valid_acc = metric.compute()['accuracy']
                average_train_loss = train_loss / eval_every[eval_time]
                average_valid_loss = valid_loss / len(valid_loader)
                train_acc_list.append(train_acc)
                valid_acc_list.append(valid_acc)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                check_point_list.append(epoch + (eval_time + 1) / eval_time_in_epoch)
                global_step_list.append(global_step)
        
                # resetting running loss values
                train_loss = 0.0
                valid_loss = 0.0
                model.train()
        
                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Acc: {:.4f}, Valid Acc: {:.4f}, Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              train_acc, valid_acc,
                              average_train_loss, average_valid_loss))
        
                # save model
                if best_valid_acc < valid_acc:
                    best_valid_acc = valid_acc
                    model.save_pretrained(model_save_path)
                    print(f'Model saved to ==> {model_save_path}')
                
                # reset progress_bar
                if global_step < num_epochs * len(train_loader):
                    progress_bar = tqdm(range(eval_every[(eval_time + 1) % eval_time_in_epoch]), leave=True)
                
                eval_time += 1
                
    save_metrics(metrics_save_path + '/accuracy_metrics.pt', train_acc_list, valid_acc_list, check_point_list)
    save_metrics(metrics_save_path + '/loss_metrics.pt', train_loss_list, valid_loss_list, check_point_list)

def main(args):
    if not os.path.exists(args.model_saving_directory):
        model_name_or_path = "bert-base-uncased"
    else:
        model_name_or_path = args.model_saving_directory
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=6,
        id2label={0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'},
        label2id={'sadness':0, 'joy':1, 'love':2, 'anger':3, 'fear':4, 'surprise':5}
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    train_dataloader = load_data(args.train_path,
                                 args.tokenizer_saving_directory,
                                 args.batch_size, True)
    valid_dataloader = load_data(args.valid_path,
                                 args.tokenizer_saving_directory,
                                 args.batch_size)
    print('Start Training!')
    train(model, optimizer, train_loader=train_dataloader, valid_loader=valid_dataloader,
          model_save_path=args.model_saving_directory,
          metrics_save_path=args.metrics_saving_directory,
          num_epochs=args.epoch, eval_time_in_epoch=args.eval_time,
          best_valid_acc=args.best_valid_accuracy)
    print('Finished Training!')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
