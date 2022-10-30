# -*- coding: utf-8 -*-

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import Dataset
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
    parser.add_argument('-mp', '--model-path', type=str, default='./model',
                        help='specify the path of pretraining model saving file')
    parser.add_argument('-tp', '--tokenizer-path', type=str, default='./tokenizer',
                        help='specify the path of pretraining tokenizer saving file')
    parser.add_argument('-mds', '--model-saving-directory', type=str, default='./model',
                        help='specify working directory to save model, default ./model')
    parser.add_argument('-mts', '--metrics-saving-directory', type=str, default='.',
                        help='specify working directory to save metrics files, default current')
    parser.add_argument('-e', '--epoch', type=int, default=5,
                        help='the number times that the learning algorithm work through the entire training dataset, default 5')
    parser.add_argument('-et', '--eval-time', type=int, default=2,
                        help='the number evaluate times in a epoch, default 2')
    parser.add_argument('-b', '--batch-size', type=int, default=4,
                        help='batch size of training loader, default 4')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                        help='learning rate, default 0.001')
    return parser

def train(model, optimizer, train_loader, valid_loader, num_epochs,
          eval_time_in_epoch, model_save_path, metrics_save_path,
          best_valid_loss = float("Inf")):
    # eval every N step
    eval_every = []
    for i in range(eval_time_in_epoch):
        if i == eval_time_in_epoch - 1:
            eval_every.append(len(train_loader) // eval_time_in_epoch +
                              len(train_loader) % eval_time_in_epoch)
        else:
            eval_every.append(len(train_loader) // eval_time_in_epoch)
    
    # initialize running loss values
    training_loss = 0.0
    validation_loss = 0.0
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
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
            training_loss += loss.item()
            global_step += 1
            
            # evaluation step
            if (len(global_steps_list) == 0 and global_step == eval_every[0]) or \
               (len(global_steps_list) > 0 and global_step - global_steps_list[-1] == eval_every[eval_time]):
                progress_bar.close()
                
                # validation loop
                model.eval()
                with torch.no_grad():      
                    for batch in tqdm(valid_loader):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(**batch)
                        
                        loss = outputs.loss
                        validation_loss += loss.item()
                
                # evaluation
                average_train_loss = training_loss / eval_every[eval_time]
                average_valid_loss = validation_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)
        
                # resetting running loss values
                training_loss = 0.0
                validation_loss = 0.0
                model.train()
        
                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss))
        
                # save model
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    model.save_pretrained(model_save_path)
                    print(f'Model saved to ==> {model_save_path}')
                
                # reset progress_bar
                if global_step < num_epochs * len(train_loader):
                    progress_bar = tqdm(range(eval_every[(eval_time + 1) % eval_time_in_epoch]), leave=True)
                
                eval_time += 1
                
    save_metrics(metrics_save_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

def main(args):
    if not os.path.exists(args.model_path):
        model_name_or_path = "bert-base-uncased"
    else:
        model_name_or_path = args.model_path
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=6,
        id2label={0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'},
        label2id={'sadness':0, 'joy':1, 'love':2, 'anger':3, 'fear':4, 'surprise':5}
    ).to(device)
    
    if not os.path.exists(args.tokenizer_path):
        tokenizer_name_or_path = "bert-base-uncased"
    else:
        tokenizer_name_or_path = args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    train_dataset = Dataset.from_csv(args.train_path)
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text"])
    tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
    tokenized_train_dataset.set_format("torch")
    train_dataloader = DataLoader(tokenized_train_dataset, shuffle=True, batch_size=args.batch_size)
    
    valid_dataset = Dataset.from_csv(args.valid_path)
    tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True)
    tokenized_valid_dataset = tokenized_valid_dataset.remove_columns(["text"])
    tokenized_valid_dataset = tokenized_valid_dataset.rename_column("label", "labels")
    tokenized_valid_dataset.set_format("torch")
    valid_dataloader = DataLoader(tokenized_valid_dataset, shuffle=True, batch_size=args.batch_size)
    
    train(model, optimizer, train_loader=train_dataloader, valid_loader=valid_dataloader,
          model_save_path=args.model_saving_directory,
          metrics_save_path=args.metrics_saving_directory,
          num_epochs=args.epoch, eval_time_in_epoch=args.eval_time)
    print('Finished Training!')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)