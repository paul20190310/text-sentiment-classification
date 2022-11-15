# -*- coding: utf-8 -*-

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import argparse
import os


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_parser():
    parser = argparse.ArgumentParser(description='test model by testing dataset')
    parser.add_argument('model_path', type=str,
                        help='specify the path of the pytorch model saving file')
    parser.add_argument('dataset_path', type=str,
                        help='specify the path of the testing dataset file')
    parser.add_argument('-tp', '--tokenizer-path', type=str, default='./tokenizer',
                        help='specify the path of pretraining tokenizer saving file')
    parser.add_argument('-b', '--batch-size', type=int, default=4,
                        help='batch size of testing loader, default 4')
    return parser

def load_data(data_file_path, tokenizer_path, batch_size, is_shuffle=False):
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    dataset = Dataset.from_csv(data_file_path)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    return DataLoader(tokenized_dataset, shuffle=is_shuffle, batch_size=batch_size)

def test(model, test_loader):
    metric = load_metric("accuracy")
    testing_loss = 0.0
    
    model.eval()
    for batch in tqdm(test_loader, leave=True):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        testing_loss += loss.item()
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    average_testing_loss = testing_loss / len(test_loader)
    accuracy = metric.compute()['accuracy']
    print('loss:', average_testing_loss)
    print('accuracy:', accuracy)
    return average_testing_loss, accuracy

def main(args):
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    test_dataloader = load_data(args.dataset_path,
                                args.tokenizer_path,
                                args.batch_size)
    print('Start Testing!')
    test(model, test_loader=test_dataloader)
    print('Finished Testing!')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
