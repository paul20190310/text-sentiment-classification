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
    
    if not os.path.exists(args.tokenizer_path):
        tokenizer_name_or_path = "bert-base-uncased"
    else:
        tokenizer_name_or_path = args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    test_dataset = Dataset.from_csv(args.dataset_path)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(["text"])
    tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels")
    tokenized_test_dataset.set_format("torch")
    test_dataloader = DataLoader(tokenized_test_dataset.select(range(200)), shuffle=True, batch_size=args.batch_size)
    
    test(model, test_loader=test_dataloader)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
