# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchtext.data import TabularDataset, BucketIterator

from tqdm.auto import tqdm
import argparse

from Mylstm import MyLSTM
from save_load_util import load_model


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_parser():
    parser = argparse.ArgumentParser(description='test model by testing dataset')
    parser.add_argument('model_path', type=str,
                        help='specify the path of the pytorch model saving file')
    parser.add_argument('dataset_path', type=str,
                        help='specify the path of the testing dataset file')
    parser.add_argument('-tf', '--text-field-path', type=str, default='./field/text_field.pth',
                        help='specify the path of the text field saving file')
    parser.add_argument('-lf', '--label-field-path', type=str, default='./field/label_field.pth',
                        help='specify the path of the label field saving file')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='batch size of testing loader, default 32')
    return parser

def test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    correct, total_data_size, loss = 0, 0, 0.0
    
    model.eval()
    with torch.no_grad():
        for ((text, text_len), labels), _ in tqdm(test_loader, leave=True):
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to('cpu')
            outputs = model(text, text_len)
            
            loss += criterion(outputs, labels).item()
            _, predicted_label = torch.max(outputs.data, 1)
            total_data_size += labels.size(0)
            correct += (predicted_label == labels).sum().item()
    accuracy = correct / total_data_size
    loss = loss / len(test_loader)
    print('accuracy: {:.4f}'.format(accuracy))
    print('loss: {:.4f}'.format(loss))
    return accuracy, loss

def main(args):
    text_field = torch.load(args.text_field_path)
    label_field = torch.load(args.label_field_path)
    fields = [('text', text_field), ('label', label_field)]

    test_data = TabularDataset(path=args.dataset_path,
                               format='CSV', fields=fields, skip_header=True)
    test_iter = BucketIterator(test_data, batch_size=args.batch_size,
                               sort_key=lambda x: len(x.text),
                               device=device, sort=True, sort_within_batch=True)
    
    model = MyLSTM().to(device)
    load_model(args.model_path, model, print_prompts=False)
    test(model, test_iter)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
