# -*- coding: utf-8 -*-

import torch
import argparse
import os
from torchtext.data import Field, TabularDataset


def get_parser():
    parser = argparse.ArgumentParser(description='build data field files by corpus')
    parser.add_argument('corpus_path', type=str,
                        help='specify the path of the corpus file')
    parser.add_argument('-s', '--saving-directory', type=str, default='.',
                        help='specify saving directory for field files, default current')
    return parser

def build_field(corpus_path, save_dir):
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(tokenize='spacy', tokenizer_language='en_core_web_sm',
                       lower=True, include_lengths=True, batch_first=True)
    
    corpus = TabularDataset(path=corpus_path,
                            format='CSV',
                            fields=[('text', text_field)],
                            skip_header=True)
    text_field.build_vocab(corpus, min_freq=3)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(text_field, save_dir + '/text_field.pth')
    torch.save(label_field, save_dir + '/label_field.pth')

def main(args):
    build_field(args.corpus_path, args.saving_directory)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)