# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='split .csv file dataset into trainning set and validation set')
    parser.add_argument('filename', type=str,
                        help='specify destination .csv file')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('-df', '--destination-folder', type=str, default='./data',
                        help='Specify the destination folder')
    parser.add_argument('-dtrain', '--destination-train-name', type=str, default='train.csv',
                        help='Specify the destination training set filename')
    parser.add_argument('-dvalid', '--destination-valid-name', type=str, default='valid.csv',
                        help='Specify the destination validation set filename')
    parser.add_argument('-r', '--train-ratio', type=float, default=0.80,
                        help='Specify the ratio of the training set')
    parser.add_argument('-t', '--split-test', action='store_true',
                        help='After split the validation set, split the validation set into half to become the test set')
    parser.add_argument('-dtest', '--destination-test-name', type=str, default='test.csv',
                        help='Specify the destination testing set filename')
    return parser

def split_data(filename,
               train_ratio,
               seed,
               destination_folder,
               destination_train_filename,
               destination_valid_filename,
               destination_test_filename,
               split_test = False):
    df_raw = pd.read_csv(filename)  # Read raw data
    df_train, df_valid = train_test_split(
        df_raw,
        train_size = train_ratio,
        random_state = seed
    )
    
    if split_test:
        df_valid, df_test = train_test_split(
            df_valid,
            train_size = 0.5,
            random_state = seed
        )
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    df_train.to_csv(destination_folder + '/' + destination_train_filename, index=False)
    df_valid.to_csv(destination_folder + '/' + destination_valid_filename, index=False)
    if split_test:
        df_test.to_csv(destination_folder + '/' + destination_test_filename, index=False)

def main(args):
    split_data(filename=args.filename, train_ratio=args.train_ratio,
               seed=args.seed, destination_folder=args.destination_folder,
               destination_train_filename=args.destination_train_name,
               destination_valid_filename=args.destination_valid_name,
               destination_test_filename=args.destination_test_name,
               split_test=args.split_test)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
