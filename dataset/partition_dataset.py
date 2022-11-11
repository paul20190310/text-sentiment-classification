# -*- coding: utf-8 -*-

import pandas as pd
import argparse
import os
import random


def get_parser():
    parser = argparse.ArgumentParser(description='split .csv file dataset into partition data for specific conditions')
    parser.add_argument('filename', type=str,
                        help='specify destination .csv file')
    parser.add_argument('-r', '--partition-ratio', type=float, nargs='+',
                        help='Specify the data distribution ratio')
    parser.add_argument('-f', '--partition-file_name', type=str, nargs='+',
                        help='Specify the data destination file name')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('-df', '--destination-folder', type=str, default='./partition',
                        help='Specify the destination folder')
    parser.add_argument('-l', '--label-oriented', action='store_true',
                        help='Whether to use the label as the partition guide.  If this is enabled, --partition-ratio should be number of label for each partition data')
    return parser

def partition_data(filename,
                   partition_ratio,
                   partition_file_name,
                   seed,
                   destination_folder,
                   is_label_oriented = False):
    df_raw = pd.read_csv(filename)  # Read raw data
    df_shuffle = df_raw.sample(frac=1, random_state=seed, ignore_index=True)
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    if not is_label_oriented:
        start_index = 0
        for ratio, filename in zip(partition_ratio, partition_file_name):
            end_index = start_index + int(len(df_shuffle) * ratio)
            if end_index > len(df_shuffle):
                end_index = len(df_shuffle)
            df_partition = df_shuffle[start_index:end_index]
            df_partition.to_csv(destination_folder + '/' + filename, index=False)
            start_index = end_index
    else:
        random.seed(seed)
        label_list = random.sample(list(range(6)), k=6)  # six labels
        label_list.append(-1)
        cur_label = 0
        for label_num, filename in zip(partition_ratio, partition_file_name):
            pd_label = pd.DataFrame()
            for i in range(int(label_num)):
                pd_label = pd.concat([pd_label, df_raw[df_raw['label'] == label_list[cur_label]]])
                if cur_label < 6:
                    cur_label += 1
            pd_label = pd_label.reset_index(drop=True)
            pd_label = pd_label.sample(frac=1, random_state=seed, ignore_index=True)
            pd_label.to_csv(destination_folder + '/' + filename, index=False)

def main(args):
    partition_data(filename=args.filename,
                   partition_ratio=args.partition_ratio,
                   partition_file_name=args.partition_file_name,
                   seed=args.seed,
                   destination_folder=args.destination_folder,
                   is_label_oriented=args.label_oriented)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)