# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import argparse
from save_load_util import load_metrics


def get_parser():
    parser = argparse.ArgumentParser(description='draw the trainning loss by metrics file')
    parser.add_argument('metrics_path', type=str,
                        help='specify the path of the metrics file')
    parser.add_argument('-x', '--x-axis-str', type=str, default='Global Steps',
                        help='specify string show at x-axis')
    parser.add_argument('-s', '--saving-path', type=str, default=None,
                        help='specify saving path of loss chart')
    return parser

def draw_loss_chart(metrics_file_path, x_axis_str, saving_path):
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(metrics_file_path)
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel(x_axis_str)
    plt.ylabel('Loss')
    plt.legend()
    if saving_path:
        plt.savefig(saving_path)
    else:
        plt.show()

def main(args):
    draw_loss_chart(args.metrics_path, args.x_axis_str, args.saving_path)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)