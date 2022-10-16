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
    return parser

def draw_loss_chart(metrics_file_path, x_axis_str):
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(metrics_file_path)
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel(x_axis_str)
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main(args):
    draw_loss_chart(args.metrics_path, args.x_axis_str)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)