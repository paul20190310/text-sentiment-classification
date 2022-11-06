# -*- coding: utf-8 -*-

import threading
import subprocess
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='federated learning simulation')
    parser.add_argument('server_command', type=str,
                        help='specify the server running command')
    parser.add_argument('client_command', type=str, nargs='+',
                        help='specify the client running command')
    return parser

def execute_process(command):
    subprocess.call(command.split(), shell=True)

def main(args):
    t_list = []
    t = threading.Thread(target=execute_process, args=[args.server_command])
    t_list.append(t)
    
    for client_command in args.client_command:
        t = threading.Thread(target=execute_process, args=[client_command])
        t_list.append(t)
    
    # start job
    for t in t_list:
        t.start()
    
    # wait job
    for t in t_list:
        t.join()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)