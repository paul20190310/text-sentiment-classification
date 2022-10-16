# -*- coding: utf-8 -*-

import torch
import argparse

from Mylstm import MyLSTM
from save_load_util import load_model


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_parser():
    parser = argparse.ArgumentParser(description='inference model')
    parser.add_argument('model_path', type=str,
                        help='specify the path of the pytorch model saving file')
    parser.add_argument('-s', '--inference-sentence', type=str, default=None,
                        help='sentence for inference')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Enable text input interactive mode inference')
    parser.add_argument('-tf', '--text-field-path', type=str, default='./field/text_field.pth',
                        help='specify the path of the text field saving file')
    return parser

def predict(model, vocab, sentence):
    words = sentence.split()
    words_index = [vocab[word] for word in words]
    text = torch.tensor([words_index]).to(device)
    text_len = torch.tensor([len(text[0])]).long()
    output = model(text, text_len)
    
    _, predicted_label = torch.max(output.data, 1)
    return int(predicted_label)

def get_classification(label):
    classification = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    return classification[label]

def inference(model, vocab, sentence, interactive_mode):
    model.eval()
    if not interactive_mode:
        label = predict(model, vocab, sentence.lower())
        print('The emotion of this sentence is \'', get_classification(label), '\'', sep='')
    else:
        print('Type \'exit()\' to exit')
        sentence = input('Please input a sentence> ')
        while sentence.lower() != 'exit()':
            label = predict(model, vocab, sentence.lower())
            print('The emotion of this sentence is \'', get_classification(label), '\'', sep='')
            sentence = input('Please input a sentence> ')
        print('bye~')

def main(args):
    if not args.interactive and args.inference_sentence == None:
        print('Requires a given inference sentence or -i to enable interactive mode')
    else:
        text_field = torch.load(args.text_field_path)  # load text field
        model = MyLSTM().to(device)
        load_model(args.model_path, model)
        inference(model, text_field.vocab,
                  sentence=args.inference_sentence,
                  interactive_mode=args.interactive)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
