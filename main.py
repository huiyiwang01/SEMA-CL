##################################################################################################################################################################################
# Slight variations in training and model behavior may occur across different hyperparameter settings and computing environments, e.g., package versions and GPU configurations. #
##################################################################################################################################################################################
import json
import argparse
from datetime import datetime
from trainer import train
from eval import eval

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json
    if not args["eval"]:
        train(args)
    else:
        eval(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/sema_inr_10task.json',
                        help='Json file of settings.')
    parser.add_argument('--eval', type=bool, default=False)
    return parser

if __name__ == '__main__':
    main()
