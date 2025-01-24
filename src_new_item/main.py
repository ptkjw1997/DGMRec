# coding: utf-8

import os
import argparse
from utils.quick_start import quick_start

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='DGMRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--gpu_id', '-g', type=str, default='1', help='gpu_id')

    parser.add_argument('--missing_modal', type=int, default=1, help='missing_modal')
    parser.add_argument('--missing_ratio', type=str, default='0.666', help='missing_ratio')
    parser.add_argument('--new_items', type=int, default=1, help='new_items')

    args, _ = parser.parse_known_args()
    
    config_dict = {
        'gpu_id': args.gpu_id,
        'missing_modal' : args.missing_modal,
        'missing_ratio' : eval(args.missing_ratio)
    }


    args, _ = parser.parse_known_args()
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=False)