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
    parser.add_argument('--missing_ratio', type=str, default=None,
                        help='missing_ratio; defaults to the dataset yaml value '
                             '(0.666 for the Amazon datasets, 0.6 for tiktok)')
    parser.add_argument('--missing_imputation', type=str, default='[1]',
                        help='imputation for missing features (0: zero, 1: mean, 2: NN); list form, swept as a hyper-parameter')
    parser.add_argument('--new_items', type=int, default=0,
                        help='new_items setting (0: missing-only [default], 1: missing + new-item)')
    parser.add_argument('--seed', type=str, default=None,
                        help='override random seed list, e.g. "[999]" or "[999,42]"')
    parser.add_argument('--save_best_model', type=int, default=0,
                        help='save best.pth checkpoint on validation improvement (needed for retrieval eval)')
    parser.add_argument('--config_override', type=str, default=None,
                        help='JSON dict of config keys to override, e.g. \'{"n_mm_layers": [1], "alignBMTemp": 0.2}\'')

    args, _ = parser.parse_known_args()

    config_dict = {
        'gpu_id': args.gpu_id,
        'missing_modal' : args.missing_modal,
        'missing_imputation' : eval(args.missing_imputation),
        'new_items' : args.new_items,
        'save_best_model' : bool(args.save_best_model),
    }
    if args.missing_ratio is not None:
        config_dict['missing_ratio'] = eval(args.missing_ratio)
    if args.seed is not None:
        config_dict['seed'] = eval(args.seed)
    if args.config_override:
        import json
        config_dict.update(json.loads(args.config_override))


    args, _ = parser.parse_known_args()
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=False)
