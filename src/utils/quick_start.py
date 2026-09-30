# coding: utf-8

from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True):
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')

    if config.get('new_items'):
        config['inter_file_name'] = f'{config["dataset"]}_del.inter'
        config['item_graph_dict_file'] = 'item_graph_dict_del.npy'

    logger.info(config)

    dataset = RecDataset(config)
    logger.info(str(dataset))

    split_out = dataset.split()
    if config.get('new_items'):
        (train_dataset, valid_dataset, test_dataset), test_dataset_newitem = split_out
    else:
        train_dataset, valid_dataset, test_dataset = split_out
        test_dataset_newitem = test_dataset
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))
    if config.get('new_items'):
        logger.info('\n====Testing (new-item)====\n' + str(test_dataset_newitem))

    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    eval_test_dataset = test_dataset_newitem if config.get('new_items') else test_dataset
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, eval_test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for i, hyper_tuple in enumerate(combinators):
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        train_data.pretrain_setup()
        model = get_model(config['model'])(config, train_data).to(config['device'])
        model.logger = logger
        logger.info(model)

        trainer = get_trainer()(config, model)
        
        print(os.getcwd())
        save_dir = config['save_name'][:-4] + "-" + str(hyper_tuple)
        model.save_dir = save_dir
        
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model, save_dir = save_dir)
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))

    if config.get('result_json'):
        import json
        def _clean(d):
            return {k: float(v) for k, v in d.items()}
        payload = {
            'model': config['model'],
            'dataset': config['dataset'],
            'hyper_parameters': list(config['hyper_parameters']),
            'best_params': [str(x) for x in hyper_ret[best_test_idx][0]],
            'best_valid': _clean(hyper_ret[best_test_idx][1]),
            'best_test': _clean(hyper_ret[best_test_idx][2]),
            'all_runs': [
                {'params': [str(x) for x in p], 'valid': _clean(k), 'test': _clean(v)}
                for (p, k, v) in hyper_ret
            ],
        }
        os.makedirs(os.path.dirname(config['result_json']) or '.', exist_ok=True)
        with open(config['result_json'], 'w') as f:
            json.dump(payload, f, indent=2)
