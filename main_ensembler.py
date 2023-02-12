import argparse
from loading_utils import prepare_data, prepare_af_data
from utils import create_logger, save_args, create_trainer, load_config, save_config
from ensembler import Ensembler
import numpy as np
import torch
import random

def main():
    # initialize a parser
    parser = argparse.ArgumentParser()
    # argument dataset accepts a string, defaultis IMDB
    parser.add_argument('--dataset', type=str, default='Yoruba', choices=['Yoruba', 'Hausa'])
    # the directory storing data
    parser.add_argument('--data_root', type=str, default="")
    # the directory storing the logs during training/testing
    parser.add_argument('--log_root', type=str, default="",
                        help='output directory to save logs in training/testing')
    # which trainer to use. 
    # chose between number of bert models to use
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        choices=['bert-base-uncased', 'bert-base-cased',
                                 'bert-large-uncased', 'bert-base-multilingual-cased'],
                        help='backbone selection')
    # Preprocessing Related
    # max sentence length
    parser.add_argument('--max_sen_len', type=int, default=512,
                        help='max sentence length, longer sentences will be truncated')
    # special tokens used in bert tokenizer
    parser.add_argument('--special_token_offsets', type=int, default=2,
                        help='number of special tokens used in bert tokenizer for text classification')
    # truncate mode ?
    parser.add_argument('--truncate_mode', type=str, default='last',
                        choices=['hybrid, last'], help='last: last 510 tokens, hybrid: first 128 + last 382')
    parser.add_argument('--manualSeed', type=int, default=1234, help='random seed for reproducibility')
    parser.add_argument('--noise_type', default='uniform_m',
                        choices=['uniform_m', 'sflip'],
                        help='noise types: uniform_m: uniform noise, sflip: single-flip noise')
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help='noise level for injected noise')
    parser.add_argument('--nl_batch_size', type=int, default=16,
                        help='noisy labeled samples per batch, can be understood as the training batch size')
    parser.add_argument('--noisy_label_seed', type=int, default=1234, help='random seed for reproducibility')
    parser.add_argument('--bert_dropout_rate', type=float, default=0.1)
    parser.add_argument('--eval_batch_size', type=int, default=50,
                        help='evaluation batch size during testing')
    parser.add_argument('--fast_eval', action='store_true',
                        help='use 10% of the test set for evaluation, to speed up the evaluation prcoess')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='freeze the bert backbone, i.e. use bert as feature extractor')

    # run as ensembler 
    parser.add_argument('--model_names', nargs="+", action='store', help='models to create ensembles with')

    args = parser.parse_args()

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.backends.cudnn.benchmark = False

    # Create the Handler for logging records/messages to a file
    logger, log_dir = create_logger(args.log_root, args)
    save_args(log_dir, args)
    logger.info("Inference started")
    num_classes_map = {'Yoruba':7, 'Hausa':5}

    logger.info(f'log dir: {log_dir}')
    num_classes = num_classes_map[args.dataset]
    r_state = np.random.RandomState(args.noisy_label_seed)

    if args.dataset in ['Yoruba', 'Hausa']:
        has_ul=False
        nl_set, _, _, t_set, _, _ = prepare_af_data(args, logger, num_classes, has_ul)
    else:
        raise NotImplementedError(f"dataset {args.dataset} not supported")

    model_config = load_config(args)
    model_config['num_classes'] = num_classes

    # create an ensemble class 
    ensemble = Ensembler(args.model_names, args.eval_batch_size, logger, args, model_config)
    ensemble.sort_items_by_FM_loss(nl_set)
    # sort items in descending order of loss 

if __name__=='__main__':
    main()