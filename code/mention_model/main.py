import argparse
import os
from os import path
import hashlib
import logging

from collections import OrderedDict
import subprocess

from experiment import Experiment

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '-data_dir', default='/home/shtoshni/Research/litbank_coref/data',
        help='Root directory of data', type=str)
    parser.add_argument('-base_model_dir',
                        default='/home/shtoshni/Research/litbank_coref/models',
                        help='Root folder storing model runs', type=str)
    parser.add_argument('-model', default='base', type=str,
                        help='BERT model type')
    parser.add_argument('-max_span_length',
                        help='Max span length', default=20, type=int)

    parser.add_argument('-hsize', default=300, type=int,
                        help='Hidden size used in the model')
    parser.add_argument('-enc_rnn', default=False, action="store_true",
                        help='If true use an RNN on top of BERT embeddings.')

    parser.add_argument('--batch_size', '-bsize',
                        help='Batch size', default=1, type=int)
    parser.add_argument('-single_window', default=False, action='store_true',
                        help='When true, do training on the first BERT window.')
    parser.add_argument('-feedback', default=False, action='store_true',
                        help='When true, do training on less data.')
    parser.add_argument('-dropout_rate', default=0.1, type=float,
                        help='Dropout rate')
    parser.add_argument('--max_epochs', '-mepochs',
                        help='Maximum number of epochs', default=50, type=int)
    parser.add_argument('-seed', default=0,
                        help='Random seed to get different runs', type=int)
    parser.add_argument('-init_lr', help="Initial learning rate",
                        default=0.001, type=float)
    parser.add_argument('-eval', help="Evaluate model",
                        default=False, action="store_true")
    parser.add_argument('-slurm_id', help="Slurm ID",
                        default=None, type=str)

    args = parser.parse_args()

    # Get model directory name
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    imp_opts = ['model', 'max_span_length', 'hsize', 'enc_rnn', 'single_window',
                'dropout_rate', 'batch_size', 'seed', 'init_lr', 'feedback']
    for key, val in vars(args).items():
        if key in imp_opts:
            opt_dict[key] = val

    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    model_name = "litbank_ment_" + str(hash_idx)

    model_dir = path.join(args.base_model_dir, model_name)
    args.model_dir = model_dir
    best_model_dir = path.join(model_dir, 'best_models')
    args.best_model_dir = best_model_dir
    if not path.exists(model_dir):
        os.makedirs(model_dir)
    if not path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    # Log directory for Tensorflow Summary
    log_dir = path.join(model_dir, "logs")
    if not path.exists(log_dir):
        os.makedirs(log_dir)

    # Slurm args
    if not args.slurm_id:
        p = subprocess.Popen(['tensorboard', '--logdir',  log_dir],
                             stdout=subprocess.PIPE, stderr=None)

    config_file = path.join(model_dir, 'config')
    with open(config_file, 'w') as f:
        for key, val in opt_dict.items():
            logging.info('%s: %s' % (key, val))
            f.write('%s: %s\n' % (key, val))

    try:
        Experiment(**vars(args))
    finally:
        if not args.slurm_id:
            p.kill()


if __name__ == "__main__":
    main()
