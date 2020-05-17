import argparse
import os
from os import path
import hashlib
import logging
import subprocess
from collections import OrderedDict

from experiment import Experiment

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '-base_data_dir', default='/home/shtoshni/Research/litbank_coref/data/',
        help='Root directory of data', type=str)
    parser.add_argument('-base_model_dir',
                        default='/home/shtoshni/Research/litbank_coref/models',
                        help='Root folder storing model runs', type=str)
    parser.add_argument(
        '-conll_scorer', type=str, help='Root folder storing model runs',
        default="/home/shtoshni/Research/litbank_coref/lrec2020-coref/"
        "reference-coreference-scorers/scorer.pl")

    parser.add_argument('-model', default='base', type=str,
                        help='BERT model type')
    parser.add_argument('-model_loc', default=None, type=str,
                        help='SpanBERT model location')
    parser.add_argument('-max_segment_len', default=512, type=int,
                        help='Max segment length of BERT segments.')
    parser.add_argument('-cross_val_split', default=0, type=int,
                        help='Cross validation split to be used.')

    # Encoder variables
    # parser.add_argument('-hsize', default=400, type=int,
    #                     help='Hidden size used in the model')
    parser.add_argument('-use_doc_rnn', default=False, action="store_true",
                        help='If true use an RNN on top of BERT embeddings.')
    parser.add_argument('-ment_emb', default='endpoint', choices=['attn', 'max', 'endpoint'],
                        type=str, help='If true use an RNN on top of mention embeddings.')

    # Memory variables
    parser.add_argument('-mem_type', default='fixed_mem', choices=['fixed_mem', 'lru'],
                        help="Memory type.")
    parser.add_argument('-num_cells', default=20, type=int,
                        help="Number of memory cells.")
    parser.add_argument('-mem_size', default=None, type=int,
                        help='Memory size used in the model')
    parser.add_argument('-mlp_size', default=1000, type=int,
                        help='MLP size used in the model')
    parser.add_argument('-mlp_depth', default=1, type=int,
                        help='Number of hidden layers')
    parser.add_argument('-entity_rep', default='avg', type=str,
                        choices=['lstm', 'gru', 'max', 'avg'],
                        help='Entity representation.')
    parser.add_argument('-emb_size', default=20, type=int,
                        help='Embedding size of features.')
    parser.add_argument('-use_last_mention', default=False, action="store_true",
                        help="Use last mention along with the global features if True.")

    # Training params
    parser.add_argument('--batch_size', '-bsize',
                        help='Batch size', default=1, type=int)
    parser.add_argument('-over_loss_wt', help='Weight of overwrite loss',
                        default=0.01, type=float)
    parser.add_argument('-max_span_length', default=10,
                        help='Random seed to get different runs', type=int)
    parser.add_argument('-feedback', default=False, action='store_true',
                        help='When true, do training on less data.')
    parser.add_argument('-dropout_rate', default=0.5, type=float,
                        help='Dropout rate')
    parser.add_argument('-max_epochs',
                        help='Maximum number of epochs', default=40, type=int)
    parser.add_argument('-seed', default=0,
                        help='Random seed to get different runs', type=int)
    parser.add_argument('-init_lr', help="Initial learning rate",
                        default=5e-4, type=float)
    parser.add_argument('-no_singletons', help="No singletons.",
                        default=False, action="store_true")
    parser.add_argument('-eval', help="Evaluate model",
                        default=False, action="store_true")
    parser.add_argument('-slurm_id', help="Slurm ID",
                        default=None, type=str)

    args = parser.parse_args()

    # Get model directory name
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    imp_opts = ['model', 'max_segment_len', 'use_doc_rnn',  'ment_emb',  # Encoder params
                'mem_type', 'num_cells', 'mem_size', 'entity_rep', 'mlp_size', 'mlp_depth',
                'emb_size', 'use_last_mention',  # Memory params
                'max_epochs', 'dropout_rate', 'batch_size', 'seed', 'init_lr',
                'feedback', 'cross_val_split', 'over_loss_wt',  # Training params
                ]
    for key, val in vars(args).items():
        if key in imp_opts:
            opt_dict[key] = val

    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    model_name = "autoreg_" + str(hash_idx)

    model_dir = path.join(args.base_model_dir, model_name)
    args.model_dir = model_dir
    best_model_dir = path.join(model_dir, 'best_models')
    args.best_model_dir = best_model_dir
    if not path.exists(model_dir):
        os.makedirs(model_dir)
    if not path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    args.data_dir = path.join(
        args.base_data_dir, f'autoregressive/{args.mem_type}/{args.cross_val_split}/{args.num_cells}')
    # print(args.data_dir)
    args.conll_data_dir = path.join(
        args.base_data_dir, "litbank_tenfold_splits/{}".format(args.cross_val_split))
    print(args.data_dir)
    # Log directory for Tensorflow Summary
    log_dir = path.join(model_dir, "logs")
    if not path.exists(log_dir):
        os.makedirs(log_dir)

    # Slurm args
    if not args.slurm_id:
        tensorboard_process = subprocess.Popen(['tensorboard', '--logdir',  log_dir],
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
            tensorboard_process.kill()


if __name__ == "__main__":
    main()
