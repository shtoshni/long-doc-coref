import argparse
import os
from os import path
import hashlib
import logging
import subprocess
from collections import OrderedDict

from experiment import Experiment
from mention_model.utils import get_mention_model_name

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '-base_data_dir', default='../../data/', help='Root directory of data', type=str)
    parser.add_argument('-base_model_dir', default='../../models',
                        help='Root folder storing model runs', type=str)
    parser.add_argument(
        '-dataset', default='litbank', choices=['litbank', 'ontonotes'], type=str)
    parser.add_argument(
        '-conll_scorer', type=str, help='Root folder storing model runs',
        default="../../lrec2020-coref/reference-coreference-scorers/scorer.pl")

    parser.add_argument('-model_size', default='base', type=str,
                        help='BERT model type')
    parser.add_argument('-doc_enc', default='overlap', type=str,
                        choices=['independent', 'overlap'], help='BERT model type')
    parser.add_argument('-pretrained_bert_dir', default='../../resources', type=str,
                        help='SpanBERT model location')
    parser.add_argument('-max_segment_len', default=512, type=int,
                        help='Max segment length of BERT segments.')

    # Mention variables
    parser.add_argument('-max_span_width', default=20, type=int,
                        help='Max span width.')
    parser.add_argument('-ment_emb', default='endpoint', choices=['attn', 'endpoint'],
                        type=str, help='If true use an RNN on top of mention embeddings.')
    parser.add_argument('-top_span_ratio', default=0.3, type=float,
                        help='Ratio of top spans proposed as mentions.')
    parser.add_argument('-train_span_model', default=False, action="store_true",
                        help="Whether to keep the mention detection model frozen or fine-tuned.")

    # Memory variables
    parser.add_argument('-mem_type', default='fixed_mem',
                        choices=['fixed_mem', 'lru', 'unbounded'],
                        help="Memory type.")
    parser.add_argument('-num_cells', default=20, type=int,
                        help="Number of memory cells.")
    parser.add_argument('-mem_size', default=None, type=int,
                        help='Memory size used in the model')
    parser.add_argument('-mlp_size', default=1000, type=int,
                        help='MLP size used in the model')
    parser.add_argument('-coref_mlp_depth', default=1, type=int,
                        help='Number of hidden layers in Coref MLP')
    parser.add_argument('-mlp_depth', default=1, type=int,
                        help='Number of hidden layers in other MLPs')
    parser.add_argument('-entity_rep', default='avg', type=str,
                        choices=['lstm', 'gru', 'max', 'avg', 'rec_avg'],
                        help='Entity representation.')
    parser.add_argument('-emb_size', default=20, type=int,
                        help='Embedding size of features.')
    parser.add_argument('-use_last_mention', default=False, action="store_true",
                        help="Use last mention along with the global features if True.")

    # Training params
    parser.add_argument('-cross_val_split', default=0, type=int,
                        help='Cross validation split to be used.')
    parser.add_argument('--batch_size', '-bsize',
                        help='Batch size', default=1, type=int)
    parser.add_argument('-new_ent_wt', help='Weight of new entity term in coref loss',
                        default=1.0, type=float)
    parser.add_argument('-ignore_wt', help='Weight of Ignore term in cross entropy loss',
                        default=1.0, type=float)
    parser.add_argument('-over_loss_wt', help='Weight of overwrite loss',
                        default=1.0, type=float)
    parser.add_argument('-num_train_docs', default=None, type=int,
                        help='Number of training docs.')
    parser.add_argument('-sample_ignores', help='Sample ignores during training',
                        default=1.0, type=float)
    parser.add_argument('-dropout_rate', default=0.3, type=float,
                        help='Dropout rate')
    parser.add_argument('-checkpoint', help="Use checkpoint",
                        default=False, action="store_true")
    parser.add_argument('-max_epochs',
                        help='Maximum number of epochs', default=30, type=int)
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
    imp_opts = ['model_size', 'max_segment_len',  # Encoder params
                'ment_emb', "doc_enc", 'max_span_width', 'top_span_ratio', 'train_span_model',  # Mention model
                'mem_type', 'num_cells', 'mem_size', 'entity_rep', 'mlp_size', 'mlp_depth',
                'coref_mlp_depth', 'emb_size', 'use_last_mention',  # Memory params
                'max_epochs', 'dropout_rate', 'batch_size', 'seed', 'init_lr',
                'ignore_wt', 'over_loss_wt',  "new_ent_wt", 'sample_ignores',  # weights & sampling
                'dataset', 'num_train_docs', 'cross_val_split',   # Training params
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

    if args.dataset == 'litbank':
        args.data_dir = path.join(args.base_data_dir, f'{args.dataset}/{args.doc_enc}/{args.cross_val_split}')
    else:
        args.data_dir = path.join(args.base_data_dir, f'{args.dataset}/{args.doc_enc}')

    args.conll_data_dir = path.join(
        args.base_data_dir, "litbank_tenfold_splits/{}".format(args.cross_val_split))
    print(args.data_dir)

    # Get mention model name
    args.pretrained_mention_model = path.join(
        path.join(args.base_model_dir, get_mention_model_name(args)), "best_models/model.pth")
    print(args.pretrained_mention_model)

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
