import argparse
import os
from os import path
import hashlib
import logging
from collections import OrderedDict

from auto_memory_model.experiment import Experiment
from mention_model.utils import get_mention_model_name

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument(
        '-base_data_dir', default='../data/', help='Root directory of data', type=str)
    parser.add_argument('-base_model_dir', default='../models',
                        help='Root folder storing model runs', type=str)
    parser.add_argument(
        '-dataset', default='litbank', choices=['litbank', 'ontonotes'], type=str)
    parser.add_argument(
        '-conll_scorer', type=str, help='Root folder storing model runs',
        default="../resources/lrec2020-coref/reference-coreference-scorers/scorer.pl")

    parser.add_argument('-model_size', default='large', type=str,
                        help='BERT model type')
    parser.add_argument('-doc_enc', default='overlap', type=str,
                        choices=['independent', 'overlap'], help='BERT model type')
    parser.add_argument('-pretrained_bert_dir', default='../resources', type=str,
                        help='SpanBERT model location')
    parser.add_argument('-max_segment_len', default=512, type=int,
                        help='Max segment length of BERT segments.')

    # Mention variables
    parser.add_argument('-max_span_width', default=20, type=int, help='Max span width.')
    parser.add_argument('-ment_emb', default='attn', choices=['attn', 'endpoint'], type=str)
    parser.add_argument('-top_span_ratio', default=0.3, type=float,
                        help='Ratio of top spans proposed as mentions.')

    # Memory variables
    parser.add_argument('-mem_type', default='learned',
                        choices=['learned', 'lru', 'unbounded', 'unbounded_no_ignore'],
                        help="Memory type.")
    parser.add_argument('-num_cells', default=20, type=int,
                        help="Number of memory cells.")
    parser.add_argument('-mlp_size', default=3000, type=int,
                        help='MLP size used in the model')
    parser.add_argument('-mlp_depth', default=1, type=int,
                        help='Number of hidden layers in other MLPs')
    parser.add_argument('-entity_rep', default='wt_avg', type=str,
                        choices=['learned_avg', 'wt_avg'], help='Entity representation.')
    parser.add_argument('-emb_size', default=20, type=int,
                        help='Embedding size of features.')

    # Training params
    parser.add_argument('-cross_val_split', default=0, type=int,
                        help='Cross validation split to be used.')
    parser.add_argument('-new_ent_wt', help='Weight of new entity term in coref loss',
                        default=1.0, type=float)
    parser.add_argument('-num_train_docs', default=None, type=int,
                        help='Number of training docs.')
    parser.add_argument('-num_eval_docs', default=None, type=int,
                        help='Number of evaluation docs.')
    parser.add_argument('-max_training_segments', default=None, type=int,
                        help='Maximum number of BERT segments in a document.')
    parser.add_argument('-sample_invalid', help='Sample prob. of invalid mentions during training',
                        default=0.2, type=float)
    parser.add_argument('-dropout_rate', default=0.3, type=float,
                        help='Dropout rate')
    parser.add_argument('-label_smoothing_wt', default=0.0, type=float,
                        help='Label Smoothing')
    parser.add_argument('-max_epochs',
                        help='Maximum number of epochs', default=30, type=int)
    parser.add_argument('-seed', default=0,
                        help='Random seed to get different runs', type=int)
    parser.add_argument('-init_lr', help="Initial learning rate",
                        default=2e-4, type=float)
    parser.add_argument('-train_with_singletons', help="Train on singletons.",
                        default=False, action="store_true")
    parser.add_argument('-eval', help="Evaluate model",
                        default=False, action="store_true")
    parser.add_argument('-slurm_id', help="Slurm ID",
                        default=None, type=str)

    args = parser.parse_args()

    if args.dataset == 'litbank':
        args.max_span_width = 20
    elif args.dataset == 'ontonotes':
        args.max_span_width = 30
    else:
        args.max_span_width = 20

    # Get model directory name
    opt_dict = OrderedDict()
    # Only include important options in hash computation
    imp_opts = ['model_size', 'max_segment_len',  # Encoder params
                'ment_emb', "doc_enc", 'max_span_width', 'top_span_ratio', # Mention model
                'mem_type', 'num_cells', 'entity_rep', 'mlp_size', 'mlp_depth',  # Memory params
                'dropout_rate', 'seed', 'init_lr',
                "new_ent_wt", 'sample_invalid',  'max_training_segments', 'label_smoothing_wt',  # weights & sampling
                'num_train_docs', 'cross_val_split', 'train_with_singletons',   # Dataset params
                ]
    for key, val in vars(args).items():
        if key in imp_opts:
            opt_dict[key] = val

    str_repr = str(opt_dict.items())
    hash_idx = hashlib.md5(str_repr.encode("utf-8")).hexdigest()
    model_name = f"coref_{args.dataset}_" + str(hash_idx)

    model_dir = path.join(args.base_model_dir, model_name)
    args.model_dir = model_dir
    print(model_dir)
    best_model_dir = path.join(model_dir, 'best_models')
    args.best_model_dir = best_model_dir
    if not path.exists(model_dir):
        os.makedirs(model_dir)
    if not path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    if args.dataset == 'litbank':
        args.data_dir = path.join(args.base_data_dir, f'{args.dataset}/{args.doc_enc}/{args.cross_val_split}')
        args.conll_data_dir = path.join(args.base_data_dir, f'{args.dataset}/conll/{args.cross_val_split}')
    elif args.dataset == 'ontonotes':
        if args.train_with_singletons:
            enc_str = "_singletons"
        else:
            enc_str = ""
        args.data_dir = path.join(args.base_data_dir, f'{args.dataset}/{args.doc_enc}{enc_str}')
        args.conll_data_dir = path.join(args.base_data_dir, f'{args.dataset}/conll')

    print(args.data_dir)

    # Get mention model name
    args.pretrained_mention_model = path.join(
        path.join(args.base_model_dir, get_mention_model_name(args)), "best_models/model.pth")
    print(args.pretrained_mention_model)

    # Log directory for Tensorflow Summary
    log_dir = path.join(model_dir, "logs")
    if not path.exists(log_dir):
        os.makedirs(log_dir)

    config_file = path.join(model_dir, 'config')
    with open(config_file, 'w') as f:
        for key, val in opt_dict.items():
            logging.info('%s: %s' % (key, val))
            f.write('%s: %s\n' % (key, val))

    Experiment(args, **vars(args))


if __name__ == "__main__":
    main()
