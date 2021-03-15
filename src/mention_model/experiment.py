import sys
from os import path

import time
import logging
import torch
from collections import defaultdict, OrderedDict
from copy import deepcopy

import numpy as np
import pytorch_utils.utils as utils
from mention_model.controller import Controller
from data_utils.utils import load_data
from coref_utils.utils import remove_singletons

EPS = 1e-8
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class Experiment:
    def __init__(self, data_dir=None,
                 model_dir=None, best_model_dir=None, pretrained_model=None,
                 # Model params
                 seed=0, init_lr=1e-3, max_gradient_norm=5.0,
                 max_epochs=20, max_segment_len=128, eval=False,
                 num_train_docs=None, sample_ontonotes_prob=0.0,
                 singleton_file=None,
                 # Other params
                 slurm_id=None, train_with_singletons=True,
                 **kwargs):

        self.pretrained_model = pretrained_model
        self.slurm_id = slurm_id
        # Set the random seed first
        self.seed = seed
        # Prepare data info
        self.train_examples = {}
        self.dev_examples = {}
        self.test_examples = {}
        for cur_dataset, dataset_dir in data_dir.items():
            train_examples, dev_examples, test_examples \
                = load_data(dataset_dir, max_segment_len, dataset=cur_dataset, singleton_file=singleton_file)
            if num_train_docs is not None:
                train_examples = train_examples[:num_train_docs]

            self.train_examples[cur_dataset] = train_examples
            self.dev_examples[cur_dataset] = dev_examples
            self.test_examples[cur_dataset] = test_examples

        self.data_iter_map = {"train": self.train_examples,
                              "dev": self.dev_examples,
                              "test": self.test_examples}

        self.sample_ontonotes_prob = sample_ontonotes_prob

        if train_with_singletons is False:
            print("Removing singletons")
            self.train_examples, self.dev_examples, self.test_examples = \
                [remove_singletons(x) for x in [self.train_examples, self.dev_examples, self.test_examples]]

        # Get model paths
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model_path = path.join(model_dir, 'model.pth')
        self.best_model_path = path.join(best_model_dir, 'model.pth')

        # Initialize model and training metadata
        self.model = Controller(**kwargs)
        self.model = self.model.cuda()

        self.initialize_setup(init_lr=init_lr)
        self.model = self.model.cuda()
        utils.print_model_info(self.model)

        if not eval:
            if self.pretrained_model is not None:
                model_state_dict = torch.load(self.pretrained_model)
                print(model_state_dict.keys())
                self.model.load_state_dict(model_state_dict, strict=False)
                self.eval_model(split='dev')
                # self.eval_model(split='test')
                sys.exit()
            else:
                self.train(max_epochs=max_epochs,
                           max_gradient_norm=max_gradient_norm)
        # Finally evaluate model
        self.final_eval(model_dir)

    def initialize_setup(self, init_lr, lr_decay=10):
        """Initialize model and training info."""
        self.train_info = {}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr, eps=1e-6)
        self.optim_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3,
            min_lr=0.1 * init_lr, verbose=True)
        self.train_info['epoch'] = 0
        self.train_info['val_perf'] = 0.0
        self.train_info['threshold'] = 0.0
        self.train_info['global_steps'] = 0

        expected_training_examples = (len(self.train_examples['litbank'])
                                      + int(self.sample_ontonotes_prob * len(self.train_examples['ontonotes'])))
        print("Expected number of training steps:", expected_training_examples)

        if not path.exists(self.model_path):
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            if self.pretrained_model is not None:
                checkpoint = torch.load(self.pretrained_model)
                self.model.load_state_dict(checkpoint, strict=False)
        else:
            logging.info('Loading previous model: %s' % self.model_path)
            # Load model
            self.load_model(self.model_path)

    def train(self, max_epochs, max_gradient_norm):
        """Train model"""
        model = self.model
        epochs_done = self.train_info['epoch']
        optimizer = self.optimizer
        scheduler = self.optim_scheduler

        for epoch in range(epochs_done, max_epochs):
            print("\n\nStart Epoch %d" % (epoch + 1))
            start_time = time.time()
            model.train()

            num_ontonotes_examples = int(self.sample_ontonotes_prob * len(self.train_examples['ontonotes']))
            train_examples = deepcopy(self.train_examples['litbank'])
            if num_ontonotes_examples > 0:
                np.random.shuffle(self.train_examples['ontonotes'])
                train_examples += self.train_examples['ontonotes'][:num_ontonotes_examples]
            np.random.shuffle(train_examples)

            for idx, cur_example in enumerate(train_examples):
                def handle_example(train_example):
                    self.train_info['global_steps'] += 1
                    loss = model(train_example)
                    total_loss = loss['mention']

                    if torch.isnan(total_loss):
                        print("Loss is NaN")
                        sys.exit()
                    # Backprop
                    optimizer.zero_grad()
                    total_loss.backward()
                    # Perform gradient clipping and update parameters
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_gradient_norm)

                    optimizer.step()

                handle_example(deepcopy(cur_example))

                if (idx + 1) % 50 == 0:
                    print("Steps %d, Max memory %.3f" % (idx + 1, (torch.cuda.max_memory_allocated() / (1024 ** 3))))
                    torch.cuda.reset_peak_memory_stats()
                    # print("Current memory %.3f" % (torch.cuda.memory_allocated() / (1024 ** 3)))
                    # print(torch.cuda.memory_summary())

            # Update epochs done
            self.train_info['epoch'] = epoch + 1
            # Development set performance
            fscore, threshold = self.eval_model()

            scheduler.step(fscore)

            # Update model if validation performance improves
            if fscore > self.train_info['val_perf']:
                self.train_info['val_perf'] = fscore
                self.train_info['threshold'] = threshold
                logging.info('Saving best model')
                self.save_model(self.best_model_path)

            # Save model
            self.save_model(self.model_path)

            # Get elapsed time
            elapsed_time = time.time() - start_time
            logging.info("Epoch: %d, Time: %.2f, F-score: %.3f"
                         % (epoch + 1, elapsed_time, fscore))

            sys.stdout.flush()

    def eval_preds(self, pred_mention_probs, gold_mentions, threshold=0.5):
        pred_mentions = (pred_mention_probs >= threshold).float()
        total_corr = torch.sum(pred_mentions * gold_mentions)

        return total_corr, torch.sum(pred_mentions), torch.sum(gold_mentions)

    def eval_model(self, split='dev', dataset='litbank', threshold=None):
        """Eval model"""
        # Set the random seed to get consistent results
        model = self.model
        model.eval()

        dev_examples = self.data_iter_map[split][dataset]

        with torch.no_grad():
            total_recall = 0
            total_gold = 0.0
            all_golds = 0.0
            # Output file to write the outputs
            agg_results = {}
            for dev_example in dev_examples:
                preds, y, cand_starts, cand_ends, recall = model(dev_example)

                all_golds += sum([len(cluster) for cluster in dev_example["clusters"]])
                total_gold += torch.sum(y).item()
                total_recall += recall

                if threshold:
                    corr, total_preds, total_y = self.eval_preds(
                        preds, y, threshold=threshold)
                    if threshold not in agg_results:
                        agg_results[threshold] = defaultdict(float)

                    x = agg_results[threshold]
                    x['corr'] += corr
                    x['total_preds'] += total_preds
                    x['total_y'] += total_y
                    prec = x['corr']/(x['total_preds'] + EPS)
                    recall = x['corr']/x['total_y']
                    x['fscore'] = 2 * prec * recall/(prec + recall + EPS)
                else:
                    threshold_range = np.arange(0.0, 1.0, 0.05)
                    for cur_threshold in threshold_range:
                        corr, total_preds, total_y = self.eval_preds(
                            preds, y, threshold=cur_threshold)
                        if cur_threshold not in agg_results:
                            agg_results[cur_threshold] = defaultdict(float)

                        x = agg_results[cur_threshold]
                        x['corr'] += corr
                        x['total_preds'] += total_preds
                        x['total_y'] += total_y
                        prec = x['corr']/x['total_preds']
                        recall = x['corr']/x['total_y']
                        x['fscore'] = 2 * prec * recall/(prec + recall + EPS)

        if threshold:
            max_fscore = agg_results[threshold]['fscore']
        else:
            max_fscore, threshold = 0, 0.0
            for key in agg_results:
                if agg_results[key]['fscore'] > max_fscore:
                    max_fscore = agg_results[key]['fscore']
                    threshold = key

            logging.info("Max F-score: %.3f, Threshold: %.3f" %
                         (max_fscore, threshold))
        print(total_recall, total_gold)
        print(total_recall, all_golds)
        logging.info("Recall: %.3f" % (total_recall/total_gold))
        return max_fscore, threshold

    def final_eval(self, model_dir):
        """Evaluate the model on train, dev, and test"""
        # Test performance  - Load best model
        self.load_model(self.best_model_path)
        logging.info("Loading best model after epoch: %d" %
                     self.train_info['epoch'])
        logging.info("Threshold: %.3f" % self.train_info['threshold'])
        threshold = self.train_info['threshold']

        perf_file = path.join(self.model_dir, "perf.txt")
        with open(perf_file, 'w') as f:
            for dataset in ['litbank', 'ontonotes']:
                for split in ['Dev', 'Test']:
                    logging.info('\n')
                    logging.info('%s' % split)
                    split_f1, _ = self.eval_model(
                        split.lower(), dataset=dataset, threshold=threshold)
                    logging.info('Calculated F1 (%s): %.3f' % (dataset, split_f1))

                    f.write("%s\t%.4f\n" % (split, split_f1))
                logging.info("Final performance summary at %s" % perf_file)

        sys.stdout.flush()

    def load_model(self, location):
        checkpoint = torch.load(location)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.optimizer.load_state_dict(
            checkpoint['optimizer'])
        self.optim_scheduler.load_state_dict(
            checkpoint['scheduler'])
        self.train_info = checkpoint['train_info']
        torch.set_rng_state(checkpoint['rng_state'])

    def save_model(self, location):
        """Save model"""
        model_state_dict = OrderedDict(self.model.state_dict())
        for key in self.model.state_dict():
            if 'bert.' in key:
                del model_state_dict[key]
        torch.save({
            'train_info': self.train_info,
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.optim_scheduler.state_dict(),
            'rng_state': torch.get_rng_state(),
        }, location)
        logging.info("Model saved at: %s" % (location))
