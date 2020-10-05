import sys
from os import path
import os
import time
import logging
import torch
import json
from collections import defaultdict, OrderedDict
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils import action_sequences_to_clusters, classify_errors
from data_utils.utils import load_data
from coref_utils.conll import evaluate_conll
from coref_utils.utils import get_mention_to_cluster
from coref_utils.metrics import CorefEvaluator
import pytorch_utils.utils as utils
from auto_memory_model.controller.lfm_controller import LearnedFixedMemController
from auto_memory_model.controller.lru_controller import LRUController
from auto_memory_model.controller.um_controller import UnboundedMemController


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class Experiment:
    def __init__(self, data_dir=None, dataset='litbank', conll_data_dir=None,
                 model_dir=None, best_model_dir=None,
                 pretrained_mention_model=None,
                 # Model params
                 seed=0, init_lr=1e-3, max_gradient_norm=5.0,
                 max_epochs=20, max_segment_len=128, eval=False, num_train_docs=None,
                 mem_type=False,
                 no_singletons=False,
                 # Other params
                 slurm_id=None, conll_scorer=None, **kwargs):
        # Set the random seed first
        self.seed = seed
        self.pretrained_mention_model = pretrained_mention_model

        # Set dataset
        self.dataset = dataset
        if self.dataset == 'litbank':
            self.update_frequency = 10
            self.max_stuck_epochs = 10
        else:
            self.update_frequency = 100
            self.max_stuck_epochs = 10

        self.train_examples, self.dev_examples, self.test_examples \
            = load_data(data_dir, max_segment_len, dataset=self.dataset)
        if num_train_docs is not None:
            self.train_examples = self.train_examples[:num_train_docs]

        self.data_iter_map = {"train": self.train_examples,
                              "dev": self.dev_examples,
                              "test": self.test_examples}
        self.cluster_threshold = (2 if no_singletons else 1)

        self.slurm_id = slurm_id
        self.conll_scorer = conll_scorer

        if not slurm_id:
            # Initialize Summary Writer
            self.writer = SummaryWriter(path.join(model_dir, "logs"),
                                        max_queue=500)
        # Get model paths
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.conll_data_dir = conll_data_dir
        self.model_path = path.join(model_dir, 'model.pth')
        self.best_model_path = path.join(best_model_dir, 'model.pth')

        # Initialize model and training metadata
        if mem_type == 'fixed_mem':
            self.model = LearnedFixedMemController(dataset=dataset, **kwargs).cuda()
        elif mem_type == 'lru':
            self.model = LRUController(dataset=dataset, **kwargs).cuda()
        elif mem_type == 'unbounded':
            self.model = UnboundedMemController(dataset=dataset, **kwargs).cuda()
            # if self.dataset == 'litbank':
            #     self.model = UnboundedMemController(dataset=dataset, **kwargs).cuda()
            # elif self.dataset == 'ontonotes':
            #     self.model = UnboundedMemControllerOntoNotes(dataset=dataset, **kwargs).cuda()
        self.initialize_setup(init_lr=init_lr)
        utils.print_model_info(self.model)
        sys.stdout.flush()

        if not eval:
            self.train(max_epochs=max_epochs,
                       max_gradient_norm=max_gradient_norm)

        # Finally evaluate model
        self.final_eval()

    def initialize_setup(self, init_lr, lr_decay=10):
        """Initialize model and training info."""
        self.train_info = {}
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=init_lr, eps=1e-6)
        self.optim_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3,
            min_lr=0.1 * init_lr, verbose=True)
        self.train_info['epoch'] = 0
        self.train_info['val_perf'] = 0.0
        self.train_info['global_steps'] = 0
        self.train_info['num_stuck_epochs'] = 0

        if not path.exists(self.model_path):
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            # Try to initialize the mention model part
            if path.exists(self.pretrained_mention_model):
                print("Found pretrained model!!")
                checkpoint = torch.load(self.pretrained_mention_model)
                print(checkpoint['model'].keys())
                self.model.load_state_dict(checkpoint['model'], strict=False)
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
        if not self.slurm_id:
            writer = self.writer

        if self.train_info['num_stuck_epochs'] >= self.max_stuck_epochs:
            return

        for epoch in range(epochs_done, max_epochs):
            print("\n\nStart Epoch %d" % (epoch + 1))
            start_time = time.time()
            # Setup training
            model.train()
            np.random.shuffle(self.train_examples)
            batch_loss = 0
            pred_class_counter, gt_class_counter = defaultdict(int), defaultdict(int)
            # errors = OrderedDict([("WL", 0), ("FN", 0), ("WF", 0), ("WO", 0),
            #                       ("FL", 0), ("C", 0)])

            ### CHANGE THIS
            # self.update_frequency = 1
            for cur_example in self.train_examples:
                # print(cur_example["doc_key"], len(cur_example["sentence_map"]))
                if cur_example["doc_key"] == "nw/wsj/20/wsj_2013_0":
                    # Too long
                    continue

                def handle_example(example):
                    self.train_info['global_steps'] += 1
                    loss, pred_action_list, pred_mentions, gt_actions = model(example)

                    for pred_action, gt_action in zip(pred_action_list, gt_actions):
                        pred_class_counter[pred_action[1]] += 1
                        gt_class_counter[gt_action[1]] += 1

                    total_loss = loss['total']
                    if total_loss is None:
                        return None

                    if not self.slurm_id:
                        writer.add_scalar("Loss/Total", total_loss, self.train_info['global_steps'])

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

                    return total_loss.item()

                example_loss = handle_example(cur_example)
                # self.model.memory_net.reporter.report(verbose=True)
                # sys.exit()

                if self.train_info['global_steps'] % self.update_frequency == 0:
                    logging.info('{} {:.3f} Max mem {:.3f} GB'.format(
                        cur_example["doc_key"], example_loss,
                        (torch.cuda.max_memory_allocated() / (1024 ** 3)))
                    )
                    torch.cuda.reset_max_memory_allocated()

            # print("Ground Truth Actions:", gt_class_counter)
            # print("Predicted Actions:", pred_class_counter)

            sys.stdout.flush()
            # Update epochs done
            self.train_info['epoch'] = epoch + 1

            # Evaluate auto regressive performance on dev set
            val_loss = self.eval_auto_reg()
            scheduler.step(val_loss)

            # Dev performance
            fscore = self.eval_model()
            # Save model
            self.save_model(self.model_path)

            # Assume that the model didn't improve
            self.train_info['num_stuck_epochs'] += 1

            # Update model if dev performance improves
            if fscore > self.train_info['val_perf']:
                self.train_info['num_stuck_epochs'] = 0
                self.train_info['val_perf'] = fscore
                logging.info('Saving best model')
                self.save_model(self.best_model_path)

            # Get elapsed time
            elapsed_time = time.time() - start_time
            logging.info("Epoch: %d, F1: %.1f, Max F1: %.1f, Time: %.2f, Val Loss: %.3f"
                         % (epoch + 1, fscore, self.train_info['val_perf'], elapsed_time, val_loss))

            sys.stdout.flush()
            if not self.slurm_id:
                self.writer.flush()

            if self.train_info['num_stuck_epochs'] >= self.max_stuck_epochs:
                return

    def eval_auto_reg(self):
        """Train model"""
        model = self.model
        model.eval()
        errors = OrderedDict([("WL", 0), ("FN", 0), ("WF", 0),
                              ("WO", 0), ("FL", 0), ("C", 0)])  #, ("CM", 0), ("WM", 0)])
        batch_loss = 0
        # pred_class_counter, gt_class_counter = defaultdict(int), defaultdict(int)
        corr_actions, total_actions = 0, 0
        with torch.no_grad():
            for example in self.dev_examples:
                loss, pred_action_list, pred_mentions, gt_actions = model(example, teacher_forcing=True)
                # batch_errors = classify_errors(pred_action_list, gt_actions)
                # for key in errors:
                #     errors[key] += batch_errors[key]
                #
                # for pred_action, gt_action in zip(pred_action_list, gt_actions):
                #     pred_class_counter[pred_action[1]] += 1
                #     gt_class_counter[gt_action[1]] += 1
                #
                #     if tuple(pred_action) == tuple(gt_action):
                #         corr_actions += 1

                total_actions += len(gt_actions)
                total_loss = loss['total']
                batch_loss += total_loss.item()

        # logging.info("Val loss: %.3f" % batch_loss)
        # logging.info("Dev: %s", str(errors))
        # logging.info("(Teacher forced) Action accuracy: %.3f", corr_actions/total_actions)
        model.train()
        return batch_loss/len(self.dev_examples)

    def eval_model(self, split='dev', final_eval=False):
        """Eval model"""
        # Set the random seed to get consistent results
        model = self.model
        model.eval()

        data_iter = self.data_iter_map[split]

        pred_class_counter, gt_class_counter = defaultdict(int), defaultdict(int)
        num_gt_clusters, num_pred_clusters = 0, 0

        with torch.no_grad():
            log_file = path.join(self.model_dir, split + ".log.jsonl")
            with open(log_file, 'w') as f:
                # Capture the auxiliary action accuracy
                corr_actions = 0.0
                total_actions = 0.0

                # Output file to write the outputs
                evaluator = CorefEvaluator()
                oracle_evaluator = CorefEvaluator()
                coref_predictions, subtoken_maps = {}, {}

                for example in data_iter:
                    loss, action_list, pred_mentions, gt_actions = model(example)
                    for pred_action, gt_action in zip(action_list, gt_actions):
                        pred_class_counter[pred_action[1]] += 1
                        gt_class_counter[gt_action[1]] += 1

                        if tuple(pred_action) == tuple(gt_action):
                            corr_actions += 1

                    total_actions += len(action_list)

                    predicted_clusters = action_sequences_to_clusters(action_list, pred_mentions)

                    predicted_clusters, mention_to_predicted =\
                        get_mention_to_cluster(predicted_clusters, threshold=self.cluster_threshold)
                    gold_clusters, mention_to_gold =\
                        get_mention_to_cluster(example["clusters"], threshold=self.cluster_threshold)

                    coref_predictions[example["doc_key"]] = predicted_clusters
                    subtoken_maps[example["doc_key"]] = example["subtoken_map"]

                    # Update the number of clusters
                    num_gt_clusters += len(gold_clusters)
                    num_pred_clusters += len(predicted_clusters)

                    oracle_clusters = action_sequences_to_clusters(gt_actions, pred_mentions)
                    oracle_clusters, mention_to_oracle = \
                        get_mention_to_cluster(oracle_clusters, threshold=self.cluster_threshold)
                    evaluator.update(predicted_clusters, gold_clusters,
                                     mention_to_predicted, mention_to_gold)
                    oracle_evaluator.update(oracle_clusters, gold_clusters,
                                            mention_to_oracle, mention_to_gold)

                    log_example = dict(example)
                    log_example["gt_actions"] = gt_actions
                    log_example["pred_actions"] = action_list
                    log_example["predicted_clusters"] = predicted_clusters

                    f.write(json.dumps(log_example) + "\n")

                print("Ground Truth Actions:", gt_class_counter)
                print("Predicted Actions:", pred_class_counter)

                # Print individual metrics
                indv_metrics_list = ['MUC', 'Bcub', 'CEAFE']
                perf_str = ""
                for indv_metric, indv_evaluator in zip(indv_metrics_list, evaluator.evaluators):
                    perf_str += ", " + indv_metric + ": {:.1f}".format(indv_evaluator.get_f1() * 100)

                prec, rec, fscore = evaluator.get_prf()
                fscore = fscore * 100
                logging.info("F-score: %.1f %s" % (fscore, perf_str))

                if final_eval:
                    gold_path = path.join(self.conll_data_dir, f'{split}.conll')
                    prediction_file = path.join(self.model_dir, f'{split}.conll')
                    conll_results = evaluate_conll(
                        self.conll_scorer, gold_path, coref_predictions, subtoken_maps, prediction_file)
                    average_f1 = sum(results for results in conll_results.values()) / len(conll_results)
                    logging.info("(CoNLL) F-score : %.1f, MUC: %.1f, Bcub: %.1f, CEAFE: %.1f"
                                 % (average_f1, conll_results["muc"], conll_results['bcub'],
                                    conll_results['ceafe']))

                logging.info("Action accuracy: %.3f, Oracle F-score: %.3f" %
                             (corr_actions/total_actions, oracle_evaluator.get_prf()[2]))
                logging.info(log_file)

        return fscore

    def final_eval(self):
        """Evaluate the model on train, dev, and test"""
        # Test performance  - Load best model
        self.load_model(self.best_model_path)
        logging.info("Loading best model after epoch: %d" %
                     self.train_info['epoch'])

        perf_file = path.join(self.model_dir, "perf.txt")
        if self.slurm_id:
            parent_dir = path.dirname(path.normpath(self.model_dir))
            perf_dir = path.join(parent_dir, "perf")
            if not path.exists(perf_dir):
                os.makedirs(perf_dir)
            perf_file = path.join(perf_dir, self.slurm_id + ".txt")

        with open(perf_file, 'w') as f:
            for split in ['Train', 'Dev', 'Test']:
                logging.info('\n')
                logging.info('%s' % split)
                split_f1 = self.eval_model(split.lower(), final_eval=True)
                logging.info('Calculated F1: %.3f' % split_f1)

                f.write("%s\t%.4f\n" % (split, split_f1))
                if not self.slurm_id:
                    self.writer.add_scalar(
                        "F-score/{}".format(split), split_f1)
            logging.info("Final performance summary at %s" % perf_file)

        sys.stdout.flush()
        if not self.slurm_id:
            self.writer.close()

    def load_model(self, location):
        checkpoint = torch.load(location)
        self.model.load_state_dict(checkpoint['model'], strict=False)

        self.optimizer.load_state_dict(
            checkpoint['optimizer'])
        self.optim_scheduler.load_state_dict(
            checkpoint['scheduler'])
        self.train_info = checkpoint['train_info']
        torch.set_rng_state(checkpoint['rng_state'])
        np.random.set_state(checkpoint['np_rng_state'])

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
            'np_rng_state': np.random.get_state()
        }, location)
        # logging.info("Model saved at: %s" % (location))
