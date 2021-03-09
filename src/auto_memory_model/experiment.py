import sys
from os import path
import os
import time
import logging
import torch
import json
from collections import defaultdict, OrderedDict
import numpy as np
from transformers import get_linear_schedule_with_warmup

from auto_memory_model.utils import action_sequences_to_clusters
from data_utils.utils import load_data
from coref_utils.conll import evaluate_conll
from coref_utils.utils import get_mention_to_cluster
from coref_utils.metrics import CorefEvaluator
import pytorch_utils.utils as utils
from coref_utils.utils import remove_singletons
from auto_memory_model.controller.utils import pick_controller
from pytorch_utils.optimization_utils import get_inverse_square_root_decay


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


class Experiment:
    def __init__(self, args, data_dir=None, dataset='litbank',
                 model_dir=None, best_model_dir=None,
                 pretrained_mention_model=None,
                 # Model params
                 seed=0, init_lr=2e-4, max_gradient_norm=10.0,
                 max_epochs=20, max_segment_len=512, eval_model=False,
                 num_train_docs=None, num_eval_docs=None,
                 mem_type="unbounded", train_with_singletons=False,
                 eval_max_ents=None, use_gold_ments=False, use_curriculum=True,
                 lr_decay='linear', warmup_frac=0.0,
                 # Other params
                 slurm_id=None, conll_data_dir=None, conll_scorer=None, **kwargs):
        self.args = args

        # Set the random seed first
        self.seed = seed
        self.model_args = vars(args)
        self.pretrained_mention_model = pretrained_mention_model

        # Cluster threshold is used to determine the minimum size of clusters for metric calculation
        self.dataset = dataset
        self.train_examples, self.dev_examples, self.test_examples \
            = load_data(data_dir, max_segment_len, dataset=self.dataset)
        if num_train_docs is not None:
            self.train_examples = self.train_examples[:num_train_docs]
        if num_eval_docs is not None:
            self.dev_examples = self.dev_examples[:num_eval_docs]
            self.test_examples = self.test_examples[:num_eval_docs]

        self.train_with_singletons = train_with_singletons

        if train_with_singletons:
            self.cluster_threshold = 1
        else:
            self.cluster_threshold = 2
            # Remove singletons from training set
            self.train_examples = remove_singletons(self.train_examples)

        self.canonical_cluster_threshold = 1
        if self.dataset == 'litbank':
            self.update_frequency = 10  # Frequency in terms of # of documents after which logs are printed
            self.max_stuck_epochs = 5  # Maximum epochs without improvement in dev performance
            self.canonical_cluster_threshold = 1
        else:
            # OntoNotes
            self.update_frequency = 100
            self.max_stuck_epochs = 5
            self.canonical_cluster_threshold = 2

        self.data_iter_map = {"train": self.train_examples,
                              "dev": self.dev_examples,
                              "test": self.test_examples}

        self.max_epochs = max_epochs
        self.use_curriculum = use_curriculum

        self.slurm_id = slurm_id  # Useful to keep this around for grid searches

        # CoNLL scorer and data in CoNLL format. Not a requirement as the python script gets pretty much
        # the same numbers.
        self.conll_scorer = conll_scorer
        self.conll_data_dir = conll_data_dir

        # Get model paths
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model_path = path.join(model_dir, 'model.pth')
        self.best_model_path = path.join(best_model_dir, 'model.pth')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if eval_model:
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model = pick_controller(device=self.device, **checkpoint['model_args']).to(self.device)
            self.model.load_state_dict(checkpoint['model'], strict=False)
            # Finally evaluate model
            if eval_max_ents is not None:
                self.model.set_max_ents(eval_max_ents)
            if use_gold_ments is not None:
                self.model.use_gold_ments = use_gold_ments

            if args.dataset != self.model.dataset:
                # Change the default mention detection constants
                if args.max_span_width is not None:
                    self.model.max_span_width = args.max_span_width
                if args.top_span_ratio is not None:
                    self.model.top_span_ratio = args.top_span_ratio

            self.final_eval()
        else:
            # Initialize model and training metadata
            self.model = pick_controller(mem_type=mem_type, dataset=dataset, device=self.device, **kwargs)

            self.train_info, self.optimizer, self.optim_scheduler = {}, None, None
            # Train info is a dictionary to keep around important training variables
            self.train_info['epoch'] = 0
            self.train_info['val_perf'] = 0.0
            self.train_info['global_steps'] = 0
            self.train_info['num_stuck_epochs'] = 0

            self.initialize_setup(init_lr=init_lr, lr_decay=lr_decay, warmup_frac=warmup_frac)
            utils.print_model_info(self.model)
            sys.stdout.flush()

            self.train(max_epochs=max_epochs, max_gradient_norm=max_gradient_norm)

            self.load_model(self.best_model_path, model_type='best')
            logger.info("Loading best model after epoch: %d" % self.train_info['epoch'])
            self.final_eval()

    def initialize_setup(self, init_lr, lr_decay='linear', warmup_frac=0.0):
        """Initialize model + optimizer(s). Check if there's a checkpoint in which case we resume from there."""
        param_list = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_list.append(param)

        self.optimizer = torch.optim.AdamW(param_list, lr=init_lr, eps=1e-6)
        num_training_steps = len(self.train_examples) * self.max_epochs
        num_warmup_steps = warmup_frac * num_training_steps

        if lr_decay == 'inv':
            self.optim_scheduler = get_inverse_square_root_decay(self.optimizer, num_warmup_steps=num_warmup_steps)
        else:
            self.optim_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        if not path.exists(self.model_path):
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            # Try to initialize the mention model part
            if path.exists(self.pretrained_mention_model):
                logger.info("Found pretrained model!!")
                checkpoint = torch.load(self.pretrained_mention_model)
                self.model.load_state_dict(checkpoint['model'], strict=False)
        else:
            logger.info('Loading previous model: %s' % self.model_path)
            # Load model
            self.load_model(self.model_path)

    def train(self, max_epochs, max_gradient_norm):
        """Train model"""
        model = self.model
        epochs_done = self.train_info['epoch']
        optimizer = self.optimizer
        scheduler = self.optim_scheduler

        if self.train_info['num_stuck_epochs'] >= self.max_stuck_epochs:
            return

        for epoch in range(epochs_done, max_epochs):
            logger.info("\n\nStart Epoch %d" % (epoch + 1))
            start_time = time.time()
            # Setup training
            model.train()
            np.random.shuffle(self.train_examples)
            # pred_class_counter, gt_class_counter = defaultdict(int), defaultdict(int)

            if self.use_curriculum:
                max_training_segments = model.doc_encoder.max_training_segments
                if max_training_segments is not None:
                    import math
                    # Linearly increase max training segments as a function of epochs
                    cur_max_training_segments = int(math.ceil((max_training_segments * (epoch + 1))/max_epochs))
                else:
                    cur_max_training_segments = None
            else:
                # No curriculume
                cur_max_training_segments = model.doc_encoder.max_training_segments

            for cur_example in self.train_examples:
                def handle_example(example):
                    optimizer.zero_grad()

                    # Send the copy of the example, as the document could be truncated during training
                    from copy import deepcopy
                    loss = model(deepcopy(example), max_training_segments=cur_max_training_segments)[0]
                    total_loss = loss['total']
                    if total_loss is None:
                        return None

                    if torch.isnan(total_loss):
                        print("Loss is NaN")
                        sys.exit()

                    total_loss.backward()
                    # Perform gradient clipping and update parameters
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_gradient_norm)

                    optimizer.step()
                    scheduler.step()

                    self.train_info['global_steps'] += 1
                    return total_loss.item()

                example_loss = handle_example(cur_example)

                if self.train_info['global_steps'] % self.update_frequency == 0:
                    logger.info('{} {:.3f} Max mem {:.3f} GB'.format(
                        cur_example["doc_key"], example_loss,
                        (torch.cuda.max_memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0)
                    )

                    if torch.cuda.is_available():
                        try:
                            torch.cuda.reset_peak_memory_stats()
                        except AttributeError:
                            # In case of an earlier torch version
                            torch.cuda.reset_max_memory_allocated()

            sys.stdout.flush()
            # Update epochs done
            self.train_info['epoch'] = epoch + 1

            # Dev performance
            cluster_threshold = max(self.cluster_threshold, self.canonical_cluster_threshold)
            # print("Cluster threshold:", cluster_threshold)
            fscore = self.eval_model(cluster_threshold=cluster_threshold)['fscore']

            # Assume that the model didn't improve
            self.train_info['num_stuck_epochs'] += 1

            # Update model if dev performance improves
            if fscore > self.train_info['val_perf']:
                self.train_info['num_stuck_epochs'] = 0
                self.train_info['val_perf'] = fscore
                logger.info('Saving best model')
                self.save_model(self.best_model_path, model_type='best')

            # Save model
            self.save_model(self.model_path)

            # Get elapsed time
            elapsed_time = time.time() - start_time
            logger.info("Epoch: %d, F1: %.1f, Max F1: %.1f, Time: %.2f"
                        % (epoch + 1, fscore, self.train_info['val_perf'], elapsed_time))

            sys.stdout.flush()
            logger.handlers[0].flush()

            if self.train_info['num_stuck_epochs'] >= self.max_stuck_epochs:
                return

    def eval_model(self, split='dev', final_eval=False, cluster_threshold=1):
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
                        get_mention_to_cluster(predicted_clusters, threshold=cluster_threshold)
                    gold_clusters, mention_to_gold =\
                        get_mention_to_cluster(example["clusters"], threshold=cluster_threshold)

                    coref_predictions[example["doc_key"]] = predicted_clusters
                    subtoken_maps[example["doc_key"]] = example["subtoken_map"]

                    # Update the number of clusters
                    num_gt_clusters += len(gold_clusters)
                    num_pred_clusters += len(predicted_clusters)

                    oracle_clusters = action_sequences_to_clusters(gt_actions, pred_mentions)
                    oracle_clusters, mention_to_oracle = \
                        get_mention_to_cluster(oracle_clusters, threshold=cluster_threshold)
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
                result_dict = OrderedDict()
                indv_metrics_list = ['MUC', 'Bcub', 'CEAFE']
                perf_str = ""
                for indv_metric, indv_evaluator in zip(indv_metrics_list, evaluator.evaluators):
                    perf_str += ", " + indv_metric + ": {:.1f}".format(indv_evaluator.get_f1() * 100)
                    result_dict[indv_metric] = OrderedDict()
                    result_dict[indv_metric]['recall'] = round(indv_evaluator.get_recall() * 100, 1)
                    result_dict[indv_metric]['precision'] = round(indv_evaluator.get_precision() * 100, 1)
                    result_dict[indv_metric]['fscore'] = round(indv_evaluator.get_f1() * 100, 1)

                fscore = evaluator.get_f1() * 100
                result_dict['fscore'] = round(fscore, 1)
                logger.info("F-score: %.1f %s" % (fscore, perf_str))

                # Only use CoNLL evaluator script for final evaluation
                if final_eval and path.exists(self.conll_scorer) and path.exists(self.conll_data_dir):
                    gold_path = path.join(self.conll_data_dir, f'{split}.conll')
                    prediction_file = path.join(self.model_dir, f'{split}.conll')
                    conll_results = evaluate_conll(
                        self.conll_scorer, gold_path, coref_predictions, subtoken_maps, prediction_file)

                    for indv_metric in indv_metrics_list:
                        result_dict[indv_metric] = OrderedDict()
                        result_dict[indv_metric]['recall'] = round(conll_results[indv_metric.lower()]["r"], 1)
                        result_dict[indv_metric]['precision'] = round(conll_results[indv_metric.lower()]["p"], 1)
                        result_dict[indv_metric]['fscore'] = round(conll_results[indv_metric.lower()]["f"], 1)

                    average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
                    result_dict['fscore'] = round(average_f1, 1)

                    logger.info("(CoNLL) F-score : %.1f, MUC: %.1f, Bcub: %.1f, CEAFE: %.1f"
                                % (average_f1, conll_results["muc"]["f"], conll_results['bcub']["f"],
                                    conll_results['ceafe']["f"]))

                logger.info("Action accuracy: %.3f, Oracle F-score: %.3f" %
                            (corr_actions/total_actions, oracle_evaluator.get_prf()[2]))
                logger.info(log_file)
                logger.handlers[0].flush()

        return result_dict

    def final_eval(self):
        """Evaluate the model on train, dev, and test"""
        # Test performance  - Load best model
        perf_file = path.join(self.model_dir, "perf.json")
        if self.slurm_id:
            parent_dir = path.dirname(path.normpath(self.model_dir))
            perf_dir = path.join(parent_dir, "perf")
            if not path.exists(perf_dir):
                os.makedirs(perf_dir)
            perf_file = path.join(perf_dir, self.slurm_id + ".json")

        output_dict = {'model_dir': self.model_dir}
        for key, val in vars(self.args).items():
            output_dict[key] = val

        for split in ['dev', 'test']:
            # if self.train_with_singletons:
            #     cluster_thresholds = [1, 2]
            # else:
            #     cluster_thresholds = [2]
            # cluster_thresholds = [1, 2]
            cluster_thresholds = [self.canonical_cluster_threshold]
            for cluster_threshold in cluster_thresholds:
                logging.info('\n')
                logging.info('%s' % split.capitalize())
                result_dict = self.eval_model(split, final_eval=True, cluster_threshold=cluster_threshold)
                if split != 'test':
                    logging.info('Calculated F1: %.3f' % result_dict['fscore'])

                output_dict[f"{split}_{cluster_threshold}"] = result_dict
                if cluster_threshold == self.canonical_cluster_threshold:
                    output_dict[f"{split}"] = result_dict

        json.dump(output_dict, open(perf_file, 'w'), indent=2)

        logging.info("Final performance summary at %s" % perf_file)
        sys.stdout.flush()

    def load_model(self, location, model_type='last'):
        if torch.cuda.is_available():
            checkpoint = torch.load(location)
        else:
            checkpoint = torch.load(location, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.train_info = checkpoint['train_info']

        if model_type != 'best':
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.optim_scheduler.load_state_dict(checkpoint['scheduler'])
            torch.set_rng_state(checkpoint['rng_state'])
            np.random.set_state(checkpoint['np_rng_state'])

    def save_model(self, location, model_type='last'):
        """Save model"""
        save_dict = {}

        model_state_dict = OrderedDict(self.model.state_dict())
        for key in self.model.state_dict():
            if 'bert.' in key:
                del model_state_dict[key]

        save_dict['model'] = model_state_dict
        save_dict['model_args'] = self.model_args
        save_dict['train_info'] = self.train_info

        # Don't need optimizers for inference, hence not saving these for the best models.
        if model_type != 'best':
            # Regular model saved during training.
            save_dict.update({
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.optim_scheduler.state_dict(),
                'rng_state': torch.get_rng_state(),
                'np_rng_state': np.random.get_state()
            })

        torch.save(save_dict, location)
        logging.info(f"Model saved at: {location}")