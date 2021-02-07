import torch
import torch.nn as nn

from auto_memory_model.memory.streaming_lru_memory import StreamingLRUMemory
from auto_memory_model.controller.lfm_controller import LearnedFixedMemController
from coref_utils.utils import get_mention_to_cluster_idx
from pytorch_utils.label_smoothing import LabelSmoothingLoss
import numpy as np


class StreamingLRUController(LearnedFixedMemController):
    def __init__(self, **kwargs):
        super(StreamingLRUController, self).__init__(**kwargs)
        self.memory_net = StreamingLRUMemory(hsize=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + self.emb_size,
                                    drop_module=self.drop_module, **kwargs)

        # Set loss functions
        # Overwrite in LRU has only 2 classes - Overwrite, Ignore, Invalid
        self.over_loss_wts = torch.tensor([1.0] * 3).to(self.device)
        self.loss_fn['over'] = nn.CrossEntropyLoss(weight=self.over_loss_wts, reduction='mean')

    def get_actions(self, pred_mentions, mention_to_cluster, gt_clusters,
                    cell_to_cluster=None, cluster_to_cell=None,
                    cell_to_last_used=None, cluster_to_rem_mentions=None):
        pred_mentions = [tuple(mention) for mention in pred_mentions]
        actions = []
        if cell_to_cluster is None:
            cell_to_cluster = {}
            cluster_to_cell = {}

            cell_to_last_used = [0 for cell in range(self.num_cells)]  # Initialize last usage of cell
            cluster_to_rem_mentions = [len(cluster) for cluster in gt_clusters]
        # Initialize with all the mentions
        # cluster_to_rem_mentions = [len(cluster) for cluster in clusters]

        lru_list = list(range(self.num_cells))

        for mention in pred_mentions:
            used_cell_idx = None
            if mention not in mention_to_cluster:
                # Not a mention
                actions.append((-1, 'i'))
            else:
                mention_cluster = mention_to_cluster[tuple(mention)]
                if mention_cluster in cluster_to_cell:
                    # Cluster is already being tracked
                    actions.append((cluster_to_cell[mention_cluster], 'c'))
                    # Update when the cell was last used
                    used_cell_idx = cluster_to_cell[mention_cluster]
                else:
                    # Cluster is not being tracked
                    # Find the cell with the least regret that we can overwrite to
                    # If the regret is non-positive i.e. we would be missing out on >= mentions
                    # of a cluster being currently tracked than the new mention cluster then we
                    # don't perform overwrite.
                    cur_rem_mentions = cluster_to_rem_mentions[mention_cluster]
                    cell_info = []
                    for cell_idx in range(self.num_cells):
                        if cell_idx in cell_to_cluster:
                            # The cell is actually in use
                            cell_cluster = cell_to_cluster[cell_idx]
                            cell_rem_mentions = cluster_to_rem_mentions[cell_cluster]
                        else:
                            # The cell is not in use
                            cell_rem_mentions = -1
                        cell_info.append((cell_rem_mentions, cell_to_last_used[cell_idx], cell_idx,
                                          lru_list.index(cell_idx)))

                    # Sort cells by least recently used cells
                    cell_info = sorted(cell_info, key=lambda x: x[3])

                    # Remaining mentions in least recently used cell
                    lru_remaining_mentions = cell_info[0][0]

                    if cur_rem_mentions >= lru_remaining_mentions:
                        used_cell_idx = cell_info[0][2]  # Get the cell index

                    if used_cell_idx is None:
                        # Ignore the mention
                        actions.append((-1, 'n'))
                    else:
                        # Overwrite
                        actions.append((used_cell_idx, 'o'))
                        # Remove the cluster to cell reference for the replacement cell
                        # Only do this if the cell was tracking anything
                        if used_cell_idx in cell_to_cluster:
                            del cluster_to_cell[cell_to_cluster[used_cell_idx]]

                        # Add the mention to being tracked
                        cluster_to_cell[mention_cluster] = used_cell_idx
                        cell_to_cluster[used_cell_idx] = mention_cluster

                # Update the cell_to_last_used index
                for cell_idx in range(self.num_cells):
                    cell_to_last_used[cell_idx] += 1
                if used_cell_idx is not None:
                    cell_to_last_used[used_cell_idx] = 0
                    # Remove the used_cell_idx and put it at the end of the LRU list
                    lru_list.remove(used_cell_idx)
                    lru_list.append(used_cell_idx)

                # Reduce the number of mentions remaining in the current cluster
                cluster_to_rem_mentions[mention_cluster] -= 1

        return actions, cell_to_cluster, cluster_to_cell, cell_to_last_used, cluster_to_rem_mentions

    def new_ignore_tuple_to_idx(self, action_tuple_list, rand_fl_list, follow_gt):
        action_indices = []

        for idx, (cell_idx, action_str) in enumerate(action_tuple_list):
            if action_str != 'c':
                if action_str == 'o':
                    action_indices.append(0)
                elif action_str == 'n':
                    # No space
                    action_indices.append(1)
                elif action_str == 'i':
                    if follow_gt and rand_fl_list[idx] > self.sample_invalid:
                        pass
                    else:
                        action_indices.append(2)

        action_indices = torch.tensor(action_indices).to(self.device)
        return action_indices

    def forward(self, example, teacher_forcing=False):
        metadata = {}
        if self.dataset == 'ontonotes':
            metadata = {'genre': self.get_genre_embedding(example)}

        follow_gt = self.training or teacher_forcing
        pred_mentions, gt_actions = [], []
        last_memory, last_action_str = None, '<s>'
        cell_to_cluster, cluster_to_cell, cell_to_last_used, cluster_to_rem_mentions = None, None, None, None
        coref_new_list, new_ignore_list, action_list = [], [], []
        rand_fl_list = []
        ment_offset = 0

        if self.training and self.max_training_segments is not None:
            from copy import deepcopy
            example = self.truncate_document(deepcopy(example))

        if "clusters" in example:
            mention_to_cluster = get_mention_to_cluster_idx(example["clusters"])

        jump_size = 2
        for idx in range(0, len(example["sentences"]), jump_size):
            word_offset = sum([len(sent) for sent in example["sentences"][:idx]])
            num_words = sum([len(sent) for sent in example["sentences"][idx: idx + jump_size]])
            cur_example = {
                "sentences": example["sentences"][idx: idx + jump_size],
                "sentence_map": example["sentence_map"][word_offset: word_offset + num_words],
            }

            cur_pred_mentions, cur_mention_emb_list, cur_mention_score_list =\
                self.get_mention_embs(cur_example)
            # print(cur_pred_mentions)
            cur_pred_mentions = [(span_start + word_offset, span_end + word_offset)
                                 for (span_start, span_end) in cur_pred_mentions]
            # print(cur_pred_mentions)

            if "clusters" in example:
                cur_gt_actions, cell_to_cluster, cluster_to_cell, cell_to_last_used, cluster_to_rem_mentions = self.get_actions(
                    cur_pred_mentions, mention_to_cluster, example["clusters"], cell_to_cluster=cell_to_cluster,
                    cluster_to_cell=cluster_to_cell, cell_to_last_used=cell_to_last_used,
                    cluster_to_rem_mentions=cluster_to_rem_mentions)
                # print(Counter(cur_gt_actions))
            else:
                cur_gt_actions = [(-1, 'i')] * len(cur_pred_mentions)

            pred_mentions.extend(cur_pred_mentions)
            gt_actions.extend(cur_gt_actions)

            cur_rand_fl_list = np.random.random(len(cur_pred_mentions))
            if teacher_forcing:
                cur_rand_fl_list = np.zeros_like(rand_fl_list)

            rand_fl_list.extend(list(cur_rand_fl_list))

            cur_coref_new_list, cur_new_ignore_list, cur_action_list, last_memory, last_action_str, ment_offset = self.memory_net(
                cur_mention_emb_list, cur_mention_score_list, cur_gt_actions, metadata, cur_rand_fl_list,
                memory_init=last_memory, last_action_str=last_action_str, ment_offset=ment_offset,
                teacher_forcing=teacher_forcing)

            if idx > 0:
                pass
                from collections import Counter
                # print(Counter(cur_gt_actions))
                # print(last_memory['mem'].shape)
                # print(Counter(cur_action_list))

            coref_new_list.extend(cur_coref_new_list)
            new_ignore_list.extend(cur_new_ignore_list)
            action_list.extend(cur_action_list)

        # if self.training:
        #     label_smoothing_fn = LabelSmoothingLoss(smoothing=0.0, dim=1)
        # else:
        #     label_smoothing_fn = LabelSmoothingLoss(smoothing=0.0, dim=1)

        loss_fn = torch.nn.CrossEntropyLoss()

        loss = {'total': None}
        if follow_gt:
            if len(coref_new_list) > 0:
                loss = {}
                action_prob_tens = torch.stack(coref_new_list, dim=0).to(self.device)  # M x (cells + 1)
                action_indices = self.action_to_coref_new_idx(gt_actions, rand_fl_list, follow_gt)

                # coref_loss = torch.sum(
                #     label_smoothing_fn(action_prob_tens, action_indices.unsqueeze(dim=1), weight=self.coref_loss_wts))
                coref_loss = loss_fn(action_prob_tens, action_indices)
                loss['coref'] = coref_loss

                # Calculate overwrite loss
                new_ignore_tens = torch.stack(new_ignore_list, dim=0).to(self.device)
                new_ignore_indices = self.new_ignore_tuple_to_idx(gt_actions, rand_fl_list, follow_gt)
                over_loss = self.loss_fn['over'](new_ignore_tens, new_ignore_indices)

                loss['over'] = over_loss
                loss['total'] = (loss['coref'] + self.over_loss_wt * loss['over'])
            return loss, action_list, pred_mentions, gt_actions
        else:
            return 0.0, action_list, pred_mentions, gt_actions
