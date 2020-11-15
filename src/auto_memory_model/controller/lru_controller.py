import torch
import torch.nn as nn

from auto_memory_model.memory.lru_memory import LRUMemory
from auto_memory_model.controller.lfm_controller import LearnedFixedMemController
from coref_utils.utils import get_mention_to_cluster_idx
from pytorch_utils.label_smoothing import LabelSmoothingLoss
import numpy as np


class LRUController(LearnedFixedMemController):
    def __init__(self, **kwargs):
        super(LRUController, self).__init__(**kwargs)
        self.memory_net = LRUMemory(hsize=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + self.emb_size,
                                    drop_module=self.drop_module, **kwargs)

        # Set loss functions
        # Overwrite in LRU has only 2 classes - Overwrite, Ignore, Invalid
        self.over_loss_wts = torch.tensor([1.0] * 3).to(self.device)
        self.loss_fn['over'] = nn.CrossEntropyLoss(weight=self.over_loss_wts, reduction='mean')

    def get_actions(self, pred_mentions, gt_clusters):
        pred_mentions = [tuple(mention) for mention in pred_mentions]

        # Useful data structures
        mention_to_cluster = get_mention_to_cluster_idx(gt_clusters)

        actions = []
        cell_to_cluster = {}
        cell_to_last_used = [0 for cell in range(self.num_cells)]  # Initialize last usage of cell
        cluster_to_cell = {}

        # Initialize with all the mentions
        # cluster_to_rem_mentions = [len(cluster) for cluster in clusters]
        cluster_to_rem_mentions = [len(cluster) for cluster in gt_clusters]
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

        return actions

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


