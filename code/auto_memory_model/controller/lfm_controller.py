import torch
import torch.nn as nn

from auto_memory_model.memory.lfm_memory import LearnedFixedMemory
from auto_memory_model.controller.base_controller import BaseController
from auto_memory_model.utils import get_mention_to_cluster, get_ordered_mentions


class LearnedFixedMemController(BaseController):
    def __init__(self, num_cells=10, over_loss_wt=0.1, new_ent_wt=1.0, ignore_wt=1.0, **kwargs):
        super(LearnedFixedMemController, self).__init__(**kwargs)
        self.memory_net = LearnedFixedMemory(
            num_cells=num_cells, hsize=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + self.emb_size,
            drop_module=self.drop_module, **kwargs)
        self.num_cells = num_cells
        # Loss setup
        self.new_ent_wt = new_ent_wt
        self.ignore_wt = ignore_wt
        self.over_loss_wt = over_loss_wt
        # Set loss functions
        self.loss_fn = {}
        coref_loss_wts = torch.tensor([1.0] * self.num_cells + [self.new_ent_wt]).cuda()
        self.loss_fn['coref'] = nn.CrossEntropyLoss(weight=coref_loss_wts, reduction='sum')
        over_loss_wts = torch.tensor([1.0] * self.num_cells + [self.ignore_wt]).cuda()
        self.loss_fn['over'] = nn.CrossEntropyLoss(weight=over_loss_wts, reduction='sum')

    def get_actions(self, pred_mentions, gt_clusters):
        # Useful data structures
        pred_mentions = [tuple(mention) for mention in pred_mentions]
        mention_to_cluster = get_mention_to_cluster(gt_clusters)

        actions = []
        cell_to_cluster = {}
        cell_to_last_used = [0 for cell in range(self.num_cells)]  # Initialize last usage of cell
        cluster_to_cell = {}

        # Initialize with all the mentions
        # cluster_to_rem_mentions = [len(cluster) for cluster in gt_clusters]

        cluster_to_rem_mentions = []  # [len(cluster) for cluster in gt_clusters]
        set_pred_mentions = set(pred_mentions)
        for cluster in gt_clusters:
            mentions_covered_in_preds = 0
            for mention in cluster:
                if tuple(mention) in set_pred_mentions:
                    mentions_covered_in_preds += 1

            cluster_to_rem_mentions.append(mentions_covered_in_preds)

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

                        cell_info.append((cell_rem_mentions, cell_to_last_used[cell_idx], cell_idx))

                    # Sort the cells primarily by the number of remaining mentions
                    # If the remaining mentions are tied, then compare the last used cell
                    cell_info = sorted(cell_info, key=lambda x: x[0] - 1e-10 * x[1])
                    min_remaining_mentions = cell_info[0][0]

                    if cur_rem_mentions > min_remaining_mentions:
                        used_cell_idx = cell_info[0][2]  # Get the cell index

                    if used_cell_idx is None:
                        # Ignore the mention
                        actions.append((-1, 'i'))
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

                # Reduce the number of mentions remaining in the current cluster
                cluster_to_rem_mentions[mention_cluster] -= 1

        return actions

    def over_ign_tuple_to_idx(self, action_tuple_list, over_ign_prob_list):
        action_indices = []
        prob_list = []

        for (cell_idx, action_str), over_ign_prob in zip(action_tuple_list, over_ign_prob_list):
            if action_str == 'o' or action_str == 'i':
                if action_str == 'o':
                    action_indices.append(cell_idx)
                else:
                    action_indices.append(self.num_cells)
                prob_list.append(over_ign_prob)

        action_indices = torch.tensor(action_indices).cuda()
        prob_tens = torch.stack(prob_list, dim=0).cuda()
        return action_indices, prob_tens

    def action_to_coref_new_idx(self, action_tuple_list):
        action_indices = []
        for (cell_idx, action_str) in action_tuple_list:
            if action_str == 'c':
                action_indices.append(cell_idx)
            elif action_str == 'o' or action_str == 'i':
                action_indices.append(self.num_cells)
            else:
                raise NotImplementedError

        return torch.tensor(action_indices).cuda()

    def forward(self, example, teacher_forcing=False):
        """
        Encode a batch of excerpts.
        """
        gt_mentions, pred_mentions, gt_actions, mention_emb_list, mention_score_list =\
            self.get_mention_embs_and_actions(example)

        action_prob_list, action_list = self.memory_net(
            mention_emb_list, mention_score_list, gt_actions,
            teacher_forcing=teacher_forcing)  # , example[""])

        loss = {}

        coref_new_prob_list, over_ign_prob_list = zip(*action_prob_list)
        action_prob_tens = torch.stack(coref_new_prob_list, dim=0).cuda()  # M x (cells + 1)
        action_indices = self.action_to_coref_new_idx(gt_actions)

        # Calculate overwrite loss
        over_action_indices, prob_tens = self.over_ign_tuple_to_idx(
            gt_actions, over_ign_prob_list)
        over_loss = self.loss_fn['over'](prob_tens, over_action_indices)
        over_loss_weight = over_action_indices.shape[0]
        loss['over'] = over_loss/over_loss_weight

        coref_loss = self.loss_fn['coref'](action_prob_tens, action_indices)
        total_weight = len(mention_emb_list)  # Total mentions

        if self.training or teacher_forcing:
            loss['coref'] = coref_loss/total_weight
            loss['total'] = loss['coref'] + self.over_loss_wt * loss['over']
            return loss, action_list, pred_mentions, gt_actions, gt_mentions
        else:
            return coref_loss, action_list, pred_mentions, gt_actions, gt_mentions
