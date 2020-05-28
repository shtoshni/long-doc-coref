import torch
import torch.nn as nn
import numpy as np

from auto_memory_model.memory.um_memory import UnboundedMemory
from auto_memory_model.controller.base_controller import BaseController
from auto_memory_model.utils import get_mention_to_cluster, get_ordered_mentions


class UnboundedMemController(BaseController):
    def __init__(self, new_ent_wt=1.0, over_loss_wt=0.1, ignore_wt=0.1, **kwargs):
        super(UnboundedMemController, self).__init__(**kwargs)
        self.new_ent_wt = new_ent_wt
        self.over_loss_wt = over_loss_wt
        self.ignore_wt = ignore_wt

        self.memory_net = UnboundedMemory(
            hsize=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + self.emb_size,
            drop_module=self.drop_module, **kwargs)
        # Set loss functions
        self.loss_fn = {}
        # Overwrite in Unbounded has only 2 classes - Overwrite and Ignore
        over_loss_wts = torch.tensor([1.0] + [self.ignore_wt]).cuda()
        self.loss_fn['over'] = nn.CrossEntropyLoss(weight=over_loss_wts, reduction='sum', ignore_index=-100)

    @staticmethod
    def get_actions(pred_mentions, clusters):
        # Useful data structures
        mention_to_cluster = get_mention_to_cluster(clusters)

        actions = []
        cell_to_cluster = {}
        cluster_to_cell = {}

        cell_counter = 0
        for mention in pred_mentions:
            if tuple(mention) not in mention_to_cluster:
                # Not a mention
                actions.append((-1, 'i'))
            else:
                mention_cluster = mention_to_cluster[tuple(mention)]
                if mention_cluster in cluster_to_cell:
                    # Cluster is already being tracked
                    actions.append((cluster_to_cell[mention_cluster], 'c'))
                else:
                    # Cluster is not being tracked
                    # Add the mention to being tracked
                    cluster_to_cell[mention_cluster] = cell_counter
                    cell_to_cluster[cell_counter] = mention_cluster
                    actions.append((cell_counter, 'o'))
                    cell_counter += 1

        return actions

    def calculate_coref_loss(self, action_prob_list, action_tuple_list, rand_fl_list, follow_gt):
        num_cells = 0
        coref_loss = 0.0
        target_list = []

        # First filter the action tuples to sample ignores
        for idx, (cell_idx, action_str) in enumerate(action_tuple_list):
            gt_idx = None
            if action_str == 'c':
                gt_idx = cell_idx
            elif action_str == 'o':
                # Overwrite
                gt_idx = (1 if num_cells == 0 else num_cells)
                num_cells += 1
            elif action_str == 'i':
                # Ignore
                if follow_gt and rand_fl_list[idx] > self.sample_ignores:
                    continue
                else:
                    gt_idx = (1 if num_cells == 0 else num_cells)

            target = torch.tensor([gt_idx]).cuda()
            target_list.append(target)

        for idx, target in enumerate(target_list):
            logit_tens = torch.unsqueeze(action_prob_list[idx], dim=0)

            # print(target, logit_tens.shape)
            weight = torch.ones_like(action_prob_list[idx]).float().cuda()
            weight[-1] = self.new_ent_wt
            coref_loss += torch.nn.functional.cross_entropy(input=logit_tens, target=target, weight=weight)

        return coref_loss

    def over_ign_tuple_to_idx(self, action_tuple_list, rand_fl_list, follow_gt):
        action_indices = []

        for idx, (cell_idx, action_str) in enumerate(action_tuple_list):
            if action_str == 'c':
                action_indices.append(-100)
            elif action_str == 'o':
                action_indices.append(0)
            elif action_str == 'i':
                # Not a mention
                if follow_gt and rand_fl_list[idx] > self.sample_ignores:
                    pass
                else:
                    action_indices.append(1)

        action_indices = torch.tensor(action_indices).cuda()
        return action_indices

    def forward(self, example, teacher_forcing=False):
        """
        Encode a batch of excerpts.
        """
        pred_mentions, gt_actions, mention_emb_list, mention_score_list = self.get_mention_embs_and_actions(example)

        metadata = {}
        if self.dataset == 'ontonotes':
            metadata = {'genre': self.get_genre_embedding(example)}

        rand_fl_list = np.random.random(len(mention_emb_list))
        follow_gt = self.training or teacher_forcing

        action_prob_list, action_list = self.memory_net(
            mention_emb_list, mention_score_list, gt_actions, metadata, rand_fl_list,
            teacher_forcing=teacher_forcing)

        loss = {}
        coref_new_prob_list, new_ignore_list = zip(*action_prob_list)

        coref_loss = 0.0
        if follow_gt:
            coref_loss = self.calculate_coref_loss(
                coref_new_prob_list, gt_actions, rand_fl_list, follow_gt)
            loss['coref'] = coref_loss
            # Calculate overwrite loss
            new_ignore_tens = torch.stack(new_ignore_list, dim=0).cuda()
            over_action_indices = self.over_ign_tuple_to_idx(
                gt_actions, rand_fl_list, follow_gt)
            over_loss = self.loss_fn['over'](new_ignore_tens, over_action_indices)
            loss['over'] = over_loss

            loss['total'] = loss['coref'] + self.over_loss_wt * loss['over']
            return loss, action_list, pred_mentions, gt_actions
        else:
            return coref_loss, action_list, pred_mentions, gt_actions
