import torch
import numpy as np

from auto_memory_model.memory import MemoryNoInvalid
from auto_memory_model.controller import BaseController
from coref_utils.utils import get_mention_to_cluster_idx


class ControllerNoInvalid(BaseController):
    def __init__(self, **kwargs):
        super(ControllerNoInvalid, self).__init__(**kwargs)

        self.memory_net = MemoryNoInvalid(
            hsize=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + self.emb_size,
            drop_module=self.drop_module, **kwargs)

    def get_actions(self, pred_mentions, clusters, rand_fl_list, follow_gt):
        # Useful data structures
        mention_to_cluster = get_mention_to_cluster_idx(clusters)

        actions = []
        cluster_to_cell = {}

        cell_counter = 0
        for idx, mention in enumerate(pred_mentions):
            if tuple(mention) not in mention_to_cluster:
                if follow_gt and rand_fl_list[idx] > self.sample_invalid:
                    # This invalid mention is ignored during training
                    actions.append((-1, 'i'))
                else:
                    # Not a mention - Add to memory anyways.
                    # This is not a problem because singletons are removed during metric calculation.
                    actions.append((cell_counter, 'o'))
                    cell_counter += 1
            else:
                mention_cluster = mention_to_cluster[tuple(mention)]
                if mention_cluster in cluster_to_cell:
                    # Cluster is already being tracked
                    actions.append((cluster_to_cell[mention_cluster], 'c'))
                else:
                    # Cluster is not being tracked
                    # Add the mention to being tracked
                    cluster_to_cell[mention_cluster] = cell_counter
                    actions.append((cell_counter, 'o'))
                    cell_counter += 1

        return actions

    def forward(self, example, teacher_forcing=False):
        """
        Encode a batch of excerpts.
        """
        encoded_doc = self.doc_encoder(example)
        pred_starts, pred_ends, pred_scores = self.get_pred_mentions(example, encoded_doc, topk=True)

        # Sort the predicted mentions
        pred_mentions = list(zip(pred_starts.tolist(), pred_ends.tolist()))
        mention_score_list = torch.unbind(torch.unsqueeze(pred_scores, dim=1))

        mention_embs = self.get_span_embeddings(encoded_doc, pred_starts, pred_ends)
        mention_emb_list = torch.unbind(mention_embs, dim=0)

        del encoded_doc
        metadata = {}
        if self.dataset == 'ontonotes':
            metadata = {'genre': self.get_genre_embedding(example)}

        follow_gt = self.training or teacher_forcing
        rand_fl_list = np.random.random(len(mention_emb_list))
        if teacher_forcing:
            rand_fl_list = np.zeros_like(rand_fl_list)

        if "clusters" in example:
            gt_actions = self.get_actions(pred_mentions, example["clusters"], rand_fl_list, follow_gt)
        else:
            gt_actions = [(-1, 'i')] * len(pred_mentions)
        action_prob_list, action_list = self.memory_net(
            mention_emb_list, mention_score_list, gt_actions, metadata, teacher_forcing=teacher_forcing)

        loss = {'total': None}
        if follow_gt:
            if len(action_prob_list) > 0:
                coref_new_prob_list = action_prob_list
                loss = {}
                coref_loss = self.calculate_coref_loss(coref_new_prob_list, gt_actions)
                loss['coref'] = coref_loss
                loss['total'] = loss['coref']

            return loss, action_list, pred_mentions, gt_actions
        else:
            return 0.0, action_list, pred_mentions, gt_actions
