import torch
import torch.nn as nn

from auto_memory_model.memory.lru_memory import LRUMemory
from auto_memory_model.controller.lfm_controller import LearnedFixedMemController

EPS = 1e-8


class LRUController(LearnedFixedMemController):
    def __init__(self, **kwargs):
        super(LRUController, self).__init__(**kwargs)
        self.memory_net = LRUMemory(hsize=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize,
                                    drop_module=self.drop_module, **kwargs)

        # Set loss functions
        # Overwrite in LRU has only 2 classes - Overwrite (LRU cell) and Ignore
        over_loss_wts = torch.ones(2).cuda()
        self.loss_fn['over'] = nn.NLLLoss(weight=over_loss_wts, reduction='sum')

    @staticmethod
    def over_ign_tuple_to_idx(action_tuple_list, over_ign_prob_list):
        action_indices = []
        prob_list = []

        for (cell_idx, action_str), over_ign_prob in zip(action_tuple_list, over_ign_prob_list):
            if action_str == 'c':
                continue
            elif action_str == 'o':
                action_indices.append(0)
            else:
                action_indices.append(1)

            prob_list.append(over_ign_prob)

        action_indices = torch.tensor(action_indices).cuda()
        prob_tens = torch.stack(prob_list, dim=0).cuda()
        return action_indices, prob_tens

    def forward(self, example, teacher_forcing=False):
        """
        Encode a batch of excerpts.
        """
        doc_tens, sent_len_list = self.tensorize_example(example)
        encoded_output = self.encode_doc(doc_tens,  sent_len_list)

        mention_embs = self.get_mention_embeddings(
            example['ord_mentions'], encoded_output, method=self.ment_emb)
        mention_emb_list = torch.unbind(mention_embs, dim=0)

        action_prob_list, action_list = self.memory_net(
            mention_emb_list, example["actions"], example["ord_mentions"],
            teacher_forcing=teacher_forcing)

        loss = {}

        coref_new_prob_list, over_ign_prob_list = zip(*action_prob_list)
        action_prob_tens = torch.stack(coref_new_prob_list, dim=0).cuda()  # M x (cells + 1)
        action_indices = self.action_to_coref_new_idx(example["actions"])

        # Calculate overwrite loss
        over_action_indices, prob_tens = self.over_ign_tuple_to_idx(
            example["actions"], over_ign_prob_list)
        over_loss = self.loss_fn['over'](prob_tens, over_action_indices)
        over_loss_weight = over_action_indices.shape[0]
        loss['over'] = over_loss / over_loss_weight

        coref_loss = self.loss_fn['coref'](action_prob_tens, action_indices)
        total_weight = len(mention_emb_list)  # Total mentions

        if self.training or teacher_forcing:
            loss['coref'] = coref_loss/total_weight
            loss['total'] = loss['coref'] + self.over_loss_wt * loss['over']
            return loss, action_list
        else:
            return coref_loss, action_list
