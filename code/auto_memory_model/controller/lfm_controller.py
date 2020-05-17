import torch
import torch.nn as nn

from auto_memory_model.memory.lfm_memory import LearnedFixedMemory
from auto_memory_model.controller.base_controller import BaseController
# from pytorch_utils.modules import MLP
# from pytorch_memlab import profile

EPS = 1e-8


class LearnedFixedMemController(BaseController):
    def __init__(self, num_cells=10, over_loss_wt=0.1, **kwargs):
        super(LearnedFixedMemController, self).__init__(**kwargs)
        self.memory_net = LearnedFixedMemory(
            num_cells=num_cells, hsize=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize,
            drop_module=self.drop_module, **kwargs)
        self.num_cells = num_cells
        # Loss setup
        self.over_loss_wt = over_loss_wt
        # Set loss functions
        self.loss_fn = {}
        coref_loss_wts = torch.ones(self.num_cells + 1).cuda()
        self.loss_fn['coref'] = nn.CrossEntropyLoss(weight=coref_loss_wts, reduction='sum')
        over_loss_wts = torch.ones(self.num_cells + 1).cuda()
        self.loss_fn['over'] = nn.CrossEntropyLoss(weight=over_loss_wts, reduction='sum')

    def over_ign_tuple_to_idx(self, action_tuple_list, over_ign_prob_list):
        action_indices = []
        prob_list = []

        for (cell_idx, action_str), over_ign_prob in zip(action_tuple_list, over_ign_prob_list):
            if action_str == 'c':
                continue
            elif action_str == 'o':
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
            else:
                action_indices.append(self.num_cells)

        return torch.tensor(action_indices).cuda()

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
            teacher_forcing=teacher_forcing)  # , example[""])

        loss = {}

        coref_new_prob_list, over_ign_prob_list = zip(*action_prob_list)
        action_prob_tens = torch.stack(coref_new_prob_list, dim=0).cuda()  # M x (cells + 1)
        action_indices = self.action_to_coref_new_idx(example["actions"])

        # Calculate overwrite loss
        over_action_indices, prob_tens = self.over_ign_tuple_to_idx(
            example["actions"], over_ign_prob_list)
        over_loss = self.loss_fn['over'](prob_tens, over_action_indices)
        over_loss_weight = over_action_indices.shape[0]
        loss['over'] = over_loss/over_loss_weight

        coref_loss = self.loss_fn['coref'](action_prob_tens, action_indices)
        total_weight = len(mention_emb_list)  # Total mentions

        if self.training or teacher_forcing:
            loss['coref'] = coref_loss/total_weight
            loss['total'] = loss['coref'] + self.over_loss_wt * loss['over']
            return loss, action_list
        else:
            return coref_loss, action_list
