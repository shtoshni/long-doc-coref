import torch
import torch.nn as nn

from auto_memory_model.memory.um_memory import UnboundedMemory
from auto_memory_model.controller.base_controller import BaseController


class UnboundedMemController(BaseController):
    def __init__(self, new_ent_wt=1.0, **kwargs):
        super(UnboundedMemController, self).__init__(**kwargs)
        self.new_ent_wt = new_ent_wt
        self.memory_net = UnboundedMemory(
            hsize=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize,
            drop_module=self.drop_module, **kwargs)
        # Set loss functions
        self.loss_fn = {}

    def calculate_coref_loss(self, action_prob_list, action_tuple_list):
        num_cells = 1
        coref_loss = 0.0

        for idx, (cell_idx, action_str) in enumerate(action_tuple_list):
            if idx == 0:
                continue

            gt_idx = None
            if action_str == 'c':
                gt_idx = cell_idx
            else:
                # Overwrite
                gt_idx = num_cells
                num_cells += 1

            target = torch.tensor([gt_idx]).cuda()
            logit_tens = torch.unsqueeze(action_prob_list[idx], dim=0)

            # print(target, logit_tens.shape)
            weight = torch.ones_like(action_prob_list[idx]).float().cuda()
            weight[-1] = self.new_ent_wt
            coref_loss += torch.nn.functional.cross_entropy(input=logit_tens, target=target, weight=weight)

        return coref_loss

    def forward(self, example, teacher_forcing=False):
        """
        Encode a batch of excerpts.
        """
        encoded_output = self.doc_encoder(example)

        mention_embs = self.get_mention_embeddings(
            example['ord_mentions'], encoded_output, method=self.ment_emb)
        mention_emb_list = torch.unbind(mention_embs, dim=0)

        action_prob_list, action_list = self.memory_net(
            mention_emb_list, example["actions"], example["ord_mentions"],
            teacher_forcing=teacher_forcing)  # , example[""])

        coref_loss = 0.0
        if self.training or teacher_forcing:
            loss = {}
            coref_loss = self.calculate_coref_loss(action_prob_list, example["actions"])
            loss['coref'] = coref_loss/len(mention_emb_list)
            loss['total'] = loss['coref']
            return loss, action_list
        else:
            return coref_loss, action_list
