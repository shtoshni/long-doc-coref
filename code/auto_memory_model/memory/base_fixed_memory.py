import torch
import torch.nn as nn
from pytorch_utils.modules import MLP
from auto_memory_model.memory.base_memory import BaseMemory


class BaseFixedMemory(BaseMemory):
    def __init__(self, num_cells=10, **kwargs):
        super(BaseFixedMemory, self).__init__(**kwargs)

        self.num_cells = num_cells

        # Fixed memory cells need to predict fertility of memory and mentions
        self.fert_mlp = MLP(input_size=self.mem_size + self.num_feats * self.emb_size,
                            hidden_size=self.mlp_size, output_size=1, num_hidden_layers=self.mlp_depth,
                            bias=True, drop_module=self.drop_module)
        self.ment_fert_mlp = MLP(input_size=self.mem_size + (self.num_feats - 2) * self.emb_size,
                                 hidden_size=self.mlp_size, output_size=1, num_hidden_layers=self.mlp_depth,
                                 bias=True, drop_module=self.drop_module)

    def initialize_memory(self):
        """Initialize the memory to null."""
        mem = torch.zeros(self.num_cells, self.mem_size).cuda()
        ent_counter = torch.tensor([0 for i in range(self.num_cells)]).cuda()
        last_mention_idx = torch.tensor([0 for i in range(self.num_cells)]).cuda()
        return mem, ent_counter, last_mention_idx

    def get_overwrite_ign_mask(self, ent_counter):
        free_cell_mask = (ent_counter == 0.0)
        if torch.max(free_cell_mask) > 0:
            free_cell_mask = free_cell_mask * torch.arange(self.num_cells + 1, 1, -1).cuda()
            free_cell_idx = torch.max(free_cell_mask, 0)[1]
            last_unused_cell = free_cell_idx.item()
            mask = torch.zeros(self.num_cells + 2).cuda()
            mask[last_unused_cell] = 1.0
            mask[-2] = 1.0  # Not a mention
            mask[-1] = 0.0  # Not worth tracking
            return mask
        else:
            return torch.ones(self.num_cells + 2).cuda()

    def get_all_mask(self, ent_counter):
        coref_mask = self.get_coref_mask(ent_counter)
        overwrite_ign_mask = self.get_overwrite_ign_mask(ent_counter)
        return torch.cat([coref_mask, overwrite_ign_mask], dim=0)

    def forward(self, mention_emb_list, actions, mentions, teacher_forcing=False):
        pass
