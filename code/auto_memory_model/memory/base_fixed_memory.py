import torch
import torch.nn as nn
from pytorch_utils.modules import MLP
from auto_memory_model.memory.base_memory import BaseMemory


class BaseFixedMemory(BaseMemory):
    def __init__(self, num_cells=10, **kwargs):
        super(BaseFixedMemory, self).__init__(**kwargs)

        self.num_cells = num_cells

        # Fixed memory cells need to predict fertility of memory and mentions
        self.fert_mlp = MLP(input_size=self.mem_size + self.emb_size,
                            hidden_size=self.mlp_size, output_size=1, num_hidden_layers=self.mlp_depth,
                            bias=True, drop_module=self.drop_module)
        self.ment_fert_mlp = MLP(input_size=self.mem_size, hidden_size=self.mlp_size, output_size=1,
                                 num_hidden_layers=self.mlp_depth,
                                 bias=True, drop_module=self.drop_module)

    def initialize_memory(self):
        """Initialize the memory to null."""
        mem = torch.zeros(self.num_cells, self.mem_size).cuda()
        ent_counter = torch.tensor([0 for i in range(self.num_cells)]).cuda()
        last_mention_idx = torch.tensor([0 for i in range(self.num_cells)]).cuda()
        return mem, ent_counter, last_mention_idx

    def get_overwrite_ign_mask(self, ent_counter):
        # last_unused_cell = None
        # for cell_idx, cell_count in enumerate(ent_counter.tolist()):
        #     if int(cell_count) == 0:
        #         last_unused_cell = cell_idx
        #         break

        free_cell_mask = (ent_counter == 0.0)
        if torch.max(free_cell_mask) > 0:
            free_cell_mask = free_cell_mask * torch.arange(self.num_cells + 1, 1, -1).cuda()
            free_cell_idx = torch.max(free_cell_mask, 0)[1]
            last_unused_cell = free_cell_idx.item()
            # score_arr = ([0] * last_unused_cell + [1]
            #              + [0] * (self.num_cells - last_unused_cell - 1) + [1])
            # # print(score_arr, last_unused_cell, ent_counter)
            # # Return the overwrite probability vector combined with ignore prob (last entry)
            # return torch.tensor(score_arr).cuda().float()
            mask = torch.cuda.FloatTensor(1 + self.num_cells).fill_(0)
            mask[last_unused_cell] = 1.0
            mask[-1] = 1.0
            return mask
        else:
            return torch.cuda.FloatTensor(1 + self.num_cells).fill_(1)

    def get_all_mask(self, ent_counter):
        coref_mask = self.get_coref_mask(ent_counter)
        overwrite_ign_mask = self.get_overwrite_ign_mask(ent_counter)
        return torch.cat([coref_mask, overwrite_ign_mask], dim=0)

    def forward(self, mention_emb_list, actions, mentions, teacher_forcing=False):
        pass
