import torch
import torch.nn as nn
from pytorch_utils.modules import MLP
# from pytorch_memlab import profile, profile_every


class DynamicMemory(nn.Module):
    def __init__(self, num_cells=10, hsize=300, mlp_size=200, mem_size=None,
                 drop_module=None, emb_size=20, entity_rep='max',
                 use_last_mention=False,
                 **kwargs):
        super(DynamicMemory, self).__init__()
        # self.query_mlp = query_mlp
        self.hsize = hsize
        self.num_cells = num_cells
        self.mem_size = (mem_size if mem_size is not None else hsize)
        self.mlp_size = mlp_size
        self.emb_size = emb_size
        self.entity_rep = entity_rep

        self.drop_module = drop_module

        self.action_str_to_idx = {'c': 0, 'o': 1}
        self.action_idx_to_str = ['c', 'o']

        self.use_last_mention = use_last_mention

        if self.entity_rep == 'lstm':
            self.mem_rnn = nn.LSTMCell(
                input_size=self.mem_size,
                hidden_size=self.mem_size)
        elif self.entity_rep == 'gru':
            self.mem_rnn = nn.GRUCell(
                input_size=self.mem_size,
                hidden_size=self.mem_size)

        # CHANGE THIS PART
        self.query_projector = nn.Linear(self.hsize + 2 * self.emb_size, self.mem_size)
        # self.query_mlp = MLP(input_size=hsize + 2 * self.emb_size,
        #                      hidden_size=self.mlp_size, output_size=self.mem_size,
        #                      num_layers=1, bias=True, drop_module=drop_module)

        self.mem_coref_mlp = MLP(3 * self.mem_size + 2 * self.emb_size, self.mlp_size, 1,
                                 num_layers=2, bias=True, drop_module=drop_module)
        if self.use_last_mention:
            self.ment_coref_mlp = MLP(3 * self.mem_size, self.mlp_size, 1,
                                      num_layers=1, bias=True, drop_module=drop_module)

        # self.last_op_emb = nn.Embedding(2 * num_cells + 2, self.emb_size)
        self.last_action_emb = nn.Embedding(4, self.emb_size)
        self.distance_embeddings = nn.Embedding(11, self.emb_size)
        self.width_embeddings = nn.Embedding(30, self.emb_size)
        self.counter_embeddings = nn.Embedding(11, self.emb_size)

        # self.emb_max_idx = nn.Embedding(2 * self.num_cells + 2, self.emb_size)
        # Write vector
        # self.U_key = nn.Linear(2 * self.mem_size, self.mem_size, bias=True)

    def initialize_memory(self):
        """Initialize the memory to null."""
        mem = torch.zeros(self.num_cells, self.mem_size).cuda()
        self.counter = torch.tensor([0 for i in range(self.num_cells)],
                                    requires_grad=False).cuda()
        self.lru_list = list(range(self.num_cells))
        self.last_mention_idx = [0 for _ in range(self.num_cells)]
        return mem

    def get_coref_mask(self):
        cell_mask = (self.counter > 0.0).float().cuda()
        return cell_mask

    def get_overwrite_mask(self):
        lru_cell = self.lru_list[0]
        mask_arr = ([0] * lru_cell + [1] + [0] * (self.num_cells - lru_cell - 1))

        return torch.tensor(mask_arr).cuda().float()

    def get_all_mask(self):
        coref_mask = self.get_coref_mask()
        overwrite_mask = self.get_overwrite_mask()
        return torch.cat([coref_mask, overwrite_mask], dim=0)

    def get_distance_bucket(self, dist):
        assert(dist >= 0)
        if dist < 5:
            return dist+1

        elif dist >= 5 and dist <= 7:
            return 6
        elif dist >= 8 and dist <= 15:
            return 7
        elif dist >= 16 and dist <= 31:
            return 8
        elif dist >= 32 and dist <= 63:
            return 9

        return 10

    def get_counter_bucket(self, count):
        assert(count >= 0)
        if count <= 5:
            return count

        elif count > 5 and count <= 7:
            return 6
        elif count >= 8 and count <= 15:
            return 7
        elif count >= 16 and count <= 31:
            return 8
        elif count >= 32 and count <= 63:
            return 9

        return 10

    def get_mention_width_bucket(self, width):
        if width < 29:
            return width

        return 29

    def get_distance_emb(self, ment_idx):
        distance_buckets = [self.get_distance_bucket(ment_idx - self.last_mention_idx[cell_idx])
                            for cell_idx in range(self.num_cells)]
        distance_tens = torch.tensor(distance_buckets).long().cuda()
        distance_embs = self.distance_embeddings(distance_tens)
        return distance_embs

    def get_counter_emb(self):
        counter_buckets = [self.get_counter_bucket(self.counter[cell_idx])
                           for cell_idx in range(self.num_cells)]
        counter_tens = torch.tensor(counter_buckets).long().cuda()
        counter_embs = self.counter_embeddings(counter_tens)
        return counter_embs

    def predict_everything(self, ment_idx, mem_vectors, last_ment_vectors,
                           query_vector, rep_query_vector):
        """Calculate similarity between query_vector and mem_vectors.
        query_vector: M x H
        mem_vectors: H
        """
        distance_embs = self.get_distance_emb(ment_idx)
        counter_embs = self.get_counter_emb()
        # Coref Score
        global_vec = torch.cat([mem_vectors, rep_query_vector,
                                mem_vectors * rep_query_vector,
                                distance_embs, counter_embs], dim=-1)

        global_score = self.mem_coref_mlp(global_vec)
        coref_score = torch.squeeze(global_score, dim=-1)  # M

        if self.use_last_mention:
            last_ment_vec = torch.cat(
                [last_ment_vectors, rep_query_vector,
                 last_ment_vectors * rep_query_vector], dim=-1)
            last_ment_score = torch.squeeze(self.ment_coref_mlp(last_ment_vec), dim=-1)

            coref_score = coref_score + last_ment_score  # M

        coref_new_prob = torch.cat(([coref_score, torch.tensor([0.0]).cuda()]), dim=0)
        return coref_new_prob

    def get_last_op_emb(self, max_idx):
        num_ops = 2 * self.num_cells + 1
        if max_idx is None:
            max_idx = num_ops
        return self.last_op_emb(torch.tensor(max_idx).cuda())

    def get_last_action_emb(self, max_idx):
        if max_idx is None:
            action_emb = 3
        else:
            action_emb = max_idx // self.num_cells
        return self.last_action_emb(torch.tensor(action_emb).cuda())

    # @profile_every(10)
    def forward(self, mention_emb_list, actions, mentions,
                get_next_pred_errors=False):
        """Read excerpts.
        hidden_state_list: list of B x H tensors
        span_end_to_span_starts: A dictionary mapping span end to span starts
        """
        # Initialize memory
        mem_vectors = self.initialize_memory()
        last_ment_vectors = torch.zeros_like(mem_vectors)

        if self.entity_rep == 'lstm':
            cell_vectors = torch.zeros_like(mem_vectors)

        query_vector = torch.zeros((1, self.mem_size)).cuda()
        # query_cell_vector = torch.zeros((1, self.mem_size)).cuda()

        action_logit_list = []
        action_list = []  # argmax actions

        max_idx = None

        for ment_idx, (ment_emb, (span_start, span_end), (cell_idx, action_str)) in \
                enumerate(zip(mention_emb_list, mentions, actions)):
            width_bucket = self.get_mention_width_bucket(span_end - span_start)
            width_embedding = self.width_embeddings(torch.tensor(width_bucket).long().cuda())
            last_action_emb = self.get_last_action_emb(max_idx)
            query_vector = self.query_projector(
                torch.cat([ment_emb, last_action_emb, width_embedding], dim=0))
            # input_vec = torch.cat([ment_emb, last_action_emb, width_embedding], dim=0)
            # query_vector = self.query_mlp(input_vec)
            # query_vector = ment_emb

            # M x H
            rep_query_vector = query_vector.repeat(self.num_cells, 1)

            action_logit = self.predict_everything(
                ment_idx, mem_vectors, last_ment_vectors,
                query_vector, rep_query_vector)

            # Decrease counter
            # action_prob = torch.cat([coref_prob, overwrite_ignore_probs], dim=0)
            action_logit_list.append(action_logit)

            pred_max_idx = torch.argmax(action_logit).item()
            pred_action_idx = pred_max_idx // self.num_cells
            pred_action_str = self.action_idx_to_str[pred_action_idx]
            pred_cell_idx = pred_max_idx % self.num_cells

            if pred_action_str == 'o':
                # In case of overwrite pick the least recently used memory cell
                pred_cell_idx = self.lru_list[0]
            # During training this records the wrong next actions  - during testing it records the
            # predicted sequence of actions
            action_list.append((pred_cell_idx, pred_action_str))

            if self.training or get_next_pred_errors:
                # Training - Operate over the ground truth
                action_idx = self.action_str_to_idx[action_str]
                max_idx = action_idx * self.num_cells + cell_idx
                # if action_str == 'c':
                #     pass
                # else:
                #     # Don't follow the GT for LRU memory
                #     cell_idx = self.lru_list[0]
            else:
                # Inference time
                max_idx, cell_idx = pred_max_idx, pred_cell_idx

            # Update the memory
            cell_mask = (torch.arange(0, self.num_cells) == cell_idx).float().cuda()
            mask = torch.unsqueeze(cell_mask, dim=1)
            mask = mask.repeat(1, self.mem_size)

            if action_str == 'c':
                # Update memory vector corresponding to cell_idx
                # Mem RNN
                if self.entity_rep == 'lstm':
                    cand_vec, cand_cell_vec = self.mem_rnn(
                            rep_query_vector, (mem_vectors, cell_vectors))
                    cell_vectors = cell_vectors * (1 - mask) + mask * cand_cell_vec
                elif self.entity_rep == 'gru':
                    cand_vec = self.mem_rnn(rep_query_vector, mem_vectors)
                    mem_vectors = mem_vectors * (1 - mask) + mask * cand_vec
                elif self.entity_rep == 'max':
                    # Max pool coref operation
                    max_pool_vec = torch.max(
                        torch.stack([mem_vectors, rep_query_vector], dim=0), dim=0)[0]
                    mem_vectors = mem_vectors * (1 - mask) + mask * max_pool_vec
                elif self.entity_rep == 'avg':
                    total_counts = torch.unsqueeze((self.counter + 1).float(), dim=1)
                    pool_vec_num = (mem_vectors * torch.unsqueeze(self.counter, dim=1)
                                    + rep_query_vector)
                    avg_pool_vec = pool_vec_num/total_counts
                    mem_vectors = mem_vectors * (1 - mask) + mask * avg_pool_vec

                # Update last mention vector
                last_ment_vectors = last_ment_vectors * (1 - mask) + mask * rep_query_vector
                self.counter = self.counter + cell_mask
                self.last_mention_idx[cell_idx] = ment_idx
            elif action_str == 'o':
                # Replace the cell content
                mem_vectors = mem_vectors * (1 - mask) + mask * rep_query_vector

                # Update last mention vector
                last_ment_vectors = last_ment_vectors * (1 - mask) + mask * rep_query_vector

                self.counter = self.counter * (1 - cell_mask) + cell_mask
                self.last_mention_idx[cell_idx] = ment_idx

            self.lru_list.remove(cell_idx)
            self.lru_list.append(cell_idx)

        return action_logit_list, action_list
