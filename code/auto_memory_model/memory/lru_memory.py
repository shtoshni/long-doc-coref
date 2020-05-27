import torch
from auto_memory_model.memory.base_fixed_memory import BaseFixedMemory


class LRUMemory(BaseFixedMemory):
    def __init__(self, **kwargs):
        super(LRUMemory, self).__init__(**kwargs)

    def predict_action(self, query_vector, ment_score, mem_vectors, last_ment_vectors,
                       ent_counter, feature_embs, ment_feature_embs, lru_list):
        coref_new_scores = self.get_coref_new_log_prob(
            query_vector, ment_score, mem_vectors, last_ment_vectors, ent_counter, feature_embs)
        # Overwrite vs Ignore
        lru_cell = lru_list[0]
        mem_fert_input = torch.cat([mem_vectors[lru_cell, :], feature_embs[lru_cell, :]], dim=0)
        mem_fert = self.fert_mlp(mem_fert_input)

        ment_fert = self.ment_fert_mlp(torch.cat([query_vector, ment_feature_embs], dim=-1))
        over_ign_score = torch.cat([mem_fert, torch.tensor([-ment_score]).cuda(), ment_fert], dim=0)

        return coref_new_scores, over_ign_score

    def forward(self, mention_emb_list, mention_scores, gt_actions, metadata,
                teacher_forcing=False):
        # Initialize memory
        mem_vectors, ent_counter, last_mention_idx = self.initialize_memory()
        last_ment_vectors = torch.zeros_like(mem_vectors)
        lru_list = list(range(self.num_cells))

        if self.entity_rep == 'lstm':
            cell_vectors = torch.zeros_like(mem_vectors)

        action_logit_list = []
        action_list = []  # argmax actions
        last_action_str = '<s>'

        for ment_idx, (ment_emb, ment_score, (gt_cell_idx, gt_action_str)) in \
                enumerate(zip(mention_emb_list, mention_scores, gt_actions)):
            query_vector = ment_emb
            metadata['last_action'] = self.action_str_to_idx[last_action_str]
            feature_embs = self.get_feature_embs(ment_idx, last_mention_idx, ent_counter, metadata)
            ment_feature_embs = self.get_ment_feature_embs(metadata)

            coref_new_scores, over_ign_score = self.predict_action(
                query_vector, ment_score, mem_vectors, last_ment_vectors,
                ent_counter, feature_embs, ment_feature_embs, lru_list)

            coref_new_max_idx = torch.argmax(coref_new_scores).item()
            if coref_new_max_idx < self.num_cells:
                pred_action_str = 'c'
                pred_cell_idx = coref_new_max_idx
            else:
                over_ign_max_idx = torch.argmax(over_ign_score).item()
                if over_ign_max_idx == 0:
                    pred_action_str = 'o'
                    pred_cell_idx = lru_list[0]
                elif over_ign_max_idx == 1:
                    pred_action_str = 'i'
                    pred_cell_idx = -1
                elif over_ign_max_idx == 2:
                    pred_action_str = 'n'
                    pred_cell_idx = -1

            # During training this records the next actions  - during testing it records the
            # predicted sequence of actions
            action_logit_list.append((coref_new_scores, over_ign_score))
            action_list.append((pred_cell_idx, pred_action_str))

            if self.training or teacher_forcing:
                # Training - Operate over the ground truth
                action_str = gt_action_str
                cell_idx = gt_cell_idx
            else:
                # Inference time
                action_str = pred_action_str
                cell_idx = pred_cell_idx

            last_action_str = action_str

            # Update the memory
            rep_query_vector = query_vector.repeat(self.num_cells, 1)
            cell_mask = (torch.arange(0, self.num_cells) == cell_idx).float().cuda()
            mask = torch.unsqueeze(cell_mask, dim=1)
            mask = mask.repeat(1, self.mem_size)

            if action_str == 'c':
                # Update memory vector corresponding to cell_idx
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
                    total_counts = torch.unsqueeze((ent_counter + 1).float(), dim=1)
                    pool_vec_num = (mem_vectors * torch.unsqueeze(ent_counter, dim=1)
                                    + rep_query_vector)
                    avg_pool_vec = pool_vec_num/total_counts
                    mem_vectors = mem_vectors * (1 - mask) + mask * avg_pool_vec

                # Update last mention vector
                last_ment_vectors = last_ment_vectors * (1 - mask) + mask * rep_query_vector
                ent_counter = ent_counter + cell_mask
                last_mention_idx[cell_idx] = ment_idx
            elif action_str == 'o':
                # Replace the cell content
                mem_vectors = mem_vectors * (1 - mask) + mask * rep_query_vector
                # Update last mention vector
                last_ment_vectors = last_ment_vectors * (1 - mask) + mask * rep_query_vector

                ent_counter = ent_counter * (1 - cell_mask) + cell_mask
                last_mention_idx[cell_idx] = ment_idx

            if action_str in ['o', 'c']:
                # Coref or overwrite was chosen
                lru_list.remove(cell_idx)
                lru_list.append(cell_idx)

        return action_logit_list, action_list
