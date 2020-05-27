import torch
from auto_memory_model.memory.base_fixed_memory import BaseFixedMemory


class LearnedFixedMemory(BaseFixedMemory):
    def __init__(self, **kwargs):
        super(LearnedFixedMemory, self).__init__(**kwargs)

    def predict_action(self, query_vector, ment_score, mem_vectors, last_ment_vectors,
                       ent_counter, feature_embs, ment_feature_embs):
        coref_new_scores = self.get_coref_new_log_prob(
            query_vector, ment_score, mem_vectors, last_ment_vectors, ent_counter, feature_embs)
        # Fertility Score
        mem_fert_input = torch.cat([mem_vectors, feature_embs], dim=-1)
        mem_fert = torch.squeeze(self.fert_mlp(mem_fert_input), dim=-1)

        ment_fert = self.ment_fert_mlp(torch.cat([query_vector, ment_feature_embs], dim=-1))
        overwrite_ign_no_space_scores = torch.cat([mem_fert, -ment_score, ment_fert], dim=0)

        overwrite_ign_mask = self.get_overwrite_ign_mask(ent_counter)
        overwrite_ign_no_space_scores = overwrite_ign_no_space_scores * overwrite_ign_mask + (1 - overwrite_ign_mask) * (-1e4)

        return coref_new_scores, overwrite_ign_no_space_scores

    def interpret_scores(self, coref_new_scores, overwrite_ign_no_space_scores):
        pred_max_idx = torch.argmax(coref_new_scores).item()
        if pred_max_idx < self.num_cells:
            # Coref
            return pred_max_idx, 'c'
        elif pred_max_idx == self.num_cells:
            # Overwrite/Ignore
            over_max_idx = torch.argmax(overwrite_ign_no_space_scores).item()
            if over_max_idx < self.num_cells:
                return over_max_idx, 'o'
            elif over_max_idx == self.num_cells:
                return -1, 'i'
            elif over_max_idx == self.num_cells + 1:
                # No space
                return -1, 'n'
        else:
            raise NotImplementedError

    def forward(self, mention_emb_list, mention_scores, gt_actions, metadata,
                teacher_forcing=False):
        # Initialize memory
        mem_vectors, ent_counter, last_mention_idx = self.initialize_memory()
        last_ment_vectors = torch.zeros_like(mem_vectors)

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

            coref_new_scores, overwrite_ign_no_space_scores = self.predict_action(
                query_vector, ment_score, mem_vectors, last_ment_vectors,
                ent_counter, feature_embs, ment_feature_embs)

            pred_cell_idx, pred_action_str = self.interpret_scores(coref_new_scores, overwrite_ign_no_space_scores)
            # During training this records the next actions  - during testing it records the
            # predicted sequence of actions
            action_list.append((pred_cell_idx, pred_action_str))
            action_logit_list.append((coref_new_scores, overwrite_ign_no_space_scores))

            if self.training or teacher_forcing:
                # Training - Operate over the ground truth
                action_str = gt_action_str
                cell_idx = gt_cell_idx
            else:
                # Inference time
                action_str = pred_action_str
                cell_idx = pred_cell_idx

            # Update last action
            last_action_str = action_str

            # Update the memory
            rep_query_vector = query_vector.repeat(self.num_cells, 1)  # M x H
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

        return action_logit_list, action_list
