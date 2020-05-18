import torch
from auto_memory_model.memory.base_fixed_memory import BaseFixedMemory


class LearnedFixedMemory(BaseFixedMemory):
    def __init__(self, **kwargs):
        super(LearnedFixedMemory, self).__init__(**kwargs)

    def predict_action(self, query_vector, mem_vectors, last_ment_vectors,
                       ment_idx, ent_counter, last_mention_idx):
        distance_embs = self.get_distance_emb(ment_idx, last_mention_idx)
        counter_embs = self.get_counter_emb(ent_counter)

        coref_new_log_prob = self.get_coref_new_log_prob(
            query_vector, mem_vectors, last_ment_vectors,
            ent_counter, distance_embs, counter_embs)
        # Fertility Score
        # Memory fertility scores
        mem_fert_scores = self.mem_fert_mlp(
            torch.cat([mem_vectors, distance_embs, counter_embs], dim=-1))
        mem_fert_scores = torch.squeeze(mem_fert_scores, dim=-1)
        # Mention fertility score
        ment_fert_score = self.ment_fert_mlp(query_vector)

        fert_score = torch.cat([mem_fert_scores, ment_fert_score], dim=0)
        overwrite_ign_mask = self.get_overwrite_ign_mask(ent_counter)
        # print(overwrite_ign_mask)
        overwrite_ign_scores = fert_score * overwrite_ign_mask + (1 - overwrite_ign_mask) * (-1e4)
        overwrite_ign_log_prob = torch.nn.functional.log_softmax(overwrite_ign_scores, dim=0)

        norm_overwrite_ign_log_prob = (coref_new_log_prob[self.num_cells] + overwrite_ign_log_prob)
        all_log_prob = torch.cat([coref_new_log_prob[:self.num_cells],
                                  norm_overwrite_ign_log_prob], dim=0)
        return all_log_prob, coref_new_log_prob, overwrite_ign_log_prob

    def forward(self, mention_emb_list, actions, mentions,
                teacher_forcing=False):
        # Initialize memory
        mem_vectors, ent_counter, last_mention_idx = self.initialize_memory()
        last_ment_vectors = torch.zeros_like(mem_vectors)

        if self.entity_rep == 'lstm':
            cell_vectors = torch.zeros_like(mem_vectors)

        action_logit_list = []
        action_list = []  # argmax actions
        action_str = '<s>'

        for ment_idx, (ment_emb, (span_start, span_end), (gt_cell_idx, gt_action_str)) in \
                enumerate(zip(mention_emb_list, mentions, actions)):
            width_bucket = self.get_mention_width_bucket(span_end - span_start)
            width_embedding = self.width_embeddings(torch.tensor(width_bucket).long().cuda())
            last_action_emb = self.get_last_action_emb(action_str)
            query_vector = self.query_projector(
                torch.cat([ment_emb, last_action_emb, width_embedding], dim=0))

            all_log_probs, coref_new_log_prob, overwrite_ign_log_prob = self.predict_action(
                query_vector, mem_vectors, last_ment_vectors,
                ment_idx, ent_counter, last_mention_idx)

            action_logit_list.append((coref_new_log_prob, overwrite_ign_log_prob))

            pred_max_idx = torch.argmax(all_log_probs).item()
            pred_cell_idx = pred_max_idx % self.num_cells
            pred_action_idx = pred_max_idx // self.num_cells
            pred_action_str = self.action_idx_to_str[pred_action_idx]
            # During training this records the next actions  - during testing it records the
            # predicted sequence of actions
            action_list.append((pred_cell_idx, pred_action_str))

            if self.training or teacher_forcing:
                # Training - Operate over the ground truth
                action_str = gt_action_str
                cell_idx = gt_cell_idx
            else:
                # Inference time
                action_str = pred_action_str
                cell_idx = pred_cell_idx

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
