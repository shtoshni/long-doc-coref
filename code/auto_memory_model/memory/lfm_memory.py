import torch
from auto_memory_model.memory.base_fixed_memory import BaseFixedMemory


class LearnedFixedMemory(BaseFixedMemory):
    def __init__(self, **kwargs):
        super(LearnedFixedMemory, self).__init__(**kwargs)

    def predict_action(self, query_vector, ment_score, mem_vectors, last_ment_vectors,
                       ment_idx, ent_counter, last_mention_idx):
        distance_embs = self.get_distance_emb(ment_idx, last_mention_idx)
        counter_embs = torch.zeros_like(distance_embs).cuda()
        # counter_embs = self.get_counter_emb(ent_counter)

        coref_new_scores = self.get_coref_new_log_prob(
            query_vector, ment_score, mem_vectors, last_ment_vectors,
            ent_counter, distance_embs, counter_embs)
        # Fertility Score
        # # Memory + Mention fertility input
        # mem_fert_input = torch.cat([mem_vectors, distance_embs, counter_embs], dim=-1)
        # # Mention fertility input
        # ment_distance_emb = torch.squeeze(self.distance_embeddings(torch.tensor([0]).cuda()), dim=0)
        # ment_counter_emb = torch.squeeze(self.counter_embeddings(torch.tensor([0]).cuda()), dim=0)
        # ment_fert_input = torch.unsqueeze(
        #     torch.cat([query_vector, ment_distance_emb, ment_counter_emb], dim=0), dim=0)
        # # Fertility scores
        # fert_input = torch.cat([mem_fert_input, ment_fert_input], dim=0)
        # fert_scores = torch.squeeze(self.fert_mlp(fert_input), dim=-1)
        # fert_scores[self.num_cells] -= ment_score

        mem_fert_input = torch.cat([mem_vectors, distance_embs, counter_embs], dim=-1)
        mem_fert = torch.squeeze(self.fert_mlp(mem_fert_input), dim=-1)

        ment_fert = self.ment_fert_mlp(query_vector) - ment_score
        fert_scores = torch.cat([mem_fert, ment_fert], dim=0)

        overwrite_ign_mask = self.get_overwrite_ign_mask(ent_counter)
        overwrite_ign_scores = fert_scores * overwrite_ign_mask + (1 - overwrite_ign_mask) * (-1e4)

        return coref_new_scores, overwrite_ign_scores

    def interpret_scores(self, coref_new_scores, overwrite_ign_scores):
        pred_max_idx = torch.argmax(coref_new_scores).item()
        if pred_max_idx < self.num_cells:
            # Coref
            return pred_max_idx, 'c'
        elif pred_max_idx == self.num_cells:
            # Overwrite/Ignore
            over_max_idx = torch.argmax(overwrite_ign_scores).item()
            if over_max_idx < self.num_cells:
                return over_max_idx, 'o'
            else:
                return -1, 'i'
        else:
            raise NotImplementedError

    def forward(self, mention_emb_list, mention_scores, gt_actions,
                teacher_forcing=False):
        # Initialize memory
        mem_vectors, ent_counter, last_mention_idx = self.initialize_memory()
        last_ment_vectors = torch.zeros_like(mem_vectors)

        if self.entity_rep == 'lstm':
            cell_vectors = torch.zeros_like(mem_vectors)

        action_logit_list = []
        action_list = []  # argmax actions
        # action_str = '<s>'

        for ment_idx, (ment_emb, ment_score, (gt_cell_idx, gt_action_str)) in \
                enumerate(zip(mention_emb_list, mention_scores, gt_actions)):
            # last_action_emb = self.get_last_action_emb(action_str)
            # query_vector = self.query_projector(
            #     torch.cat([ment_emb, last_action_emb], dim=0))
            query_vector = ment_emb

            coref_new_scores, overwrite_ign_scores = self.predict_action(
                query_vector, ment_score, mem_vectors, last_ment_vectors,
                ment_idx, ent_counter, last_mention_idx)

            action_logit_list.append((coref_new_scores, overwrite_ign_scores))

            # pred_max_idx = torch.argmax(all_log_probs).item()
            # pred_cell_idx = pred_max_idx % self.num_cells
            # pred_action_idx = pred_max_idx // self.num_cells
            pred_cell_idx, pred_action_str = self.interpret_scores(coref_new_scores, overwrite_ign_scores)
            # pred_action_str = self.action_idx_to_str[pred_action_idx]
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
