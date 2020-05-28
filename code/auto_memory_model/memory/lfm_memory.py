import torch
from auto_memory_model.memory.base_fixed_memory import BaseFixedMemory
import numpy as np

class LearnedFixedMemory(BaseFixedMemory):
    def __init__(self, **kwargs):
        super(LearnedFixedMemory, self).__init__(**kwargs)

    def get_overwrite_ign_mask(self, ent_counter):
        free_cell_mask = (ent_counter == 0.0)
        if torch.max(free_cell_mask) > 0:
            free_cell_mask = free_cell_mask * torch.arange(self.num_cells + 1, 1, -1).cuda()
            free_cell_idx = torch.max(free_cell_mask, 0)[1]
            last_unused_cell = free_cell_idx.item()
            mask = torch.zeros(self.num_cells + 2).cuda()
            mask[last_unused_cell] = 1.0
            mask[-1] = 1.0  # Not a mention
            return mask
        else:
            return torch.ones(self.num_cells + 2).cuda()

    def predict_new_or_ignore(self, query_vector, ment_score, mem_vectors,
                              ent_counter, feature_embs, ment_feature_embs):
        # Fertility Score
        # mem_fert_input = torch.cat([mem_vectors, feature_embs], dim=-1)
        # mem_fert = torch.squeeze(self.fert_mlp(mem_fert_input), dim=-1)
        #
        # ment_fert = self.ment_fert_mlp(torch.cat([query_vector, ment_feature_embs], dim=-1))
        # overwrite_ign_no_space_scores = torch.cat([mem_fert, ment_fert, -ment_score], dim=0)

        mem_fert_input = torch.cat([mem_vectors, feature_embs], dim=-1)
        ment_fert_input = torch.unsqueeze(torch.cat([query_vector, ment_feature_embs], dim=-1), dim=0)
        fert_input = torch.cat([mem_fert_input, ment_fert_input], dim=0)
        fert_scores = torch.squeeze(self.fert_mlp(fert_input), dim=-1)

        overwrite_ign_no_space_scores = torch.cat([fert_scores, -ment_score], dim=0)

        overwrite_ign_mask = self.get_overwrite_ign_mask(ent_counter)
        overwrite_ign_no_space_scores = overwrite_ign_no_space_scores * overwrite_ign_mask + (1 - overwrite_ign_mask) * (-1e4)

        return overwrite_ign_no_space_scores

    def interpret_coref_new_score(self, coref_new_scores):
        pred_max_idx = torch.argmax(coref_new_scores).item()
        if pred_max_idx < self.num_cells:
            # Coref
            return pred_max_idx, 'c'
        elif pred_max_idx == self.num_cells:
            # Overwrite/No Space/Ignore
            return -1, None

    def interpret_new_ignore_score(self, overwrite_ign_no_space_scores):
        over_max_idx = torch.argmax(overwrite_ign_no_space_scores).item()
        if over_max_idx < self.num_cells:
            return over_max_idx, 'o'
        elif over_max_idx == self.num_cells:
            return -1, 'n'
        elif over_max_idx == self.num_cells + 1:
            # No space
            return -1, 'i'

    def forward(self, mention_emb_list, mention_scores, gt_actions, metadata, rand_fl_list,
                teacher_forcing=False):
        # Initialize memory
        mem_vectors, ent_counter, last_mention_idx = self.initialize_memory()
        last_ment_vectors = torch.zeros_like(mem_vectors)

        if self.entity_rep == 'lstm':
            cell_vectors = torch.zeros_like(mem_vectors)

        action_list = []
        coref_new_list = []  # argmax actions
        new_ignore_list = []
        last_action_str = '<s>'

        follow_gt = self.training or teacher_forcing

        for ment_idx, (ment_emb, ment_score, (gt_cell_idx, gt_action_str)) in \
                enumerate(zip(mention_emb_list, mention_scores, gt_actions)):
            query_vector = ment_emb
            metadata['last_action'] = self.action_str_to_idx[last_action_str]
            feature_embs = self.get_feature_embs(ment_idx, last_mention_idx, ent_counter, metadata)
            ment_feature_embs = self.get_ment_feature_embs(metadata)

            # cond1 = (follow_gt and gt_action_str == 'i' and rand_fl_list[ment_idx] > self.sample_ignores)
            if not (follow_gt and gt_action_str == 'i' and rand_fl_list[ment_idx] > self.sample_ignores):
                coref_new_scores = self.get_coref_new_log_prob(query_vector, ment_score, mem_vectors,
                                                               last_ment_vectors, ent_counter, feature_embs)
                coref_new_list.append(coref_new_scores)

                pred_cell_idx, pred_action_str = self.interpret_coref_new_score(coref_new_scores)

                if (follow_gt and gt_action_str != 'c') or ((not follow_gt) and pred_action_str != 'c'):
                    new_ignore_score = self.predict_new_or_ignore(
                        query_vector, ment_score, mem_vectors,
                        ent_counter, feature_embs, ment_feature_embs)
                    pred_cell_idx, pred_action_str = self.interpret_new_ignore_score(new_ignore_score)
                    new_ignore_list.append(new_ignore_score)

                # During training this records the next actions  - during testing it records the
                # predicted sequence of actions
                action_list.append((pred_cell_idx, pred_action_str))

            if follow_gt:
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
                elif self.entity_rep == 'rec_avg':
                    avg_pool_vec = (mem_vectors + rep_query_vector)/2
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

        return coref_new_list, new_ignore_list, action_list
