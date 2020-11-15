import torch
from auto_memory_model.memory import BaseFixedMemory


class LearnedFixedMemory(BaseFixedMemory):
    def __init__(self, **kwargs):
        super(LearnedFixedMemory, self).__init__(**kwargs)

    def get_overwrite_ign_mask(self, ent_counter):
        free_cell_mask = (ent_counter == 0.0).float().to(device=self.device)
        if torch.max(free_cell_mask) > 0:
            free_cell_mask = free_cell_mask * torch.arange(self.num_cells + 1, 1, -1).float().to(device=self.device)
            free_cell_idx = torch.max(free_cell_mask, 0)[1]
            last_unused_cell = free_cell_idx.item()
            mask = torch.zeros(self.num_cells + 2).to(device=self.device)
            mask[last_unused_cell] = 1.0
            mask[-1] = 1.0  # Not a mention
            return mask
        else:
            return torch.ones(self.num_cells + 2).to(device=self.device)

    def predict_new_or_ignore(self, query_vector, ment_score, mem_vectors,
                              ent_counter, feature_embs, ment_feature_embs):
        # Fertility Score
        mem_fert_input = torch.cat([mem_vectors, feature_embs], dim=-1)
        ment_fert_input = torch.unsqueeze(torch.cat([query_vector, ment_feature_embs], dim=-1), dim=0)
        fert_input = torch.cat([mem_fert_input, ment_fert_input], dim=0)

        # del mem_fert_input
        # del ment_fert_input

        fert_scores = self.fert_mlp(fert_input)
        fert_scores = torch.squeeze(fert_scores, dim=-1)
        # del fert_input

        new_or_ignore_scores = torch.cat([fert_scores, -ment_score], dim=0)

        overwrite_ign_mask = self.get_overwrite_ign_mask(ent_counter)
        new_or_ignore_scores = new_or_ignore_scores * overwrite_ign_mask + (1 - overwrite_ign_mask) * (-1e4)

        return new_or_ignore_scores

    def interpret_new_ignore_score(self, overwrite_ign_no_space_scores):
        max_idx = torch.argmax(overwrite_ign_no_space_scores).item()
        if max_idx < self.num_cells:
            return max_idx, 'o'
        elif max_idx == self.num_cells:
            # No space - The new entity is not "fertile"
            return -1, 'n'
        elif max_idx == self.num_cells + 1:
            # Invalid mention - Low mention score
            return -1, 'i'

    def forward(self, mention_emb_list, mention_scores, gt_actions, metadata, rand_fl_list,
                teacher_forcing=False):
        # Initialize memory
        mem_vectors, ent_counter, last_mention_idx = self.initialize_memory()

        action_list = []
        coref_new_list = []  # argmax actions
        new_ignore_list = []
        last_action_str = '<s>'

        follow_gt = self.training or teacher_forcing

        for ment_idx, (ment_emb, ment_score, (gt_cell_idx, gt_action_str)) in \
                enumerate(zip(mention_emb_list, mention_scores, gt_actions)):
            query_vector = ment_emb

            if not (follow_gt and gt_action_str == 'i' and rand_fl_list[ment_idx] > self.sample_invalid):
                # This part of the code executes in the following cases:
                # (a) Inference
                # (b) Training and the mention is not an invalid or
                # (c) Training and mention is an invalid mention and randomly sampled float is less than invalid
                # sampling probability
                metadata['last_action'] = self.action_str_to_idx[last_action_str]
                feature_embs = self.get_feature_embs(ment_idx, last_mention_idx, ent_counter, metadata)
                ment_feature_embs = self.get_ment_feature_embs(metadata)

                coref_new_scores = self.get_coref_new_scores(query_vector, ment_score, mem_vectors,
                                                             ent_counter, feature_embs)
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
            else:
                continue

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
            # rep_query_vector = query_vector.repeat(self.num_cells, 1)  # M x H
            cell_mask = (torch.arange(0, self.num_cells) == cell_idx).float().to(device=self.device)
            mask = torch.unsqueeze(cell_mask, dim=1)
            mask = mask.repeat(1, self.mem_size)

            if action_str == 'c':
                # Update memory vector corresponding to cell_idx
                mem_vectors = self.coref_update(mem_vectors, query_vector, cell_idx, mask, ent_counter)

                # Update last mention vector
                ent_counter = ent_counter + cell_mask
                last_mention_idx[cell_idx] = ment_idx
            elif action_str == 'o':
                # Replace the cell content
                mem_vectors = mem_vectors * (1 - mask) + mask * torch.unsqueeze(query_vector, dim=0)

                last_mention_idx[cell_idx] = ment_idx
                ent_counter = ent_counter * (1 - cell_mask) + cell_mask

        return coref_new_list, new_ignore_list, action_list
