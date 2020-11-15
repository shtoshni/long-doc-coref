import torch
from auto_memory_model.memory import BaseFixedMemory


class LRUMemory(BaseFixedMemory):
    def __init__(self, **kwargs):
        super(LRUMemory, self).__init__(**kwargs)

    def predict_new_or_ignore(self, query_vector, ment_score, mem_vectors,
                              feature_embs, ment_feature_embs, lru_list):
        lru_cell = lru_list[0]
        mem_fert_input = torch.cat([mem_vectors[lru_cell, :], feature_embs[lru_cell, :]], dim=0)
        ment_fert_input = torch.cat([query_vector, ment_feature_embs], dim=-1)
        fert_input = torch.stack([mem_fert_input, ment_fert_input], dim=0)
        fert_scores = torch.squeeze(self.fert_mlp(fert_input), dim=-1)

        over_ign_score = torch.cat([fert_scores, -ment_score], dim=0)
        return over_ign_score

    @staticmethod
    def interpret_new_ignore_score(overwrite_ign_no_space_scores, lru_cell_idx):
        over_max_idx = torch.argmax(overwrite_ign_no_space_scores).item()
        if over_max_idx == 0:
            return lru_cell_idx, 'o'
        elif over_max_idx == 1:
            # No space
            return -1, 'n'
        elif over_max_idx == 2:
            # Invalid mention
            return -1, 'i'

    def forward(self, mention_emb_list, mention_scores, gt_actions, metadata, rand_fl_list,
                teacher_forcing=False):
        # Initialize memory
        mem_vectors, ent_counter, last_mention_idx = self.initialize_memory()
        lru_list = list(range(self.num_cells))

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

            if not (follow_gt and gt_action_str == 'i' and rand_fl_list[ment_idx] > self.sample_invalid):
                # This part of the code executes in the following cases:
                # (a) Inference
                # (b) Training and the mention is not an invalid or
                # (c) Training and mention is an invalid mention and randomly sampled float is less than invalid
                # sampling probability
                coref_new_scores = self.get_coref_new_scores(query_vector, ment_score, mem_vectors,
                                                             ent_counter, feature_embs)
                coref_new_list.append(coref_new_scores)

                pred_cell_idx, pred_action_str = self.interpret_coref_new_score(coref_new_scores)

                if (follow_gt and gt_action_str != 'c') or ((not follow_gt) and pred_action_str != 'c'):
                    new_ignore_score = self.predict_new_or_ignore(
                        query_vector, ment_score, mem_vectors,
                        feature_embs, ment_feature_embs, lru_list)
                    pred_cell_idx, pred_action_str = self.interpret_new_ignore_score(new_ignore_score, lru_list[0])
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

            last_action_str = action_str

            # Update the memory
            rep_query_vector = query_vector.repeat(self.num_cells, 1)
            cell_mask = (torch.arange(0, self.num_cells) == cell_idx).float().to(self.device)
            mask = torch.unsqueeze(cell_mask, dim=1)
            mask = mask.repeat(1, self.mem_size)

            if action_str == 'c':
                mem_vectors = self.coref_update(mem_vectors, query_vector, cell_idx, mask, ent_counter)

                ent_counter = ent_counter + cell_mask
                last_mention_idx[cell_idx] = ment_idx
            elif action_str == 'o':
                # Replace the cell content
                mem_vectors = mem_vectors * (1 - mask) + mask * rep_query_vector

                ent_counter = ent_counter * (1 - cell_mask) + cell_mask
                last_mention_idx[cell_idx] = ment_idx

            if action_str in ['o', 'c']:
                # Coref or overwrite was chosen; place the cell_idx in use at the back of the list
                lru_list.remove(cell_idx)
                lru_list.append(cell_idx)

        return coref_new_list, new_ignore_list, action_list
