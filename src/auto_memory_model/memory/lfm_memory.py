import torch
from auto_memory_model.memory import BaseMemory


class LearnedFixedMemory(BaseMemory):
    def __init__(self, num_cells=10, **kwargs):
        super(LearnedFixedMemory, self).__init__(**kwargs)
        self.num_cells = num_cells

    def initialize_memory(self):
        """Initialize the memory to null with only 1 memory cell to begin with."""
        mem = torch.zeros(1, self.mem_size).to(self.device)
        ent_counter = torch.tensor([0.0]).to(self.device)
        last_mention_idx = torch.zeros(1).long().to(self.device)
        return mem, ent_counter, last_mention_idx

    def predict_new_or_ignore(self, query_vector, mem_vectors, feature_embs, ment_feature_embs):
        # Fertility Score
        mem_fert_input = torch.cat([mem_vectors, feature_embs], dim=-1)
        ment_fert_input = torch.unsqueeze(torch.cat([query_vector, ment_feature_embs], dim=-1), dim=0)
        fert_input = torch.cat([mem_fert_input, ment_fert_input], dim=0)

        fert_scores = self.fert_mlp(fert_input)
        fert_scores = torch.squeeze(fert_scores, dim=-1)

        new_or_ignore_scores = fert_scores

        output = new_or_ignore_scores,
        max_idx = torch.argmax(new_or_ignore_scores).item()
        if max_idx < self.num_cells:
            return output + (max_idx, 'o')
        else:
            # No space - The new entity is not "fertile"
            return output + (-1, 'n')

    @staticmethod
    def assign_cluster(coref_new_scores, first_overwrite):
        if first_overwrite:
            num_ents = 0
        else:
            num_ents = coref_new_scores.shape[0] - 1

        pred_max_idx = torch.argmax(coref_new_scores).item()
        if pred_max_idx < num_ents:
            # Coref
            return pred_max_idx, 'c'
        else:
            # New cluster
            return num_ents, 'o'

    def forward(self, mention_emb_list, mention_scores, gt_actions, metadata, rand_fl_list,
                teacher_forcing=False):
        # Initialize memory
        mem_vectors, ent_counter, last_mention_idx = self.initialize_memory()

        coref_new_list = []
        new_ignore_list = []
        entity_or_not_list = []
        action_list = []  # argmax actions
        first_overwrite = True
        last_action_str = '<s>'

        follow_gt = self.training or teacher_forcing

        for ment_idx, (ment_emb, ment_score, (gt_cell_idx, gt_action_str)) in \
                enumerate(zip(mention_emb_list, mention_scores, gt_actions)):
            query_vector = ment_emb
            metadata['last_action'] = self.action_str_to_idx[last_action_str]
            feature_embs = self.get_feature_embs(ment_idx, last_mention_idx, ent_counter, metadata)

            if not (follow_gt and gt_action_str == 'i' and rand_fl_list[ment_idx] > self.sample_invalid):
                # This part of the code executes in the following cases:
                # (a) Inference
                # (b) Training and the mention is not an invalid or
                # (c) Training and mention is an invalid mention and randomly sampled float is less than invalid
                # sampling probability

                overwrite_ign_scores = torch.cat([torch.tensor([0.0]).to(self.device), -ment_score], dim=0)
                entity_or_not_list.append(overwrite_ign_scores)
                pred_action_str = None
                if ment_score < 0:
                    pred_action_str = 'i'

                if (follow_gt and gt_action_str == 'i') or ((not follow_gt) and pred_action_str == 'i'):
                    action_list.append((-1, 'i'))
                    continue

                coref_new_scores = self.get_coref_new_scores(query_vector, mem_vectors, ent_counter, feature_embs)

                pred_cell_idx, pred_action_str = self.assign_cluster(coref_new_scores, first_overwrite)
                if pred_cell_idx > self.num_cells:
                    # Reached memory capacity
                    new_or_ignore_scores, pred_cell_idx, pred_action_str = self.predict_new_or_ignore(
                        query_vector, mem_vectors, feature_embs, feature_embs)
                    new_ignore_list.append(new_or_ignore_scores)

                coref_new_list.append(coref_new_scores)
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

            if first_overwrite and action_str == 'o':
                first_overwrite = False
                # We start with a single empty memory cell
                mem_vectors = torch.unsqueeze(query_vector, dim=0)
                ent_counter = torch.tensor([1.0]).to(self.device)
                last_mention_idx[0] = ment_idx
            else:
                num_ents = mem_vectors.shape[0]
                # Update the memory
                cell_mask = (torch.arange(0, num_ents) == cell_idx).float().to(self.device)
                mask = torch.unsqueeze(cell_mask, dim=1)
                mask = mask.repeat(1, self.mem_size)

                # print(cell_idx, action_str, mem_vectors.shape[0])
                if action_str == 'c':
                    mem_vectors = self.coref_update(mem_vectors, query_vector, cell_idx, mask, ent_counter)

                    ent_counter = ent_counter + cell_mask
                    last_mention_idx[cell_idx] = ment_idx
                elif action_str == 'o':
                    # Append the new vector
                    mem_vectors = torch.cat([mem_vectors, torch.unsqueeze(query_vector, dim=0)], dim=0)

                    ent_counter = torch.cat([ent_counter, torch.tensor([1.0]).to(self.device)], dim=0)
                    last_mention_idx = torch.cat([last_mention_idx, torch.tensor([ment_idx]).to(self.device)], dim=0)

        return entity_or_not_list, coref_new_list, new_ignore_list, action_list
