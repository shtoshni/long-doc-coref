import torch
from auto_memory_model.memory import BaseMemory


class UnboundedMemory(BaseMemory):
    def __init__(self, **kwargs):
        super(UnboundedMemory, self).__init__(**kwargs)

    def initialize_memory(self):
        """Initialize the memory to null with only 1 memory cell to begin with."""
        mem = torch.zeros(1, self.mem_size).to(self.device)
        ent_counter = torch.tensor([0.0]).to(self.device)
        last_mention_idx = torch.zeros(1).long().to(self.device)
        return mem, ent_counter, last_mention_idx

    def predict_action(self, query_vector, ment_score, mem_vectors,
                       ent_counter, feature_embs):
        coref_new_scores = self.get_coref_new_scores(
            query_vector, ment_score, mem_vectors, ent_counter, feature_embs)

        # Negate the mention score
        not_a_ment_score = -ment_score

        over_ign_score = torch.cat([torch.tensor([0.0]).to(self.device), not_a_ment_score], dim=0).to(self.device)
        return coref_new_scores, over_ign_score

    def interpret_scores(self, coref_new_scores, overwrite_ign_scores, first_overwrite):
        if first_overwrite:
            num_ents = 0
            num_cells = 1
        else:
            num_ents = coref_new_scores.shape[0] - 1
            num_cells = num_ents

        pred_max_idx = torch.argmax(coref_new_scores).item()
        if pred_max_idx < num_cells:
            # Coref
            return pred_max_idx, 'c'
        elif pred_max_idx == num_cells:
            # Overwrite/Invalid mention
            over_max_idx = torch.argmax(overwrite_ign_scores).item()
            if over_max_idx == 0:
                return num_ents, 'o'
            else:
                # Invalid mention
                return -1, 'i'
        else:
            raise NotImplementedError

    def forward(self, mention_emb_list, mention_scores, gt_actions, metadata, rand_fl_list,
                teacher_forcing=False):
        # Initialize memory
        mem_vectors, ent_counter, last_mention_idx = self.initialize_memory()

        action_logit_list = []
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
                coref_new_scores, overwrite_ign_scores = self.predict_action(
                    query_vector, ment_score, mem_vectors,
                    ent_counter, feature_embs)

                pred_cell_idx, pred_action_str = self.interpret_scores(
                    coref_new_scores, overwrite_ign_scores, first_overwrite)
                action_logit_list.append((coref_new_scores, overwrite_ign_scores))
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

        return action_logit_list, action_list
