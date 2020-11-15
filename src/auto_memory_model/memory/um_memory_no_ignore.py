import torch
from auto_memory_model.memory import UnboundedMemory


class UnboundedMemoryNoIgnore(UnboundedMemory):
    def __init__(self, **kwargs):
        super(UnboundedMemoryNoIgnore, self).__init__(**kwargs)

    def predict_action(self, query_vector, ment_score, mem_vectors,
                       ent_counter, feature_embs):
        coref_new_scores = self.get_coref_new_scores(
            query_vector, ment_score, mem_vectors, ent_counter, feature_embs)
        return coref_new_scores

    def interpret_scores(self, coref_new_scores, first_overwrite):
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
            return num_ents, 'o'
        else:
            raise NotImplementedError

    def forward(self, mention_emb_list, mention_scores, gt_actions, metadata, teacher_forcing=False):
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

            if not (follow_gt and gt_action_str == 'i'):
                # This part of the code executes in the following cases:
                # (a) Inference
                # (b) Training and the mention is not an invalid or
                coref_new_scores = self.predict_action(
                    query_vector, ment_score, mem_vectors,
                    ent_counter, feature_embs)

                pred_cell_idx, pred_action_str = self.interpret_scores(
                    coref_new_scores, first_overwrite)
                action_logit_list.append(coref_new_scores)
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

            if first_overwrite:
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
                else:
                    # Append the new vector
                    mem_vectors = torch.cat([mem_vectors, torch.unsqueeze(query_vector, dim=0)], dim=0)

                    ent_counter = torch.cat([ent_counter, torch.tensor([1.0]).to(self.device)], dim=0)
                    last_mention_idx = torch.cat([last_mention_idx, torch.tensor([ment_idx]).to(self.device)], dim=0)

        return action_logit_list, action_list
