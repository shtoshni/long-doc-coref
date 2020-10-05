import torch
from pytorch_utils.modules import MLP
from auto_memory_model.memory.base_fixed_memory import BaseMemory


class UnboundedMemoryOntoNotes(BaseMemory):
    def __init__(self, **kwargs):
        super(UnboundedMemoryOntoNotes, self).__init__(**kwargs)

    def initialize_memory(self):
        """Initialize the memory to null with only 1 memory cell to begin with."""
        mem = torch.zeros(1, self.mem_size).cuda()
        ent_counter = torch.tensor([0.0]).cuda()
        last_mention_idx = torch.zeros(1).long().cuda()
        return mem, ent_counter, last_mention_idx

    def predict_action(self, query_vector, ment_score, mem_vectors, last_ment_vectors,
                       ent_counter, feature_embs):
        coref_new_scores = self.get_coref_new_log_prob(
            query_vector, ment_score, mem_vectors, last_ment_vectors, ent_counter, feature_embs)

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
            return num_ents, 'o'
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
        first_overwrite = True

        for ment_idx, (ment_emb, ment_score, (gt_cell_idx, gt_action_str)) in \
                enumerate(zip(mention_emb_list, mention_scores, gt_actions)):
            query_vector = ment_emb
            feature_embs = self.get_feature_embs(ment_idx, last_mention_idx, ent_counter, metadata)
            coref_new_scores = self.predict_action(
                query_vector, ment_score, mem_vectors, last_ment_vectors,
                ent_counter, feature_embs)

            action_logit_list.append(coref_new_scores)
            pred_cell_idx, pred_action_str = self.interpret_scores(
                coref_new_scores, first_overwrite)

            if self.training or teacher_forcing:
                # Training - Operate over the ground truth
                action_str = gt_action_str
                cell_idx = gt_cell_idx
            else:
                # Inference time
                action_str = pred_action_str
                cell_idx = pred_cell_idx

            action_list.append((pred_cell_idx, pred_action_str))

            if first_overwrite and action_str == 'o':
                first_overwrite = False
                # We start with a single empty memory cell
                mem_vectors = torch.unsqueeze(query_vector, dim=0)
                last_ment_vectors = torch.unsqueeze(query_vector, dim=0)
                ent_counter = torch.tensor([1.0]).cuda()
                last_mention_idx[0] = ment_idx
            else:
                # During training this records the next actions  - during testing it records the
                # predicted sequence of actions
                num_ents = coref_new_scores.shape[0] - 1
                # Update the memory
                rep_query_vector = query_vector.repeat(num_ents, 1)  # M x H
                cell_mask = (torch.arange(0, num_ents) == cell_idx).float().cuda()
                mask = torch.unsqueeze(cell_mask, dim=1)
                mask = mask.repeat(1, self.mem_size)

                # print(cell_idx, action_str, mem_vectors.shape[0])
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
                    # Append the new vector
                    mem_vectors = torch.cat([mem_vectors, torch.unsqueeze(query_vector, dim=0)], dim=0)
                    # Update last mention vector
                    last_ment_vectors = torch.cat([last_ment_vectors, torch.unsqueeze(query_vector, dim=0)], dim=0)

                    ent_counter = torch.cat([ent_counter, torch.tensor([1.0]).cuda()], dim=0)
                    last_mention_idx = torch.cat([last_mention_idx, torch.tensor([ment_idx]).cuda()], dim=0)
                    # last_mention_idx.append(ment_idx)

        return action_logit_list, action_list
