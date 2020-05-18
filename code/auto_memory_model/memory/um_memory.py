import torch
from auto_memory_model.memory.base_fixed_memory import BaseMemory


class UnboundedMemory(BaseMemory):
    def __init__(self, **kwargs):
        super(UnboundedMemory, self).__init__(**kwargs)

    def initialize_memory(self):
        """Initialize the memory to null with only 1 memory cell to begin with."""
        mem = torch.zeros(1, self.mem_size).cuda()
        ent_counter = torch.tensor([0]).cuda()
        last_mention_idx = [0]
        return mem, ent_counter, last_mention_idx

    def predict_action(self, query_vector, mem_vectors, last_ment_vectors,
                       ment_idx, ent_counter, last_mention_idx):
        distance_embs = self.get_distance_emb(ment_idx, last_mention_idx)
        counter_embs = self.get_counter_emb(ent_counter)

        coref_new_log_prob = self.get_coref_new_log_prob(
            query_vector, mem_vectors, last_ment_vectors,
            ent_counter, distance_embs, counter_embs)

        return coref_new_log_prob

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

            coref_new_log_prob = self.predict_action(
                query_vector, mem_vectors, last_ment_vectors,
                ment_idx, ent_counter, last_mention_idx)

            action_logit_list.append(coref_new_log_prob)

            if ment_idx == 0:
                # We start with a single empty memory cell
                mem_vectors = torch.unsqueeze(query_vector, dim=0)
                last_ment_vectors = torch.unsqueeze(query_vector, dim=0)
                ent_counter = torch.tensor([1.0]).cuda()
                last_mention_idx[0] = 0

                action_list.append((0, 'o'))
            else:
                pred_max_idx = torch.argmax(coref_new_log_prob).item()
                num_ents = coref_new_log_prob.shape[0] - 1

                if pred_max_idx == num_ents:
                    pred_action_str = 'o'
                    pred_cell_idx = num_ents
                else:
                    pred_action_str = 'c'
                    pred_cell_idx = pred_max_idx

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
                    last_mention_idx.append(ment_idx)

        return action_logit_list, action_list
