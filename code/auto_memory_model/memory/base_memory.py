import torch
import torch.nn as nn
from pytorch_utils.modules import MLP
import math

LOG2 = math.log(2)

class BaseMemory(nn.Module):
    def __init__(self, hsize=300, mlp_size=200, mlp_depth=1, coref_mlp_depth=1,
                 mem_size=None, drop_module=None, emb_size=20, entity_rep='max',
                 use_last_mention=False, dataset='litbank',
                 **kwargs):
        super(BaseMemory, self).__init__()
        self.dataset = dataset
        if self.dataset == 'litbank':
            self.num_feats = 3
        elif self.dataset == 'ontonotes':
            self.num_feats = 4

        self.hsize = hsize
        self.mem_size = (mem_size if mem_size is not None else hsize)
        self.mlp_size = mlp_size
        self.mlp_depth = mlp_depth
        self.emb_size = emb_size
        self.entity_rep = entity_rep

        self.drop_module = drop_module

        self.action_str_to_idx = {'c': 0, 'o': 1, 'i': 2, 'n': 3, '<s>': 4}
        self.action_idx_to_str = ['c', 'o', 'i', 'n', '<s>']

        self.use_last_mention = use_last_mention

        if self.entity_rep == 'lstm':
            self.mem_rnn = nn.LSTMCell(
                input_size=self.mem_size,
                hidden_size=self.mem_size)
        elif self.entity_rep == 'gru':
            self.mem_rnn = nn.GRUCell(
                input_size=self.mem_size,
                hidden_size=self.mem_size)

        self.mem_coref_mlp = MLP(3 * self.mem_size + self.num_feats * self.emb_size, self.mlp_size, 1,
                                 num_hidden_layers=coref_mlp_depth, bias=True, drop_module=drop_module)
        if self.use_last_mention:
            self.ment_coref_mlp = MLP(3 * self.mem_size, self.mlp_size, 1,
                                      num_hidden_layers=mlp_depth, bias=True, drop_module=drop_module)

        self.last_action_embeddings = nn.Embedding(5, self.emb_size)
        self.distance_embeddings = nn.Embedding(10, self.emb_size)
        self.counter_embeddings = nn.Embedding(10, self.emb_size)

    @staticmethod
    def get_distance_bucket(distances):
        logspace_idx = torch.floor(torch.log(distances.float()) / LOG2).long() + 3
        use_identity = (distances <= 4).long()
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return torch.clamp(combined_idx, 0, 9)

    @staticmethod
    def get_counter_bucket(count):
        logspace_idx = torch.floor(torch.log(count.float()) / LOG2).long() + 3
        use_identity = (count <= 4).long()
        combined_idx = use_identity * count + (1 - use_identity) * logspace_idx
        return torch.clamp(combined_idx, 0, 9)

    def get_distance_emb(self, ment_idx, last_mention_idx):
        distance_tens = self.get_distance_bucket(ment_idx - last_mention_idx)
        distance_embs = self.distance_embeddings(distance_tens)
        return distance_embs

    def get_counter_emb(self, ent_counter):
        counter_buckets = self.get_counter_bucket(ent_counter.long())
        counter_embs = self.counter_embeddings(counter_buckets)
        return counter_embs

    def get_last_action_emb(self, action_str):
        action_emb = self.action_str_to_idx[action_str]
        return self.last_action_emb(torch.tensor(action_emb).cuda())

    @staticmethod
    def get_coref_mask(ent_counter):
        cell_mask = (ent_counter > 0.0).float()
        return cell_mask

    def get_feature_embs(self, ment_idx, last_mention_idx, ent_counter, metadata):
        distance_embs = self.get_distance_emb(ment_idx, last_mention_idx)
        counter_embs = self.get_counter_emb(ent_counter)

        feature_embs_list = [distance_embs, counter_embs]

        if 'genre' in metadata:
            genre_emb = metadata['genre']
            num_cells = distance_embs.shape[0]
            genre_emb = torch.unsqueeze(genre_emb, dim=0).repeat(num_cells, 1)
            feature_embs_list.append(genre_emb)

        if 'last_action' in metadata:
            last_action_idx = torch.tensor(metadata['last_action']).long().cuda()
            last_action_emb = self.last_action_embeddings(last_action_idx)
            num_cells = distance_embs.shape[0]
            last_action_emb = torch.unsqueeze(last_action_emb, dim=0).repeat(num_cells, 1)
            feature_embs_list.append(last_action_emb)

        feature_embs = self.drop_module(torch.cat(feature_embs_list, dim=-1))
        return feature_embs

    def get_ment_feature_embs(self, metadata):
        feature_embs_list = []

        if 'genre' in metadata:
            genre_emb = metadata['genre']
            feature_embs_list.append(genre_emb)

        if 'last_action' in metadata:
            last_action_idx = torch.tensor(metadata['last_action']).long().cuda()
            last_action_emb = self.last_action_embeddings(last_action_idx)
            feature_embs_list.append(last_action_emb)

        feature_embs = self.drop_module(torch.cat(feature_embs_list, dim=-1))
        return feature_embs

    def get_coref_new_log_prob(self, query_vector, ment_score, mem_vectors, last_ment_vectors,
                               ent_counter, feature_embs):
        # Repeat the query vector for comparison against all cells
        num_cells = mem_vectors.shape[0]
        rep_query_vector = query_vector.repeat(num_cells, 1)  # M x H

        # Coref Score
        pair_vec = torch.cat([mem_vectors, rep_query_vector, mem_vectors * rep_query_vector,
                              feature_embs], dim=-1)

        pair_score = self.mem_coref_mlp(pair_vec)
        coref_score = torch.squeeze(pair_score, dim=-1) + ment_score  # M

        if self.use_last_mention:
            last_ment_vec = torch.cat(
                [last_ment_vectors, rep_query_vector,
                 last_ment_vectors * rep_query_vector], dim=-1)
            last_ment_score = torch.squeeze(self.ment_coref_mlp(last_ment_vec), dim=-1)
            coref_score = coref_score + last_ment_score  # M

        coref_new_mask = torch.cat([self.get_coref_mask(ent_counter), torch.tensor([1.0]).cuda()], dim=0)
        coref_new_scores = torch.cat(([coref_score, torch.tensor([0.0]).cuda()]), dim=0)

        coref_new_not_scores = coref_new_scores * coref_new_mask + (1 - coref_new_mask) * (-1e4)
        return coref_new_not_scores

    def forward(self, mention_emb_list, actions, mentions, teacher_forcing=False):
        pass
