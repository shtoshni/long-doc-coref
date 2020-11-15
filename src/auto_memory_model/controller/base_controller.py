import torch
import torch.nn as nn

from document_encoder.independent import IndependentDocEncoder
from document_encoder.overlap import OverlapDocEncoder
from pytorch_utils.modules import MLP


class BaseController(nn.Module):
    def __init__(self,
                 dropout_rate=0.5, max_span_width=20, top_span_ratio=0.4,
                 ment_emb='endpoint', doc_enc='independent', mlp_size=1000,
                 emb_size=20, sample_invalid=1.0, label_smoothing_wt=0.0,
                 dataset='litbank', device='cuda', **kwargs):
        super(BaseController, self).__init__()

        self.device = device
        self.dataset = dataset

        self.max_span_width = max_span_width
        self.top_span_ratio = top_span_ratio
        self.sample_invalid = sample_invalid
        self.label_smoothing_wt = label_smoothing_wt

        if doc_enc == 'independent':
            self.doc_encoder = IndependentDocEncoder(device=self.device, **kwargs)
        else:
            self.doc_encoder = OverlapDocEncoder(device=self.device, **kwargs)

        self.hsize = self.doc_encoder.hsize
        self.mlp_size = mlp_size
        self.emb_size = emb_size
        self.drop_module = nn.Dropout(p=dropout_rate)
        self.ment_emb = ment_emb
        self.ment_emb_to_size_factor = {'attn': 3, 'endpoint': 2, 'max': 1}

        if self.dataset == 'ontonotes':
            # Ontonotes - Genre embedding
            self.genre_list = ["bc", "bn", "mz", "nw", "pt", "tc", "wb"]
            self.genre_to_idx = dict()
            for idx, genre in enumerate(self.genre_list):
                self.genre_to_idx[genre] = idx

            self.genre_embeddings = nn.Embedding(len(self.genre_list), self.emb_size)

        if self.ment_emb == 'attn':
            self.mention_attn = nn.Linear(self.hsize, 1)

        # Mention modeling part
        self.span_width_embeddings = nn.Embedding(self.max_span_width, self.emb_size)
        self.span_width_prior_embeddings = nn.Embedding(self.max_span_width, self.emb_size)
        self.mention_mlp = MLP(input_size=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + self.emb_size,
                               hidden_size=self.mlp_size,
                               output_size=1, num_hidden_layers=1, bias=True,
                               drop_module=self.drop_module)
        self.span_width_mlp = MLP(input_size=self.emb_size, hidden_size=self.mlp_size,
                                  output_size=1, num_hidden_layers=1, bias=True,
                                  drop_module=self.drop_module)

        self.memory_net = None
        self.loss_fn = {}

    def get_span_embeddings(self, encoded_doc, ment_starts, ment_ends):
        span_emb_list = [encoded_doc[ment_starts, :], encoded_doc[ment_ends, :]]
        # Add span width embeddings
        span_width_indices = ment_ends - ment_starts
        span_width_embs = self.drop_module(self.span_width_embeddings(span_width_indices))
        span_emb_list.append(span_width_embs)

        if self.ment_emb == 'attn':
            num_words = encoded_doc.shape[0]  # T
            num_c = ment_starts.shape[0]  # C
            doc_range = torch.unsqueeze(torch.arange(num_words), 0).repeat(num_c, 1).to(self.device)  # [C x T]
            ment_masks = ((doc_range >= torch.unsqueeze(ment_starts, dim=1)) &
                          (doc_range <= torch.unsqueeze(ment_ends, dim=1)))  # [C x T]

            del doc_range
            del ment_starts
            del ment_ends

            word_attn = torch.squeeze(self.mention_attn(encoded_doc), dim=1)  # [T]
            mention_word_attn = nn.functional.softmax(
                (1 - ment_masks.float()) * (-1e10) + torch.unsqueeze(word_attn, dim=0), dim=1)  # [C x T]

            del word_attn
            del ment_masks

            attention_term = torch.matmul(mention_word_attn, encoded_doc)  # K x H
            span_emb_list.append(attention_term)

        return torch.cat(span_emb_list, dim=1)

    def get_mention_width_scores(self, cand_starts, cand_ends):
        span_width_idx = cand_ends - cand_starts
        span_width_embs = self.span_width_prior_embeddings(span_width_idx)
        width_scores = torch.squeeze(self.span_width_mlp(span_width_embs), dim=-1)

        return width_scores

    def get_candidate_endpoints(self, encoded_doc, example):
        num_words = encoded_doc.shape[0]

        sent_map = torch.tensor(example["sentence_map"], device=self.device)
        # num_words x max_span_width
        cand_starts = torch.unsqueeze(torch.arange(num_words), dim=1).repeat(1, self.max_span_width).to(
            device=self.device
        )
        cand_ends = cand_starts + torch.unsqueeze(torch.arange(self.max_span_width), dim=0).to(device=self.device)

        cand_start_sent_indices = sent_map[cand_starts]
        # Avoid getting sentence indices for cand_ends >= num_words
        corr_cand_ends = torch.min(cand_ends, torch.ones_like(cand_ends).to(self.device) * (num_words - 1))
        cand_end_sent_indices = sent_map[corr_cand_ends]

        # End before document ends & Same sentence
        constraint1 = (cand_ends < num_words)
        constraint2 = (cand_start_sent_indices == cand_end_sent_indices)
        cand_mask = constraint1 & constraint2
        flat_cand_mask = cand_mask.reshape(-1)

        # Filter and flatten the candidate end points
        filt_cand_starts = cand_starts.reshape(-1)[flat_cand_mask]  # (num_candidates,)
        filt_cand_ends = cand_ends.reshape(-1)[flat_cand_mask]  # (num_candidates,)
        return filt_cand_starts, filt_cand_ends

    def get_pred_mentions(self, example, encoded_doc):
        num_words = encoded_doc.shape[0]

        filt_cand_starts, filt_cand_ends = self.get_candidate_endpoints(encoded_doc, example)

        span_embs = self.get_span_embeddings(encoded_doc, filt_cand_starts, filt_cand_ends)

        mention_logits = torch.squeeze(self.mention_mlp(span_embs), dim=-1)
        # Span embeddings not needed anymore
        del span_embs
        mention_logits += self.get_mention_width_scores(filt_cand_starts, filt_cand_ends)

        k = int(self.top_span_ratio * num_words)
        topk_indices = torch.topk(mention_logits, k)[1]

        topk_starts = filt_cand_starts[topk_indices]
        topk_ends = filt_cand_ends[topk_indices]
        topk_scores = mention_logits[topk_indices]

        # Sort the mentions by (start) and tiebreak with (end)
        sort_scores = topk_starts + 1e-5 * topk_ends
        _, sorted_indices = torch.sort(sort_scores, 0)

        return topk_starts[sorted_indices], topk_ends[sorted_indices], topk_scores[sorted_indices]

    def get_mention_embs_and_actions(self, example):
        encoded_doc = self.doc_encoder(example)
        pred_starts, pred_ends, pred_scores = self.get_pred_mentions(example, encoded_doc)

        # Sort the predicted mentions
        pred_mentions = list(zip(pred_starts.tolist(), pred_ends.tolist()))
        pred_scores = torch.unbind(torch.unsqueeze(pred_scores, dim=1))

        if "clusters" in example:
            gt_actions = self.get_actions(pred_mentions, example["clusters"])
        else:
            gt_actions = [(-1, 'i')] * len(pred_mentions)

        mention_embs = self.get_span_embeddings(encoded_doc, pred_starts, pred_ends)

        del encoded_doc

        mention_emb_list = torch.unbind(mention_embs, dim=0)
        return pred_mentions, gt_actions, mention_emb_list, pred_scores

    def get_genre_embedding(self, examples):
        genre = examples["doc_key"][:2]
        genre_idx = self.genre_to_idx[genre]
        return self.genre_embeddings(torch.tensor(genre_idx, device=self.device))

    def forward(self, example, teacher_forcing=False):
        pass
