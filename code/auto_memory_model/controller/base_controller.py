import torch
import torch.nn as nn

from pytorch_utils.utils import get_sequence_mask, get_span_mask
from document_encoder.independent import IndependentDocEncoder
from document_encoder.overlap import OverlapDocEncoder
from auto_memory_model.utils import get_ordered_mentions
from pytorch_utils.modules import MLP
import numpy as np


class BaseController(nn.Module):
    def __init__(self,
                 dropout_rate=0.5, max_span_width=20, top_span_ratio=0.4,
                 ment_emb='endpoint', doc_enc='independent', mlp_size=1000,
                 train_span_model=False, sample_ignores=1.0, **kwargs):
        super(BaseController, self).__init__()
        self.max_span_width = max_span_width
        self.top_span_ratio = top_span_ratio
        self.sample_ignores = sample_ignores

        if doc_enc == 'independent':
            self.doc_encoder = IndependentDocEncoder(**kwargs)
        else:
            self.doc_encoder = OverlapDocEncoder(**kwargs)

        self.hsize = self.doc_encoder.hsize
        self.mlp_size = mlp_size
        self.drop_module = nn.Dropout(p=dropout_rate, inplace=False)
        self.ment_emb = ment_emb
        self.ment_emb_to_size_factor = {'attn': 3, 'endpoint': 2, 'max': 1}

        if self.ment_emb == 'attn':
            self.mention_attn = nn.Linear(self.hsize, 1)

        # Mention modeling part
        self.span_width_embeddings = nn.Embedding(self.max_span_width, 20)
        self.span_width_prior_embeddings = nn.Embedding(self.max_span_width, 20)
        self.mention_mlp = MLP(input_size=self.ment_emb_to_size_factor[self.ment_emb] * self.hsize + 20,
                               hidden_size=self.mlp_size,
                               output_size=1, num_hidden_layers=1, bias=True,
                               drop_module=self.drop_module)
        self.span_width_mlp = MLP(input_size=20, hidden_size=self.mlp_size,
                                  output_size=1, num_hidden_layers=1, bias=True,
                                  drop_module=self.drop_module)

        if not train_span_model:
            for param in self.mention_mlp.parameters():
                param.requires_grad = False
            for param in self.span_width_mlp.parameters():
                param.requires_grad = False
            for param in self.span_width_embeddings.parameters():
                param.requires_grad = False
            for param in self.span_width_prior_embeddings.parameters():
                param.requires_grad = False

        self.memory_net = None
        self.loss_fn = {}
        # self.mention_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    def get_document_enc(self, example):
        if self.doc_enc == 'independent':
            encoded_output = self.doc_enc(example)
        else:
            # Overlap
            encoded_output = None

        return encoded_output

    def get_span_embeddings(self, encoded_doc, ment_starts, ment_ends):
        span_emb_list = [encoded_doc[ment_starts, :], encoded_doc[ment_ends, :]]
        # Add span width embeddings
        span_width_indices = ment_ends - ment_starts
        span_width_embs = self.span_width_embeddings(span_width_indices)
        span_emb_list.append(span_width_embs)

        if self.ment_emb == 'attn':
            num_words = encoded_doc.shape[0]  # T
            num_c = ment_starts.shape[0]  # C
            doc_range = torch.unsqueeze(torch.arange(num_words), 0).repeat(num_c, 1).cuda()  # [C x T]
            ment_masks = ((doc_range >= torch.unsqueeze(ment_starts, dim=1)) &
                          (doc_range <= torch.unsqueeze(ment_ends, dim=1)))  # [C x T]
            word_attn = torch.squeeze(self.mention_attn(encoded_doc), dim=1)  # [T]
            mention_word_attn = nn.functional.softmax(
                (1 - ment_masks.float()) * (-1e10) + torch.unsqueeze(word_attn, dim=0), dim=1)  # [C x T]
            attention_term = torch.matmul(mention_word_attn, encoded_doc)  # K x H

            span_emb_list.append(attention_term)

        return torch.cat(span_emb_list, dim=1)

    def get_mention_scores(self, span_embs, cand_starts, cand_ends):
        mention_logits = torch.squeeze(self.mention_mlp(span_embs), dim=-1)

        span_width_idx = cand_ends - cand_starts
        span_width_embs = self.span_width_prior_embeddings(span_width_idx)
        width_scores = torch.squeeze(self.span_width_mlp(span_width_embs), dim=-1)

        mention_logits += width_scores

        return mention_logits

    def get_pred_mentions(self, example, encoded_doc):
        num_words = encoded_doc.shape[0]

        sent_map = torch.tensor(example["sentence_map"]).cuda()
        # num_words x max_span_width
        cand_starts = torch.unsqueeze(torch.arange(num_words), dim=1).repeat(1, self.max_span_width).cuda()
        cand_ends = cand_starts + torch.unsqueeze(torch.arange(self.max_span_width), dim=0).cuda()

        cand_start_sent_indices = sent_map[cand_starts]
        # Avoid getting sentence indices for cand_ends >= num_words
        corr_cand_ends = torch.min(cand_ends, torch.ones_like(cand_ends).cuda() * (num_words - 1))
        cand_end_sent_indices = sent_map[corr_cand_ends]

        # End before document ends & Same sentence
        constraint1 = (cand_ends < num_words)
        constraint2 = (cand_start_sent_indices == cand_end_sent_indices)
        cand_mask = constraint1 & constraint2
        flat_cand_mask = cand_mask.reshape(-1)

        # Filter and flatten the candidate end points
        filt_cand_starts = cand_starts.reshape(-1)[flat_cand_mask]  # (num_candidates,)
        filt_cand_ends = cand_ends.reshape(-1)[flat_cand_mask]  # (num_candidates,)

        span_embs = self.get_span_embeddings(encoded_doc, filt_cand_starts, filt_cand_ends)
        mention_scores = self.get_mention_scores(span_embs, filt_cand_starts, filt_cand_ends)

        k = int(self.top_span_ratio * num_words)
        topk_indices = torch.topk(mention_scores, k)[1]

        topk_starts = filt_cand_starts[topk_indices]
        topk_ends = filt_cand_ends[topk_indices]
        topk_scores = mention_scores[topk_indices]

        # Sort the mentions by (start) and tiebreak with (end)
        sort_scores = topk_starts + 1e-5 * topk_ends
        _, sorted_indices = torch.sort(sort_scores, 0)

        return topk_starts[sorted_indices], topk_ends[sorted_indices], topk_scores[sorted_indices]

    def get_mention_embs_and_actions(self, example):
        encoded_doc = self.doc_encoder(example)

        gt_mentions = get_ordered_mentions(example["clusters"])
        pred_starts, pred_ends, pred_scores = self.get_pred_mentions(example, encoded_doc)

        # Sort the predicted mentions
        pred_mentions = list(zip(pred_starts.tolist(), pred_ends.tolist()))
        # print(list(pred_mentions))
        pred_scores = torch.unbind(pred_scores)
        # pred_mentions_scores = zip(pred_mentions, pred_scores)
        # pred_mentions_scores = sorted(pred_mentions_scores, key=lambda x: x[0][0] + 1e-5 * x[0][1])
        # pred_mentions, pred_scores = zip(*pred_mentions_scores)
        # print(pred_mentions)
        # # print(pred_scores)
        # pred_starts, pred_ends = zip(*pred_mentions)
        # pred_starts = torch.tensor(pred_starts).cuda()
        # pred_ends = torch.tensor(pred_ends).cuda()

        gt_actions = self.get_actions(pred_mentions, example["clusters"])
        mention_embs = self.get_span_embeddings(
            encoded_doc, pred_starts, pred_ends)
        mention_emb_list = torch.unbind(mention_embs, dim=0)

        if self.training and self.sample_ignores < 1.0:
            # Subsample from non-mentions
            sub_gt_actions = list(gt_actions)
            sub_mention_emb_list = list(mention_emb_list)
            sub_pred_mentions = list(pred_mentions)
            sub_pred_scores = list(pred_scores)

            rand_fl_list = list(np.random.random(len(gt_actions)))
            for gt_action, mention_emb, pred_mention, pred_score, rand_fl in zip(
                    gt_actions, mention_emb_list, pred_mentions, pred_scores, rand_fl_list):
                add_instance = True
                if gt_action[1] == 'i':
                    if rand_fl > self.sample_ignores:
                        add_instance = False

                if add_instance:
                    sub_gt_actions.append(gt_action)
                    sub_mention_emb_list.append(mention_emb)
                    sub_pred_mentions.append(pred_mention)
                    sub_pred_scores.append(pred_score)

            gt_actions = sub_gt_actions
            mention_emb_list = sub_mention_emb_list
            pred_mentions = sub_pred_mentions
            pred_scores = sub_pred_scores

        # mention_score_list = torch.unbind(pred_scores, dim=0)
        return gt_mentions, pred_mentions, gt_actions, mention_emb_list, pred_scores

    def forward(self, example, teacher_forcing=False):
        pass