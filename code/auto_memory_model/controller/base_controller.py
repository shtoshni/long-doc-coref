import torch
import torch.nn as nn

from pytorch_utils.utils import get_sequence_mask, get_span_mask
from document_encoder.independent import IndependentDocEncoder
from document_encoder.overlap import OverlapDocEncoder
from auto_memory_model.utils import get_ordered_mentions


class BaseController(nn.Module):
    def __init__(self, model='base', model_loc=None,
                 dropout_rate=0.5, max_span_width=20,
                 ment_emb='endpoint', doc_enc='independent', **kwargs):
        super(BaseController, self).__init__()
        self.max_span_width = max_span_width

        if doc_enc == 'independent':
            self.doc_encoder = IndependentDocEncoder(model=model, model_loc=model_loc)
        else:
            self.doc_encoder = OverlapDocEncoder(model=model, model_loc=model_loc)

        self.hsize = self.doc_encoder.hsize
        self.drop_module = nn.Dropout(p=dropout_rate, inplace=False)
        self.ment_emb = ment_emb
        self.ment_emb_to_size_factor = {'attn': 3, 'endpoint': 2, 'max': 1}

        if self.ment_emb == 'attn':
            self.mention_attn = nn.Linear(self.hsize, 1)

        self.memory_net = None
        self.loss_fn = {}

    def get_mention_embeddings(self, encoded_doc, ment_starts, ment_ends):
        ment_emb_list = [encoded_doc[ment_starts, :], encoded_doc[ment_ends, :]]

        if self.ment_emb == 'endpoint':
            return torch.cat(ment_emb_list, dim=-1)
        else:
            num_words = encoded_doc.shape[0]  # T
            num_c = ment_starts.shape[0]  # C
            doc_range = torch.unsqueeze(torch.arange(num_words), 0).repeat(num_c, 1).cuda()  # [C x T]
            ment_masks = ((doc_range >= torch.unsqueeze(ment_starts, dim=1)) &
                          (doc_range <= torch.unsqueeze(ment_ends, dim=1)))  # [C x T]
            word_attn = torch.squeeze(self.mention_attn(encoded_doc), dim=1)  # [T]
            mention_word_attn = nn.functional.softmax(
                (1 - ment_masks.float()) * (-1e10) + torch.unsqueeze(word_attn, dim=0), dim=1)  # [C x T]
            attention_term = torch.matmul(mention_word_attn, encoded_doc)  # K x H

            ment_emb_list.append(attention_term)
            return torch.cat(ment_emb_list, dim=1)

    def get_document_enc(self, example):
        if self.doc_enc == 'independent':
            encoded_output = self.doc_enc(example)
        else:
            # Overlap
            encoded_output = None

        return encoded_output

    def get_mention_embs_and_actions(self, example):
        encoded_output = self.doc_encoder(example)

        gt_mentions = get_ordered_mentions(example["clusters"])
        pred_mentions = gt_mentions
        gt_actions = self.get_actions(pred_mentions, example["clusters"])

        cand_starts, cand_ends = zip(*pred_mentions)
        mention_embs = self.get_mention_embeddings(
            encoded_output, torch.tensor(cand_starts).cuda(), torch.tensor(cand_ends).cuda())
        mention_emb_list = torch.unbind(mention_embs, dim=0)
        return gt_mentions, pred_mentions, gt_actions, mention_emb_list

    def forward(self, example, teacher_forcing=False):
        pass