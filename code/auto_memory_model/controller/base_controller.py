import torch
import torch.nn as nn

from pytorch_utils.utils import get_sequence_mask, get_span_mask
from document_encoder.independent import IndependentDocEncoder
from document_encoder.overlap import OverlapDocEncoder


class BaseController(nn.Module):
    def __init__(self, model='base', model_loc=None,
                 dropout_rate=0.5, max_span_length=20, use_doc_rnn=False,
                 ment_emb='attn', doc_enc='independent', **kwargs):
        super(BaseController, self).__init__()
        self.max_span_length = max_span_length

        if doc_enc == 'independent':
            self.doc_encoder = IndependentDocEncoder(model=model, model_loc=model_loc)
        else:
            self.doc_encoder = OverlapDocEncoder(model=model, model_loc=model_loc)

        self.hsize = self.doc_encoder.hsize
        self.drop_module = nn.Dropout(p=dropout_rate, inplace=False)
        self.use_doc_rnn = use_doc_rnn
        self.ment_emb = ment_emb
        self.ment_emb_to_size_factor = {'attn': 3, 'endpoint': 2, 'max': 1}

        if self.ment_emb == 'attn':
            self.attention_params = nn.Linear(self.hsize, 1)

        self.memory_net = None
        self.loss_fn = {}

    def get_mention_embeddings(self, mentions, doc_enc, method='endpoint'):
        span_start_list, span_end_list = zip(*mentions)
        span_start = torch.tensor(span_start_list).cuda()

        # Add 1 to span_end - After processing with Joshi et al's code, we need to add 1
        span_end = torch.tensor(span_end_list).cuda()

        rep_doc_enc = doc_enc.unsqueeze(dim=0)
        span_masks = get_span_mask(span_start, span_end + 1, rep_doc_enc.shape[1])  # K x T

        if method == 'endpoint':
            mention_start_vec = doc_enc[span_start, :]
            mention_end_vec = doc_enc[span_end, :]
            return torch.cat([mention_start_vec, mention_end_vec], dim=1)
        elif method == 'max':
            span_masks = get_span_mask(span_start, span_end, rep_doc_enc.shape[1])
            tmp_repr = rep_doc_enc * span_masks - 1e10 * (1 - span_masks)
            span_repr = torch.max(tmp_repr, dim=1)[0]
            return span_repr
        elif method == 'attn':
            rep_doc_enc = doc_enc.unsqueeze(dim=0)  # 1 x T x H
            attn_mask = (1 - span_masks) * (-1e10)
            attn_logits = torch.squeeze(self.attention_params(rep_doc_enc), dim=2) + attn_mask
            attention_wts = nn.functional.softmax(attn_logits, dim=1)  # K x T
            attention_term = torch.matmul(attention_wts, doc_enc)  # K x H
            mention_start_vec = doc_enc[span_start, :]
            mention_end_vec = doc_enc[span_end, :]
            return torch.cat([mention_start_vec, mention_end_vec, attention_term], dim=1)

    def get_document_enc(self, example):
        if self.doc_enc == 'independent':
            encoded_output = self.doc_enc(example)
        else:
            # Overlap
            encoded_output = None

        return encoded_output

    def forward(self, example, teacher_forcing=False):
        pass