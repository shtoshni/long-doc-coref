import torch
import torch.nn as nn

from os import path
from transformers import BertModel, BertTokenizer
from pytorch_utils.utils import get_sequence_mask, get_span_mask
# from pytorch_utils.modules import MLP

EPS = 1e-8


class BaseController(nn.Module):
    def __init__(self, model='base', model_loc=None,
                 dropout_rate=0.5, max_span_length=20, use_doc_rnn=False,
                 ment_emb='attn', **kwargs):
        super(BaseController, self).__init__()
        self.last_layers = 1
        self.max_span_length = max_span_length

        self.drop_module = nn.Dropout(p=dropout_rate, inplace=False)
        self.use_doc_rnn = use_doc_rnn
        self.ment_emb = ment_emb
        self.ment_emb_to_size_factor = {'attn': 3, 'endpoint': 2, 'max': 1}

        # Summary Writer
        if model_loc:
            self.bert = BertModel.from_pretrained(
                path.join(model_loc, "spanbert_{}".format(model)), output_hidden_states=True)
        else:
            bert_model_name = 'bert-' + model + '-cased'
            self.bert = BertModel.from_pretrained(
                bert_model_name, output_hidden_states=True)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.pad_token = 0

        for param in self.bert.parameters():
            # Don't update BERT params
            param.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size
        self.hsize = self.last_layers * bert_hidden_size

        if self.ment_emb == 'attn':
            self.attention_params = nn.Linear(self.hsize, 1)

        self.memory_net = None
        self.loss_fn = {}

    def encode_doc(self, document, text_length_list):
        """
        Encode chunks of a document.
        batch_excerpt: C x L where C is number of chunks padded upto max length of L
        text_length_list: list of length of chunks (length C)
        """
        num_chunks = len(text_length_list)
        attn_mask = get_sequence_mask(torch.tensor(text_length_list).cuda()).cuda().float()

        with torch.no_grad():
            outputs = self.bert(document, attention_mask=attn_mask)  # C x L x E

        encoded_layers = outputs[2]
        # encoded_repr = torch.cat(encoded_layers[self.start_layer_idx:self.end_layer_idx], dim=-1)
        encoded_repr = encoded_layers[-1]

        unpadded_encoded_output = []
        for i in range(num_chunks):
            unpadded_encoded_output.append(
                encoded_repr[i, :text_length_list[i], :])

        encoded_output = torch.cat(unpadded_encoded_output, dim=0)
        encoded_output = encoded_output
        return encoded_output

    def tensorize_example(self, example):
        sentences = example["sentences"]
        sent_len_list = [len(sent) for sent in sentences]
        max_sent_len = max(sent_len_list)
        padded_sent = [self.tokenizer.convert_tokens_to_ids(sent)
                       + [self.pad_token] * (max_sent_len - len(sent))
                       for sent in sentences]
        doc_tens = torch.tensor(padded_sent).cuda()
        return doc_tens, sent_len_list

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

    def forward(self, example, teacher_forcing=False):
        """
        Encode a batch of excerpts.
        """
        doc_tens, sent_len_list = self.tensorize_example(example)
        encoded_output = self.encode_doc(doc_tens,  sent_len_list)

        mention_embs = self.get_mention_embeddings(
            example['ord_mentions'], encoded_output, method=self.ment_emb)
        mention_emb_list = torch.unbind(mention_embs, dim=0)

        action_prob_list, action_list = self.memory_net(
            mention_emb_list, example["actions"], example["ord_mentions"],
            teacher_forcing=teacher_forcing)


