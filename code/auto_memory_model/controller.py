import torch
import torch.nn as nn

from os import path
from transformers import BertModel, BertTokenizer
from pytorch_utils.utils import get_sequence_mask, get_span_mask
from memory import DynamicMemory
import lru_memory
# from pytorch_utils.modules import MLP
# from pytorch_memlab import profile

EPS = 1e-8


class Controller(nn.Module):
    def __init__(self, model='base', model_loc=None,
                 dropout_rate=0.5, max_span_length=20, use_doc_rnn=False,
                 num_cells=10, use_ment_rnn=False, ment_emb='attn',
                 coref_loss_wt=0.5, use_lru=False,
                 **kwargs):
        super(Controller, self).__init__()
        self.last_layers = 1
        self.max_span_length = max_span_length

        self.use_lru = use_lru
        self.num_cells = num_cells

        self.drop_module = nn.Dropout(p=dropout_rate, inplace=False)
        self.use_doc_rnn = use_doc_rnn
        self.ment_emb = ment_emb
        ment_emb_to_size_factor = {'attn': 3, 'endpoint': 2, 'max': 1}

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

        # if not self.use_doc_rnn:
        #     # self.proj_layer = MLP(input_size=bert_emb_dim,
        #     #                       hidden_size=hsize, output_size=hsize,
        #     #                       num_layers=1, bias=True)
        #     self.proj_layer = nn.Linear(bert_emb_dim, hsize)
        # else:
        #     # self.enc_rnn = nn.GRU(bert_emb_dim, hsize, batch_first=True)
        #     self.doc_rnn = nn.LSTM(bert_emb_dim, hsize//2, bidirectional=True, batch_first=True)

        hsize = self.last_layers * bert_hidden_size

        if self.ment_emb == 'attn':
            self.attention_params = nn.Linear(hsize, 1)

        if self.use_lru:
            self.memory_net = lru_memory.DynamicMemory(
                num_cells=num_cells, hsize=ment_emb_to_size_factor[self.ment_emb] * hsize,
                drop_module=self.drop_module, **kwargs)
        else:
            self.memory_net = DynamicMemory(
                num_cells=num_cells, hsize=ment_emb_to_size_factor[self.ment_emb] * hsize,
                drop_module=self.drop_module, **kwargs)

        # Loss setup
        self.coref_loss_wt = coref_loss_wt
        if self.use_lru:
            loss_wts = torch.tensor([self.coref_loss_wt] * self.num_cells + [1.0]).cuda()
        else:
            loss_wts = torch.tensor([self.coref_loss_wt] * self.num_cells
                                    + [1.0] * self.num_cells + [1.0]).cuda()

        self.loss_fn = nn.CrossEntropyLoss(weight=loss_wts, reduction='sum')
        # self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

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

        # if not self.use_doc_rnn:
        #     encoded_output = self.proj_layer(encoded_output)
        # else:
        #     encoded_output, _ = self.doc_rnn(torch.unsqueeze(encoded_output, dim=0))
        #     encoded_output = torch.squeeze(encoded_output, dim=0)

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

    # @profile
    def get_mention_embeddings(self, mentions, doc_enc, method='endpoint'):
        span_start_list, span_end_list = zip(*mentions)
        span_start = torch.tensor(span_start_list).cuda()

        # Add 1 to span_end - After processing with Joshi et al's code, we need to add 1
        span_end = torch.tensor(span_end_list).cuda() + 1

        if method == 'endpoint':
            mention_start_vec = doc_enc[span_start, :]
            mention_end_vec = doc_enc[span_end - 1, :]
            return torch.cat([mention_start_vec, mention_end_vec], dim=1)
        elif method == 'max':
            rep_doc_enc = doc_enc.unsqueeze(dim=0)
            span_masks = get_span_mask(span_start, span_end, rep_doc_enc.shape[1])
            tmp_repr = rep_doc_enc * span_masks - 1e10 * (1 - span_masks)
            span_repr = torch.max(tmp_repr, dim=1)[0]
            return span_repr
        elif method == 'attn':
            rep_doc_enc = doc_enc.unsqueeze(dim=0)  # 1 x T x H
            span_masks = get_span_mask(span_start, span_end, rep_doc_enc.shape[1])  # K x T
            attn_mask = (1 - span_masks) * (-1e10)
            attn_logits = torch.squeeze(self.attention_params(rep_doc_enc), dim=2) + attn_mask
            attention_wts = nn.functional.softmax(attn_logits, dim=1)  # K x T
            attention_term = torch.matmul(attention_wts, doc_enc)  # K x H
            mention_start_vec = doc_enc[span_start, :]
            mention_end_vec = doc_enc[span_end - 1, :]
            return torch.cat([mention_start_vec, mention_end_vec, attention_term], dim=1)

    def action_tuple_to_idx(self, action_tuple_list):
        action_indices = []
        for (cell_idx, action_str) in action_tuple_list:
            base_idx = 0
            if action_str == 'o':
                base_idx = 1
            elif action_str == 'i':
                base_idx = 2
                cell_idx = 0

            action_idx = base_idx * self.num_cells + cell_idx
            action_indices.append(action_idx)

        return torch.tensor(action_indices).cuda()

    def action_tuple_to_idx_lru(self, action_tuple_list):
        action_indices = []
        for (cell_idx, action_str) in action_tuple_list:
            if action_str == 'c':
                action_indices.append(cell_idx)
            else:
                action_indices.append(self.num_cells)

        return torch.tensor(action_indices).cuda()

    def forward(self, example, get_next_pred_errors=False):
        """
        Encode a batch of excerpts.
        """
        doc_tens, sent_len_list = self.tensorize_example(example)
        encoded_output = self.encode_doc(doc_tens,  sent_len_list)

        mention_embs = self.get_mention_embeddings(
            example['ord_mentions'], encoded_output, method=self.ment_emb)
        mention_emb_list = torch.unbind(mention_embs, dim=0)

        action_prob_list, action_list = self.memory_net(
            mention_emb_list, example["actions"],
            example["ord_mentions"],
            get_next_pred_errors=get_next_pred_errors)  # , example[""])
        action_prob_tens = torch.stack(action_prob_list, dim=0).cuda()  # M x (2 * cells + 1)

        if self.use_lru:
            action_indices = self.action_tuple_to_idx_lru(example["actions"])
        else:
            action_indices = self.action_tuple_to_idx(example["actions"])
        coref_loss = self.loss_fn(action_prob_tens, action_indices)
        # Total mentions
        total_weight = len(mention_emb_list)

        if self.training or get_next_pred_errors:
            loss = {}
            loss['coref'] = coref_loss/total_weight
            return loss, action_list
        else:
            return coref_loss, action_list
