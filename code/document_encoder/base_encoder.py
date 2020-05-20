import torch
import torch.nn as nn
from os import path
from pytorch_utils.utils import get_sequence_mask, get_span_mask
from transformers import BertModel, BertTokenizer


class BaseDocEncoder(nn.Module):
    def __init__(self, model='base', model_loc=None):
        super(BaseDocEncoder, self).__init__()
        self.last_layers = 1

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