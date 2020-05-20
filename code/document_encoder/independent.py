import torch
import torch.nn as nn
from os import path
from pytorch_utils.utils import get_sequence_mask, get_span_mask
from document_encoder.base_encoder import BaseDocEncoder


class IndependentDocEncoder(BaseDocEncoder):
    def __init__(self, **kwargs):
        super(IndependentDocEncoder, self).__init__(**kwargs)

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

    def forward(self, example):
        return self.encode_doc(*self.tensorize_example(example))