import torch.nn as nn
from os import path
from transformers import BertModel, BertTokenizer, AutoModel


class BaseDocEncoder(nn.Module):
    def __init__(self, model_size='base', pretrained_bert_dir=None, max_training_segments=4, device="cuda", **kwargs):
        super(BaseDocEncoder, self).__init__()
        self.device = device

        self.max_training_segments = max_training_segments

        # Check if the pretrained bert directory argument has the spanbert models
        if pretrained_bert_dir and path.exists(path.join(pretrained_bert_dir, "spanbert_{}".format(model_size))):
            self.bert = BertModel.from_pretrained(
                path.join(pretrained_bert_dir, "spanbert_{}".format(model_size)), output_hidden_states=False)
        else:
            # Use the model from huggingface
            self.bert = AutoModel.from_pretrained(f"shtoshni/spanbert_coreference_{model_size}")

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', output_hidden_states=False)
        self.pad_token = 0

        for param in self.bert.parameters():
            # Don't update BERT params
            param.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size
        self.hsize = bert_hidden_size

    def forward(self, example):
        return
