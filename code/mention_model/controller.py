import torch
import torch.nn as nn

from transformers import BertModel
from pytorch_utils.utils import get_sequence_mask

EPS = 1e-8


class Controller(nn.Module):
    def __init__(self, model='base',
                 hsize=100, dropout_rate=0.5, max_span_length=20,
                 enc_rnn=False, **kwargs):
        super(Controller, self).__init__()
        self.last_layers = 4
        if model == 'large':
            self.start_layer_idx = 19
            self.end_layer_idx = self.start_layer_idx + self.last_layers
        elif model == 'base':
            self.start_layer_idx = 9
            self.end_layer_idx = self.start_layer_idx + self.last_layers

        self.max_span_length = max_span_length
        self.drop_module = nn.Dropout(p=dropout_rate, inplace=True)

        # Summary Writer
        self.bert = BertModel.from_pretrained('bert-' + model + '-cased',
                                              output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size
        bert_emb_dim = self.last_layers * bert_hidden_size

        self.weighing_params = nn.Parameter(
            torch.ones(self.bert.config.num_hidden_layers))

        self.use_rnn = enc_rnn

        if not self.use_rnn:
            self.proj_layer = nn.Linear(bert_emb_dim, hsize)
        else:
            self.enc_rnn = nn.GRU(bert_emb_dim, hsize, batch_first=True)

        self.label_net = nn.Sequential(
            nn.Linear(2 * hsize, hsize),
            nn.Tanh(),
            nn.LayerNorm(hsize),
            nn.Dropout(0.2),
            nn.Linear(hsize, 1),
        )

    def encode_batch_window(self, batch_excerpt, input_mask):
        """
        Encode a batch of excerpts.
        batch_excerpt: B x L
        input_mask: B x L
        """
        batch_size, max_len = batch_excerpt.size()

        with torch.no_grad():
            outputs = self.bert(
                batch_excerpt, attention_mask=input_mask)  # B x L x E

        encoded_layers = outputs[2]

        encoded_repr = torch.cat(encoded_layers[self.start_layer_idx:self.end_layer_idx], dim=-1)
        if not self.use_rnn:
            encoded_repr = self.proj_layer(encoded_repr)
        else:
            encoded_repr, _ = self.enc_rnn(encoded_repr)

        return encoded_repr

    def encode_excerpt(self, batch_data):
        """
        Encode a batch of excerpts.
        batch_excerpt: B x L
        input_mask: B x L
        """
        text, text_length = batch_data.text
        # text, text_length = text.cuda(), text_length.cuda()
        text_length_list = text_length.tolist()

        total_length = torch.sum(text_length, dim=-1)
        total_length_list = total_length.tolist()
        max_input_length = max(total_length_list)

        batch_size, max_windows, max_len = text.size()

        # Process the different windows independently
        indep_windows = torch.reshape(text, [-1, max_len])
        indep_window_lens = torch.reshape(text_length, [-1])
        attn_mask = get_sequence_mask(indep_window_lens).cuda().float()

        encoder_output = self.encode_batch_window(indep_windows, attn_mask)
        encoder_hidden_size = encoder_output.shape[-1]
        encoder_output = torch.reshape(encoder_output,
                                       [batch_size, max_windows, max_len, encoder_hidden_size])

        final_encoded_output = []
        for i in range(batch_size):
            cur_encoded_output = []
            for j in range(max_windows):
                cur_encoded_output.append(
                    encoder_output[i, j, :text_length_list[i][j], :])
            # Need for padding
            padding_length = max_input_length - total_length_list[i]
            if padding_length:
                padding_tens = torch.zeros((padding_length, encoder_hidden_size)).cuda()
                cur_encoded_output.append(padding_tens)

            final_encoded_output.append(torch.cat(cur_encoded_output, dim=0))

        final_output = torch.stack(final_encoded_output)
        input_mask = get_sequence_mask(total_length).cuda().float()

        return final_output, input_mask

    def predict_mention_prob(self, encoder_outputs, input_mask):
        batch_size, time_steps = input_mask.size()
        # B x T1 x L
        mention_logits = torch.ones(batch_size, time_steps,
                                    self.max_span_length) * (-1e9)
        mention_logits = mention_logits.cuda()

        for span_length in range(1, self.max_span_length + 1):
            if span_length == 1:
                start_tens = end_tens = encoder_outputs
            else:
                start_tens = encoder_outputs[:, :-(span_length - 1), :]
                end_tens = encoder_outputs[:, (span_length - 1):, :]

            span_logits = self.label_net(
                torch.cat(
                    # We take the end hidden vector inside the span
                    [start_tens, end_tens],
                    dim=-1)
            )
            mention_logits[:, (span_length-1):, span_length-1] = torch.squeeze(span_logits, dim=-1)

        input_mask = input_mask.unsqueeze(dim=2)
        mention_logits = mention_logits * input_mask.expand(
            -1, -1, self.max_span_length)
        mention_logits = mention_logits
        return mention_logits

    def get_gold_mentions(self, gold_starts, gold_ends, input_mask):
        batch_size, time_steps = input_mask.size()
        # B x T1 x L
        gold_mentions = torch.zeros(batch_size, time_steps,
                                    self.max_span_length).cuda()
        max_mentions = gold_starts.shape[1]
        for i in range(batch_size):
            for j in range(max_mentions):
                span_length = gold_ends[i, j] - gold_starts[i, j]
                if span_length > 0:
                    # Not a padded gold mention
                    if span_length <= self.max_span_length:
                        gold_mentions[i, gold_ends[i, j], span_length - 1] = 1.0

        return gold_mentions

    def forward(self, batch_data):
        """
        Encode a batch of excerpts.
        """
        text, text_length = batch_data.text
        # input_mask = get_sequence_mask(text_length).cuda().float()

        encoded_output, input_mask = self.encode_excerpt(batch_data)

        pred_mention_logits = self.predict_mention_prob(
            encoded_output, input_mask)

        gold_mentions = self.get_gold_mentions(
            batch_data.gold_starts[0].cuda(),
            batch_data.gold_ends[0].cuda(), input_mask)

        mention_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_mention_logits, gold_mentions, reduction='sum')

        total_weight = torch.sum(input_mask) * self.max_span_length

        if self.training:
            loss = {}
            loss['mention'] = mention_loss/total_weight
            return loss

        else:
            pred_mention_probs = torch.sigmoid(pred_mention_logits)
            return mention_loss, total_weight, pred_mention_probs, gold_mentions
