import torch
import torch.nn as nn

from auto_memory_model.controller.base_controller import BaseController
from pytorch_utils.modules import MLP


class Controller(BaseController):
    def __init__(self, mlp_size=1024, mlp_depth=1, max_span_width=30, **kwargs):
        super(Controller, self).__init__(**kwargs)
        self.max_span_width = max_span_width
        self.mlp_size = mlp_size
        self.mlp_depth = mlp_depth
        self.mention_mlp = MLP(input_size=2 * self.hsize, hidden_size=self.mlp_size,
                               output_size=1, num_hidden_layers=self.mlp_depth, bias=True,
                               drop_module=self.drop_module)
        self.span_width_embeddings = nn.Embedding(self.max_span_width, 20)
        self.span_width_mlp = MLP(input_size=20, hidden_size=self.mlp_size,
                                  output_size=1, num_hidden_layers=1, bias=True,
                                  drop_module=self.drop_module)
        self.mention_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    def get_mention_scores(self, span_embs, cand_starts, cand_ends):
        mention_logits = torch.squeeze(self.mention_mlp(span_embs), dim=-1)

        span_width_idx = cand_ends - cand_starts
        span_width_embs = self.span_width_embeddings(span_width_idx)
        width_scores = torch.squeeze(self.span_width_mlp(span_width_embs), dim=-1)

        mention_logits += width_scores

        return mention_logits

    def get_gold_mentions(self, clusters, cand_starts, flat_cand_mask):
        gold_ments = torch.zeros_like(cand_starts).cuda()
        for cluster in clusters:
            for (span_start, span_end) in cluster:
                span_width = span_end - span_start + 1
                if span_width <= self.max_span_width:
                    span_width_idx = span_width - 1
                    gold_ments[span_start, span_width_idx] = 1

        filt_gold_ments = gold_ments.reshape(-1)[flat_cand_mask].float()

        assert(torch.sum(gold_ments) == torch.sum(filt_gold_ments))  # Filtering shouldn't remove gold mentions

        return filt_gold_ments

    def forward(self, example):
        """
        Encode a batch of excerpts.
        """
        encoded_doc = self.doc_encoder(example)
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
        cand_mask = (constraint1 & constraint2).bool()
        flat_cand_mask = cand_mask.reshape(-1)

        # Filter and flatten the candidate end points
        filt_cand_starts = cand_starts.reshape(-1)[flat_cand_mask]  # (num_candidates,)
        filt_cand_ends = cand_ends.reshape(-1)[flat_cand_mask]  # (num_candidates,)

        span_embs = torch.cat([encoded_doc[filt_cand_starts, :], encoded_doc[filt_cand_ends, :]], dim=-1)
        mention_scores = self.get_mention_scores(span_embs, filt_cand_starts, filt_cand_ends)

        filt_gold_mentions = self.get_gold_mentions(example["clusters"], cand_starts, flat_cand_mask)

        mention_loss = self.mention_loss_fn(mention_scores, filt_gold_mentions)
        total_weight = filt_cand_starts.shape[0]

        if self.training:
            loss = {'mention': mention_loss / total_weight}
            return loss

        else:
            pred_mention_probs = torch.sigmoid(mention_scores)
            return mention_loss, total_weight, pred_mention_probs, filt_gold_mentions, filt_cand_starts, filt_cand_ends
