import torch
import torch.nn as nn

from auto_memory_model.controller.base_controller import BaseController
from pytorch_utils.modules import MLP


class Controller(BaseController):
    def __init__(self, **kwargs):
        super(Controller, self).__init__(**kwargs)
        self.mention_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    def get_gold_mentions(self, clusters, num_words, flat_cand_mask):
        gold_ments = torch.zeros(num_words, self.max_span_width).cuda()
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

        filt_cand_starts, filt_cand_ends, flat_cand_mask = self.get_candidate_endpoints(
            encoded_doc, example, return_mask=True)

        span_embs = self.get_span_embeddings(encoded_doc, self.get_genre_embedding(example),
                                             filt_cand_starts, filt_cand_ends)

        mention_logits = torch.squeeze(self.mention_mlp(span_embs), dim=-1)
        # Span embeddings not needed anymore
        mention_logits += self.get_mention_width_scores(filt_cand_starts, filt_cand_ends)

        filt_gold_mentions = self.get_gold_mentions(example["clusters"], num_words, flat_cand_mask)
        # print(torch.sum(filt_gold_mentions))

        if self.training:
            mention_loss = self.mention_loss_fn(mention_logits, filt_gold_mentions)
            total_weight = filt_cand_starts.shape[0]

            loss = {'mention': mention_loss / total_weight}
            return loss

        else:
            pred_mention_probs = torch.sigmoid(mention_logits)
            # Calculate Recall
            k = int(self.top_span_ratio * num_words)
            topk_indices = torch.topk(mention_logits, k)[1]
            topk_indices_mask = torch.zeros_like(mention_logits).cuda()
            topk_indices_mask[topk_indices] = 1
            recall = torch.sum(filt_gold_mentions * topk_indices_mask).item()

            return pred_mention_probs, filt_gold_mentions, filt_cand_starts, filt_cand_ends, recall
