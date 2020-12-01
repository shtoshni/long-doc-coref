import torch

from transformers import BertTokenizer
from auto_memory_model.utils import action_sequences_to_clusters
from auto_memory_model.controller.utils import pick_controller
from inference.tokenize_doc import get_tokenized_doc, flatten


class Inference:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = pick_controller(device=self.device, **checkpoint['model_args']).to(self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()  # Eval mode

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def perform_coreference(self, doc):
        tokenized_doc = get_tokenized_doc(doc, self.tokenizer)
        doc_tokens = flatten(tokenized_doc["sentences"])
        subtoken_map = tokenized_doc["subtoken_map"]

        _, pred_actions, pred_mentions, _ = self.model(tokenized_doc)
        idx_clusters = action_sequences_to_clusters(pred_actions, pred_mentions)

        mentions = []
        for (ment_start, ment_end) in pred_mentions:
            mentions.append(subtoken_map[ment_start], subtoken_map[ment_end + 1])

        clusters = []
        for idx_cluster in idx_clusters:
            cur_cluster = []
            for (ment_start, ment_end) in idx_cluster:
                cur_cluster.append(((subtoken_map[ment_start], subtoken_map[ment_end + 1]),
                                    self.tokenizer.convert_tokens_to_string(doc_tokens[ment_start: ment_end + 1])))

            clusters.append(cur_cluster)

        return tokenized_doc, clusters, mentions, pred_actions

