import torch
import json
from transformers import BertTokenizer, BertTokenizerFast
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
        print(checkpoint['model_args'])
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()  # Eval mode

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    def perform_coreference(self, doc, doc_key="nw", num_sents=None):
        if isinstance(doc, str) or isinstance(doc, list):
            tokenized_doc = get_tokenized_doc(doc, self.tokenizer)
        elif isinstance(doc, dict):
            tokenized_doc = doc
        else:
            raise ValueError

        # Ontonotes model need document genre which is formatted as the first two characters of the doc key
        tokenized_doc["doc_key"] = doc_key

        # print(len(tokenized_doc["sentences"]))
        output_doc_dict = {"sentences": tokenized_doc["sentences"], "subtoken_map": tokenized_doc["subtoken_map"]}
        if num_sents is not None:
            tokenized_doc["sentences"] = tokenized_doc["sentences"][:num_sents]
            output_doc_dict["sentences"] = tokenized_doc["sentences"]
            num_words = sum([len(sentence) for sentence in tokenized_doc["sentences"]])
            output_doc_dict["subtoken_map"] = tokenized_doc["subtoken_map"][:num_words]

        doc_tokens = flatten(tokenized_doc["sentences"])
        subtoken_map = tokenized_doc["subtoken_map"]

        with torch.no_grad():
            _, pred_actions, pred_mentions, _ = self.model(tokenized_doc)

        idx_clusters = action_sequences_to_clusters(pred_actions, pred_mentions)

        mentions = []
        for (ment_start, ment_end) in pred_mentions:
            mentions.append((subtoken_map[ment_start], subtoken_map[ment_end]))

        clusters = []
        for idx_cluster in idx_clusters:
            cur_cluster = []
            for (ment_start, ment_end) in idx_cluster:
                cur_cluster.append(((subtoken_map[ment_start], subtoken_map[ment_end]),
                                    self.tokenizer.convert_tokens_to_string(doc_tokens[ment_start: ment_end + 1])))

            clusters.append(cur_cluster)

        return {"tokenized_doc": output_doc_dict, "clusters": clusters,
                "subtoken_idx_clusters": idx_clusters, "actions": pred_actions}
