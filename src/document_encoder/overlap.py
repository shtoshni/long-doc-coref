import torch
import random
from pytorch_utils.utils import get_sequence_mask
from document_encoder.base_encoder import BaseDocEncoder


class OverlapDocEncoder(BaseDocEncoder):
    def __init__(self, **kwargs):
        super(OverlapDocEncoder, self).__init__(**kwargs)

    def encode_doc(self, example):
        """
        Encode chunks of a document.
        batch_excerpt: C x L where C is number of chunks padded upto max length of L
        text_length_list: list of length of chunks (length C)
        """
        if self.training and self.max_training_segments is not None:
            example = self.truncate_document(example)
        sentences = example["real_sentences"]
        start_indices = example["start_indices"]
        end_indices = example["end_indices"]

        sentences = [(['[CLS]'] + sent + ['[SEP]']) for sent in sentences]
        sent_len_list = [len(sent) for sent in sentences]
        max_sent_len = max(sent_len_list)
        padded_sent = [self.tokenizer.convert_tokens_to_ids(sent)
                       + [self.pad_token] * (max_sent_len - len(sent))
                       for sent in sentences]
        doc_tens = torch.tensor(padded_sent).to(self.device)

        num_chunks = len(sent_len_list)
        attn_mask = get_sequence_mask(torch.tensor(sent_len_list).to(self.device)).to(self.device).float()

        with torch.no_grad():
            outputs = self.bert(doc_tens, attention_mask=attn_mask)  # C x L x E

        encoded_repr = outputs[0]

        unpadded_encoded_output = []
        offset = 1  # for [CLS] token which was not accounted during segmentation
        for i in range(num_chunks):
            unpadded_encoded_output.append(
                encoded_repr[i, offset + start_indices[i]:offset + end_indices[i], :])

        encoded_output = torch.cat(unpadded_encoded_output, dim=0)
        encoded_output = encoded_output
        return encoded_output

    def truncate_document(self, example):
        num_sentences = len(example["real_sentences"])

        if num_sentences > self.max_training_segments:
            sentence_offset = random.randint(0, num_sentences - self.max_training_segments)
            word_offset = sum([(end_idx - start_idx)
                               for start_idx, end_idx in zip(example["start_indices"][:sentence_offset],
                                                             example["end_indices"][:sentence_offset])])
            sentences = example["real_sentences"][sentence_offset: sentence_offset + self.max_training_segments]

            start_indices = example["start_indices"][sentence_offset: sentence_offset + self.max_training_segments]
            # Set first window to start at 0th token
            word_offset -= start_indices[0]
            start_indices[0] = 0

            end_indices = example["end_indices"][sentence_offset: sentence_offset + self.max_training_segments]
            # Set last window to end at last token
            end_indices[-1] = len(sentences[-1])

            num_words = sum([(end_idx - start_idx) for start_idx, end_idx in zip(start_indices, end_indices)])
            sentence_map = example["sentence_map"][word_offset: word_offset + num_words]

            # if word_offset > 0:
            #     try:
            #         assert(example["sentence_map"][word_offset] != example["sentence_map"][word_offset - 1])
            #     except AssertionError:
            #         print(example["doc_key"], sentence_offset, example["sentence_map"][word_offset - 10: word_offset])
            # if (sentence_offset + self.max_training_segments) < num_sentences:
            #     idx = word_offset + num_words - 1
            #     try:
            #         assert(example["sentence_map"][idx] != example["sentence_map"][idx + 1])
            #     except AssertionError:
            #         print(example["doc_key"], sentence_offset, example["sentence_map"][idx: idx + 10])

            clusters = []
            for orig_cluster in example["clusters"]:
                cluster = []
                for ment_start, ment_end in orig_cluster:
                    if ment_end >= word_offset and ment_start < word_offset + num_words:
                        cluster.append((ment_start - word_offset, ment_end - word_offset))

                if len(cluster):
                    clusters.append(cluster)

            example["real_sentences"] = sentences
            example["clusters"] = clusters
            example["sentence_map"] = sentence_map
            example["start_indices"] = start_indices
            example["end_indices"] = end_indices

            return example
        else:
            return example

    def forward(self, example):
        return self.encode_doc(example)
