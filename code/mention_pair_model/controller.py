import numpy as np
import torch
import torch.nn as nn
import random

class DocEncoder(nn.Module):
    def __init__(self, max_training_sentences=N):
        super(DocEncoder, self).__init__()

    @staticmethod
    def tensorize_mentions(mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)



    def truncate_example(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map,):
        max_training_sentences = self.config["max_training_sentences"]
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(
            0, num_sentences - max_training_sentences) if sentence_offset is None else sentence_offset
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset +
                             max_training_sentences].sum()
        input_ids = input_ids[sentence_offset:sentence_offset +
                              max_training_sentences, :]
        input_mask = input_mask[sentence_offset:sentence_offset +
                                max_training_sentences, :]
        speaker_ids = speaker_ids[sentence_offset:sentence_offset +
                                  max_training_sentences, :]
        text_len = text_len[sentence_offset:sentence_offset +
                            max_training_sentences]

        sentence_map = sentence_map[word_offset: word_offset + num_words]
        gold_spans = np.logical_and(
            gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]

        return input_ids, input_mask, text_len, speaker_ids, genre, is_training,  gold_starts, gold_ends, cluster_ids, sentence_map