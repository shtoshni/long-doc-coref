"""This is an adaptation of the tokenizer used for LitBank in the overlapping segments setting."""


import re
from data_processing.overlap_ontonotes import normalize_word

BERT_RE = re.compile(r'## *')
MAX_SEGMENT_LEN = 512


class DocumentState(object):
    def __init__(self):
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.segments = []
        self.real_segments = []
        self.start_indices = []
        self.end_indices = []
        self.subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []

    def finalize(self):
        subtoken_map = flatten(self.segment_subtoken_map)
        num_words = len(flatten(self.segments))
        # assert num_words == len(flatten(self.speakers))
        assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
        return {
            "sentences": self.segments,
            "real_sentences": self.real_segments,
            "start_indices": self.start_indices,
            "end_indices": self.end_indices,
            'sentence_map': [0] * num_words,  # Assume no sentence boundaries are specified
            "subtoken_map": subtoken_map,
        }


def flatten(l):
  return [item for sublist in l for item in sublist]


def split_into_segments(document_state, constraints1, constraints2):
    current = 0
    prev_current = -1
    start_idx = 0

    while current < len(document_state.subtokens):
        if prev_current == current:
            break
        # print(current, len(document_state.subtokens))
        end = min(current + MAX_SEGMENT_LEN - 1 - 2,
                  len(document_state.subtokens) - 1)
        while end >= current and not constraints1[end]:
            end -= 1
        if end < current:
            end = min(current + MAX_SEGMENT_LEN - 1 - 2,
                      len(document_state.subtokens) - 1)
            while end >= current and not constraints2[end]:
                end -= 1
            if end < current:
                raise Exception("Can't find valid segment")

        # print(end)
        if (end + 1) == len(document_state.subtokens):
            end_idx = end + 1
        else:
            last_seg_length = end - current + 1
            # Move current to the middle of last window
            ovlp_current = end - last_seg_length//2
            while ovlp_current < end and not constraints1[ovlp_current]:
                ovlp_current += 1
            # Move to next sentence start token
            ovlp_current += 1
            if ovlp_current == (end + 1):
                ovlp_current = end - last_seg_length//2
                while ovlp_current < end and not constraints2[ovlp_current]:
                    ovlp_current += 1
                # Move to next word
                ovlp_current += 1

            extra_length = (end + 1 - ovlp_current)//2
            end_idx = ovlp_current + extra_length

        document_state.real_segments.append(document_state.subtokens[current:end + 1])
        document_state.segments.append(document_state.subtokens[start_idx:end_idx])
        subtoken_map = document_state.subtoken_map[start_idx: end_idx]
        document_state.segment_subtoken_map.append(subtoken_map)

        document_state.start_indices.append(start_idx - current)
        document_state.end_indices.append(end_idx - current)
        # print(start_idx, end_idx)
        start_idx = end_idx

        if (end + 1) == len(document_state.subtokens):
            current = end + 1
        else:
            current = ovlp_current


def get_tokenized_doc(doc, tokenizer):
    document_state = DocumentState()
    if isinstance(doc, list):
        for word_idx, word in enumerate(doc):
            word = normalize_word(word)
            subtokens = tokenizer.tokenize(word)
            document_state.tokens.append(word)
            document_state.token_end += ([False]
                                         * (len(subtokens) - 1)) + [True]
            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)
    else:
        tokenized_doc = tokenizer.tokenize(doc)
        word_idx = -1
        for idx, token in enumerate(tokenized_doc):
            if not BERT_RE.match(token):
                word_idx += 1

            document_state.tokens.append(token)
            # Subtoken and token are same
            document_state.subtokens.append(token)
            if idx == len(tokenized_doc) - 1:
                # End of document
                document_state.token_end += ([True])
            else:
                next_token = tokenized_doc[idx + 1]
                if BERT_RE.match(next_token):
                    # If the next token has ## at the start then the current subtoken
                    # is clearly not the end of the token
                    document_state.token_end += ([False])
                else:
                    document_state.token_end += ([True])

            document_state.subtoken_map.append(word_idx)
            document_state.sentence_end.append(False)  # No info on sentence end

    split_into_segments(document_state, document_state.sentence_end, document_state.token_end)
    document = document_state.finalize()
    return document


if __name__ == "__main__":
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    doc = "My father’s eyes had closed upon the light of this world six months, when Ishmael opened on it."
    print(get_tokenized_doc(doc, tokenizer))

