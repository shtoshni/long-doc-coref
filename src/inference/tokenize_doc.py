"""This is an adaptation of the tokenizer used for LitBank in the overlapping segments setting."""


import re

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
            "start_indices": self.start_indices,
            "end_indices": self.end_indices,
            'sentence_map': [0] * num_words,  # Assume no sentence boundaries are specified
            "subtoken_map": subtoken_map,
        }


def flatten(l):
  return [item for sublist in l for item in sublist]


def split_into_segments(document_state, max_segment_len, constraints1, constraints2):
    current = 0
    previous_token = 0
    while current < len(document_state.subtokens):
        end = min(current + max_segment_len - 1 - 2,
                  len(document_state.subtokens) - 1)
        while end >= current and not constraints1[end]:
            end -= 1
        if end < current:
            end = min(current + max_segment_len - 1 - 2,
                      len(document_state.subtokens) - 1)
            while end >= current and not constraints2[end]:
                end -= 1
            if end < current:
                raise Exception("Can't find valid segment")
        document_state.segments.append(
            ['[CLS]'] + document_state.subtokens[current:end + 1] + ['[SEP]'])
        subtoken_map = document_state.subtoken_map[current: end + 1]
        document_state.segment_subtoken_map.append(
            [previous_token] + subtoken_map + [subtoken_map[-1]])
        current = end + 1
        previous_token = subtoken_map[-1]


def get_tokenized_doc(doc, tokenizer):
    document_state = DocumentState()
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

    split_into_segments(document_state, MAX_SEGMENT_LEN, document_state.sentence_end, document_state.token_end)
    document = document_state.finalize()
    return document


if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    doc = open("/share/data/speech/Data/kgimpel/ccarol.tok.txt").read()
    document = get_tokenized_doc(doc, tokenizer)
    import json
    output_file = "/home/shtoshni/Research/long-doc-coref/notebooks/ccarol.json"
    with open(output_file, 'w') as fp:
        json.dump(document, fp)

