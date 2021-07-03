from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re
import os
import sys
import json
import collections

from coref_utils import conll
from os import path
from transformers import BertTokenizer, BertTokenizerFast
from data_processing.overlap_utils import DocumentState, split_into_segments


def get_document(document_lines, tokenizer, segment_len):
    document_state = DocumentState(document_lines[0])
    word_idx = -1
    for line in document_lines[1]:
        row = line.split()
        sentence_end = len(row) == 0
        if not sentence_end:
            assert len(row) == 12
            # if len(row) == 11:
            #     row.append('-')
            word_idx += 1
            word = row[3]
            subtokens = tokenizer.tokenize(word)
            document_state.tokens.append(word)
            document_state.token_end += ([False]
                                         * (len(subtokens) - 1)) + [True]
            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                info = None if sidx != 0 else (row + [len(subtokens)])
                document_state.info.append(info)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)
        else:
            document_state.sentence_end[-1] = True
    # split_into_segments(document_state, segment_len, document_state.token_end)
    # split_into_segments(document_state, segment_len, document_state.sentence_end)
    constraints1 = document_state.sentence_end
    split_into_segments(document_state, segment_len,
                        constraints1, document_state.token_end)
    document = document_state.finalize()
    return document


def minimize_split(seg_len, input_dir, output_dir, split="test"):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    input_path = path.join(input_dir, "{}.conll".format(split))
    output_path = path.join(output_dir, "{}.{}.jsonlines".format(split, seg_len))
    count = 0
    print("Minimizing {}".format(input_path))
    documents = []
    with open(input_path, "r") as input_file:
        for line in input_file.readlines():
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(
                    begin_document_match.group(1), begin_document_match.group(2))
                documents.append((doc_key, []))
            elif line.startswith("#end document"):
                continue
            else:
                documents[-1][1].append(line)
    with open(output_path, "w") as output_file:
        for document_lines in documents:
            document = get_document(
                document_lines, tokenizer, seg_len)
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    print("Wrote {} documents to {}".format(count, output_path))


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for seg_len in [512]:
        minimize_split(seg_len, input_dir, output_dir)
