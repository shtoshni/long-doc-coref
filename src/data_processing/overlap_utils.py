import collections
from data_processing.utils import flatten

class DocumentState(object):
    def __init__(self, key):
        self.doc_key = key
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.info = []
        self.segments = []
        self.real_segments = []
        self.start_indices = []
        self.end_indices = []
        self.subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []
        self.pronouns = []
        self.clusters = collections.defaultdict(list)
        self.coref_stacks = collections.defaultdict(list)
        self.speakers = []
        self.segment_info = []

    def finalize(self):
        # finalized: segments, segment_subtoken_map
        # populate speakers from info
        subtoken_idx = 0
        for segment in self.segment_info:
            speakers = []
            for i, tok_info in enumerate(segment):
                if tok_info is None and (i == 0 or i == len(segment) - 1):
                    speakers.append('[SPL]')
                elif tok_info is None:
                    speakers.append(speakers[-1])
                else:
                    speakers.append(tok_info[9])
                    if tok_info[4] == 'PRP':
                        self.pronouns.append(subtoken_idx)
                subtoken_idx += 1
            self.speakers += [speakers]
        # populate sentence map

        # populate clusters
        first_subtoken_index = -1
        for seg_idx, segment in enumerate(self.segment_info):
            for i, tok_info in enumerate(segment):
                first_subtoken_index += 1
                coref = tok_info[-2] if tok_info is not None else '-'
                if coref != "-":
                    last_subtoken_index = first_subtoken_index + \
                                          tok_info[-1] - 1
                    for part in coref.split("|"):
                        if part[0] == "(":
                            if part[-1] == ")":
                                cluster_id = int(part[1:-1])
                                self.clusters[cluster_id].append(
                                    (first_subtoken_index, last_subtoken_index))
                            else:
                                cluster_id = int(part[1:])
                                self.coref_stacks[cluster_id].append(
                                    first_subtoken_index)
                        else:
                            cluster_id = int(part[:-1])
                            start = self.coref_stacks[cluster_id].pop()
                            self.clusters[cluster_id].append(
                                (start, last_subtoken_index))
        # merge clusters
        merged_clusters = []
        for c1 in self.clusters.values():
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often.)")
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))
        merged_clusters = [list(c) for c in merged_clusters]
        all_mentions = flatten(merged_clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)
        assert len(all_mentions) == len(set(all_mentions))
        num_words = len(flatten(self.segments))
        assert num_words == len(flatten(self.speakers))
        assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
        assert num_words == len(sentence_map), (num_words, len(sentence_map))
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "real_sentences": self.real_segments,
            "start_indices": self.start_indices,
            "end_indices": self.end_indices,
            "speakers": self.speakers,
            "clusters": merged_clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
        }


def split_into_segments(document_state, max_segment_len, constraints1, constraints2):
    current = 0
    prev_current = -1
    start_idx = 0
    end_idx = -1
    while current < len(document_state.subtokens):
        if prev_current == current:
            break
        # print(current, len(document_state.subtokens))
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

        # print(end)
        if (end + 1) == len(document_state.subtokens):
            end_idx = end + 1
        else:
            last_seg_length = end - current + 1
            # Move current to the middle of last window
            ovlp_current = end - last_seg_length // 2
            while ovlp_current < end and not constraints1[ovlp_current]:
                ovlp_current += 1
            # Move to next sentence start token
            ovlp_current += 1
            if ovlp_current == (end + 1):
                ovlp_current = end - last_seg_length // 2
                while ovlp_current < end and not constraints2[ovlp_current]:
                    ovlp_current += 1
                # Move to next word
                ovlp_current += 1

            extra_length = (end + 1 - ovlp_current) // 2
            end_idx = ovlp_current + extra_length

        document_state.real_segments.append(document_state.subtokens[current:end + 1])
        document_state.segments.append(document_state.subtokens[start_idx:end_idx])
        subtoken_map = document_state.subtoken_map[start_idx: end_idx]
        document_state.segment_subtoken_map.append(subtoken_map)

        info = document_state.info[start_idx: end_idx]
        document_state.segment_info.append(info)
        document_state.start_indices.append(start_idx - current)
        document_state.end_indices.append(end_idx - current)
        # print(start_idx, end_idx)
        start_idx = end_idx

        if (end + 1) == len(document_state.subtokens):
            current = end + 1
        else:
            current = ovlp_current


def get_sentence_map(segments, sentence_end):
    current = 0
    sent_map = []
    sent_end_idx = 0
    assert len(sentence_end) == sum([len(s) for s in segments])
    for segment in segments:
        # sent_map.append(current)
        for i in range(len(segment)):
            sent_map.append(current)
            current += int(sentence_end[sent_end_idx])
            sent_end_idx += 1
        # sent_map.append(current)
    return sent_map
