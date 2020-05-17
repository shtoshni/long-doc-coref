import os
import json
from collections import defaultdict, OrderedDict
from os import path
import numpy as np


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))

    return data


def get_mention_to_cluster(clusters):
    mention_to_cluster = {}
    for cluster_idx, cluster in enumerate(clusters):
        for mention in cluster:
            mention_to_cluster[tuple(mention)] = cluster_idx
    return mention_to_cluster


def get_ordered_mentions(clusters):
    """Order all the mentions in the doc w.r.t. span_start and in case of ties span_end."""
    all_mentions = []
    for cluster in clusters:
        all_mentions.extend(cluster)

    # Span start is the main criteria, and span end is used to break ties
    all_mentions = sorted(all_mentions, key=lambda x: x[0] + 1e-5 * x[1])
    return all_mentions


def get_actions(clusters, num_cells=5):
    # Useful data structures
    mention_to_cluster = get_mention_to_cluster(clusters)
    ordered_mentions = get_ordered_mentions(clusters)

    actions = []
    cell_to_cluster = {}
    cell_to_last_used = [0 for cell in range(num_cells)]  # Initialize last usage of cell
    cluster_to_cell = {}
    # Initialize with all the mentions
    cluster_to_rem_mentions = [len(cluster) for cluster in clusters]

    non_optimal_overwrites = 0
    total_overwrites = 0

    lru_list = list(range(num_cells))

    for mention in ordered_mentions:
        used_cell_idx = None
        mention_cluster = mention_to_cluster[tuple(mention)]
        if mention_cluster in cluster_to_cell:
            # Cluster is already being tracked
            actions.append((cluster_to_cell[mention_cluster], 'c'))
            # Update when the cell was last used
            used_cell_idx = cluster_to_cell[mention_cluster]
        else:
            # Cluster is not being tracked
            # Find the cell with the least regret that we can overwrite to
            # If the regret is non-positive i.e. we would be missing out on >= mentions
            # of a cluster being currently tracked than the new mention cluster then we
            # don't perform overwrite.
            cur_rem_mentions = cluster_to_rem_mentions[mention_cluster]
            cell_info = []
            for cell_idx in range(num_cells):
                if cell_idx in cell_to_cluster:
                    # The cell is actually in use
                    cell_cluster = cell_to_cluster[cell_idx]
                    cell_rem_mentions = cluster_to_rem_mentions[cell_cluster]
                else:
                    # The cell is not in use
                    cell_rem_mentions = -1

                cell_info.append((cell_rem_mentions, cell_to_last_used[cell_idx], cell_idx,
                                  lru_list.index(cell_idx)))

            # Original sorting the cells primarily by the number of remaining mentions
            # If the remaining mentions are tied, then compare the last used cell
            orig_cell_info = sorted(cell_info, key=lambda x: x[0] - 1e-10 * x[1])
            # Sort cells by least recently used cells
            cell_info = sorted(cell_info, key=lambda x: x[3])
            # Remaining mentions in least recently used cell
            lru_remaining_mentions = cell_info[0][0]

            if cur_rem_mentions > lru_remaining_mentions:
                used_cell_idx = cell_info[0][2]  # Get the cell index

            if used_cell_idx is None:
                # Ignore the mention
                actions.append((-1, 'i'))
            else:
                # Overwrite
                actions.append((used_cell_idx, 'o'))
                # Remove the cluster to cell reference for the replacement cell
                # Only do this if the cell was tracking anything
                if used_cell_idx in cell_to_cluster:
                    del cluster_to_cell[cell_to_cluster[used_cell_idx]]

                # Add the mention to being tracked
                cluster_to_cell[mention_cluster] = used_cell_idx
                cell_to_cluster[used_cell_idx] = mention_cluster

                total_overwrites += 1
                if cell_info[0][0] > orig_cell_info[0][0]:
                    non_optimal_overwrites += 1

        # Update the cell_to_last_used index
        for cell_idx in range(num_cells):
            cell_to_last_used[cell_idx] += 1
        if used_cell_idx is not None:
            cell_to_last_used[used_cell_idx] = 0
            lru_list.remove(used_cell_idx)
            lru_list.append(used_cell_idx)

        # Reduce the number of mentions remaining in the current cluster
        cluster_to_rem_mentions[mention_cluster] -= 1

    return actions, non_optimal_overwrites, total_overwrites


def action_sequences_to_clusters(actions, mentions):
    clusters = []
    cell_to_clusters = {}

    for mention, (cell_idx, action_type) in zip(mentions, actions):
        if action_type == 'i':
            # Singleton
            clusters.append([mention])
        elif action_type == 'c':
            cell_to_clusters[cell_idx].append(mention)
        else:
            # Overwrite
            if cell_idx in cell_to_clusters:
                # Remove the old cluster and initialize the new
                clusters.append(cell_to_clusters[cell_idx])
            cell_to_clusters[cell_idx] = [mention]

    for cell_idx, cluster in cell_to_clusters.items():
        clusters.append(cluster)

    return clusters


def get_mention_to_action_partition(split, seg_len, input_dir, output_dir, num_cells):
    input_file = path.join(input_dir, "{}.{}.jsonlines".format(split, seg_len))
    output_file = path.join(output_dir, "{}.{}.jsonlines".format(split, seg_len))

    data = load_jsonl(input_file)
    total_overwrites = 0
    non_optimal_overwrites = 0

    with open(output_file, 'w') as f:
        for doc in data:
            clusters = doc["clusters"]
            actions, doc_non_optimal_overwrites, doc_total_overwrites = get_actions(
                clusters, num_cells=num_cells)
            total_overwrites += doc_total_overwrites
            non_optimal_overwrites += doc_non_optimal_overwrites

            ord_mentions = get_ordered_mentions(clusters)
            pred_clusters = action_sequences_to_clusters(actions, ord_mentions)
            mention_to_cluster = get_mention_to_cluster(clusters)

            doc['actions'] = actions
            doc['oracle_clusters'] = pred_clusters
            doc['ord_mentions'] = ord_mentions
            doc['cluster_ids'] = [mention_to_cluster[tuple(mention)] for mention in ord_mentions]
            f.write(json.dumps(doc) + "\n")
        print('Fraction of non-optimal overwrites: %.2f, # of cells: %d'
              % (100 * non_optimal_overwrites / total_overwrites, num_cells))


def get_mention_to_action(cross_val_split, num_cells, seg_len, input_dir, output_dir):
    cross_val_input_dir = path.join(input_dir, str(cross_val_split))
    cross_val_dir = path.join(output_dir, str(cross_val_split))
    cell_dir = path.join(cross_val_dir, str(num_cells))
    if not path.exists(cell_dir):
        os.makedirs(cell_dir)

    for split in ["dev", "train", "test"]:
        get_mention_to_action_partition(
            split, seg_len, cross_val_input_dir, cell_dir, num_cells)


if __name__ == "__main__":
    input_dir = "/home/shtoshni/Research/litbank_coref/data/segmentation"
    output_dir = "/home/shtoshni/Research/litbank_coref/data/autoregressive/lru"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for cross_val_split in range(10):
        for num_cells in [5, 10, 20, 50]:
            for seg_len in [128, 256, 384, 512]:
            # for seg_len in [384, 512]:
                get_mention_to_action(cross_val_split, num_cells, seg_len, input_dir, output_dir)
                # for k, v in labels.items():
                #     print("{} = [{}]".format(k, ", ".join(
                #         "\"{}\"".format(label) for label in v)))
                # for k, v in stats.items():
                #     print("{} = {}".format(k, v))
