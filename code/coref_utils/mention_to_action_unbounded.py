import os
import json
from collections import defaultdict, OrderedDict
from os import path
import numpy as np
import sys


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


def get_actions(clusters):
    # Useful data structures
    mention_to_cluster = get_mention_to_cluster(clusters)
    ordered_mentions = get_ordered_mentions(clusters)

    actions = []
    cell_to_cluster = {}
    cluster_to_cell = {}

    cell_counter = 0
    for mention in ordered_mentions:
        mention_cluster = mention_to_cluster[tuple(mention)]
        if mention_cluster in cluster_to_cell:
            # Cluster is already being tracked
            actions.append((cluster_to_cell[mention_cluster], 'c'))
        else:
            # Cluster is not being tracked
            # Add the mention to being tracked
            cluster_to_cell[mention_cluster] = cell_counter
            cell_to_cluster[cell_counter] = mention_cluster
            actions.append((cell_counter, 'o'))
            cell_counter += 1

    return actions


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


def get_mention_to_action_partition(split, seg_len, input_dir, output_dir):
    input_file = path.join(input_dir, "{}.{}.jsonlines".format(split, seg_len))
    output_file = path.join(output_dir, "{}.{}.jsonlines".format(split, seg_len))

    data = load_jsonl(input_file)
    with open(output_file, 'w') as f:
        for doc in data:
            clusters = doc["clusters"]
            actions = get_actions(clusters)
            ord_mentions = get_ordered_mentions(clusters)
            pred_clusters = action_sequences_to_clusters(actions, ord_mentions)
            mention_to_cluster = get_mention_to_cluster(clusters)

            doc['actions'] = actions
            doc['oracle_clusters'] = pred_clusters
            doc['ord_mentions'] = ord_mentions
            doc['cluster_ids'] = [mention_to_cluster[tuple(mention)] for mention in ord_mentions]
            f.write(json.dumps(doc) + "\n")


def get_mention_to_action(cross_val_split, seg_len, input_dir, output_dir):
    cross_val_input_dir = path.join(input_dir, str(cross_val_split))
    cross_val_dir = path.join(output_dir, str(cross_val_split))
    if not path.exists(cross_val_dir):
        os.makedirs(cross_val_dir)

    for split in ["dev", "train", "test"]:
        get_mention_to_action_partition(
            split, seg_len, cross_val_input_dir, cross_val_dir)


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for cross_val_split in range(10):
        for seg_len in [128, 256, 384, 512]:
            get_mention_to_action(cross_val_split, seg_len, input_dir, output_dir)