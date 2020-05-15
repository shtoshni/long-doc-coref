
def mention_to_cluster(clusters, threshold=1):
    clusters = [tuple(tuple(mention) for mention in cluster)
                for cluster in clusters if len(cluster) >= threshold]
    mention_to_cluster = {}
    for cluster in clusters:
        for mention in cluster:
            mention_to_cluster[mention] = cluster
    return clusters, mention_to_cluster
