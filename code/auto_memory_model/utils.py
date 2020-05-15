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


def classify_errors(pred_action_list, gt_action_list):
    # WL - Wrong link; picks the wrong cell for a coref decision
    # WO - Wrong Overwrite; picks the wrong cell for overwrite decision
    # FL - False Link; assigns a link when it's a singleton - Coref when overwrite or ignore
    # FN - Overwrite or Ignore when there's a coref decision
    decisions = {"WL": 0, "FN": 0,
                 "WF": 0, "WO": 0,
                 "FL": 0, "C": 0}
    for cur_pred_action, cur_gt_action in zip(pred_action_list, gt_action_list):
        pred_tuple, gt_tuple = tuple(cur_pred_action), tuple(cur_gt_action)
        # (_, pred_action), (_, gt_action) = pred_tuple, gt_tuple
        pred_action = pred_tuple[1]
        gt_action = gt_tuple[1]

        if pred_tuple == gt_tuple:
            decisions['C'] += 1
        elif gt_action == 'c':
            if pred_action == 'c':
                # Chose wrong cell index
                decisions['WL'] += 1
            else:
                # False new entity prediction
                decisions['FN'] += 1
        elif gt_action == 'o':
            if pred_action == 'o':
                # Maybe check if this leads to any complications down the line
                decisions['WO'] += 1
            elif pred_action == 'i':
                decisions['WF'] += 1
            else:
                decisions['FL'] += 1
        else:
            # GT is ignore
            if pred_action == 'o':
                decisions['WF'] += 1
            elif pred_action == 'c':
                decisions['FL'] += 1
            else:
                decisions['C'] += 1

    return decisions
