import json
from os import path


def load_data(data_dir, max_segment_len, dataset='litbank'):
    all_splits = []
    for split in ["train", "dev", "test"]:
        jsonl_file = path.join(data_dir, "{}.{}.jsonlines".format(split, max_segment_len))
        with open(jsonl_file) as f:
            split_data = []
            for line in f:
                split_data.append(json.loads(line.strip()))
        all_splits.append(split_data)

    train_data, dev_data, test_data = all_splits

    if dataset == 'litbank':
        assert(len(train_data) == 80)
        assert(len(dev_data) == 10)
        assert(len(test_data) == 10)
    elif dataset == 'ontonotes':
        assert (len(train_data) == 2802)
        assert (len(dev_data) == 343)
        assert (len(test_data) == 348)

    return train_data, dev_data, test_data


