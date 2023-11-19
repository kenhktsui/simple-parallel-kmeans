import json
import os
from collections import Counter
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    """
    Algorithm:
    1. Load the clustering result, sort by file name and line number
    2. Lazy load the jsonl file to retrieve the line
    3. Write the line to the corresponding file line by line
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("clustering_result", type=str)
    parser.add_argument("data_dir", type=str)
    parser.add_argument("write_dir", type=str)
    args = parser.parse_args()

    data = []
    with open(args.clustering_result) as f:
        for l in f.readlines():
            data.append(tuple(*json.loads(l).items()))
    print(f"There are {len(data)} lines.")
    print(f"Distribution: {Counter([d[1] for d in data])}")

    def sort_func(x):
        file_name, line_number = x.split("___")
        file_name = file_name.split("__")[0]
        return (file_name, int(line_number))

    sorted_data = sorted(data, key=lambda x: sort_func(x[0]))

    def lazy_read_jsonl(filename):
        """generator from jsonl with idx"""
        with open(filename) as f:
            for idx, line in enumerate(f):
                yield idx, line

    lazy_jsonl_loader_dict = {}
    for line_idx, cluster_idx in tqdm(sorted_data):
        file_name, line_number = line_idx.split('___')
        file_name = file_name.split("__")[0]
        line_number = int(line_number)
        if file_name not in lazy_jsonl_loader_dict:
            lazy_jsonl_loader_dict[file_name] = lazy_read_jsonl(os.path.join(args.data_dir, file_name))
        file_line_idx, line = next(iter(lazy_jsonl_loader_dict[file_name]))
        assert file_line_idx == line_number, f"{file_line_idx} != {line_number}"
        if not os.path.exists(os.path.join(args.write_dir, f"{cluster_idx}.jsonl")):
            mode = "w"
        else:
            mode = "a"

        with open(os.path.join(args.write_dir, f"{cluster_idx}.jsonl"), mode) as f:
            f.write(line)
