import json
import os
from collections import Counter
import argparse
from joblib import Parallel, delayed
from glob import glob
from tqdm import tqdm


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
    parser.add_argument("--n_process", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.write_dir)

    data = []
    n_clusters = 0
    with open(args.clustering_result) as f:
        for l in f.readlines():
            d = tuple(*json.loads(l).items())
            n_clusters = max(n_clusters, d[1])
            data.append(d)
    n_clusters += 1
    print(f"There are {len(data)} lines.")
    print(f"There are {n_clusters} clusters.")
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

    def write_to_file_single_cluster(sorted_data, args, cluster_index):
        lazy_jsonl_loader_dict = {}
        for line_idx, cluster_idx in tqdm(sorted_data):
            file_name, line_number = line_idx.split('___')
            file_name = file_name.split("__")[0]
            line_number = int(line_number)
            if file_name not in lazy_jsonl_loader_dict:
                lazy_jsonl_loader_dict[file_name] = lazy_read_jsonl(os.path.join(args.data_dir, file_name))
            # still need to iterate through the generator although it's not in the cluster
            file_line_idx, line = next(iter(lazy_jsonl_loader_dict[file_name]))
            assert file_line_idx == line_number, f"{file_line_idx} != {line_number}"
            if cluster_idx != cluster_index:
                continue

            mode = "w" if not os.path.exists(os.path.join(args.write_dir, f"{cluster_idx}.jsonl")) else "a"
            with open(os.path.join(args.write_dir, f"{cluster_idx}.jsonl"), mode) as f:
                f.write(line)


    with Parallel(n_jobs=args.n_process) as parallel:
        parallel(
            delayed(write_to_file_single_cluster)(sorted_data, args, c_idx)
            for c_idx in range(0, n_clusters)
        )


    def count_jsonl(filename):
        line_count = 0
        with open(filename) as f:
            for _ in f.readlines():
                line_count += 1
        return line_count

    total_line_check = sum(count_jsonl(f) for f in glob(os.path.join(args.write_dir, "*.jsonl")))
    print(f"Count of total lines: {total_line_check} in output files, {len(sorted_data)} in clustering_result")
