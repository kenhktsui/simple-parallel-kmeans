import json
from joblib import Parallel, delayed
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer
from scipy import sparse
import joblib
import numpy as np


def embed_data(text_list):
    vectorizer = HashingVectorizer(ngram_range=(1, 1), n_features=2**20)
    embeddings = vectorizer.fit_transform(text_list)
    return embeddings


def cluster_single_chunk(filename,
                         text_col_name,
                         output_dir,
                         text_len=2000,
                         n_clusters=16,
                         start_idx=0,
                         n_processed_line=None,
                         sampled_kmeans=False):
    data = []
    with open(filename) as f:
        for idx, line in enumerate(f.readlines()):
            if idx < start_idx:
                continue
            if n_processed_line is not None and idx >= start_idx + n_processed_line:
                break
            data.append(json.loads(line)[text_col_name][:text_len])

    embeddings = embed_data(data)

    km_func = MiniBatchKMeans if sampled_kmeans else KMeans
    km = km_func(n_clusters=n_clusters, random_state=0, init="random")
    km_labels = km.fit_predict(embeddings).tolist()
    centers = [sparse.csr_matrix(m, dtype=np.float16) for m in km.cluster_centers_]

    base_filename = os.path.basename(filename)

    membership = {f"{base_filename}__{start_idx}__{n_processed_line}___{start_idx+idx}": f'{base_filename}__{start_idx}__{n_processed_line}___{label}'
                  for idx, label in enumerate(km_labels)}

    centers_dict = {}
    for c_i, c in enumerate(centers):
        centers_dict[f"{base_filename}__{start_idx}__{n_processed_line}___{c_i}"] = c

    joblib.dump(centers_dict, f"{output_dir}/{base_filename}__{start_idx}__{n_processed_line}.centers.pkl")

    with open(f"{output_dir}/{base_filename}__{start_idx}__{n_processed_line}.label.jsonl", "w") as f:
        for k, v in membership.items():
            f.write(json.dumps({k: v}) + '\n')


def cluster_single_file(filename, text_col_name, output_dir, text_len,
                        n_clusters, n_process, batch_size=10000, sampled_kmeans=False):
    line_count = 0
    with open(filename) as f:
        for _ in f.readlines():
            line_count += 1

    with Parallel(n_jobs=n_process) as parallel:
        parallel(
            delayed(cluster_single_chunk)(filename, text_col_name, output_dir, text_len=text_len, n_clusters=n_clusters,
                                          start_idx=idx, n_processed_line=batch_size, sampled_kmeans=sampled_kmeans)
            for idx in range(0, line_count, batch_size)
        )


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Clustering single file...")
    parser.add_argument("filename", type=str)
    parser.add_argument("--output_dir_parent", type=str, default='.')
    parser.add_argument("--text_col_name", type=str, default="text")
    parser.add_argument("--text_len", type=int, default=2000)
    parser.add_argument("--n_clusters", type=int, default=16)
    parser.add_argument("--n_process", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--sampled_kmeans", action="store_true")

    args = parser.parse_args()

    output_dir = f"{args.output_dir_parent}/" + args.filename.split('/')[-1].split('.')[0] + "_hashing_output"
    os.makedirs(output_dir, exist_ok=False)

    cluster_single_file(args.filename,
                        args.text_col_name,
                        output_dir,
                        args.text_len,
                        args.n_clusters,
                        args.n_process,
                        batch_size=args.batch_size,
                        sampled_kmeans=args.sampled_kmeans
                        )
