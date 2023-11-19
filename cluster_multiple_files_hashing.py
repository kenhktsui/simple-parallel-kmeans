import json
from glob import glob
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.sparse import vstack
import joblib
import numpy as np
from tqdm import tqdm
from scipy import sparse


def cluster_multiple_file(center_path, label_path, output_dir, n_clusters=16, sampled_kmeans=False):
    config = {
        "center_path": center_path,
        "label_path": label_path,
        "output_dir": output_dir,
        "n_clusters": n_clusters,
        "sampled_kmeans": sampled_kmeans
    }

    centers_path = glob(center_path)
    labels_path = glob(label_path)
    print(f"Found {len(centers_path)} centers files")
    print(f"Found {len(labels_path)} labels files")

    centers = {}
    for cp in tqdm(centers_path):
        f = joblib.load(cp)
        centers.update(f)
    print(f"There are {len(centers)} centers of subcluster to be clustered")

    center_name_list, center_embedding_list = [], []
    for center_name, embedding_list in tqdm(centers.items()):
        for embed in embedding_list:
            center_name_list.append(center_name)
            center_embedding_list.append(embed)

    center_embedding_list = vstack(center_embedding_list, format="csr")

    km_func = MiniBatchKMeans if sampled_kmeans else KMeans
    km = km_func(n_clusters=n_clusters, random_state=0)
    km_labels = km.fit_predict(center_embedding_list).tolist()
    centers = km.cluster_centers_
    print(km.inertia_)

    assert len(center_name_list) == len(km_labels)
    multiple_file_center_clustering_mapping = {k: v for k, v in zip(center_name_list, km_labels)}

    with open(f"{output_dir}/multiple_files_center_clustering_config.json", "w") as f:
        json.dump(config, f)

    with open(f"{output_dir}/multiple_files_center_clustering_mapping.jsonl", "w") as f:
        for k, v in multiple_file_center_clustering_mapping.items():
            f.write(json.dumps({k: v}) + '\n')

    joblib.dump({c_i: sparse.csr_matrix(c) for c_i, c in enumerate(centers)},
                f"{output_dir}/multiple_files_center_clustering_centers.pkl")

    labels = {}
    for lp in tqdm(labels_path):
        with open(lp) as f:
            for line in f.readlines():
                labels.update(json.loads(line))
    print(f"Found {len(labels)} labels")

    for k, v in labels.items():
        labels[k] = multiple_file_center_clustering_mapping.get(v, -1)

    with open(f"{output_dir}/multiple_files_center_clustering_result.jsonl", "w") as f:
        for k, v in labels.items():
            f.write(json.dumps({k: v}) + '\n')


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Clustering multiple files...")
    parser.add_argument("center_path", type=str)
    parser.add_argument("label_path", type=str)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--n_clusters", type=int, default="16")
    parser.add_argument("--sampled_kmeans", action="store_true")
    args = parser.parse_args()

    if args.output_dir != ".":
        os.makedirs(args.output_dir, exist_ok=False)

    cluster_multiple_file(args.center_path, args.label_path, args.output_dir,
                          n_clusters=args.n_clusters, sampled_kmeans=args.sampled_kmeans)
