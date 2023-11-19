from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import KMeans
import json
import time
import os
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score, adjusted_rand_score, adjusted_mutual_info_score


filename = "../eval.jsonl"
text_col_name = "text"


if __name__ == "__main__":
    if not os.path.exists("benchmark_hashing.jsonl"):
        data = []
        with open(filename) as f:
            for idx, line in enumerate(f.readlines()):
                data.append(json.loads(line)[text_col_name])

        hashing = HashingVectorizer(ngram_range=(1, 1), n_features=2**20)
        embeddings = hashing.fit_transform(data)

        km = KMeans(n_clusters=16, random_state=0)
        km_labels = km.fit_predict(embeddings).tolist()

        for l in range(16):
            idx_list = [idx for idx, label in enumerate(km_labels) if label == l][:10]
            for idx in idx_list:
                print(data[idx])
            print('='*100)

        with open(f"benchmark_hashing.jsonl", "w") as f:
            for i, label in enumerate(km_labels):
                f.write(json.dumps({i: label}) + '\n')
    else:
        km_labels = []
        with open(f"benchmark_hashing.jsonl") as f:
            for l in f.readlines():
                km_labels.append(json.loads(l))

        km_labels = sorted(km_labels, key=lambda x: int(list(x.keys())[0]))
        km_labels = [list(i.values())[0] for i in km_labels]

    partitioned_km_labels = []
    with open(f"../all_files_hashing/multiple_files_center_clustering_result.jsonl") as f:
        for l in f.readlines():
            partitioned_km_labels.append(json.loads(l))

    partitioned_km_labels = sorted(partitioned_km_labels, key=lambda x: int(list(x.keys())[0].split('___')[1]))
    partitioned_km_labels = [list(i.values())[0] for i in partitioned_km_labels]

    print('homogeneity_score: ', homogeneity_score(km_labels, partitioned_km_labels))
    print('completeness_score: ', completeness_score(km_labels, partitioned_km_labels))
    print('v_measure_score: ', v_measure_score(km_labels, partitioned_km_labels))
    print('adjusted_rand_score', adjusted_rand_score(km_labels, partitioned_km_labels))
    print("adjusted_mutual_info_score:", adjusted_mutual_info_score(km_labels, partitioned_km_labels))
