# A Simple Parallelizable K-Means for Large Dataset
## Motivation
How to do we do clustering of a large dataset (say 3TB) and partition it into k portions?
As we have a desired number of clusters, we can use k-means clustering to do the job.

Time complexity: O(k * n * d * i)
Space complexity: O(k * n + k * d)
where k is the number of clusters, n is the number of data points, d is the dimensionality of the data, and i is the number of iterations required for convergence.

Given fixed k, to reduce both complexity, we can:
- reducing n by dividing and conquer
- reducing d by using smaller model/ sparse vector
- reducing i

Dividing and conquer is the most promising one. We can reduce n dramatically but not d.
We can partition the dataset into smaller chunks, running (lower level) clustering on each chunk, and then running a higher level clustering on the cluster centroids)
Literally, it is similar to MapReduce.
Map step: partition the dataset into smaller chunks and running clustering on each chunk
Reduce step: running a higher level clustering on the cluster centroids and reassigning cluster membership from the lower level clustering to the high level clustering

The key here is that we don't need to save the embedding of individual record because it will be either OOM or out of disk space. 
Instead, the assumption is that clustering shall fairly represent the region of where each record stays in the representation space.

There are also limited broadcast overhead: 
- among mappers, as each mapper only run kmeans on a single partition without knowing the other processed partitions
- between mapper and reducer, as we only need to broadcast the cluster centroids and the mapping from each chunk to the reducer. Unlike other algorithms like PKMeans that will broadcast the global centroid for the next iteration.

Encodings explored:
- hashing trick
- embedding


## Algorithm
1. Map Step: Running clustering on a single file, by chunking it into n chunks
   - chunk1
     - 16 cluster centroids 
       - description: embedding of 16 cluster centroids
       - format: {"eval.jsonl__0__10000___0": [dense or sparse vector]}
     - membership in chunk1
       - description: mapping from each line to its cluster index
       - format: {f"{filename}__{start_idx}__{n_processed_line}___{start_idx+idx}": f"{filename}__{start_idx}__{n_processed_line}___{label}"} 
   - chunk2
     - 16 cluster centroids
     - membership in chunk2
   - chunk3
     - 16 cluster centroids
     - membership in chunk3
   - ...
2. Reduce Step: Running clustering on multiple files 
   - run clustering on 16 x n cluster centroids to get 16 cluster centroids
   - reassign membership of all chunks to the 16 cluster centroids
   - outputs:
     - config
     - 16 cluster centroids: {"0": [dense or sparse vector]}
     - mapping: {"eval.jsonl__90000__10000___0": 0}
     - result: {"eval.jsonl__0__10000___0": 1}

    
## Running
### Hashing
```shell
python cluster_single_file_hashing.py "eval.jsonl" --n_clusters 16
python cluster_multiple_files_hashing.py "*_hashing_output/*.centers.pkl"  "*_hashing_output/*.label.jsonl" --output_dir "all_files_hashing"  --n_clusters 16
```

```shell
python cluster_single_file_hashing.py "eval.jsonl" --n_clusters 16 --sampled_kmean
python cluster_multiple_files_hashing.py "*_hashing_sampled_kmeans_output/*.centers.pkl"  "*_hashing_sampled_kmeans_output/*.label.jsonl" --output_dir "all_files_hashing_sampled_kmeans"  --n_clusters 16
```

### Embedding
```shell
python cluster_single_file_embedding.py "eval.jsonl" --n_clusters 16
python cluster_multiple_files_embedding.py "*_embedding_output/*.centers.jsonl"  "*_embedding_output/*.label.jsonl" --output_dir "all_files_embedding"  --n_clusters 16
```

## Benchmarking
```shell
cd benchmark
python benchmark_embedding.py
python benchmark_hashing.py
python hash_vs_embedding.py
```
### Result:
Evaluation is using evaluation data of MS Marco dataset. It consists of 100k records.

|                                                  | v measure score with Full Hash |
|--------------------------------------------------|--------------------------------|
| Full Hash                                        | 1.000                          |
| Distributed Hash (11 partition)                  | 0.640                          |
| Distributed Hash (11 partition, sampled_kmeans)  | 0.542                          |               |
| Distributed Hash (102 partition)                 | 0.585                          |
| Distributed Hash (102 partition, sampled_kmeans) | 0.552                          |


|                                      | v measure score with Full Embedding |
|--------------------------------------|-------------------------------------|
| Full Embedding                       | 1.000                               |
| Distributed Embedding (11 partition) | 0.100                               |
- In terms of agreement, Distributed Hashing Trick is far closer (0.640 vs 0.100) to Full Hash than Distributed Embedding is to Full Embedding. The reason needs to be further investigated.
- Agreement decreases as the number of partitions increases 
- Agreement of sampled KMeans decreases (0.640 vs 0.542), but the gap decreases as the number of partitions increases (0.585 vs 0.552)


|                                  | v measure score with Full Hash |
|----------------------------------|--------------------------------|
| Full Embedding                   | 0.444                          |
- Agreement of Full Hash vs Full Embedding: 0.444

## Limitation
- More dataset with different scale is required to evaluate the performance of the algorithm
- The performance decreases with no of partitions.
- It is expected the performance becomes worse when n_cluster increases because the assumption that clustering shall fairly represent the region of where each record stays in the representation space is easy to break.
