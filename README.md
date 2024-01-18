# A Simple Parallelizable K-Means for Large Dataset - It works with SLURM and local machine
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
1. Map Step: Running clustering on a single file, by chunking it into n chunks `cluster_single_file_hashing.py`
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
2. Reduce Step: Running clustering on multiple files `cluster_multiple_files_hashing.py`
   - run clustering on 16 x n cluster centroids to get 16 cluster centroids
   - reassign membership of all chunks to the 16 cluster centroids
   - outputs:
     - config
     - 16 cluster centroids: {"0": [dense or sparse vector]}
     - mapping: {"eval.jsonl__90000__10000___0": 0}
     - result: {"eval.jsonl__0__10000___0": 1}
3. Produce partitions (multiple jsonl files) based on clustering result `partition_cluster_data.py`
   - outputs:
     - 0.jsonl
     - 1.jsonl
     - 2.jsonl
     - ...

## Further Optimisations
- Use minibatch KMeans to reduce memory usage
- For hashing tricks, use unigram to reduce memory usage
- For hashing tricks, only hash the first n (e.g. 2000) characters to reduce memory usage
- Save centroid centers at np.float16 to reduce disk usage.

## Installation
```shell
pip install -r requirements.txt
```

## Running
It assumes all data files are in jsonl.

### Locally
#### Hashing
```shell
python cluster_single_file_hashing.py "eval.jsonl" --n_clusters 16 --n_process 4
python cluster_multiple_files_hashing.py "*_hashing_output/*.centers.pkl"  "*_hashing_output/*.label.jsonl" --output_dir "all_files_hashing"  --n_clusters 16
python partition_cluster_data.py "all_files_hashing/multiple_files_center_clustering_result.jsonl" "." "all_files_hashing/partitions"  --n_process 4
```

#### Embedding (TODO)
```shell
python cluster_single_file_embedding.py "eval.jsonl" --n_clusters 16 --n_process 4
python cluster_multiple_files_embedding.py "*_embedding_output/*.centers.jsonl"  "*_embedding_output/*.label.jsonl" --output_dir "all_files_embedding"  --n_clusters 16
python partition_cluster_data.py "all_files_embedding/multiple_files_center_clustering_result.jsonl" "." "all_files_embedding/partitions"  --n_process 4
```

### In SLURM (typically in a supercomputer)
All shell scripts are in `slurm_template` folder. Please modify the scripts to fit your needs.
1. Map Step - Clustering each file
You can submit job arrays to SLURM, given that your file names are with an integer index.
For example, the below script will cluster 100 files with names `data_0.jsonl`, `data_1.jsonl`, ..., `data_99.jsonl` in parallel. There are 100 jobs in total.
```shell
sbatch cluster_single_file_hashing.sh
````
Or you can run sequentially with the below script 
```python
import os


folder_path = DATA_FOLDER_TO_CLUSTER
output_dir_parent = OUTPUT_DIR
files = os.listdir(folder_path)

for file_name in files:
    print(f"Started clustering {file_name}")
    if file_name.endswith(".jsonl"):
        command = f"python cluster_single_file_hashing.py {os.path.join(folder_path, file_name)} --output_dir_parent {output_dir_parent} --n_clusters 16 --batch_size 30000 --n_process 12"
        os.system(command)
        print(f"Finished clustering {file_name}")
```
2. Reduce Step
```shell
sbatch cluster_multiple_files_hashing.sh
```
3. Partition Data
```shell
sbatch partition.sh
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
| Distributed Hash (11 partition)                  | 0.546                          |
| Distributed Hash (11 partition, sampled_kmeans)  | 0.495                          |
| Distributed Hash (102 partition)                 | 0.548                          |
| Distributed Hash (102 partition, sampled_kmeans) | 0.471                          |


|                                      | v measure score with Full Embedding |
|--------------------------------------|-------------------------------------|
| Full Embedding                       | 1.000                               |
| Distributed Embedding (11 partition) | 0.100                               |
- In terms of agreement, Distributed Hashing Trick is far closer (0.546 vs 0.100) to Full Hash than Distributed Embedding is to Full Embedding. The reason needs to be further investigated.
- Using sampled KMeans results in worse agreement (0.495 vs 0.546 and 0.471 vs 0.548).

|                                  | v measure score with Full Hash |
|----------------------------------|--------------------------------|
| Full Embedding                   | 0.416                          |
- Agreement of Full Hash vs Full Embedding: 0.416

## Limitation
- More dataset with different scale is required to evaluate the performance of the algorithm
- The performance decreases with no of partitions.
- It is expected the performance becomes worse when n_cluster increases because the assumption that clustering shall fairly represent the region of where each record stays in the representation space is easy to break.
