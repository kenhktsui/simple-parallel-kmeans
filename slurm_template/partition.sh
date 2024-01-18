#!/bin/bash -l
#SBATCH --job-name=partition   # Job name
#SBATCH --output=STDOUT_TXT # Name of stdout output file
#SBATCH --error=STDERR_TXT  # Name of stderr error file
#SBATCH --partition=small       # Partition (queue) name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --cpus-per-task=16     # Number of cores (threads)
#SBATCH --mem=128G               # Memory (RAM) required (MB)
#SBATCH --time=48:00:00         # Run time (hh:mm:ss)
#SBATCH --account=PROJECT_NAME  # Project for billing

# Any other commands must follow the #SBATCH directives
srun python partition_cluster_data.py  \
          "OUTPUT_DIR/all_files_hashing_sampled_kmeans/multiple_files_center_clustering_result.jsonl" \
          "DATA_FOLDER_TO_CLUSTER" \
          "OUTPUT_DIR/all_files_hashing_sampled_kmeans/partitions" \
          --n_process $SLURM_CPUS_PER_TASK