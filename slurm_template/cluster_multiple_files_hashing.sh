#!/bin/bash -l
#SBATCH --job-name=cluster_multiple_files_hashing   # Job name
#SBATCH --output=STDOUT_TXT # Name of stdout output file
#SBATCH --error=STDERR_TXT  # Name of stderr error file
#SBATCH --partition=small       # Partition (queue) name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --cpus-per-task=16     # Number of cores (threads)
#SBATCH --mem=128G               # Memory (RAM) required (MB)
#SBATCH --time=48:00:00         # Run time (hh:mm:ss)
#SBATCH --account=PROJECT_NAME  # Project for billing

# Any other commands must follow the #SBATCH directives
srun python cluster_multiple_files_hashing.py \
        "OUTPUT_DIR/*_hashing_output/*.centers.pkl" \
        "OUTPUT_DIR/*_hashing_output/*.label.jsonl" \
        --output_dir "all_files_hashing_sampled_kmeans"  \
        --n_clusters 256