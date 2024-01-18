#!/bin/bash -l
#SBATCH --array=0-99
#SBATCH --job-name=cluster_single_file_hashing   # Job name
#SBATCH --output=STDOUT_TXT # Name of stdout output file
#SBATCH --error=STDERR_TXT  # Name of stderr error file
#SBATCH --partition=small       # Partition (queue) name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --cpus-per-task=16     # Number of cores (threads)
#SBATCH --mem=16G               # Real memory (RAM) required (MB)
#SBATCH --time=72:00:00         # Run time (hh:mm:ss)
#SBATCH --account=PROJECT_NAME  # Project for billing

# Any other commands must follow the #SBATCH directives
srun python cluster_single_file_hashing.py \
        DATA_FOLDER_TO_CLUSTER/data_${SLURM_ARRAY_TASK_ID}.jsonl \
        --output_dir_parent OUTPUT_DIR \
        --text_col_name "text" \
        --n_clusters 16 \
        --batch_size 30000 \
        --n_process 12