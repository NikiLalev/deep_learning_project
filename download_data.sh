#!/bin/bash
#SBATCH --job-name=download_imagenet
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=regular
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load modules
module purge
module load Python/3.9.6-GCCcore-11.2.0

# PATHS
export HF_HOME=/scratch/$USER/huggingface_cache
export HF_DATASETS_CACHE=/scratch/$USER/huggingface_datasets
mkdir -p $HF_HOME $HF_DATASETS_CACHE

# Activate Environment
source /scratch/$USER/envs/yolo_env/bin/activate

echo "Job started on $(date)"
echo "Downloading to: $HF_HOME"


python -c "
from huggingface_hub import snapshot_download
import os

print('Starting Download via snapshot_download...')

# Download the dataset files directly to the cache
# This uses low RAM because it streams files to disk
snapshot_download(
    repo_id='ILSVRC/imagenet-1k',
    repo_type='dataset',
    token=os.environ.get('HF_TOKEN'),
    cache_dir=os.environ['HF_HOME'],
    resume_download=True
)

print('Success! Files downloaded to cache.')
"

echo "Job finished on $(date)"
