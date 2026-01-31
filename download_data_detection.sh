#!/bin/bash
#SBATCH --job-name=download_imagenet
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=regular
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4


rm -rf /scratch/s4015843/envs/yolo_env

# Load modules
module purge
module load Python/3.11.5-GCCcore-13.2.0

# 2. Create the directory in scratch
mkdir -p /scratch/$USER/envs

# 3. Create the virtual environment
python3 -m venv /scratch/$USER/envs/yolo_env

# PATHS
export HF_HOME=/scratch/$USER/huggingface_cache
export HF_DATASETS_CACHE=/scratch/$USER/huggingface_datasets
mkdir -p $HF_HOME $HF_DATASETS_CACHE

# Activate Environment
source /scratch/$USER/envs/yolo_env/bin/activate

# 5. Install your requirements (ensure requirements.txt is in your current folder)
pip install --upgrade pip
pip install -r requirements.txt

echo "Job started on $(date)"
echo "Downloading to: $HF_HOME"


python download_data_detection.py --output_dir /scratch/$USER/voc_data

echo \"Job finished on \$(date)\"
