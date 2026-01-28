#!/bin/bash
#SBATCH --job-name=yolo_posttrain
#SBATCH --output=logs/yolo_posttrain_%j.out
#SBATCH --error=logs/yolo_posttrain_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=w.j.a.teillers@student.rug.nl

# ============================================================================
# YOLO ImageNet Posttraining - Habrok SLURM Script
# ============================================================================

echo "Job started on $(date)"
echo "Node: $SLURM_NODELIST"

# 1. Load Modules
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

# 2. Set Scratch Paths (CRITICAL)
# This tells the script to look for the data we just downloaded in /scratch
export HF_HOME=/scratch/$USER/huggingface_cache
export HF_DATASETS_CACHE=/scratch/$USER/huggingface_datasets
export TMPDIR=/scratch/$USER/tmp
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TMPDIR

echo "HF_HOME: $HF_HOME"

# 3. Activate Environment
source /scratch/$USER/envs/yolo_env/bin/activate

# 4. Navigate to project
cd $SLURM_SUBMIT_DIR

# 5. Run Training
echo "Starting YOLO pascal Posttraining..."
python train_habrok.py

echo "Job finished on $(date)"
