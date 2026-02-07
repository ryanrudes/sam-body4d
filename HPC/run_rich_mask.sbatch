#!/bin/bash
#SBATCH --job-name=hmr
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-36:00:00
#SBATCH --output=/users/ac1mga/data/output/hmr/logs/%x-%j.log
#SBATCH --error=/users/ac1mga/data/output/hmr/logs/%x-%j.log

# === Load Env ===
module purge
module load CUDA/11.8.0
module load Anaconda3/2024.02-1
module load GCCcore/12.3.0
module load binutils/2.40-GCCcore-12.3.0
eval "$(conda shell.bash hook)"
conda activate body4d

# === Set Workspace ===
cd /users/ac1mga/projects/sam-body4d

# === Inference ===
CUDA_VISIBLE_DEVICES=0 python scripts/eval/run_rich_mask_kp.py \
    --data_dir /users/ac1mga/data/datasets/hmr/RICH \
    --output_dir /users/ac1mga/data/predictions/hmr/RICH-mask-text