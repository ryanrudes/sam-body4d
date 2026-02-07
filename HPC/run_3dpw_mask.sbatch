#!/bin/bash
#SBATCH --job-name=hmr
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --output=/users/ac1mga/data/output/hmr/logs/%x-%j.out
#SBATCH --error=/users/ac1mga/data/output/hmr/logs/%x-%j.err

# === Load Env ===
module purge
module load CUDA/11.8.0
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate body4d

# === Set Workspace ===
cd /users/ac1mga/projects/hmr/sam-body4d

# === Inference ===
CUDA_VISIBLE_DEVICES=0 python scripts/eval/run_3dpw_mask_kp.py \
    --data_dir /users/ac1mga/data/datasets/hmr/3DPW \
    --output_dir /users/ac1mga/predictions/hmr/3DPW-mask-text