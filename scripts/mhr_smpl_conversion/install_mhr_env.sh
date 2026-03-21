#!/usr/bin/env bash
set -eo pipefail

ENV_NAME="mhr2smpl"

echo "===> Checking conda installation..."

# Ensure conda is available
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda not found. Please install conda first."
    exit 1
fi

echo "===> Checking if environment '${ENV_NAME}' exists..."

# Create the environment only if it does not already exist
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists. Skipping creation."
else
    echo "===> Creating environment '${ENV_NAME}'"
    conda create -y -n "${ENV_NAME}" python=3.12
fi

echo "===> Installing pymomentum-gpu via conda"
conda run -n "${ENV_NAME}" conda install -y -c conda-forge pymomentum-gpu

echo "===> Upgrading pip"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

echo "===> Installing Python dependencies via pip"
conda run -n "${ENV_NAME}" python -m pip install \
    scikit-learn \
    smplx \
    mhr \
    tqdm \
    opencv-python \
    einops \
    colorlog

echo "===> Installing chumpy without build isolation"
conda run -n "${ENV_NAME}" python -m pip install chumpy --no-build-isolation

echo "===> Installing pytorch3d from pytorch3d-nightly without dependency resolution"
if conda run -n "${ENV_NAME}" conda install -y -c pytorch3d-nightly pytorch3d --no-deps; then
    echo "pytorch3d installed successfully."
else
    echo "Warning: pytorch3d installation failed. Continuing without it."
fi

echo "===> Installation complete"
echo "Environment '${ENV_NAME}' is ready."