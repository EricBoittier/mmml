#!/bin/bash
# Setup script for JAX with CUDA 13 / RTX 5090 support

ENV_NAME="jax_cuda13"

echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.11 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing JAX with CUDA 12 (closest available, may work with CUDA 13)..."
# CUDA 13 support may require nightly builds
pip install --upgrade pip

# Try official CUDA 12 build first (often works with CUDA 13 due to backward compat)
pip install --upgrade "jax[cuda12]"

# Install other dependencies
pip install numpy pandas matplotlib optax flax e3x ase

echo ""
echo "Environment setup complete!"
echo "Activate with: conda activate $ENV_NAME"
echo ""
echo "If CUDA graph issues persist, try JAX nightly:"
echo "  pip install --upgrade jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html"
