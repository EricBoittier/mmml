# syntax=docker/dockerfile:1.6
# Multi-stage Dockerfile for MMML
# Supports both CPU and GPU variants

ARG BASE_IMAGE=python:3.11-slim
ARG CUDA_VERSION=12.1.1

# ============================================================================
# Stage 1: Base image with system dependencies
# ============================================================================
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    cmake \
    ninja-build \
    git \
    curl \
    wget \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libxc-dev \
    libhdf5-dev \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    liblzma-dev \
    libsqlite3-dev \
    libreadline-dev \
    zlib1g-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    rm -rf /root/.local/share

ENV UV_LINK_MODE=copy \
    UV_PYTHON=python3.11

WORKDIR /workspace/mmml

# ============================================================================
# Stage 2: Dependencies installation
# ============================================================================
FROM base AS dependencies

# Copy only dependency files first for better caching
COPY pyproject.toml uv.lock ./
COPY setup ./setup

# Install dependencies without the project itself
RUN uv sync --frozen --no-install-project

# ============================================================================
# Stage 3: Final runtime image (CPU)
# ============================================================================
FROM dependencies AS runtime-cpu

# Copy the entire project
COPY . .

# Install the project itself
RUN uv sync --frozen

# Set up environment variables
ENV VIRTUAL_ENV=/workspace/mmml/.venv \
    PATH="/workspace/mmml/.venv/bin:${PATH}" \
    CHARMM_HOME=/workspace/mmml/setup/charmm \
    CHARMM_LIB_DIR=/workspace/mmml/setup/charmm \
    LD_LIBRARY_PATH=/workspace/mmml/setup/charmm

# Create CHARMM setup file
RUN printf 'export CHARMM_HOME=%s\nexport CHARMM_LIB_DIR=%s\n' \
    "$CHARMM_HOME" "$CHARMM_LIB_DIR" > mmml/CHARMMSETUP

# Create a convenient symlink
RUN ln -s /workspace/mmml /root/mmml

CMD ["bash"]

# ============================================================================
# Stage 4: GPU variant (CUDA)
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 AS runtime-gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=python3.11

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    gfortran \
    cmake \
    ninja-build \
    git \
    curl \
    wget \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libxc-dev \
    libhdf5-dev \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    liblzma-dev \
    libsqlite3-dev \
    libreadline-dev \
    zlib1g-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    rm -rf /root/.local/share

WORKDIR /workspace/mmml

# Copy dependency files
COPY pyproject.toml uv.lock ./
COPY setup ./setup

# Install dependencies with GPU extras
RUN uv sync --frozen --no-install-project --extra gpu

# Copy the entire project
COPY . .

# Install the project with GPU extras
RUN uv sync --frozen --extra gpu

# Set up environment variables
ENV VIRTUAL_ENV=/workspace/mmml/.venv \
    PATH="/workspace/mmml/.venv/bin:${PATH}" \
    CHARMM_HOME=/workspace/mmml/setup/charmm \
    CHARMM_LIB_DIR=/workspace/mmml/setup/charmm \
    LD_LIBRARY_PATH=/workspace/mmml/setup/charmm \
    CUDA_HOME=/usr/local/cuda \
    XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Create CHARMM setup file
RUN printf 'export CHARMM_HOME=%s\nexport CHARMM_LIB_DIR=%s\n' \
    "$CHARMM_HOME" "$CHARMM_LIB_DIR" > mmml/CHARMMSETUP

# Create a convenient symlink
RUN ln -s /workspace/mmml /root/mmml

CMD ["bash"]
