# syntax=docker/dockerfile:1.6

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON=python3.11

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
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

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && rm -rf /root/.local/share

WORKDIR /workspace/mmml

COPY pyproject.toml uv.lock ./
COPY setup ./setup
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

RUN ln -s /workspace/mmml /root/mmml

ENV VIRTUAL_ENV=/workspace/mmml/.venv \
    PATH="/workspace/mmml/.venv/bin:${PATH}" \
    CHARMM_HOME=/workspace/mmml/setup/charmm \
    CHARMM_LIB_DIR=/workspace/mmml/setup/charmm \
    LD_LIBRARY_PATH=/workspace/mmml/setup/charmm

RUN printf 'export CHARMM_HOME=%s\nexport CHARMM_LIB_DIR=%s\n' "$CHARMM_HOME" "$CHARMM_LIB_DIR" > mmml/CHARMMSETUP

CMD ["bash"]
