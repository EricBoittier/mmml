.PHONY: help install install-gpu install-dev install-all clean test docker-build docker-run micromamba-create micromamba-create-gpu micromamba-create-gpu-cuda13 micromamba-create-full micromamba-update micromamba-remove docker-clean lfs-summary lfs-audit lfs-setup-symlinks

help:
	@echo "MMML - Makefile Commands"
	@echo "========================"
	@echo ""
	@echo "Installation:"
	@echo "  make install          - Install with uv (CPU only)"
	@echo "  make install-gpu      - Install with uv (GPU/CUDA 12)"
	@echo "  make install-gpu-cuda13 - Install with uv (GPU/CUDA 13)"
	@echo "  make install-dev      - Install with development dependencies"
	@echo "  make install-all      - Install all optional dependencies"
	@echo ""
	@echo "Micromamba:"
	@echo "  make micromamba-create     - Create micromamba environment (CPU)"
	@echo "  make micromamba-create-gpu - Create micromamba environment (GPU/CUDA 12)"
	@echo "  make micromamba-create-gpu-cuda13 - Create micromamba environment (GPU/CUDA 13)"
	@echo "  make micromamba-create-full - Create micromamba environment (all features)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build-cpu  - Build CPU Docker image"
	@echo "  make docker-build-gpu  - Build GPU Docker image (CUDA 12)"
	@echo "  make docker-build-gpu-cuda13 - Build GPU Docker image (CUDA 13 + Python 3.14)"
	@echo "  make docker-run-cpu    - Run CPU Docker container"
	@echo "  make docker-run-gpu    - Run GPU Docker container (CUDA 12)"
	@echo "  make docker-run-gpu-cuda13 - Run GPU Docker container (CUDA 13 + Python 3.14)"
	@echo "  make docker-jupyter    - Start Jupyter Lab in Docker"
	@echo "  make docker-clean      - Remove all Docker containers and images"
	@echo ""
	@echo "Testing:"
	@echo "  make test              - Run all tests"
	@echo "  make test-quick        - Run quick tests only"
	@echo "  make test-coverage     - Run tests with coverage report"
	@echo ""
	@echo "Training (PhysNetJAX):"
	@echo "  make physnet-train         TRAIN=train.npz [VALID=valid.npz] [NATOMS=60] [BATCH=32] [EPOCHS=100] [LR=0.001] [NAME=run] [CHARGES=false]"
	@echo "  make physnet-train-adv     TRAIN=train.npz [VALID=valid.npz] [NATOMS=60] [BATCH=32] [EPOCHS=100] [LR=0.001] [NAME=run] BATCH_SHAPE=512 NBLEN=16384"
	@echo "  make physnet-train-chg     TRAIN=train.npz [VALID=valid.npz] CHARGES=true (adds dipole/charges loss weights)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             - Remove build artifacts and caches"
	@echo "  make clean-all         - Remove everything including venv"
	@echo ""
	@echo "Data Utilities:"
	@echo "  make split-8-1-1       - Split an NPZ into 8:1:1 train/valid/test"
	@echo ""
	@echo "PySCF/GPU4PySCF (requires GPU + gpu4pyscf):"
	@echo "  make pyscf-example    - Run water DFT example (energy + gradient)"
	@echo "  make pyscf-dft        - Run pyscf-dft CLI on water (energy only)"
	@echo ""
	@echo "Git LFS:"
	@echo "  make lfs-summary       - Show LFS file count and total size"
	@echo "  make lfs-audit         - Save LFS file list (sorted by size) to lfs_audit.txt"
	@echo "  make lfs-setup-symlinks - Replace duplicate grids_esp with symlinks to preclassified_data"
	@echo "  make lfs-remove-hooks  - Remove LFS hooks (silence warning when git-lfs not installed)"
	@echo ""

# ==============================================================================
# Installation with uv
# ==============================================================================

install:
	uv sync

install-gpu:
	uv pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "jax[cuda12]" "jaxlib[cuda12]"
	uv sync --extra gpu

install-gpu-cuda13:
	uv pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html jax jaxlib jax-cuda13-plugin jax-cuda13-pjrt
	uv sync --extra gpu-cuda13

install-dev:
	uv sync --extra dev

install-all:
	uv sync --extra all

# ==============================================================================
# Micromamba environments
# ==============================================================================

micromamba-create:
	micromamba env create -f setup/environment.yml

micromamba-create-gpu:
	micromamba env create -f setup/environment-gpu.yml

micromamba-create-gpu-cuda13:
	micromamba env create -f setup/environment-gpu-cuda13.yml

micromamba-create-full:
	micromamba env create -f setup/environment-full.yml

micromamba-update:
	micromamba env update -f setup/environment.yml --prune

micromamba-remove:
	micromamba env remove -n mmml -y

# ==============================================================================
# Docker
# ==============================================================================

docker-build-cpu:
	docker build --target runtime-cpu -t mmml:cpu .

docker-build-gpu:
	docker build --target runtime-gpu -t mmml:gpu .

docker-build-gpu-cuda13:
	docker build --target runtime-gpu-cuda13 -t mmml:gpu-cuda13 .

docker-run-cpu:
	docker run -it --rm -v $$(pwd):/workspace/mmml mmml:cpu

docker-run-gpu:
	docker run -it --rm --gpus all -v $$(pwd):/workspace/mmml mmml:gpu

docker-run-gpu-cuda13:
	docker run -it --rm --gpus all -v $$(pwd):/workspace/mmml mmml:gpu-cuda13

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

docker-jupyter:
	docker-compose up -d mmml-jupyter
	@echo "Jupyter Lab is running at http://localhost:8888"

docker-clean:
	docker-compose down -v
	docker rmi mmml:cpu mmml:gpu mmml:gpu-cuda13 mmml:jupyter 2>/dev/null || true

# ==============================================================================
# Testing
# ==============================================================================

test:
	cd tests && uv run pytest .

test-quick:
	uv run pytest -q tests/functionality/mmml/test_mmml_calc.py::test_ev2kcalmol_constant

test-coverage:
	uv run pytest --cov=mmml --cov-report=html --cov-report=term tests/

test-data:
	@if [ -z "$(MMML_DATA)" ] || [ -z "$(MMML_CKPT)" ]; then \
		echo "Error: MMML_DATA and MMML_CKPT must be set"; \
		exit 1; \
	fi
	uv run pytest tests/functionality/mmml/test_mmml_calc.py::test_ml_energy_matches_reference_when_data_available

# ==============================================================================
# Code quality
# ==============================================================================

lint:
	uv run ruff check mmml/

format:
	uv run ruff format mmml/

type-check:
	uv run mypy mmml/

# ==============================================================================
# Documentation
# ==============================================================================

docs-build:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# ==============================================================================
# Cleanup
# ==============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean
	rm -rf .venv/
	rm -rf mmml.egg-info/

# ==============================================================================
# Development
# ==============================================================================

dev-setup: install-dev
	@echo "Development environment ready!"
	@echo "Activate with: source .venv/bin/activate"

notebook:
	uv run jupyter lab

# ==============================================================================
# CHARMM setup
# ==============================================================================

charmm-setup:
	bash setup/install.sh
	@echo "CHARMM setup complete"
	@echo "Source the environment: source CHARMMSETUP"

# ==============================================================================
# Utilities
# ==============================================================================

show-deps:
	uv tree

show-outdated:
	uv pip list --outdated

freeze:
	uv pip freeze > requirements-frozen.txt

upgrade:
	uv sync --upgrade

# ==============================================================================
# Git LFS
# ==============================================================================

lfs-summary:
	@echo "LFS files: $$(git lfs ls-files 2>/dev/null | wc -l)"
	@git lfs ls-files -s 2>/dev/null | grep -oE '\([0-9.]+ (KB|MB|GB)\)' | \
	  awk -F'[ ()]' '{u=$$3; v=$$2; \
	    if(u=="KB")t+=v*1024; else if(u=="MB")t+=v*1024*1024; else if(u=="GB")t+=v*1024*1024*1024} \
	    END{printf "Total: %.1f MB\n", t/1024/1024}'

lfs-audit:
	git lfs ls-files -s 2>/dev/null | grep -E '\([0-9.]+ (KB|MB|GB)\)' | sort -t'(' -k2 -V -r > lfs_audit.txt
	@echo "Saved to lfs_audit.txt ($$(wc -l < lfs_audit.txt) files)"

lfs-setup-symlinks:
	bash scripts/setup_grid_symlinks.sh

# Remove LFS hooks to silence "git-lfs not found" warning when LFS is not installed.
# Run: make lfs-remove-hooks
lfs-remove-hooks:
	@for h in post-checkout post-commit pre-push post-merge; do \
	  if [ -f .git/hooks/$$h ]; then \
	    rm .git/hooks/$$h && echo "Removed .git/hooks/$$h"; \
	  fi; \
	done
	@echo "LFS hooks removed. git pull/checkout will no longer warn about missing git-lfs."

# ==============================================================================
# Data utilities
# ==============================================================================

INPUT ?=
OUTDIR ?=
split-8-1-1:
	@if [ -z "$(INPUT)" ]; then echo "Error: set INPUT=<data.npz>"; exit 1; fi
	uv run python scripts/split_npz_8_1_1.py $(INPUT) $(if $(OUTDIR),--out-dir $(OUTDIR),)

# ==============================================================================
# PySCF/GPU4PySCF examples
# ==============================================================================

PYSCF_MOL ?= "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0"

pyscf-example:
	$(PY) examples/pyscf4gpu/water_energy.py

pyscf-dft:
	$(PY) -m mmml.cli pyscf-dft --mol $(PYSCF_MOL) --energy --output pyscf_water_output.pkl

# ==============================================================================
# Training helpers (PhysNetJAX via Hydra)
# ==============================================================================

PY ?= uv run python

# Common variables with defaults
TRAIN ?=
VALID ?=
NATOMS ?= 60
BATCH ?= 32
EPOCHS ?= 100
LR ?= 0.001
NAME ?= physnet_run
SEED ?= 42
CHARGES ?= false

# Advanced batching defaults
BATCH_SHAPE ?= 512
NBLEN ?= 16384

physnet-train:
	@if [ -z "$(TRAIN)" ]; then echo "Error: set TRAIN=<train.npz>"; exit 1; fi
	$(PY) scripts/physnet_hydra_train.py \
	  data.train_file=$(TRAIN) \
	  $(if $(VALID),data.valid_file=$(VALID),) \
	  model.natoms=$(NATOMS) \
	  train.batch_size=$(BATCH) \
	  train.max_epochs=$(EPOCHS) \
	  train.learning_rate=$(LR) \
	  logging.name=$(NAME) \
	  train.seed=$(SEED) \
	  model.charges=$(CHARGES)

physnet-train-adv:
	@if [ -z "$(TRAIN)" ]; then echo "Error: set TRAIN=<train.npz>"; exit 1; fi
	$(PY) scripts/physnet_hydra_train.py \
	  data.train_file=$(TRAIN) \
	  $(if $(VALID),data.valid_file=$(VALID),) \
	  model.natoms=$(NATOMS) \
	  train.batch_size=$(BATCH) \
	  train.max_epochs=$(EPOCHS) \
	  train.learning_rate=$(LR) \
	  logging.name=$(NAME) \
	  train.seed=$(SEED) \
	  model.charges=$(CHARGES) \
	  batching.method=advanced batching.batch_shape=$(BATCH_SHAPE) batching.batch_nbl_len=$(NBLEN)

physnet-train-chg:
	@if [ -z "$(TRAIN)" ]; then echo "Error: set TRAIN=<train.npz>"; exit 1; fi
	$(PY) scripts/physnet_hydra_train.py \
	  data.train_file=$(TRAIN) \
	  $(if $(VALID),data.valid_file=$(VALID),) \
	  model.natoms=$(NATOMS) \
	  train.batch_size=$(BATCH) \
	  train.max_epochs=$(EPOCHS) \
	  train.learning_rate=$(LR) \
	  logging.name=$(NAME) \
	  train.seed=$(SEED) \
	  model.charges=true \
	  loss.dipole_weight=25.0 loss.charges_weight=10.0
