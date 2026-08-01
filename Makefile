.PHONY: help install install-native install-full doctor install-gpu install-dev install-all install-all-offline-cuda13 install-all-offline-cuda12 install-jupyter-kernel clean test docker-build docker-run micromamba-create micromamba-create-gpu micromamba-create-gpu-cuda13 micromamba-create-full micromamba-update micromamba-remove docker-clean lfs-summary lfs-audit lfs-setup-symlinks lfs-remove-hooks install-hooks docs-build docs-strict docs-pdf docs-serve lint-dupes merge-check test-ci

help:
	@echo "MMML - Makefile Commands"
	@echo "========================"
	@echo ""
	@echo "Installation (from a fresh clone: make install-full):"
	@echo "  make install-full     - Python deps + native (libcharmm, packmol) + doctor"
	@echo "  make install          - Python deps only, via uv"
	@echo "  make install-native   - Native only: libcharmm + packmol (uv cannot build these)"
	@echo "  make doctor           - Is this machine ready? (JAX, CHARMM, Packmol)"
	@echo ""
	@echo "  make install-md-cpu   - Install CPU MD smoke extras (Vesin, mdanalysis)"
	@echo "  make install-gpu      - Install with uv (GPU/CUDA 13, default)"
	@echo "  make install-gpu-cuda12 - Install with uv (GPU/CUDA 12)"
	@echo "  make install-dev      - Install with development dependencies"
	@echo "  make install-all      - Install all optional dependencies"
	@echo "  make install-all-offline-cuda13 - Offline install, all extras, CUDA 13 (dedupes cuda12 plugin)"
	@echo "  make install-all-offline-cuda12 - Offline install, all extras, CUDA 12 (dedupes cuda13 plugin)"
	@echo "  make install-jupyter-kernel - Register this venv as a Jupyter kernel (name: mmml)"
	@echo ""
	@echo "Micromamba:"
	@echo "  make micromamba-create     - Create micromamba environment (CPU)"
	@echo "  make micromamba-create-gpu - Create micromamba environment (GPU/CUDA 12)"
	@echo "  make micromamba-create-gpu-cuda13 - Create micromamba environment (GPU/CUDA 13)"
	@echo "  make micromamba-create-full - Create micromamba environment (all features, CUDA 13)"
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
	@echo "  make test              - Run unit/integration tests (skip live PyCHARMM/GPU/MLpot)"
	@echo "  make test-all          - Run full pytest suite (needs mpirun for charmm_mpi live tests)"
	@echo "  make test-quick        - Run quick tests only"
	@echo "  make test-coverage     - Run tests with coverage report"
	@echo "  make test-ci           - Local stand-in for CI's build job (hides libcharmm)"
	@echo "  make lint-dupes        - Duplicate defs / conflict markers (bad-merge detector)"
	@echo "  make merge-check       - Pre-merge gate: lint-dupes + lint + imports + docs"
	@echo "  make deadcode          - Report dead/unused code (Ruff + Vulture)"
	@echo "  make deadcode-fix      - Auto-fix safe unused-code issues with Ruff"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs-build        - Build MkDocs HTML site"
	@echo "  make docs-strict       - Build MkDocs HTML site with strict checks"
	@echo "  make docs-pdf          - Build PDF docs at site/mmml-docs.pdf"
	@echo "  make docs-serve        - Serve docs locally with MkDocs"
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
	@echo "  make pyscf-check-gpu  - Check CuPy/CUDA compatibility"
	@echo ""
	@echo "Git LFS:"
	@echo "  make lfs-summary       - Show LFS file count and total size"
	@echo "  make lfs-audit         - Save LFS file list (sorted by size) to lfs_audit.txt"
	@echo "  make lfs-setup-symlinks - Replace duplicate grids_esp with symlinks to preclassified_data"
	@echo "  make lfs-remove-hooks  - Remove LFS hooks (silence warning when git-lfs not installed)"
	@echo ""
	@echo "Git hooks:"
	@echo "  make install-hooks     - Install pre-commit hook that auto-regenerates CI-checked docs"
	@echo ""

# ==============================================================================
# Installation with uv
# ==============================================================================

install:
	uv sync

# The native half: libcharmm (CMake/Fortran) + packmol. uv cannot build these --
# they are not Python packages -- so they get their own door.
# No env vars are needed afterwards: mmml auto-discovers setup/charmm.
install-native:
	./scripts/rebuild_charmm_mlpot.sh
	@echo ""
	@$(MAKE) --no-print-directory doctor

# Everything, from a fresh clone.
install-full: install install-native

# Is this machine ready to run MMML?
doctor:
	uv run mmml doctor

install-md-cpu:
	uv sync --extra md-cpu

install-gpu:
	uv pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html jax jaxlib jax-cuda13-plugin jax-cuda13-pjrt
	uv sync --extra gpu

install-gpu-cuda12:
	uv pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "jax[cuda12]" "jaxlib[cuda12]"
	uv sync --extra gpu-cuda12

install-dev:
	uv sync --extra dev

install-all:
	uv sync --extra all

# Offline installs: use only the local uv cache / already-downloaded wheels
# (no network access), then strip any leftover jax-cuda12/cuda13 plugin from
# the *other* CUDA major version. uv sync alone won't remove those if they
# were pip-installed manually in a previous session — mixing both plugins
# in one venv makes JAX fail with CUDA_ERROR_UNKNOWN even though nvidia-smi
# and cupy see the GPU fine.
install-all-offline-cuda13:
	uv sync --offline --extra all-cuda13
	.venv/bin/pip uninstall -y jax-cuda12-pjrt jax-cuda12-plugin 2>/dev/null || true

install-all-offline-cuda12:
	uv sync --offline --extra all-cuda12
	.venv/bin/pip uninstall -y jax-cuda13-pjrt jax-cuda13-plugin 2>/dev/null || true

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
	uv run pytest tests -m "not pycharmm and not gpu and not mlpot"

test-all:
	uv run pytest tests

test-quick:
	uv run pytest -q tests/functionality/mmml_tests/test_mmml_calc.py::test_ev2kcalmol_constant

test-coverage:
	uv run pytest --cov=mmml --cov-report=html --cov-report=term tests/

# The honest local verdict on a test run. pytest's exit code cannot be trusted
# here for two reasons: it is 0 when every selected test skips, and it is 0 once
# libcharmm has been loaded (CHARMM's Fortran STOP replaces the exit status at
# teardown, and can kill the session mid-run). The JUnit report is the only
# record that survives both, so `|| true` below is deliberate -- the gate that
# follows is what decides pass/fail.
# Run: make test-shape
test-shape:
	mkdir -p .ci-reports
	uv run pytest tests/ -q -p no:cacheprovider --junitxml=.ci-reports/junit-local.xml || true
	uv run python scripts/ci/check_test_report.py .ci-reports/junit-local.xml \
	  --label "local suite" --min-passed 3000 --max-skipped-frac 0.25

test-data:
	@if [ -z "$(MMML_DATA)" ] || [ -z "$(MMML_CKPT)" ]; then \
		echo "Error: MMML_DATA and MMML_CKPT must be set"; \
		exit 1; \
	fi
	uv run pytest tests/functionality/mmml_tests/test_mmml_calc.py::test_ml_energy_matches_reference_when_data_available

# ==============================================================================
# Code quality
# ==============================================================================

lint:
	uv run ruff check mmml/ scripts/ setup/charmm/tool/pycharmm/pycharmm/

# Duplicated definitions / dead imports / syntax breakage, repo-wide.
# A bad merge that concatenates two versions of a file shows up here as F811
# "Redefinition of unused X" -- which is exactly how three duplicated blocks
# (mm_bonded.py, linear_distance.py, umbrella/energy.py) reached main after the
# PR #140 merge. Two of them were not cosmetic: a duplicate @register_term made
# `import mmml.md.energy.terms` raise, and a duplicated block was the only home
# of a helper that was still being called. Unlike `make lint` this also covers
# shipped code (`make lint` covers the same dirs, so this is the same verdict)
# and, advisory-only, the dirs lint never sees. tests/ examples/ workflows/ carry
# pre-existing redefinitions, so they are reported but do not fail the gate --
# a gate that is red on day one gets ignored. Conflict markers are always fatal.
# Run: make lint-dupes
lint-dupes:
	@uv run ruff check --select F811 mmml/ scripts/ setup/charmm/tool/pycharmm/pycharmm/
	@if git grep -nE '^(<<<<<<< |>>>>>>> )' -- '*.py' '*.sh' '*.yaml' '*.yml' '*.toml' '*.md'; then \
	  echo "lint-dupes: unresolved conflict markers above" >&2; exit 1; \
	fi
	@echo "--- advisory: redefinitions outside lint scope (not fatal) ---"
	@uv run ruff check --select F811 --quiet tests/ examples/ workflows/ || true
	@echo "lint-dupes: shipped code has no duplicate definitions; no conflict markers"

# Local stand-in for the CI `build` job on a machine that HAS libcharmm.
# CI installs no libcharmm, so every live-PyCHARMM test self-skips there. Locally
# they run instead and abort the session inside test_charmm_mpi.py on a native
# CHARMM exit, truncating the run long before the real failures. Hiding the
# library reproduces CI's skip behaviour and gives an honest signal.
#
# MMML_DISABLE_CHARMM is the *only* reliable way to do that: pointing
# CHARMM_LIB_DIR at /nonexistent used to leave `resolve_charmm_paths()` still
# returning the real setup/charmm tree (a lib-less explicit override is treated
# as stale and discarded), so this target only half-hid the build and did not
# reproduce CI. See charmm_paths.charmm_disabled.
# Run: make test-ci
test-ci:
	MMML_DISABLE_CHARMM=1 \
	  uv run pytest tests/ -q -p no:cacheprovider

# Pre-merge gate: everything CI checks first, in the order it fails.
# Run: make merge-check
merge-check: lint-dupes lint
	uv run python -c "import mmml.md.energy.terms; print('energy terms import ok')"
	uv run python scripts/generate_cli_docs.py --check
	uv run python scripts/generate_package_architecture.py --check
	@echo "merge-check: OK -- now run 'make test-ci' for the full suite"

format:
	uv run ruff format mmml/ scripts/

type-check:
	uv run mypy mmml/

deadcode:
	uv run ruff check --select F401,F841,F541 mmml/ scripts/ setup/charmm/tool/pycharmm/pycharmm/
	uvx vulture mmml scripts setup/charmm/tool/pycharmm/pycharmm --min-confidence 80

deadcode-fix:
	uv run ruff check --fix --select F401,F841,F541 mmml/ scripts/

# ==============================================================================
# Documentation
# ==============================================================================

docs-build:
	uv run python scripts/generate_cli_docs.py
	uv run python scripts/generate_docs_figures.py
	uv run python scripts/generate_crystal_lit_compare.py
	uv run python scripts/plot_mlpot_settings.py
	uv run --extra dev mkdocs build

docs-strict:
	uv run python scripts/generate_cli_docs.py
	uv run python scripts/generate_docs_figures.py
	uv run python scripts/generate_crystal_lit_compare.py
	uv run python scripts/plot_mlpot_settings.py
	uv run --extra dev mkdocs build --strict

docs-pdf:
	uv run --extra dev --with reportlab python scripts/build_docs_pdf.py

docs-serve:
	uv run --extra dev mkdocs serve

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

# Register this venv as a Jupyter kernel (e.g. for a cluster-wide JupyterHub)
# so it shows up as "mmml" in the kernel picker without activating the venv.
install-jupyter-kernel:
	uv sync --extra notebooks
	.venv/bin/python -m ipykernel install --user --name mmml --display-name "mmml (.venv)"
	@echo "Registered kernel 'mmml (.venv)' — refresh Jupyter's kernel list to see it."

# ==============================================================================
# CHARMM setup
# ==============================================================================

charmm-setup:
	bash setup/install.sh
	@echo "CHARMM setup complete"
	@echo "Build libcharmm: bash scripts/rebuild_charmm_mlpot.sh (paths auto-discovered from setup/charmm)"

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

# Install the repo's git hooks (currently a pre-commit that auto-regenerates the
# CI-checked generated docs so commits never carry stale copies).
# Run: make install-hooks
install-hooks:
	@hooks_dir="$$(git rev-parse --git-path hooks)"; \
	mkdir -p "$$hooks_dir"; \
	ln -sf "$$(pwd)/scripts/git-hooks/pre-commit" "$$hooks_dir/pre-commit"; \
	chmod +x scripts/git-hooks/pre-commit; \
	echo "Installed pre-commit hook -> $$hooks_dir/pre-commit"

# ==============================================================================
# Data utilities
# ==============================================================================

split-8-1-1:
	@echo "split-8-1-1 is deprecated: scripts/split_npz_8_1_1.py was removed."
	@echo "Use the dataset split commands in mmml.cli.misc.split_dataset instead."

# ==============================================================================
# PySCF/GPU4PySCF examples
# ==============================================================================

PYSCF_MOL ?= "O 0 0 0; H 0.96 0 0; H -0.24 0.93 0"

pyscf-example:
	$(PY) examples/pyscf4gpu/water_energy.py

pyscf-dft:
	$(PY) -m mmml.cli pyscf-dft --mol $(PYSCF_MOL) --energy --output pyscf_water_output

# Diagnose CuPy/CUDA compatibility (run on GPU node if using pyscf-dft)
pyscf-check-gpu:
	$(PY) scripts/check_cupy_gpu.py

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
	@echo "Use: uv run mmml physnet-train --help"

physnet-train-adv:
	@echo "Use: uv run mmml physnet-train --config your.yaml"

physnet-train-chg:
	@echo "Use: uv run mmml physnet-train with --dipole-weight / --charges-weight"
