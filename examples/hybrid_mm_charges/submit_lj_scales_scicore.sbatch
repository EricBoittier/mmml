#!/bin/bash
# Trainable CGenFF LJ σ/ε scales: prepare → train → deploy, on SciCORE.
#
# Closes the "never run end-to-end" gap for issue #133: produces the first real
# hybrid_mm.json (trained scale vectors) and then exercises the only MD path
# that can consume it (jax_mic switched MM, NOT periodic_external).
#
# Submit from the repo root on a SciCORE login node:
#
#     mkdir -p artifacts/lj_scales_dcm          # Slurm will not create this
#     sbatch examples/hybrid_mm_charges/submit_lj_scales_scicore.sbatch
#
# Overridable via the environment (all have defaults):
#     MMML_REPO        repo root                (default: $HOME/mmml)
#     MMML_LJ_DATASET  input NPZ                (default: examples/dcm_mp2_psf_order.npz)
#     MMML_LJ_EPOCHS   training epochs          (default: 500)
#     MMML_LJ_NTRAIN / MMML_LJ_NVALID           (default: 8000 / 1000)
#     MMML_LJ_SKIP_MD  set to 1 to stop after training
#
#SBATCH --job-name=lj-scales-dcm
#SBATCH --partition=rtx4090
#SBATCH --qos=rtx4090-6hours
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=artifacts/lj_scales_dcm/slurm-%j.out
#SBATCH --error=artifacts/lj_scales_dcm/slurm-%j.err

set -uo pipefail

REPO="${MMML_REPO:-$HOME/mmml}"
cd "$REPO" || { echo "ERROR: no repo at $REPO" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Module stack. This MUST match what the local PyCHARMM/libcharmm.so was built
# against — a mismatched GCCcore gives an ABI/symbol failure on `import pycharmm`.
#
# scripts/scicore_env.sh still defaults to CMake/3.27.6-GCCcore-13.2.0 with
# foss/2023b (GCC 13.2.0), which is NOT what the current SciCORE PyCHARMM was
# compiled with. Override here rather than editing the shared prolog.
#
# NB: the prolog runs both `module load` calls with `|| true`, so a wrong or
# missing module name is silent. The pycharmm assertion below is what actually
# catches it.
# ---------------------------------------------------------------------------
export MMML_SCICORE_CMAKE="${MMML_SCICORE_CMAKE:-CMake/3.31.8-GCCcore-14.3.0}"
# Toolchain must be the GCC 14.3.0-consistent one; foss/2023b would drag in
# GCC 13.2.0 and fight the CMake module above. Override if the name differs.
export MMML_SCICORE_TOOLCHAIN="${MMML_SCICORE_TOOLCHAIN:-foss/2025a}"

# SciCORE prolog: initializes Lmod in a non-login shell, loads the toolchain +
# CMake selected above, and sets the MMML_NO_MPI_* guards that keep the generic
# launcher's fallbacks disabled (see docs/scicore.md).
#
# Sourced *before* `set -e` on purpose: scripts/scicore_env.sh documents that it
# contains tolerated non-zero commands and that caller errexit "would abort it on
# the first" of them. It manages `set -u` itself (relax, then restore).
source scripts/scicore_env.sh
source .venv/bin/activate

echo "--- module stack ---"
module list 2>&1 | sed 's/^/  /' || true

# Everything past the prolog is fail-fast.
set -e

RUN_DIR="$REPO/artifacts/lj_scales_dcm"
ENRICHED="$RUN_DIR/dcm_mp2_cgenff.npz"
CKPT_DIR="$RUN_DIR/ckpts"
TAG="hybrid_mm_fixed_lj_scales"
mkdir -p "$RUN_DIR" "$CKPT_DIR"

DATASET="${MMML_LJ_DATASET:-$REPO/examples/dcm_mp2_psf_order.npz}"
EPOCHS="${MMML_LJ_EPOCHS:-500}"
NTRAIN="${MMML_LJ_NTRAIN:-8000}"
NVALID="${MMML_LJ_NVALID:-1000}"

echo "=== lj-scales-dcm ==="
echo "  host      : $(hostname)"
echo "  repo      : $REPO"
echo "  dataset   : $DATASET"
echo "  epochs    : $EPOCHS   (n_train=$NTRAIN n_valid=$NVALID)"
echo "  run dir   : $RUN_DIR"

# ---------------------------------------------------------------------------
# Preflight. These inputs live on the workstation where this script was written
# (/home/boittier/mmml/examples), NOT necessarily on the SciCORE filesystem —
# fail here with an actionable message rather than 20 minutes into the queue.
# ---------------------------------------------------------------------------
if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: dataset not found: $DATASET" >&2
  echo "       Copy an MP2 NPZ across, or point MMML_LJ_DATASET at one." >&2
  echo "       It must be PSF-ordered — dcm_mp2_psf_order.npz has Z=[6,17,17,1,1]" >&2
  echo "       (C,Cl,Cl,H,H); the non-PSF DCM file is [6,1,1,17,17] and will" >&2
  echo "       mis-assign CGenFF types." >&2
  exit 2
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import jax; print('JAX devices:', jax.devices())"

# Fail here, not 3 hours in at Stage 3. The prolog's `module load ... || true`
# means a wrong GCCcore is silent until libcharmm.so fails to dlopen, and the
# MD leg is the last thing this job does — so import PyCHARMM up front.
if ! python -c "import pycharmm" >/dev/null 2>&1; then
  echo "ERROR: 'import pycharmm' failed — the loaded module stack does not match" >&2
  echo "       the one libcharmm.so was built against." >&2
  echo "       toolchain=$MMML_SCICORE_TOOLCHAIN cmake=$MMML_SCICORE_CMAKE" >&2
  echo "       Fix by exporting the correct names before sbatch, e.g.:" >&2
  echo "         export MMML_SCICORE_TOOLCHAIN=<foss/YYYYx>" >&2
  echo "         export MMML_SCICORE_CMAKE=CMake/3.31.8-GCCcore-14.3.0" >&2
  echo "       Diagnose with:  ldd \$(python -c 'import pycharmm,os;print(os.path.dirname(pycharmm.__file__))')/../*.so | grep 'not found'" >&2
  echo "       (Set MMML_LJ_SKIP_MD=1 to run prep+train only, which needs no CHARMM.)" >&2
  exit 5
fi
echo "PyCHARMM import OK"

# ---------------------------------------------------------------------------
# Stages. These delegate to the numbered ladder in examples/lj_scales/ rather
# than re-implementing prep/train/MD here — one definition of each step, so a
# fix in the ladder reaches the cluster job automatically.
#
#   01  prepare-mm-dataset + schema assertion
#   02  PSF-ordering + type histogram
#   05  physnet-train with learn_mm_lj_scales
#   06  report which scales moved
#   07  md-system liquid_nvt (jax_mic)
#
# The ladder's _env.sh reads the same variables exported above.
# ---------------------------------------------------------------------------
export LJ_DATASET="$DATASET"
export ARTIFACTS_DIR="$RUN_DIR"
export LJ_ENRICHED="$ENRICHED"
export LJ_CKPT_DIR="$CKPT_DIR"
export LJ_TAG="$TAG"
export LJ_EPOCHS="$EPOCHS"
export LJ_NTRAIN="$NTRAIN"
export LJ_NVALID="$NVALID"
export LJ_DEVICE="${LJ_DEVICE:-gpu}"
export LJ_WORKERS="${SLURM_CPUS_PER_TASK:-4}"

bash examples/lj_scales/01_prepare_dataset.sh
uv run python examples/lj_scales/02_inspect_dataset.py
bash examples/lj_scales/05_train.sh
uv run python examples/lj_scales/06_inspect_scales.py

if [[ "${MMML_LJ_SKIP_MD:-0}" == "1" ]]; then
  echo "MMML_LJ_SKIP_MD=1 — stopping after training."
  exit 0
fi

bash examples/lj_scales/07_deploy_md.sh

echo "=== done: $RUN_DIR ==="
