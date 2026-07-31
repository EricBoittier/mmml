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
# Stage 1 — CGenFF assignment. Adds cgenff_type_idx / cgenff_charge / mol_id,
# without which hybrid_mm training cannot run at all.
# ---------------------------------------------------------------------------
if [[ -f "$ENRICHED" ]]; then
  echo "--- Stage 1: reusing existing $ENRICHED"
else
  echo "--- Stage 1: prepare-mm-dataset"
  mmml prepare-mm-dataset \
    --data "$DATASET" \
    --output "$ENRICHED" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}"
fi

python - "$ENRICHED" <<'PY'
import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
need = {"cgenff_type_idx", "cgenff_charge", "mol_id"}
missing = need - set(d.files)
if missing:
    raise SystemExit(f"ERROR: enriched NPZ missing {sorted(missing)}")
print(f"Stage 1 OK: {len(d['E'])} frames, keys={sorted(d.files)}")
PY

# ---------------------------------------------------------------------------
# Stage 2 — train ML + learnable LJ scales under MIC.
# lr_solver must stay 'mic': ewald / nvalchemiops_pme force LJ off, so there
# would be no LJ term to differentiate. See docs/hybrid-mm-lj-scales.md
# § "Why you train LJ without Ewald".
# ---------------------------------------------------------------------------
echo "--- Stage 2: physnet-train (learn_mm_lj_scales)"
mmml physnet-train \
  --config examples/hybrid_mm_charges/train_fixed_lj_scales.yaml \
  --data "$ENRICHED" \
  --ckpt-dir "$CKPT_DIR" \
  --tag "$TAG" \
  --n-train "$NTRAIN" \
  --n-valid "$NVALID" \
  --num-epochs "$EPOCHS"

SIDECAR="$(find "$CKPT_DIR" -name hybrid_mm.json -newer "$ENRICHED" 2>/dev/null | head -1)"
if [[ -z "$SIDECAR" ]]; then
  echo "ERROR: training produced no hybrid_mm.json under $CKPT_DIR" >&2
  echo "       learn_mm_lj_scales may have been forced off — check the log for" >&2
  echo "       'Learnable MM LJ scales enabled' and confirm lr_solver=mic." >&2
  exit 3
fi
echo "Stage 2 OK: $SIDECAR"
python -c "
import json,sys
d=json.load(open('$SIDECAR'))
s,e=d['mm_lj_sigma_scale'],d['mm_lj_epsilon_scale']
print(f'  learned {len(s)} type scales')
print(f'  sigma   min={min(s):.4f} max={max(s):.4f}')
print(f'  epsilon min={min(e):.4f} max={max(e):.4f}')
moved=sum(1 for x in s+e if abs(x-1.0)>1e-3)
print(f'  {moved} of {len(s)+len(e)} scales moved off 1.0')
"

if [[ "${MMML_LJ_SKIP_MD:-0}" == "1" ]]; then
  echo "MMML_LJ_SKIP_MD=1 — stopping after training."
  exit 0
fi

# ---------------------------------------------------------------------------
# Stage 3 — deploy in a condensed-phase run. jax_mic is mandatory here: the
# switched-MM pair loop is what consumes ep_scale/sig_scale, and MLpot now
# raises rather than silently ignoring the sidecar under periodic_external.
# ---------------------------------------------------------------------------
CKPT="$(find "$CKPT_DIR" -name "params.json" -newer "$ENRICHED" 2>/dev/null | head -1)"
if [[ -z "$CKPT" ]]; then
  echo "ERROR: no params.json under $CKPT_DIR" >&2
  exit 4
fi
echo "--- Stage 3: md-system liquid_nvt (checkpoint=$CKPT)"
mmml md-system \
  --config examples/hybrid_mm_charges/md_fixed_lj_scales.yaml \
  --only liquid_nvt \
  --checkpoint "$CKPT" \
  --mm-lj-scales-file "$SIDECAR" \
  --output-dir "$RUN_DIR/liquid_nvt"

echo "=== done: $RUN_DIR ==="
