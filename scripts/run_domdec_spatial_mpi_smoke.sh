#!/usr/bin/env bash
# run_domdec_spatial_mpi_smoke.sh
#
# Two-step Tier 3 smoke: DOMDEC + Spatial MLpot on a DCM cluster.
#
# Step 1 (np=1): build and write PSF/CRD for DCM:20.
# Step 2 (np=4): callback-only DOMDEC path check (no checkpoint needed).
# Step 3 (np=4, opt-in): live CHARMM ENER with DOMDEC + spatial MPI.
#
# Usage:
#   # Callback-only (no checkpoint):
#   bash scripts/run_domdec_spatial_mpi_smoke.sh
#
#   # Live ENER (checkpoint required):
#   MMML_CKPT=/path/to/checkpoint.json bash scripts/run_domdec_spatial_mpi_smoke.sh --live
#
# Environment:
#   MMML_CKPT          PhysNet checkpoint path (required for --live)
#   SMOKE_NP           Number of MPI ranks for the smoke (default: 4)
#   SMOKE_N_MOL        Number of DCM monomers (default: 20)
#   SMOKE_BOX          Box side length in Å (default: 40)
#   SMOKE_CUTNB        Nonbond cutoff in Å (default: 10)
#   SMOKE_PREBUILT_DIR Artifact directory (default: artifacts/domdec_spatial_smoke)

set -euo pipefail

LIVE=0
for arg in "$@"; do
    case "$arg" in
        --live) LIVE=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

SMOKE_NP=${SMOKE_NP:-4}
SMOKE_N_MOL=${SMOKE_N_MOL:-20}
SMOKE_BOX=${SMOKE_BOX:-40}
SMOKE_CUTNB=${SMOKE_CUTNB:-10}
SMOKE_PREBUILT_DIR=${SMOKE_PREBUILT_DIR:-artifacts/domdec_spatial_smoke}

SCRIPT="tests/functionality/mlpot/10_domdec_spatial_mpi_smoke.py"

echo "================================================================"
echo "DOMDEC + Spatial MLpot Smoke"
echo "  np=$SMOKE_NP  n_monomers=$SMOKE_N_MOL  box=${SMOKE_BOX}Å  cutnb=${SMOKE_CUTNB}Å"
echo "================================================================"
echo ""

# ----------------------------------------------------------------
# Step 1 — Build prebuilt PSF/CRD (np=1)
# ----------------------------------------------------------------
PSF_FILE="${SMOKE_PREBUILT_DIR}/dcm_${SMOKE_N_MOL}mer.psf"
if [[ -f "$PSF_FILE" ]]; then
    echo "Step 1: prebuilt artifacts found at $PSF_FILE — skipping build."
else
    echo "Step 1: building DCM:${SMOKE_N_MOL} PSF/CRD (np=1)..."
    MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh python \
        "$SCRIPT" \
        --prepare-prebuilt-only \
        --residue DCM \
        --n-molecules "$SMOKE_N_MOL" \
        --box-side "$SMOKE_BOX" \
        --prebuilt-dir "$SMOKE_PREBUILT_DIR"
    echo ""
fi

# ----------------------------------------------------------------
# Step 2 — Callback-only DOMDEC path check (no checkpoint)
# ----------------------------------------------------------------
echo "Step 2: callback-only smoke (mocked DOMDEC, np=$SMOKE_NP)..."
MMML_MPI_NP="$SMOKE_NP" MMML_MLPOT_SPATIAL_MPI=1 \
    ./scripts/mmml-charmm-mpirun.sh python \
    "$SCRIPT" \
    --residue DCM \
    --n-molecules "$SMOKE_N_MOL" \
    --box-side "$SMOKE_BOX"
echo ""

# ----------------------------------------------------------------
# Step 3 — Live CHARMM ENER (opt-in, requires checkpoint)
# ----------------------------------------------------------------
if [[ "$LIVE" -eq 1 ]]; then
    if [[ -z "${MMML_CKPT:-}" ]]; then
        echo "ERROR: --live requires MMML_CKPT to be set." >&2
        exit 1
    fi
    echo "Step 3: live CHARMM ENER (DOMDEC + spatial MPI, np=$SMOKE_NP)..."
    MMML_MPI_NP="$SMOKE_NP" MMML_MLPOT_SPATIAL_MPI=1 \
        ./scripts/mmml-charmm-mpirun.sh python \
        "$SCRIPT" \
        --charmm-ener \
        --checkpoint "$MMML_CKPT" \
        --residue DCM \
        --n-molecules "$SMOKE_N_MOL" \
        --box-side "$SMOKE_BOX" \
        --cutnb "$SMOKE_CUTNB" \
        --prebuilt-dir "$SMOKE_PREBUILT_DIR"
    echo ""
else
    echo "Step 3: skipped (pass --live and set MMML_CKPT for live CHARMM ENER)."
fi

echo "================================================================"
echo "Smoke complete."
echo "================================================================"
