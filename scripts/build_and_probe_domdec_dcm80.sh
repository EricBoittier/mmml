#!/usr/bin/env bash
# Build an 82 Å DCM box for DOMDEC ndir=8 probing, then run the atom-map test.
#
# Requirements:
#   - mmml venv active  (source ~/mmml/.venv/bin/activate  or  conda activate mmml)
#   - mpirun available
#   - CHARMM_HOME / CHARMM_LIB_DIR set (mmml configure)
#
# DOMDEC sizing (cutnb=10 Å, ndir=8):
#   domain_width = 82/8 = 10.25 Å > cutnb=10 Å  ✓
#   min_box      = 8 × 10 = 80 Å                 (using 82 Å for margin)
#
# Molecule count: 80 DCM in 82 Å keeps the same number density as the
#   existing 10 DCM / 40 Å test system (scales 10 × (82/40)³ ≈ 86 → use 80).
#
# Usage:
#   bash scripts/build_and_probe_domdec_dcm80.sh [output_dir]
#
set -euo pipefail

OUTDIR="${1:-${HOME}/tests/boxes/domdec_dcm80_l82}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NRANKS=8
NDIR=8
BOX=82
CUTNB=10
CTOFNB=9
CTONNB=8

echo "========================================"
echo "  Step 1: build box  (DCM:80, ${BOX} Å)"
echo "========================================"
mmml liquid-box \
    --composition "DCM:80" \
    --box-size ${BOX} \
    --output-dir "${OUTDIR}" \
    --profile standard \
    --seed 42

PSF="${OUTDIR}/model.psf"
CRD="${OUTDIR}/model.crd"

if [[ ! -f "${PSF}" || ! -f "${CRD}" ]]; then
    echo "ERROR: liquid-box did not produce model.psf / model.crd in ${OUTDIR}"
    exit 1
fi

echo ""
echo "========================================"
echo "  Step 2: single-rank sanity check"
echo "========================================"
python "${SCRIPT_DIR}/probe_domdec_atoms_live.py" \
    --psf "${PSF}" \
    --crd "${CRD}" \
    --box ${BOX} \
    --ndir 1 \
    --cutnb ${CUTNB} \
    --ctofnb ${CTOFNB} \
    --ctonnb ${CTONNB} \
    --no-ewald

echo "--- rank 00 (single-rank) ---"
cat "${PWD}/domdec_probe_rank00.txt"

echo ""
echo "========================================"
echo "  Step 3: 8-rank DOMDEC probe"
echo "========================================"
cd "${OUTDIR}"
mpirun -np ${NRANKS} python "${SCRIPT_DIR}/probe_domdec_atoms_live.py" \
    --psf "${PSF}" \
    --crd "${CRD}" \
    --box ${BOX} \
    --ndir ${NDIR} \
    --cutnb ${CUTNB} \
    --ctofnb ${CTOFNB} \
    --ctonnb ${CTONNB}

echo ""
echo "========================================"
echo "  Results"
echo "========================================"
echo "--- rank 00 (should show natoml>0 and local_idx populated) ---"
cat domdec_probe_rank00.txt

echo ""
echo "--- all ranks summary ---"
grep -h "natoml=\|active=\|NDIR=" domdec_probe_rank*.txt

echo ""
echo "--- rank 0 cross-rank disjoint check ---"
grep -A2 "Cross-rank" domdec_probe_rank00.txt || echo "(cross-rank check only runs on rank 0)"
