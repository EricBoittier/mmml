#!/usr/bin/env bash
# Build a DCM box large enough for DOMDEC ndir=8, then run the atom-map probe.
#
# Requirements:
#   - mmml venv active
#   - mpirun available
#   - CHARMM_HOME / CHARMM_LIB_DIR set (mmml configure)
#
# DOMDEC sizing rule: domain_width = box/ndir >= cutnb
#   → min_box = ndir × cutnb
#
# Two presets (pass as first arg):
#
#   sparse (default)
#     DCM:80  box=82 Å  cutnb=10  domain_width=10.25 Å ✓
#     ~same number density as existing 10-mol/40 Å test system
#
#   dense
#     DCM:200 box=50 Å  cutnb=6   domain_width=6.25 Å ✓
#     ~17% of liquid DCM density (good stress test without full production size)
#     NOTE: true liquid density in 50 Å needs ~1175 DCM molecules.
#           For a liquid-density DOMDEC test use --box-auto density below.
#
# Usage:
#   bash scripts/build_and_probe_domdec_dcm80.sh [sparse|dense] [output_dir]
#
set -euo pipefail

PRESET="${1:-sparse}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${PRESET}" in
  dense)
    NMOL=200
    BOX=50          # domain_width = 50/8 = 6.25 > cutnb=6 ✓
    CUTNB=6
    CTOFNB=5.5
    CTONNB=4.5
    OUTDIR="${2:-${HOME}/tests/boxes/domdec_dcm200_l50}"
    ;;
  sparse|*)
    NMOL=80
    BOX=82          # domain_width = 82/8 = 10.25 > cutnb=10 ✓
    CUTNB=10
    CTOFNB=9
    CTONNB=8
    OUTDIR="${2:-${HOME}/tests/boxes/domdec_dcm80_l82}"
    ;;
esac

NRANKS=8
NDIR=8
DOMAIN_WIDTH=$(python3 -c "print(f'{${BOX}/${NDIR}:.2f}')" 2>/dev/null || echo "${BOX}/${NDIR}")

echo "========================================"
echo "  Preset : ${PRESET}"
echo "  Box    : DCM:${NMOL}  ${BOX} Å"
echo "  cutnb  : ${CUTNB} Å   domain_width=${DOMAIN_WIDTH} Å"
echo "  outdir : ${OUTDIR}"
echo "========================================"

# ---- Step 1: build -------------------------------------------------------
echo ""
echo ">>> Step 1: mmml liquid-box"
mmml liquid-box \
    --composition "DCM:${NMOL}" \
    --box-size ${BOX} \
    --output-dir "${OUTDIR}" \
    --profile standard \
    --seed 42

PSF="${OUTDIR}/model.psf"
CRD="${OUTDIR}/model.crd"
[[ -f "${PSF}" && -f "${CRD}" ]] || { echo "ERROR: model.psf/crd missing in ${OUTDIR}"; exit 1; }
echo "Box built: ${PSF}"

# ---- Step 2: single-rank sanity ------------------------------------------
echo ""
echo ">>> Step 2: single-rank sanity (cutoff-only)"
cd "${OUTDIR}"
python "${SCRIPT_DIR}/probe_domdec_atoms_live.py" \
    --psf "${PSF}" --crd "${CRD}" \
    --box ${BOX} --ndir 1 \
    --cutnb ${CUTNB} --ctofnb ${CTOFNB} --ctonnb ${CTONNB} \
    --no-ewald

echo "--- rank 00 ---"
cat domdec_probe_rank00.txt

# ---- Step 3: 8-rank DOMDEC -----------------------------------------------
# Disable the broken OpenFabrics/InfiniBand BTL so MPI falls back to
# shared-memory (within one node) or TCP.  Prevents spin-wait deadlock
# when mlx5_* devices are present but non-functional.
MPIFLAGS="--mca btl ^openib --mca shmem mmap"
echo ""
echo ">>> Step 3: mpirun -np ${NRANKS}  ndir=${NDIR}  cutnb=${CUTNB}"
echo "    MPI flags: ${MPIFLAGS}"
mpirun -np ${NRANKS} ${MPIFLAGS} python "${SCRIPT_DIR}/probe_domdec_atoms_live.py" \
    --psf "${PSF}" --crd "${CRD}" \
    --box ${BOX} --ndir ${NDIR} \
    --cutnb ${CUTNB} --ctofnb ${CTOFNB} --ctonnb ${CTONNB}

echo ""
echo "========================================"
echo "  Results"
echo "========================================"
echo "--- rank 00 ---"
cat domdec_probe_rank00.txt
echo ""
echo "--- per-rank natoml (should total ~${NMOL}×5 atoms across 8 ranks) ---"
grep "natoml=" domdec_probe_rank*.txt
echo ""
echo "--- cross-rank disjoint (rank 0 only) ---"
grep -A3 "Cross-rank" domdec_probe_rank00.txt || echo "(cross-rank lines not found)"
