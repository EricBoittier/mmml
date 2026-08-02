#!/usr/bin/env bash
# Step 03 — write the periodic structure out for MD, viewers and other tools.
#
# `mmml build-crystal` reads the deposited CIF, lets ASE apply the symmetry
# operators, matches each molecule onto the CHARMM `ACO` atom names from the
# make-res template, and writes a PDB with a correct CRYST1 record plus an
# extxyz carrying the full cell.
#
# Nothing downstream in this ladder needs the output: the lattice energy is
# computed on the unit cell with an explicit lattice sum, which is both cheaper
# and more accurate than tiling. This step exists so the structure you just
# validated can leave the ladder.
#
#   ACO_PHASE=pbca_5k bash 03_build_supercell.sh
#   ACO_SUPERCELL=2,2,1 bash 03_build_supercell.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/acetone_crystal/_env.sh"
cd "${ROOT}"
aco_crystal_banner

# Map the phase key onto the build-crystal preset name.
case "${ACO_PHASE}" in
  pbca_150k) PRESET="aco" ;;
  pbca_5k)   PRESET="aco5k" ;;
  pbca_110k) PRESET="aco110k" ;;
  cmcm_160k) PRESET="acocmcm" ;;
  cmcm_15kbar)
    echo "03: the 15 kbar Cmcm phase has rotationally disordered methyls." >&2
    echo "    Its 12 half-occupancy hydrogens per molecule cannot be mapped onto" >&2
    echo "    the 6 atoms of CGenFF ACO. Pick an ordered phase:" >&2
    echo "      ACO_PHASE=pbca_150k bash examples/acetone_crystal/03_build_supercell.sh" >&2
    exit 1 ;;
  *)
    echo "03: unknown ACO_PHASE '${ACO_PHASE}'" >&2
    exit 1 ;;
esac

OUT_PDB="${ARTIFACTS_DIR}/acetone_${ACO_PHASE}.pdb"
OUT_XYZ="${ARTIFACTS_DIR}/acetone_${ACO_PHASE}.extxyz"

echo "=== 03: building ${ACO_PHASE} (preset ${PRESET}), supercell ${ACO_SUPERCELL} ==="

uv run mmml build-crystal \
  --literature "${PRESET}" \
  --supercell "${ACO_SUPERCELL}" \
  -o "${OUT_PDB}"

uv run mmml build-crystal \
  --literature "${PRESET}" \
  --supercell "${ACO_SUPERCELL}" \
  -o "${OUT_XYZ}"

echo
echo "Wrote:"
echo "  ${OUT_PDB}"
echo "  ${OUT_XYZ}"
cat <<'EOF'

A note on what you can do with this. The cell is orthorhombic but strongly
non-cubic, and mmml's periodic MD paths are cubic-only: prepare_charmm_pbc
installs a cubic CHARMM IMAGE, and the md-system box resolution averages the
three edge lengths into one. Handing this structure to `md-system --box-size`
would therefore run a differently shaped box than the one you built, silently.

So: use these files for visualisation, for other codes, and for the static
lattice energy in step 04. Running crystal MD in mmml needs orthorhombic PBC
support first -- see the README for what that involves.
EOF
