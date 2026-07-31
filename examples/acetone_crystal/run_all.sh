#!/usr/bin/env bash
# Build solid acetone from its published crystal structure and compute the
# sublimation enthalpy.
#
# The whole ladder is CPU-only, needs no CHARMM, no GPU and no trained
# checkpoint, and finishes in well under a minute. Step 03 is skipped by
# default only because its output is not used downstream.
#
#   ACO_PHASE=pbca_5k bash run_all.sh      # work with a different phase
#   ACO_BUILD=1 bash run_all.sh            # also write PDB/extxyz (step 03)
#   ACO_SCALES=/path/hybrid_mm.json        # evaluate with learned LJ scales
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/acetone_crystal/_env.sh"
cd "${ROOT}"
DIR="examples/acetone_crystal"
aco_crystal_banner

echo "=== solid acetone: structure and sublimation enthalpy ==="

uv run python "${DIR}/00_check_env.py"
uv run python "${DIR}/01_phases.py"
uv run python "${DIR}/02_contacts.py"

if [[ "${ACO_BUILD:-0}" == "1" ]]; then
  bash "${DIR}/03_build_supercell.sh"
else
  echo
  echo "SKIP 03 (structure export): ACO_BUILD=1 to write PDB/extxyz."
fi

uv run python "${DIR}/04_lattice_energy.py"
uv run python "${DIR}/05_sublimation.py"

echo "=== ALL ACETONE CRYSTAL STEPS PASSED ==="
