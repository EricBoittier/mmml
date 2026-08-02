#!/usr/bin/env bash
# Crystalline dichloromethane: check the deposited structures, test the paper's
# claim about what holds the crystal together, relax to ambient pressure and
# compare with experiment.
#
# CPU-only, no CHARMM, no GPU, no trained checkpoint. Under a minute end to end.
#
#   DCM_PHASE=pbcn_163gpa bash run_all.sh   # work from the other pressure point
#   DCM_SCALES=/path/hybrid_mm.json         # evaluate with learned LJ scales
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/dcm_crystal/_env.sh"
cd "${ROOT}"
DIR="examples/dcm_crystal"
dcm_crystal_banner

echo "=== crystalline dichloromethane: cohesion and sublimation ==="

uv run python "${DIR}/00_check_env.py"
uv run python "${DIR}/01_phases.py"
uv run python "${DIR}/02_contacts.py"
uv run python "${DIR}/03_cohesion.py"
uv run python "${DIR}/04_lattice_energy.py"
uv run python "${DIR}/05_relax_and_sublimation.py"

echo "=== ALL DCM CRYSTAL STEPS PASSED ==="
