#!/usr/bin/env bash
# Run the whole sweep end to end: boxes → backends → electrostatics → ML/MM.
#
#   DRY_RUN=1 ./run_all.sh          # print every command first (recommended)
#   ./run_all.sh
#   BOX_SIZE=32 PS_PROD=50 ./run_all.sh
#
# Preflight (`mmml doctor`) runs first: on a stale libcharmm the MLpot atom
# limit falls back to max_Nml=100, which every cell in this matrix exceeds
# (the smallest is DCM:103 → 515 atoms), and the run would abort at setup.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

banner "Preflight"
if [[ "${SKIP_DOCTOR:-0}" != "1" ]]; then
  mmml doctor || {
    echo "mmml doctor reported problems — fix them or re-run with SKIP_DOCTOR=1" >&2
    exit 1
  }
  cat <<'EOF'

Check the line "CHARMM MLpot limits" above. If it says
"conservative fallback (libcharmm.so older than api_func.F90)" then max_Nml=100
and every cell here will fail. Rebuild first:

    ./scripts/rebuild_charmm_mlpot.sh --clean

EOF
fi

bash "${here}/01_build_boxes.sh"
bash "${here}/02_backends.sh"
bash "${here}/03_electrostatics.sh"
bash "${here}/04_ml_mm.sh"

banner "Done — results under ${OUT_ROOT}"
