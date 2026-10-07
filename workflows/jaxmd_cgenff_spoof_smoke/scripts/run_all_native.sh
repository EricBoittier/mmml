#!/usr/bin/env bash
# Run native PyCHARMM counterparts of the DCM/ACO spoof smoke jobs.
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  export KARML_PYTHON="$REPO_ROOT/.venv/bin/python"
elif [[ -x "/mmhome/boittier/home/karml/.venv/bin/python" ]]; then
  export KARML_PYTHON="/mmhome/boittier/home/karml/.venv/bin/python"
else
  export KARML_PYTHON="${KARML_PYTHON:-python3}"
fi

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export KARML_WORKFLOW_CONFIG="${KARML_WORKFLOW_CONFIG:-$WORKFLOW_ROOT/config.native_charmm.yaml}"

if [[ -z "${KARML_CKPT:-}" ]]; then
  if [[ -f "$REPO_ROOT/examples/ckpts_json/DESdimers_params.json" ]]; then
    export KARML_CKPT="$REPO_ROOT/examples/ckpts_json/DESdimers_params.json"
  elif [[ -f /mmhome/boittier/home/karml/examples/ckpts_json/DESdimers_params.json ]]; then
    export KARML_CKPT=/mmhome/boittier/home/karml/examples/ckpts_json/DESdimers_params.json
  fi
fi

echo "REPO_ROOT=$REPO_ROOT"
echo "KARML_PYTHON=$KARML_PYTHON"
echo "KARML_WORKFLOW_CONFIG=$KARML_WORKFLOW_CONFIG"

JOBS=(dcm_vac_nve dcm_pbc_nve aco_vac_nve aco_pbc_nve)
if [[ $# -gt 0 ]]; then
  JOBS=("$@")
fi

fail=0
for job in "${JOBS[@]}"; do
  echo
  echo "######## native $job ########"
  if ! "$KARML_PYTHON" "$WORKFLOW_ROOT/scripts/run_job.py" "$job" --config "$KARML_WORKFLOW_CONFIG"; then
    fail=1
  fi
done
exit "$fail"
