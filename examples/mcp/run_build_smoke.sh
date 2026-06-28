#!/usr/bin/env bash
# MCP build_smoke driver — geometry, periodic box, hybrid MD (ASE / JAX-MD / PyCHARMM).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${1:-build001}"
MODE="${MODE:-smoke}"
DRY_RUN="${DRY_RUN:-0}"
cd "${ROOT}"

if [[ -f "${ROOT}/examples/md_cpu/_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/examples/md_cpu/_env.sh"
fi

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

py() { uv run python "$@"; }

echo "=== MCP build_smoke run_id=${RUN_ID} mode=${MODE} ==="
py -c "from mmml.mcp.recipes import configure_run; import json; print(json.dumps(configure_run('${RUN_ID}', recipe='build_smoke', mode='${MODE}'), indent=2))"

_stages=(make_res box_build hybrid_md_ase hybrid_md_jaxmd hybrid_md_pycharmm)
for stage in "${_stages[@]}"; do
  echo "--- stage: ${stage} ---"
  if [[ "${DRY_RUN}" == "1" ]]; then
    py -c "
from mmml.mcp.recipes import run_recipe_stage
import json
print(json.dumps(run_recipe_stage('${RUN_ID}', '${stage}', mode='${MODE}', dry_run=True), indent=2))
"
  else
    py -c "
from mmml.mcp.recipes import run_recipe_stage
import json
r = run_recipe_stage('${RUN_ID}', '${stage}', mode='${MODE}')
print(json.dumps(r, indent=2))
if r.get('state') == 'failed':
    raise SystemExit(1)
"
  fi
done

echo "=== done: artifacts/mcp_runs/${RUN_ID} ==="
