#!/usr/bin/env bash
# Quick post-mortem for one burst matrix cell (run on cluster after a Slurm failure).
# Usage: bash scripts/diag_cell.sh RUN_TAG
#   e.g. bash scripts/diag_cell.sh dcm_100_t300_l28
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
RUN_TAG="${1:?usage: diag_cell.sh RUN_TAG}"

cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

"$PY" -c "
import json
import sys
from pathlib import Path

workflow = Path('${WORKFLOW_ROOT}')
sys.path.insert(0, str(workflow / 'scripts'))
from campaign_lib import cell_from_tag, load_config, paths_for_run

cfg = load_config(workflow / 'config.yaml')
cell = cell_from_tag(cfg, '${RUN_TAG}')
paths = paths_for_run(cfg, cell)
out = paths['out_dir']
init = out / 'pycharmm_init'

print('=== ${RUN_TAG} ===')
print(f'cell: {cell.solvent} N={cell.n_monomers} T={cell.temperature}K L={cell.box_size}Å')
print(f'out_dir: {out}')
print(f'done.txt: {paths[\"done\"].is_file()} ({paths[\"done\"].stat().st_size if paths[\"done\"].is_file() else 0} B)')

summary = paths['campaign_summary']
if summary.is_file():
    jobs = json.loads(summary.read_text()).get('jobs', [])
    if jobs:
        last = jobs[-1]
        print(f'campaign_summary: {len(jobs)} job(s) recorded')
        print(f'  last job: {last.get(\"job_id\")} exit={last.get(\"exit_code\")}')
    else:
        print('campaign_summary: empty jobs list')
else:
    print('campaign_summary: missing')

handoff = init / 'handoff' / 'state.npz'
print(f'pycharmm_init handoff: {handoff.is_file()}')

for pattern in ('heat_*.res', 'heat_*.dcd', 'mini_*.res'):
    hits = sorted(init.glob(pattern))
    if hits:
        for p in hits[:3]:
            print(f'  {p.name}: {p.stat().st_size} B')
        if len(hits) > 3:
            print(f'  ... {len(hits) - 3} more {pattern}')

pretreat = init / 'pretreat'
if pretreat.is_dir():
    print(f'pretreat/: {len(list(pretreat.rglob(\"*\")))} files')
"

LOG_GLOB="${WORKFLOW_ROOT}/.snakemake/slurm_logs/rule_run_burst/*${RUN_TAG}*"
echo ""
echo "Slurm logs (newest first):"
shopt -s nullglob
logs=( $(ls -t ${LOG_GLOB}/*.log 2>/dev/null || true) )
if ((${#logs[@]} == 0)); then
  echo "  (none under ${LOG_GLOB})"
else
  latest="${logs[0]}"
  echo "  ${latest}"
  echo "--- tail ---"
  tail -40 "$latest"
  echo "--- errors ---"
  grep -E 'intra-monomer|overlap|ERROR|Traceback|exit code|CHARMM|FAILED|CANCELLED|Killed' "$latest" | tail -20 || true
fi

STDOUT="${REPO_ROOT}/artifacts/pbc_solvent_burst/${RUN_TAG}/stdout.log"
if [[ -f "$STDOUT" && -s "$STDOUT" ]]; then
  echo ""
  echo "stdout.log tail:"
  tail -20 "$STDOUT"
fi
