#!/usr/bin/env bash
# Shared helpers for dcm_density_setup_compare debug scripts (pc-studix / Slurm).
# Run from the workflow directory on the login node:
#   cd ~/mmml/workflows/dcm_density_setup_compare
# shellcheck shell=bash

set -euo pipefail

debug_workflow_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

debug_repo_root() {
  local wf
  wf="$(debug_workflow_root)"
  cd "$wf/../.." && pwd
}

debug_bootstrap_cluster() {
  local repo
  repo="$(debug_repo_root)"
  if [[ -f "$repo/scripts/resolve_mmml_env.sh" ]]; then
    # shellcheck source=../../../scripts/resolve_mmml_env.sh
    source "$repo/scripts/resolve_mmml_env.sh"
    mmml_resolve_env "$repo"
  fi
}

debug_python() {
  if [[ -n "${MMML_PYTHON:-}" ]]; then
    echo "$MMML_PYTHON"
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  else
    echo python3
  fi
}

debug_artifact_root() {
  local wf repo cfg raw py
  wf="$(debug_workflow_root)"
  repo="$(debug_repo_root)"
  cfg="$wf/config.yaml"
  if [[ -f "$cfg" ]]; then
    py="$(debug_python)"
    raw="$("$py" -c "
import sys
sys.path.insert(0, '${wf}/scripts')
from campaign_lib import load_config
from pathlib import Path
cfg = load_config(Path('${cfg}'))
print(cfg.get('output_root', 'artifacts/dcm_density_setup_compare'))
" 2>/dev/null || true)"
    if [[ -z "${raw:-}" ]]; then
      raw="$(grep -E '^output_root:' "$cfg" | sed 's/^output_root:[[:space:]]*//' | tr -d '"' || true)"
    fi
  fi
  if [[ -n "${raw:-}" ]]; then
    if [[ "$raw" = /* ]]; then
      echo "$raw"
    else
      echo "$repo/$raw"
    fi
  else
    echo "$repo/artifacts/dcm_density_setup_compare"
  fi
}

debug_cell_log() {
  local tag="${1:?tag required}"
  echo "$(debug_artifact_root)/$tag/stdout.log"
}

debug_cell_dir() {
  local tag="${1:?tag required}"
  echo "$(debug_artifact_root)/$tag"
}

debug_slurm_log_dir() {
  echo "$(debug_workflow_root)/.snakemake/slurm_logs/rule_run_setup_compare"
}

# Pattern groups (extended regex for grep -E).
readonly DBG_PAT_ABORT='pycharmm_mlpot: error:|RuntimeError|coordinates still too strained|dynamics skipped|Pre-dynamics GRMS [0-9]+.*>|Campaign summary reports failed|Failed leg '
readonly DBG_PAT_GRMS='Hybrid GRMS|CHARMM GRMS|GRMS thresholds|max_before_dyn|intervention=|Pre-dynamics GRMS|post-rescue gate|post-overlap-rescue|max_grms'
readonly DBG_PAT_MINI='Post MLpot SD|MLpot SD pass|watchdog|rollback|partial|monomer repack polish|SD pass 1'
readonly DBG_PAT_HEAT='heat segment [0-9]+/[0-9]+|heat_thermostat|fly-off|checkpoint ladder|inter-monomer atom overlap|overlap rescue|Packmol repack|PhysNet|separation fallback|Monomer health'
readonly DBG_PAT_LEGS='pycharmm_init|pycharmm_equi|pycharmm_prod|jaxmd_prod|ase_prod|exit_code|job_id|resume skip complete job'
readonly DBG_PAT_MPI='apply_bonded_mm_only_block|selective COEFF BLOCK|bonded-MM-mini: skipping'
readonly DBG_PAT_SLURM='Error in rule|external_jobid|Trying to restart|FAILED|CANCELLED|OOM|DUE TO TIME LIMIT|Killed|OUT_OF_MEMORY'

debug_grep_section() {
  local title="$1"
  local file="$2"
  local pattern="$3"
  local max="${4:-80}"

  echo "=== $title ==="
  if [[ ! -f "$file" ]]; then
    echo "(missing: $file)"
    echo
    return 0
  fi
  if grep -nE "$pattern" "$file" 2>/dev/null | tail -n "$max"; then
    :
  else
    echo "(no matches)"
  fi
  echo
}

debug_extract_abort() {
  local file="$1"
  grep -m1 -oE 'pycharmm_mlpot: error: [^[:cntrl:]]+|post-overlap-rescue hybrid GRMS [0-9.]+ > [0-9]+' "$file" 2>/dev/null || true
}

debug_extract_mini_grms() {
  local file="$1"
  grep -E 'Post MLpot SD pass 1|monomer repack polish left hybrid GRMS' "$file" 2>/dev/null \
    | tail -1 \
    | grep -oE '[0-9]+\.[0-9]+ kcal/mol/Å' \
    | head -1 || true
}

debug_cell_done() {
  local tag="$1"
  local done_file
  done_file="$(debug_cell_dir "$tag")/done.txt"
  if [[ -s "$done_file" ]]; then
    echo OK
  else
    echo --
  fi
}

debug_slurm_job_summary() {
  local jobid="${1:?job id required}"
  if ! command -v sacct >/dev/null 2>&1; then
    echo "(sacct not available)"
    return 0
  fi
  sacct -j "$jobid" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList -P -n 2>/dev/null \
    | head -5 || echo "(no sacct record for $jobid)"
}

debug_user_gpu_queue() {
  if ! command -v squeue >/dev/null 2>&1; then
    echo "(squeue not available)"
    return 0
  fi
  squeue -u "${USER:?USER unset}" -o '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R' 2>/dev/null \
    | grep -E 'setup_compare|md-system|mmml|JOBID' || true
}

debug_find_slurm_log() {
  local jobid="$1"
  local dir
  dir="$(debug_slurm_log_dir)"
  find "$dir" -name "${jobid}.log" -print 2>/dev/null | head -1
}

debug_print_campaign_summary() {
  local summary="$1"
  local py
  py="$(debug_python)"
  "$py" - <<PY
import json
from pathlib import Path
p = Path("$summary")
if not p.is_file():
    raise SystemExit(0)
data = json.loads(p.read_text())
jobs = data.get("jobs", data if isinstance(data, list) else [])
for j in jobs:
    rc = int(j.get("exit_code", 0))
    mark = "FAIL" if rc else "ok "
    print(f"  [{mark}] {j.get('job_id')} backend={j.get('backend')} exit_code={rc}")
failed = [j.get("job_id") for j in jobs if int(j.get("exit_code", 0)) != 0]
if failed:
    print("failed:", failed)
PY
}
