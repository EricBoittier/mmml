#!/usr/bin/env bash
# Shared helpers for dcm_density_setup_compare debug scripts.
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

debug_artifact_root() {
  local wf repo cfg raw
  wf="$(debug_workflow_root)"
  repo="$(debug_repo_root)"
  cfg="$wf/config.yaml"
  if [[ -f "$cfg" ]]; then
    raw="$(grep -E '^output_root:' "$cfg" | sed 's/^output_root:[[:space:]]*//' | tr -d '"' || true)"
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

# Pattern groups (extended regex for grep -E).
readonly DBG_PAT_ABORT='pycharmm_mlpot: error:|RuntimeError|coordinates still too strained|dynamics skipped|Pre-dynamics GRMS [0-9]+.*>|Campaign summary reports failed|Failed leg '
readonly DBG_PAT_GRMS='Hybrid GRMS|CHARMM GRMS|GRMS thresholds|max_before_dyn|intervention=|Pre-dynamics GRMS|post-rescue gate|post-overlap-rescue|max_grms'
readonly DBG_PAT_MINI='Post MLpot SD|MLpot SD pass|watchdog|rollback|partial|monomer repack polish|SD pass 1'
readonly DBG_PAT_HEAT='heat segment [0-9]+/[0-9]+|heat_thermostat|fly-off|checkpoint ladder|inter-monomer atom overlap|overlap rescue|Packmol repack|PhysNet|separation fallback|Monomer health'
readonly DBG_PAT_LEGS='pycharmm_init|pycharmm_equi|pycharmm_prod|jaxmd_prod|ase_prod|exit_code|job_id|resume skip complete job'
readonly DBG_PAT_MPI='apply_bonded_mm_only_block|selective COEFF BLOCK|bonded-MM-mini: skipping'
readonly DBG_PAT_SLURM='Error in rule|external_jobid|Trying to restart|FAILED|CANCELLED|OOM|DUE TO TIME LIMIT|Killed'

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
