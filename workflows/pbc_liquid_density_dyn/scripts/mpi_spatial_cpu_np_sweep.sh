#!/usr/bin/env bash
# Sweep MMML_MPI_NP vs OMP_NUM_THREADS for Tier-2 spatial MPI on CPU (DCM:100).
#
# Usage (from repo root):
#   export MMML_CKPT=$PWD/examples/ckpts_json/DESdimers_params.json
#   bash workflows/pbc_liquid_density_dyn/scripts/mpi_spatial_cpu_np_sweep.sh
#
# Optional:
#   CPUS_PER_TASK=8 NP_LIST="1 2 4 8" MINI_NSTEP=20 ML_BATCH=128
#   CPROFILE=1 OUT_ROOT=artifacts/mpi_spatial_cpu_np_sweep
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

# shellcheck source=../../../scripts/pc_bach_env.sh
source "$REPO_ROOT/scripts/pc_bach_env.sh"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export MMML_MLPOT_DEVICE="${MMML_MLPOT_DEVICE:-cpu}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export MMML_MLPOT_PROFILE="${MMML_MLPOT_PROFILE:-1}"
export MMML_JAX_COMPILE_TIMERS="${MMML_JAX_COMPILE_TIMERS:-1}"
# Override mmml-charmm-mpirun default (disables compile thread bump under MPI).
export MMML_NO_JAX_COMPILE_THREADS="${MMML_NO_JAX_COMPILE_THREADS:-0}"
export MMML_JAX_COMPILE_THREADS="${MMML_JAX_COMPILE_THREADS:-8}"
export MMML_FORCE_JAX_COMPILE_THREADS="${MMML_FORCE_JAX_COMPILE_THREADS:-1}"

CKPT="${MMML_CKPT:-$REPO_ROOT/examples/ckpts_json/DESdimers_params.json}"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT (set MMML_CKPT)" >&2
  exit 1
fi

CPUS="${CPUS_PER_TASK:-8}"
NP_LIST="${NP_LIST:-1 2 4 8}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/artifacts/mpi_spatial_cpu_np_sweep}"
BOX="${BOX_SIZE:-25.0}"
MINI_NSTEP="${MINI_NSTEP:-20}"
ML_BATCH="${ML_BATCH:-128}"
CPROFILE="${CPROFILE:-1}"
N_ML=500  # DCM:100 × 5 ML atoms/monomer

mkdir -p "$OUT_ROOT"

eval "$(
  "$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml "$N_ML" --pbc --box-size "$BOX" \
    | grep '^export '
)"

RESULTS="$OUT_ROOT/results.csv"
echo "np,omp,wall_s,exit_code,spatial_mpi,host,cpus_per_task,ml_batch,mini_nstep" >"$RESULTS"

echo "=== Spatial MPI CPU NP sweep (DCM:100 L=${BOX} mini_nstep=${MINI_NSTEP}) ==="
echo "CPUS_PER_TASK=$CPUS NP_LIST=$NP_LIST ML_BATCH=$ML_BATCH CPROFILE=$CPROFILE"
echo "CHARMM_LIB_DIR=${CHARMM_LIB_DIR:-<unset>}"

for np in $NP_LIST; do
  if (( np > CPUS )); then
    echo "skip np=$np (> CPUS_PER_TASK=$CPUS)" >&2
    continue
  fi
  omp=$((CPUS / np))
  [[ "$omp" -lt 1 ]] && omp=1
  tag="np${np}_omp${omp}"
  run_dir="$OUT_ROOT/$tag"
  mkdir -p "$run_dir"

  cfg="$run_dir/md_system.yaml"
  sed \
    -e "s|REPLACE_OUT|$run_dir|g" \
    -e "s|REPLACE_CKPT|$CKPT|g" \
    -e "s|REPLACE_BOX|$BOX|g" \
    -e "s|REPLACE_MINI_NSTEP|$MINI_NSTEP|g" \
    -e "s|REPLACE_ML_BATCH|$ML_BATCH|g" \
    "$WORKFLOW_ROOT/benchmarks/dcm100_spatial_mpi_cpu.yaml.tpl" >"$cfg"

  export MMML_MPI_NP="$np"
  export OMP_NUM_THREADS="$omp"
  export MMML_NO_MPI_RERUN=1
  if (( np > 1 )); then
    export MMML_MLPOT_SPATIAL_MPI=1
  else
    unset MMML_MLPOT_SPATIAL_MPI || true
  fi

  echo "--- $tag (MMML_MPI_NP=$np OMP_NUM_THREADS=$omp spatial=$([[ $np -gt 1 ]] && echo 1 || echo 0)) ---"
  log="$run_dir/stdout.log"
  prof="$run_dir/md_system.prof"
  set +e
  start=$(date +%s.%N)
  md_argv=(md-system --config "$cfg" --mlpot-profile --ml-spatial-mpi --reuse-packmol-cache)
  if [[ "$CPROFILE" == 1 ]]; then
    "$REPO_ROOT/scripts/mmml-charmm-mpirun.sh" \
      python -m cProfile -o "$prof" -m mmml.cli.__main__ "${md_argv[@]}" \
      >"$log" 2>&1
  else
    "$REPO_ROOT/scripts/mmml-charmm-mpirun.sh" "${md_argv[@]}" >"$log" 2>&1
  fi
  rc=$?
  end=$(date +%s.%N)
  set -e
  wall=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')
  spatial=$([[ $np -gt 1 ]] && echo 1 || echo 0)

  echo "$np,$omp,$wall,$rc,$spatial,$(hostname),$CPUS,$ML_BATCH,$MINI_NSTEP" >>"$RESULTS"
  echo "  wall=${wall}s exit=$rc log=$log"
  if [[ -f "$prof" ]]; then
    "$PY" - "$prof" "$run_dir/cprofile_top.txt" <<'PY' || true
import pstats
import sys
from pathlib import Path

prof_path, out_path = sys.argv[1], sys.argv[2]
stats = pstats.Stats(prof_path)
stats.sort_stats("cumulative")
with Path(out_path).open("w", encoding="utf-8") as fh:
    stats.stream = fh
    stats.print_stats(40)
PY
    echo "  cProfile top -> $run_dir/cprofile_top.txt"
  fi
  if [[ -f "$run_dir/stage_summary.json" ]]; then
    cp -f "$run_dir/stage_summary.json" "$run_dir/stage_summary_${tag}.json" 2>/dev/null || true
  fi
  grep -E 'MLpot profile:|JAX compile|mmml: JAX compile timers' "$log" | tail -8 || true
done

echo ""
echo "Wrote $RESULTS"
column -t -s, "$RESULTS" 2>/dev/null || cat "$RESULTS"
