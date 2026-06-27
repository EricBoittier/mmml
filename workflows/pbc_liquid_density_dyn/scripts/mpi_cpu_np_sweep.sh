#!/usr/bin/env bash
# Sweep MMML_MPI_NP vs OMP_NUM_THREADS on a fixed DCM:32 CPU mini.
#
# Usage (from repo root):
#   export MMML_CKPT=$PWD/examples/ckpts_json/DESdimers_params.json
#   bash workflows/pbc_liquid_density_dyn/scripts/mpi_cpu_np_sweep.sh
#
# Optional: CPUS_PER_TASK=8 NP_LIST="1 2 4" OUT_ROOT=artifacts/mpi_cpu_np_sweep
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

CKPT="${MMML_CKPT:-$REPO_ROOT/examples/ckpts_json/DESdimers_params.json}"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT (set MMML_CKPT)" >&2
  exit 1
fi

CPUS="${CPUS_PER_TASK:-8}"
NP_LIST="${NP_LIST:-1 2 4 8}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/artifacts/mpi_cpu_np_sweep}"
BOX=25.0
N_ML=160  # DCM:32 × 5 ML atoms/monomer

mkdir -p "$OUT_ROOT"

eval "$(
  "$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml "$N_ML" --pbc --box-size "$BOX" \
    | grep '^export '
)"

RESULTS="$OUT_ROOT/results.csv"
echo "np,omp,wall_s,exit_code,host,cpus_per_task" >"$RESULTS"

echo "=== MPI CPU NP sweep (DCM:32 L=${BOX} mini_nstep=40) ==="
echo "CPUS_PER_TASK=$CPUS NP_LIST=$NP_LIST CHARMM_LIB_DIR=${CHARMM_LIB_DIR:-<unset>}"

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
  sed -e "s|REPLACE_OUT|$run_dir|g" -e "s|REPLACE_CKPT|$CKPT|g" \
    "$WORKFLOW_ROOT/benchmarks/dcm32_mini_mpi.yaml.tpl" >"$cfg"

  export MMML_MPI_NP="$np"
  export OMP_NUM_THREADS="$omp"
  export MMML_NO_MPI_RERUN=1

  echo "--- $tag (MMML_MPI_NP=$np OMP_NUM_THREADS=$omp) ---"
  log="$run_dir/stdout.log"
  set +e
  start=$(date +%s.%N)
  "$REPO_ROOT/scripts/mmml-charmm-mpirun.sh" md-system \
    --config "$cfg" \
    --mlpot-profile \
    >"$log" 2>&1
  rc=$?
  end=$(date +%s.%N)
  set -e
  wall=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.2f", e-s}')

  echo "$np,$omp,$wall,$rc,$(hostname),$CPUS" >>"$RESULTS"
  echo "  wall=${wall}s exit=$rc log=$log"
  if [[ -f "$run_dir/stage_summary.json" ]]; then
    cp -f "$run_dir/stage_summary.json" "$run_dir/stage_summary_${tag}.json" 2>/dev/null || true
  fi
  grep -E 'MLpot profile:|JAX compile' "$log" | tail -5 || true
done

echo ""
echo "Wrote $RESULTS"
column -t -s, "$RESULTS" 2>/dev/null || cat "$RESULTS"
