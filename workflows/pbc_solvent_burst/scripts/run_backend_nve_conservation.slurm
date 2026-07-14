#!/usr/bin/env bash
# Matched PyCHARMM/JAX-MD NVE energy-conservation check from one saved handoff.
#SBATCH --job-name=mmml-nve-compare
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000
#SBATCH --time=01:00:00
#SBATCH --output=../../artifacts/diagnostics/nve-compare-%j.out
#SBATCH --error=../../artifacts/diagnostics/nve-compare-%j.err

set -euo pipefail

WORKFLOW_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
export JAX_ENABLE_X64=1
export MMML_ML_DTYPE=float64
export MMML_TRACE_DYNAMICS_COMMAND=1

eval "$("$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml 100 --pbc --box-size 32 \
  | tee /dev/stderr | grep '^export ')"

ROOT="${MMML_NVE_COMPARE_ROOT:-$REPO_ROOT/artifacts/diagnostics/dcm10_nve_backend_compare_v2}"
HANDOFF="$REPO_ROOT/artifacts/pbc_recovery_gate_smoke_v2/dcm_10/pycharmm_equi_00/handoff/state.npz"
TEMPLATE="$REPO_ROOT/artifacts/pbc_recovery_gate_smoke_v2/dcm_10/pycharmm_equi_00/handoff/final.res"
NATIVE_RESTART="$REPO_ROOT/artifacts/pbc_recovery_gate_smoke_v2/dcm_10/pycharmm_equi_00/equi.res"
CHECKPOINT="$REPO_ROOT/artifacts/spooky_so3lr_muon_tight/epoch-0001_params.json"
mkdir -p "$ROOT/pycharmm" "$ROOT/jaxmd"

COMMON=(
  --checkpoint "$CHECKPOINT" --composition DCM:10 --box-size 32.0
  --temperature 150.0 --dt-fs 0.1 --ps 0.02 --continue-from "$HANDOFF"
  --continue-velocities --handoff-template-res "$TEMPLATE"
  --include-mm --mm-switch-on 6.0 --mm-cutoff 4.0 --ml-switch-width 1.0
  --ml-compute-dtype float64 --ml-batch-size 512
  --no-pre-minimize --no-bonded-mm-mini --dynamics-overlap-action warn
)

MMML_NVE_C_API_HANDOFF=1 "$MMML_PYTHON" -m mmml.cli md-system \
  --backend pycharmm --setup pbc_nve --md-stages nve --ps-nve 0.02 \
  --output-dir "$ROOT/pycharmm" --job-name dcm10_pycharmm_nve \
  --dyn-nprint 1 --dcd-nsavc 0 --dcd-max-frames 0 --pre-nve-charmm-update \
  "${COMMON[@]}" >"$ROOT/pycharmm/stdout.log" 2>"$ROOT/pycharmm/stderr.log"

"$MMML_PYTHON" "$REPO_ROOT/workflows/dcm3_nve_cutoff_sweep/scripts/analyze_nve_energy.py" \
  --log "$ROOT/pycharmm/stdout.log" --output "$ROOT/pycharmm/energy_report.json" \
  --npz "$ROOT/pycharmm/energy_trace.npz"

"$MMML_PYTHON" -m mmml.cli md-system \
  --backend jaxmd --setup pbc_nve --output-dir "$ROOT/jaxmd" \
  --job-name dcm10_jaxmd_nve --steps-per-recording 1 --jax-md-update-interval 1 \
  --jaxmd-minimize-steps 0 --jaxmd-pbc-minimize-steps 0 \
  --no-handoff-velocity-remove-drift \
  --no-calculator-pre-minimize --no-charmm-pre-minimize \
  "${COMMON[@]}" >"$ROOT/jaxmd/stdout.log" 2>"$ROOT/jaxmd/stderr.log"

"$MMML_PYTHON" "$REPO_ROOT/scripts/analyze_jaxmd_nve_energy.py" \
  --h5 "$ROOT/jaxmd/pbc_nve_jaxmd_nve.h5" \
  --output "$ROOT/jaxmd/energy_report.json"
