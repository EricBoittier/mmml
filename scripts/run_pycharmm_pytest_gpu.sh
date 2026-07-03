#!/usr/bin/env bash
# PyCHARMM pytest selection for GPU nodes (CHARMM + OpenMPI + JAX CUDA).
#
# Prerequisites (on the node):
#   source mmml/CHARMMSETUP   # or export CHARMM_HOME / CHARMM_LIB_DIR
#   uv sync --extra gpu
#   export MMML_CKPT=examples/ckpts_json/DESdimers_params.json
#
# Usage:
#   ./scripts/run_pycharmm_pytest_gpu.sh              # default: gpu slice
#   ./scripts/run_pycharmm_pytest_gpu.sh mlpot -q     # selection as 1st arg (see README)
#   MMML_PYTEST_SELECTION=smoke ./scripts/run_pycharmm_pytest_gpu.sh -q
#   ./scripts/run_pycharmm_pytest_gpu.sh tests/functionality/mlpot/test_mlpot_energy_matches_ase.py -q
#
# Selections (1st positional arg or MMML_PYTEST_SELECTION):
#   gpu      — -m "pycharmm and gpu"  (default; ML + CHARMM integration + short NVE/heat smoke)
#   pycharmm — -m pycharmm            (all live PyCHARMM tests)
#   smoke    — -m "pycharmm and not gpu" (CHARMM-only, no checkpoint/GPU ML)
#   mlpot    — -m mlpot               (MLpot-focused subset incl. test_mlpot_dynamics_smoke)
#   live     — optimizer + dynamics live validation (test_live_optimizers_dynamics.py)
#   quick    — fast mocked CLI/unit checks before the heavy suite
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=resolve_mmml_env.sh
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"

# shellcheck source=setup_jax_cuda_env.sh
source "$ROOT/scripts/setup_jax_cuda_env.sh"

export MMML_MLPOT_DEVICE="${MMML_MLPOT_DEVICE:-gpu}"
export MMML_CKPT="${MMML_CKPT:-$ROOT/examples/ckpts_json/DESdimers_params.json}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

_KNOWN_SELECTIONS=(gpu pycharmm smoke mlpot live quick)

_is_known_selection() {
  local name="$1"
  local sel
  for sel in "${_KNOWN_SELECTIONS[@]}"; do
    if [[ "$sel" == "$name" ]]; then
      return 0
    fi
  done
  return 1
}

SELECTION="${MMML_PYTEST_SELECTION:-gpu}"
if [[ $# -gt 0 ]] && _is_known_selection "$1"; then
  SELECTION="$1"
  shift
fi

_run_pytest() {
  local -a pytest_args=(--color=yes "$@")
  exec "$ROOT/scripts/mmml-charmm-mpirun.sh" python -m pytest "${pytest_args[@]}"
}

case "$SELECTION" in
  gpu)
    _run_pytest -m 'pycharmm and gpu' "$@"
    ;;
  pycharmm)
    _run_pytest -m pycharmm "$@"
    ;;
  smoke)
    _run_pytest -m 'pycharmm and not gpu' "$@"
    ;;
  mlpot)
    _run_pytest -m mlpot "$@"
    ;;
  live)
    _run_pytest tests/functionality/mlpot/test_live_optimizers_dynamics.py "$@"
    ;;
  quick)
    _run_pytest \
      tests/unit/test_monomer_constraints.py \
      tests/unit/test_md_system_pycharmm_cmd.py \
      tests/unit/test_assert_dynamics_ready.py \
      tests/unit/test_charmm_output_settings.py \
      tests/functionality/mlpot/test_pycharmm_conversion.py \
      "$@"
    ;;
  *)
    echo "run_pycharmm_pytest_gpu: unknown selection=$SELECTION" >&2
    echo "Valid: gpu, pycharmm, smoke, mlpot, live, quick" >&2
    exit 2
    ;;
esac
