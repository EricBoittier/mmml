#!/usr/bin/env bash
# PyCHARMM pytest selection for GPU nodes (CHARMM + OpenMPI + JAX CUDA).
#
# Prerequisites (on the node):
#   source mmml/CHARMMSETUP   # or export CHARMM_HOME / CHARMM_LIB_DIR
#   uv sync --extra gpu
#   export MMML_CKPT=examples/ckpts_json/DESdimers_params.json
#
# Usage:
#   ./scripts/run_pycharmm_pytest_gpu.sh
#   MMML_PYTEST_SELECTION=smoke ./scripts/run_pycharmm_pytest_gpu.sh
#   ./scripts/run_pycharmm_pytest_gpu.sh tests/functionality/mlpot/test_mlpot_energy_matches_ase.py -q
#
# Selections (MMML_PYTEST_SELECTION):
#   gpu      — -m "pycharmm and gpu"  (default; ML + CHARMM integration)
#   pycharmm — -m pycharmm            (all live PyCHARMM tests)
#   smoke    — -m "pycharmm and not gpu" (CHARMM-only, no checkpoint/GPU ML)
#   mlpot    — -m mlpot               (MLpot-focused subset)
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

SELECTION="${MMML_PYTEST_SELECTION:-gpu}"
MARK_EXPR=""
case "$SELECTION" in
  gpu) MARK_EXPR='pycharmm and gpu' ;;
  pycharmm) MARK_EXPR='pycharmm' ;;
  smoke) MARK_EXPR='pycharmm and not gpu' ;;
  mlpot) MARK_EXPR='mlpot' ;;
  quick)
    exec "$ROOT/scripts/mmml-charmm-mpirun.sh" python -m pytest --color=yes \
      tests/unit/test_monomer_constraints.py \
      tests/unit/test_md_system_pycharmm_cmd.py \
      tests/unit/test_assert_dynamics_ready.py \
      tests/unit/test_charmm_output_settings.py \
      tests/functionality/mlpot/test_pycharmm_conversion.py \
      "$@"
    ;;
  *)
    echo "run_pycharmm_pytest_gpu: unknown MMML_PYTEST_SELECTION=$SELECTION" >&2
    echo "Valid: gpu, pycharmm, smoke, mlpot, quick" >&2
    exit 2
    ;;
esac

if [[ $# -gt 0 ]]; then
  exec "$ROOT/scripts/mmml-charmm-mpirun.sh" python -m pytest --color=yes -m "$MARK_EXPR" "$@"
fi

exec "$ROOT/scripts/mmml-charmm-mpirun.sh" python -m pytest --color=yes -m "$MARK_EXPR"
