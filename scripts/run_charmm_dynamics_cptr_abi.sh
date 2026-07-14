#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_ENABLE_X64=1
export MMML_ML_DTYPE=float64
export OMP_NUM_THREADS=1

bash scripts/rebuild_charmm_mlpot.sh --clean --no-domdec --skip-packmol
source scripts/resolve_mmml_env.sh
mmml_resolve_env "$ROOT"

"$MMML_PYTHON" -m pytest tests/unit/test_charmm_dynamics_c_abi.py -q
"$MMML_PYTHON" scripts/probe_charmm_dynamics_velocity_abi.py \
  --library setup/charmm/lib/libcharmm.so \
  --output artifacts/diagnostics/charmm_dynamics_velocity_abi_gpu08.json
