#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-1}"
if ! [[ "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 POSITIVE_STEPS" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/resolve_mmml_env.sh
mmml_resolve_env "$ROOT"

PS="$($MMML_PYTHON -c "print(int('$STEPS') * 0.0001)")"
OUT="$ROOT/artifacts/diagnostics/dcm10_nve_cptr_${STEPS}step"
rm -rf "$OUT"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_ENABLE_X64=1
export MMML_ML_DTYPE=float64
export OMP_NUM_THREADS=1
export MMML_NVE_C_API_HANDOFF=1
export MMML_TRACE_DYNAMICS_COMMAND=1
export CHARMM_LIB_DIR="$ROOT/setup/charmm/lib"
export LD_LIBRARY_PATH="$CHARMM_LIB_DIR:${LD_LIBRARY_PATH:-}"

"$MMML_PYTHON" -m mmml.cli md-system \
  --backend pycharmm --setup pbc_nve --md-stages nve \
  --ps-nve "$PS" --ps "$PS" --dt-fs 0.1 \
  --output-dir "$OUT" --job-name "dcm10_cptr_nve_${STEPS}" \
  --checkpoint "$ROOT/artifacts/spooky_so3lr_muon_tight/epoch-0001_params.json" \
  --composition DCM:10 --box-size 32.0 --temperature 150.0 \
  --continue-from "$ROOT/artifacts/pbc_recovery_gate_smoke_v2/dcm_10/pycharmm_equi_00/handoff/state.npz" \
  --continue-velocities \
  --handoff-template-res "$ROOT/artifacts/pbc_recovery_gate_smoke_v2/dcm_10/pycharmm_equi_00/handoff/final.res" \
  --include-mm --mm-switch-on 6.0 --mm-cutoff 4.0 --ml-switch-width 1.0 \
  --ml-compute-dtype float64 --ml-batch-size 512 \
  --dyn-nprint 1 --dcd-nsavc 0 --dcd-max-frames 0 \
  --no-pre-minimize --no-bonded-mm-mini \
  --no-calculator-pre-minimize --no-charmm-pre-minimize \
  --dynamics-overlap-action warn \
  >"$OUT/stdout.log" 2>"$OUT/stderr.log"
