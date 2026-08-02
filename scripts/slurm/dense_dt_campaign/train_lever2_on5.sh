#!/usr/bin/env bash
# GPU fine-tune hybrid LJ-scales at mm_switch_on=5 (soft lever-2 / DDC_HANDOFF=soft).
#
# Warm-starts epoch222 (trained at on=8). Writes under artifacts/lj_scales/ckpts/
# with tag hybrid_mm_lever2_on5_ft (override via DDC_ON5_TAG / env below).
#
# Prefer the sbatch wrapper:
#   bash scripts/slurm/dense_dt_campaign/submit_train_lever2_on5.sh
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${MMML_ROOT:-}" && -d "${MMML_ROOT}" ]]; then
  ROOT="$(cd "${MMML_ROOT}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "$ROOT"

CONFIG="${DDC_ON5_CONFIG:-examples/hybrid_mm_charges/train_fixed_lj_scales_on5.yaml}"
DATA="${DDC_ON5_DATA:-artifacts/lj_scales/dataset_cgenff.npz}"
CKPT="${DDC_ON5_CKPT:-artifacts/lj_scales/ckpts/params_hybrid_mm_fixed_lj_scales_epoch222.json}"
CKPT_DIR="${DDC_ON5_CKPT_DIR:-artifacts/lj_scales/ckpts}"
TAG="${DDC_ON5_TAG:-hybrid_mm_lever2_on5_ft}"
EPOCHS="${DDC_ON5_EPOCHS:-50}"
# Keep ~85/15 split of the 37950-frame DCM CGenFF set (same spirit as full train).
N_TRAIN="${DDC_ON5_N_TRAIN:-32000}"
N_VALID="${DDC_ON5_N_VALID:-5950}"
SEED="${DDC_ON5_SEED:-42}"
BATCH="${DDC_ON5_BATCH:-64}"

source .venv/bin/activate
export PATH="${HOME}/.local/bin:${PATH}"
export UV_NO_SYNC="${UV_NO_SYNC:-1}"
export PYTHONUNBUFFERED=1

# Force GPU. Do NOT honour a stale login-shell JAX_PLATFORMS=cpu (see
# examples/lj_scales/_env.sh). Submit with a clean env, not --export=ALL.
export LJ_DEVICE=gpu
export JAX_PLATFORMS=cuda
export MMML_MLPOT_DEVICE=gpu
export MMML_JAX_WARMUP_DEVICE=gpu
export MMML_MM_NL_DEVICE=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

# Miniconda libtinfo on the login node pollutes LD_LIBRARY_PATH via --export=ALL
# and can break CUDA plugin init. Prefer venv NVIDIA wheels, then system.
_NV_LIB=""
for _d in "${ROOT}"/.venv/lib/python*/site-packages/nvidia/*/lib; do
  [[ -d "$_d" ]] && _NV_LIB="${_NV_LIB:+${_NV_LIB}:}${_d}"
done
if [[ -n "${_NV_LIB}" ]]; then
  export LD_LIBRARY_PATH="${_NV_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# Drop bare /opt/miniconda3/lib if it is the only/first entry and nvidia is present.
case ":${LD_LIBRARY_PATH:-}:" in
  *:/opt/miniconda3/lib:*)
    LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '^/opt/miniconda3/lib$' | paste -sd: -)"
    export LD_LIBRARY_PATH
    ;;
esac
unset _NV_LIB _d

mkdir -p "$CKPT_DIR" artifacts/lj_scales/dense_dt_campaign/logs

[[ -f "$CONFIG" ]] || { echo "ERROR: missing config $CONFIG" >&2; exit 2; }
[[ -f "$DATA" ]] || { echo "ERROR: missing data $DATA" >&2; exit 2; }
[[ -f "$CKPT" ]] || { echo "ERROR: missing warm-start $CKPT" >&2; exit 2; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" || -n "${SLURM_JOB_ID:-}" ]] || {
  echo "ERROR: no Slurm GPU allocation (CUDA_VISIBLE_DEVICES unset, not in a job)." >&2
  echo "       Submit via: bash scripts/slurm/dense_dt_campaign/submit_train_lever2_on5.sh" >&2
  exit 2
}

echo "=== lever-2 on=5 retrain ==="
echo "  host     : $(hostname)"
echo "  job      : ${SLURM_JOB_ID:-local}"
echo "  root     : $ROOT"
echo "  config   : $CONFIG"
echo "  data     : $DATA"
echo "  warmstart: $CKPT"
echo "  ckpt_dir : $CKPT_DIR"
echo "  tag      : $TAG"
echo "  epochs   : $EPOCHS  n_train=$N_TRAIN n_valid=$N_VALID batch=$BATCH seed=$SEED"
echo "  handoff  : mm_switch_on=5.0 ml_switch_width=1.5 mm_switch_width=5.0"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "  JAX_PLATFORMS=$JAX_PLATFORMS MMML_MLPOT_DEVICE=$MMML_MLPOT_DEVICE"
date -Is
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true

# Hard gate: refuse silent CPU fallback (job 206061 burned ~210 s/epoch on CPU).
python - <<'PY'
import os, sys
import jax
devs = jax.devices()
print("JAX devices:", devs)
print("JAX_PLATFORMS=", os.environ.get("JAX_PLATFORMS"))
ok = any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devs)
if not ok:
    print("ERROR: expected CudaDevice, got", devs, file=sys.stderr)
    print("       Refuse to train on CPU under a GPU Slurm allocation.", file=sys.stderr)
    sys.exit(4)
PY

uv run mmml physnet-train \
  --config "$CONFIG" \
  --data "$DATA" \
  --valid-data "" \
  --ckpt-dir "$CKPT_DIR" \
  --tag "$TAG" \
  --n-train "$N_TRAIN" \
  --n-valid "$N_VALID" \
  --num-epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --seed "$SEED" \
  --mm-switch-on 5.0 \
  --ml-switch-width 1.5 \
  --mm-switch-width 5.0 \
  --hybrid-mm \
  --learn-mm-lj-scales \
  --lr-solver mic \
  --match-checkpoint-architecture \
  --physnet-checkpoint "$CKPT"

# Newest sidecar under this tag (uuid run dir).
SIDECAR="$(find "$CKPT_DIR" -type f -path "*${TAG}*/hybrid_mm.json" -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -1 | cut -d' ' -f2- || true)"
if [[ -z "${SIDECAR}" ]]; then
  echo "ERROR: no hybrid_mm.json under ${CKPT_DIR} for tag ${TAG}" >&2
  exit 3
fi
echo "=== done ==="
echo "  sidecar : $SIDECAR"
echo "  next    : re-run contact-ok dimer scans / ablate_overbind on the new best ckpt"
date -Is
