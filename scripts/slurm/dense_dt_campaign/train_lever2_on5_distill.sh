#!/usr/bin/env bash
# Distilled short FT at mm_switch_on=5 (freeze LJ scales, lr=1e-4, 15 epochs).
#
#   bash scripts/slurm/dense_dt_campaign/submit_train_lever2_on5_distill.sh
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${MMML_ROOT:-}" && -d "${MMML_ROOT}" ]]; then
  ROOT="$(cd "${MMML_ROOT}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "$ROOT"

CONFIG="${DDC_ON5D_CONFIG:-examples/hybrid_mm_charges/train_fixed_lj_scales_on5_distill.yaml}"
DATA="${DDC_ON5D_DATA:-artifacts/lj_scales/dataset_cgenff.npz}"
CKPT="${DDC_ON5D_CKPT:-artifacts/lj_scales/ckpts/params_hybrid_mm_fixed_lj_scales_epoch222.json}"
TEACHER="${DDC_ON5D_TEACHER:-$CKPT}"
SCALE_SIDECAR="${DDC_ON5D_SCALE_SIDECAR:-artifacts/lj_scales/ckpts/hybrid_mm_fixed_lj_scales-4d68132d-c686-4ded-9887-efc16d5b2638/hybrid_mm.json}"
CKPT_DIR="${DDC_ON5D_CKPT_DIR:-artifacts/lj_scales/ckpts}"
TAG="${DDC_ON5D_TAG:-hybrid_mm_lever2_on5_distill}"
EPOCHS="${DDC_ON5D_EPOCHS:-15}"
N_TRAIN="${DDC_ON5D_N_TRAIN:-32000}"
N_VALID="${DDC_ON5D_N_VALID:-5950}"
SEED="${DDC_ON5D_SEED:-42}"
BATCH="${DDC_ON5D_BATCH:-64}"
LR="${DDC_ON5D_LR:-0.0001}"
DISTILL_ALPHA="${DDC_ON5D_DISTILL_ALPHA:-0.35}"

source .venv/bin/activate
export PATH="${HOME}/.local/bin:${PATH}"
export UV_NO_SYNC="${UV_NO_SYNC:-1}"
export PYTHONUNBUFFERED=1
export LJ_DEVICE=gpu
export JAX_PLATFORMS=cuda
export MMML_MLPOT_DEVICE=gpu
export MMML_JAX_WARMUP_DEVICE=gpu
export MMML_MM_NL_DEVICE=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

_NV_LIB=""
for _d in "${ROOT}"/.venv/lib/python*/site-packages/nvidia/*/lib; do
  [[ -d "$_d" ]] && _NV_LIB="${_NV_LIB:+${_NV_LIB}:}${_d}"
done
if [[ -n "${_NV_LIB}" ]]; then
  export LD_LIBRARY_PATH="${_NV_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
case ":${LD_LIBRARY_PATH:-}:" in
  *:/opt/miniconda3/lib:*)
    LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '^/opt/miniconda3/lib$' | paste -sd: -)"
    export LD_LIBRARY_PATH
    ;;
esac
unset _NV_LIB _d

mkdir -p "$CKPT_DIR" artifacts/lj_scales/dense_dt_campaign/logs

for f in "$CONFIG" "$DATA" "$CKPT" "$TEACHER" "$SCALE_SIDECAR"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 2; }
done
[[ -n "${CUDA_VISIBLE_DEVICES:-}" || -n "${SLURM_JOB_ID:-}" ]] || {
  echo "ERROR: submit via submit_train_lever2_on5_distill.sh (need GPU allocation)" >&2
  exit 2
}

echo "=== lever-2 on=5 DISTILL FT ==="
echo "  host/job : $(hostname) / ${SLURM_JOB_ID:-local}"
echo "  config   : $CONFIG"
echo "  warm+tch : $CKPT"
echo "  scales   : $SCALE_SIDECAR (frozen into sidecar after train)"
echo "  tag      : $TAG"
echo "  epochs=$EPOCHS lr=$LR batch=$BATCH distill_alpha=$DISTILL_ALPHA"
echo "  handoff  : mm_switch_on=5.0 (LJ scales NOT learned)"
date -Is
nvidia-smi --query-gpu=index,name,memory.free --format=csv || true
python - <<'PY'
import os, sys
import jax
devs = jax.devices()
print("JAX devices:", devs)
ok = any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devs)
if not ok:
    print("ERROR: expected CudaDevice", file=sys.stderr)
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
  --learning-rate "$LR" \
  --seed "$SEED" \
  --mm-switch-on 5.0 \
  --ml-switch-width 1.5 \
  --mm-switch-width 5.0 \
  --hybrid-mm \
  --no-learn-mm-lj-scales \
  --lr-solver mic \
  --match-checkpoint-architecture \
  --physnet-checkpoint "$CKPT" \
  --distill \
  --distill-alpha "$DISTILL_ALPHA" \
  --distill-targets energy forces dipole \
  --teacher-checkpoint "$TEACHER" \
  --early-stop-patience 8 \
  --best

RUN_DIR="$(find "$CKPT_DIR" -maxdepth 1 -type d -name "${TAG}-*" -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -1 | cut -d' ' -f2- || true)"
[[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || { echo "ERROR: no run dir for $TAG" >&2; exit 3; }
SIDECAR="$RUN_DIR/hybrid_mm.json"
[[ -f "$SIDECAR" ]] || { echo "ERROR: missing $SIDECAR" >&2; exit 3; }

# Freeze epoch222 LJ scales into the new sidecar for MD deploy.
uv run python - <<PY
import json
from pathlib import Path
src = Path("$SCALE_SIDECAR")
dst = Path("$SIDECAR")
a, b = json.loads(src.read_text()), json.loads(dst.read_text())
for k in (
    "mm_lj_sigma_scale",
    "mm_lj_epsilon_scale",
    "mm_lj_sigma_scale_bounds",
    "mm_lj_epsilon_scale_bounds",
    "mm_lj_trainable_mask",
    "mm_lj_type_frame_counts",
    "cgenff_type_names",
):
    if k in a:
        b[k] = a[k]
b["learn_mm_lj_scales"] = True  # scales present for MD loader; not re-trained here
b["mm_switch_on"] = 5.0
b["ml_switch_width"] = 1.5
b["mm_switch_width"] = 5.0
b["scales_source"] = str(src)
b["ft_recipe"] = "distill_on5_freeze_lj_lr1e-4"
dst.write_text(json.dumps(b, indent=2) + "\n")
print(f"merged LJ scales from {src} -> {dst}")
PY

echo "=== train done ==="
echo "  run_dir : $RUN_DIR"
echo "  sidecar : $SIDECAR"
echo "  next    : bash scripts/slurm/dense_dt_campaign/submit_eval_lever2_on5_distill.sh"
date -Is
