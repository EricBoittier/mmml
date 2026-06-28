#!/usr/bin/env bash
# Pin a local Snakemake job to one GPU via file locks (dual-GPU nodes).
# Usage: with_local_gpu.sh command [args...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: with_local_gpu.sh command [args...]" >&2
  exit 2
fi

SLOTS="${MMML_LOCAL_GPU_SLOTS:-2}"
DIR="${MMML_GPU_LOCK_DIR:-/tmp/mmml_gpu_slots_${USER}}"
mkdir -p "$DIR"

if [[ "$SLOTS" -le 1 ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  exec "$@"
fi

# Round-robin slot id, then block on that GPU's lock (queues when both are busy).
idx="$(
  flock "$DIR/assign.lock" bash -c "
    c=\$(cat '$DIR/counter' 2>/dev/null || echo 0)
    echo \$((c % $SLOTS)) > '$DIR/counter'
    echo \$((c % $SLOTS))
  "
)"

echo "with_local_gpu: slot=${idx} CUDA_VISIBLE_DEVICES=${idx} cmd=$*" >&2
exec flock -w 86400 "$DIR/gpu${idx}.lock" env CUDA_VISIBLE_DEVICES="$idx" "$@"
