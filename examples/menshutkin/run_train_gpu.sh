#!/usr/bin/env bash
# Long-range PhysNet retrain for the Menshutkin campaign. Usage: run_train_gpu.sh [gpu]
set -u
cd /mmhome/andreychev/mmml/mmml
source examples/menshutkin/_env.sh
export CUDA_VISIBLE_DEVICES="${1:-0}"
exec /mmhome/andreychev/mmml/mmml/.venv/bin/mmml physnet-train \
  --config physnet_train_longrange.yaml
