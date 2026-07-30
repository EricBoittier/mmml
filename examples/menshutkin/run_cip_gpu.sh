#!/usr/bin/env bash
# Solvated PMF from reactants through the CIP to the solvent-separated ion pair.
#   run_cip_gpu.sh <solvent> <gpu> [extra args]
#
# Uses model_ext.json, the same converged checkpoint as the gas-phase run, so
# the two profiles are directly comparable. The long-range retrain
# (ckpts/menshutkin_longrange) is NOT the default: it was stopped at epoch 1436
# of 5000 and still has off-path holes -- a minimiser exploring a solvated box
# found -1171 eV where this model gives -526. Set MENSH_LONGRANGE=1 to use its
# newest epoch instead, or MENSH_CKPT_FORCE to pin any checkpoint.
set -u
SOLVENT="${1:-water}"; GPU="${2:-1}"; shift 2 || true
cd /mmhome/andreychev/mmml/mmml
source examples/menshutkin/_env.sh

if [ -n "${MENSH_CKPT_FORCE:-}" ]; then
  export MENSH_CKPT="$MENSH_CKPT_FORCE"
elif [ "${MENSH_LONGRANGE:-0}" = "1" ]; then
  CKPT_ROOT=/mmhome/andreychev/mmml/mmml/ckpts/menshutkin_longrange
  RUN_DIR=$(ls -d "$CKPT_ROOT"/longrange-* 2>/dev/null | tail -1)
  LATEST=$(ls "$RUN_DIR" | sed -n 's/^epoch-\([0-9]*\)$/\1/p' | sort -n | tail -1)
  export MENSH_CKPT="$RUN_DIR/epoch-$LATEST"
fi
if [ ! -e "$MENSH_CKPT" ]; then
  echo "no checkpoint at $MENSH_CKPT" >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES=$GPU
echo "model: $MENSH_CKPT"

exec /mmhome/andreychev/mmml/mmml/.venv/bin/python -u \
  examples/menshutkin/07_solvated_pmf.py \
  --solvent "$SOLVENT" \
  --xi-min -1.3 --xi-max 6.0 --fine 0.15 --fine-to 2.0 --coarse 0.25 \
  --dt-fs 0.25 --minimize-dt-fs 0.10 \
  --equil-ps 1.0 --prod-ps 2.0 --record-every 10 \
  --output-dir /mmhome/andreychev/mmml/mmml/artifacts/menshutkin/pmf_cip/"$SOLVENT" "$@"
