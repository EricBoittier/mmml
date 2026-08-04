#!/usr/bin/env bash
# GPU Slurm job: run the mmml asv suite and refresh the HTML report.
#
# Submit:
#   sbatch ~/mmml/benchmarks/slurm_bench_gpu.sh
#   sbatch --export=ALL,BENCH_PATTERN=bench_md_driver ~/mmml/benchmarks/slurm_bench_gpu.sh
#
# Monitor:
#   tail -f ~/tests/runs/slurm-mmml-bench-*.out
#
# Results land in $REPO_ROOT/benchmarks/results/<machine>/ — commit them to keep
# the history, since asv's regression view is only as long as what is checked in.
#
#SBATCH --job-name=mmml-bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/mmml}"
cd "${REPO_ROOT}"

# GPU: benchmarking the CPU fallback by accident is the classic way to waste a
# GPU allocation, so fail loudly instead of silently falling back.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export MMML_BENCH_X64="${MMML_BENCH_X64:-1}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-${MMML_BENCH_X64}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MMML_CKPT="${MMML_CKPT:-${REPO_ROOT}/examples/ckpts_json/DESdimers_params.json}"
# asv identifies results by machine name; without this every node writes to a
# different series and the history fragments.
export ASV_MACHINE="${ASV_MACHINE:-${SLURM_JOB_PARTITION:-gpu}-$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | awk -F= '/GRES=/{print $NF; exit}' || echo gpu)}"

echo "=== mmml asv benchmark job ==="
echo "host          : $(hostname)"
echo "repo          : ${REPO_ROOT} ($(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "asv machine   : ${ASV_MACHINE}"
echo "JAX_PLATFORMS : ${JAX_PLATFORMS}"
echo "x64           : ${MMML_BENCH_X64}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

.venv/bin/python - <<'PY'
import jax
devices = jax.devices()
print(f"jax devices   : {devices}")
if devices[0].platform != "gpu":
    raise SystemExit(
        f"refusing to burn a GPU allocation on the {devices[0].platform} backend"
    )
PY

if [[ ! -f "$HOME/.asv-machine.json" ]]; then
  .venv/bin/asv machine --yes --machine "${ASV_MACHINE}"
fi

# --set-commit-hash is required for results to be saved at all under
# environment_type=existing; see the note in benchmarks/run_bench.sh.
RUN_ARGS=(
  run
  --machine "${ASV_MACHINE}"
  --set-commit-hash "$(git rev-parse HEAD)"
)
if [[ -n "${BENCH_PATTERN:-}" ]]; then
  RUN_ARGS+=(--bench "${BENCH_PATTERN}")
fi

.venv/bin/asv "${RUN_ARGS[@]}"
.venv/bin/asv publish

echo "=== done: benchmarks/html/index.html refreshed ==="
