#!/usr/bin/env bash
# GPU Slurm job: correctness probes, then the mmml asv suite, then HTML reports.
#
# Submit:
#   sbatch ~/mmml/benchmarks/slurm_bench_gpu.sh
#   sbatch --export=ALL,BENCH_PATTERN=bench_md_driver ~/mmml/benchmarks/slurm_bench_gpu.sh
#
# Monitor:
#   tail -f ~/tests/runs/slurm-mmml-bench-*.out
#
# Opens in a browser:
#   benchmarks/html/gpu-report.html   # correctness + timing snapshot
#   benchmarks/html/index.html        # full asv graphs (also: uv run asv preview)
#
# Results JSON lands in $REPO_ROOT/benchmarks/results/<machine>/ — commit that
# to keep history. CI does not publish these numbers; asv publish is local.
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
# GPU allocation. Pin CUDA here (before JAX imports) and let gpu_bench.py
# refuse any other backend.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export MMML_BENCH_X64="${MMML_BENCH_X64:-1}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-${MMML_BENCH_X64}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MMML_CKPT="${MMML_CKPT:-${REPO_ROOT}/examples/ckpts_json/DESdimers_params.json}"
# asv identifies results by machine name; without this every node writes to a
# different series and the history fragments.
export ASV_MACHINE="${ASV_MACHINE:-${SLURM_JOB_PARTITION:-gpu}-$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | awk -F= '/GRES=/{print $NF; exit}' || echo gpu)}"

echo "=== mmml GPU asv benchmark job ==="
echo "host          : $(hostname)"
echo "repo          : ${REPO_ROOT} ($(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "asv machine   : ${ASV_MACHINE}"
echo "JAX_PLATFORMS : ${JAX_PLATFORMS}"
echo "x64           : ${MMML_BENCH_X64}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

ARGS=()
if [[ -n "${BENCH_PATTERN:-}" ]]; then
  ARGS+=(--bench "${BENCH_PATTERN}")
fi
if [[ "${BENCH_APPEND_SAMPLES:-0}" == "1" ]]; then
  ARGS+=(--append-samples)
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ -x "${PYTHON}" ]]; then
  "${PYTHON}" "${REPO_ROOT}/benchmarks/gpu_bench.py" "${ARGS[@]}"
else
  uv run python "${REPO_ROOT}/benchmarks/gpu_bench.py" "${ARGS[@]}"
fi

echo "=== done: benchmarks/html/gpu-report.html  (+ asv index.html) ==="
