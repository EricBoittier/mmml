#!/usr/bin/env bash
# Run the mmml asv suite against the current checkout and refresh the HTML report.
#
#   bash benchmarks/run_bench.sh                    # everything
#   bash benchmarks/run_bench.sh bench_md_driver    # one module
#   bash benchmarks/run_bench.sh MDSystemSize       # one class
#
# Results accumulate under benchmarks/results/<machine>/ — one JSON per commit,
# never overwritten across commits — so repeated runs build history rather than
# replacing it. Re-running the SAME commit does replace that commit's entry;
# pass BENCH_APPEND_SAMPLES=1 to merge new samples into it instead.
#
# The HTML in benchmarks/html/ is fully regenerated from benchmarks/results/ on
# every publish, so it is safe to delete but pointless to edit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BENCH_PATTERN="${1:-}"

# Match the production MD path (examples/md_cpu/_env.sh). Timings are only
# comparable between runs that agree on this — asv records nothing about it, so
# changing it means starting a new results series, not extending the old one.
export MMML_BENCH_X64="${MMML_BENCH_X64:-1}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-${MMML_BENCH_X64}}"
# One thread per process: JAX's CPU backend and NumPy will otherwise both try to
# fill the machine and the samples turn into scheduler noise.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MMML_CKPT="${MMML_CKPT:-${REPO_ROOT}/examples/ckpts_json/DESdimers_params.json}"

ASV=(.venv/bin/asv)
if [[ ! -x "${ASV[0]}" ]]; then
  ASV=(uv run asv)
fi

if [[ ! -f "$HOME/.asv-machine.json" ]]; then
  echo "==> registering this machine with asv"
  "${ASV[@]}" machine --yes
fi

# --set-commit-hash is REQUIRED, not cosmetic: with environment_type=existing
# asv did not build the tree it is timing, so it refuses to attribute results to
# a commit unless told which one (asv/commands/run.py: skip_save). Without it the
# run prints numbers and saves nothing, and `asv publish` has no history to draw.
COMMIT_HASH="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain -- ':!benchmarks/results' ':!benchmarks/html')" ]]; then
  echo "WARNING: working tree is dirty — results will be labelled ${COMMIT_HASH:0:8}"
  echo "         but describe the tree as it is right now, not that commit."
fi

RUN_ARGS=(run --set-commit-hash "${COMMIT_HASH}")
if [[ -n "${BENCH_PATTERN}" ]]; then
  RUN_ARGS+=(--bench "${BENCH_PATTERN}")
fi
if [[ "${BENCH_APPEND_SAMPLES:-0}" == "1" ]]; then
  RUN_ARGS+=(--append-samples)
fi

echo "==> asv run  (x64=${MMML_BENCH_X64}  pattern='${BENCH_PATTERN:-all}'  commit=${COMMIT_HASH:0:8})"
"${ASV[@]}" "${RUN_ARGS[@]}"

echo "==> asv publish"
"${ASV[@]}" publish

cat <<EOF

Report written to benchmarks/html/index.html
View it with:   uv run asv preview        # serves benchmarks/html on localhost
EOF
