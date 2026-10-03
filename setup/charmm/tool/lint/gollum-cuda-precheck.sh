#!/usr/bin/env bash
# gollum-cuda-precheck.sh -- fast CUDA syntax pre-check for gollum (or any
# cuda=ON build host), to be run *before* the ~10 min ninja build so that
# undefined-identifier / missing-include errors in .cu / blade .cxx files
# (the MR!380 gpuCheck class) surface in seconds instead of after a full build.
#
# It just points tool/lint/charmm-lint at a cuda-enabled build's
# compile_commands.json. Because that build has cuda=ON, the .cu (and
# KEY_BLADE-gated .cxx) entries are present and get checked; on a macOS
# no-cuda build they simply aren't there.
#
# Usage:
#   tool/lint/gollum-cuda-precheck.sh [BUILD_DIR] [-- <extra charmm-lint args>]
#
#   BUILD_DIR   dir containing compile_commands.json + build.ninja
#               (default: build/cmake)
#
# Environment (forwarded to charmm-lint's CUDA backend):
#   CHARMM_LINT_CUDA_CLANG   clang++ used for -x cuda -fsyntax-only (default clang++)
#   CHARMM_LINT_CUDA_ARCH    e.g. sm_80 for the rtx6000ada nodes (default sm_70)
#   CHARMM_LINT_CUDA_PATH    --cuda-path if the toolkit is module-installed
#
# Exit status is nonzero if any CUDA/C++ file fails its syntax check.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="build/cmake"
if [[ $# -gt 0 && "$1" != "--" ]]; then
  build_dir="$1"; shift
fi
[[ "${1:-}" == "--" ]] && shift

# Default the GPU arch to the gollum rtx6000ada (Ada, sm_89) if not set.
export CHARMM_LINT_CUDA_ARCH="${CHARMM_LINT_CUDA_ARCH:-sm_89}"

echo "gollum-cuda-precheck: linting CUDA/C++ in ${build_dir} (arch ${CHARMM_LINT_CUDA_ARCH})"
exec "${here}/charmm-lint" --build-dir "${build_dir}" --lang cuda --lang c --all "$@"
