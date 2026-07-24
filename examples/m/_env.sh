# Source from repo root:  source examples/m/_env.sh

_repo_has_pyproject() {
  [[ -f "${1}/pyproject.toml" ]]
}

if _repo_has_pyproject "${ROOT:-}"; then
  REPO_ROOT="$(cd "${ROOT}" && pwd)"
elif _repo_has_pyproject "${REPO_ROOT:-}"; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
elif [[ -n "${BASH_VERSION:-}" ]]; then
  _ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_ENV_DIR}/../.." && pwd)"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _ENV_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"
  REPO_ROOT="$(cd "${_ENV_DIR}/../.." && pwd)"
else
  _ENV_DIR="$(cd "$(dirname "$0")" && pwd)"
  REPO_ROOT="$(cd "${_ENV_DIR}/../.." && pwd)"
fi
export REPO_ROOT

EXAMPLE_DIR="${REPO_ROOT}/examples/m"
export EXAMPLE_DIR

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export MMML_MLPOT_DEVICE="${MMML_MLPOT_DEVICE:-cpu}"
export MMML_JAX_WARMUP_DEVICE="${MMML_JAX_WARMUP_DEVICE:-cpu}"

# Checkpoint + dataset from commit 30eb7a01f7fcf1d42a795f188526a80e547110fd
export MMML_CKPT="${MMML_CKPT:-${EXAMPLE_DIR}/kl.json}"
export MMML_DATA="${MMML_DATA:-${EXAMPLE_DIR}/nh3_ch3cl_filtered.npz}"
export MMML_CGENFF_EXTRA_RTF="${MMML_CGENFF_EXTRA_RTF:-${EXAMPLE_DIR}/top_ch3cl.rtf}"
export MMML_CGENFF_EXTRA_PRM="${MMML_CGENFF_EXTRA_PRM:-${EXAMPLE_DIR}/par_ch3cl.prm}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/nh3_ch3cl}"
mkdir -p "${ARTIFACTS_DIR}"
export ARTIFACTS_DIR

# Default composition: ammonia + chloromethane (needs EXTRA_RTF for CH3CL)
export MMML_COMPOSITION="${MMML_COMPOSITION:-AMM1:1,CH3CL:1}"
export MMML_SPACING="${MMML_SPACING:-4.0}"
