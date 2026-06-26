# Source from repo root:  source examples/md_cpu/_env.sh
# (bash or zsh). Child scripts set ROOT before sourcing when possible.

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

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export MMML_MLPOT_DEVICE="${MMML_MLPOT_DEVICE:-cpu}"
export MMML_JAX_WARMUP_DEVICE="${MMML_JAX_WARMUP_DEVICE:-cpu}"
export MMML_CKPT="${MMML_CKPT:-${REPO_ROOT}/examples/ckpts_json/DESdimers_params.json}"
export MMML_MM_NL_BACKEND="${MMML_MM_NL_BACKEND:-auto}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/md_cpu}"
mkdir -p "${ARTIFACTS_DIR}"
export ARTIFACTS_DIR

CKPT_JSON="${REPO_ROOT}/examples/ckpts_json/DESdimers_params.json"
export CKPT_JSON
