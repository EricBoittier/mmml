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
#
# A pre-set MMML_CKPT wins on purpose (so a run can be pointed at another
# model), but that override is easy to forget about — an MMML_CKPT left in a
# login profile silently evaluates a different checkpoint than the example
# claims. Record where the value came from and surface it in the banner.
if [[ -n "${MMML_CKPT:-}" ]]; then
  MMML_CKPT_SOURCE="environment (pre-set MMML_CKPT)"
else
  MMML_CKPT="${EXAMPLE_DIR}/model_ext.json"
  MMML_CKPT_SOURCE="examples/m default"
fi
export MMML_CKPT MMML_CKPT_SOURCE

if [[ -n "${MMML_DATA:-}" ]]; then
  MMML_DATA_SOURCE="environment (pre-set MMML_DATA)"
else
  MMML_DATA="${EXAMPLE_DIR}/nh3_ch3cl_filtered.npz"
  MMML_DATA_SOURCE="examples/m default"
fi
export MMML_DATA MMML_DATA_SOURCE

# Print the resolved inputs once per pipeline. The guard is exported so nested
# `bash examples/m/0X_*.sh` steps inherit it and do not repeat the banner.
mmml_example_env_banner() {
  if [[ "${MMML_EXAMPLE_ENV_BANNER_SHOWN:-0}" == "1" ]]; then
    return 0
  fi
  export MMML_EXAMPLE_ENV_BANNER_SHOWN=1
  printf 'examples/m inputs\n'
  printf '  checkpoint : %s\n' "${MMML_CKPT}"
  printf '               (%s)\n' "${MMML_CKPT_SOURCE}"
  printf '  dataset    : %s\n' "${MMML_DATA}"
  printf '               (%s)\n' "${MMML_DATA_SOURCE}"
  if [[ ! -e "${MMML_CKPT}" ]]; then
    printf '  WARNING: checkpoint path does not exist\n' >&2
  elif [[ -d "${MMML_CKPT}" ]]; then
    printf '  note: checkpoint is a directory; the newest epoch-* run inside it is used\n'
  fi
  if [[ "${MMML_CKPT_SOURCE}" == environment* ]]; then
    printf '  note: MMML_CKPT came from the environment, not from examples/m.\n'
    printf '        Run `unset MMML_CKPT` to use %s/model_ext.json.\n' "${EXAMPLE_DIR}"
  fi
  printf '\n'
}
export MMML_CGENFF_EXTRA_RTF="${MMML_CGENFF_EXTRA_RTF:-${EXAMPLE_DIR}/top_ch3cl.rtf}"
export MMML_CGENFF_EXTRA_PRM="${MMML_CGENFF_EXTRA_PRM:-${EXAMPLE_DIR}/par_ch3cl.prm}"
# Vacuum/all-ML jax_mic empties CHARMM nonbond lists — use JAX MM pairs.
export MMML_MM_PAIR_SOURCE="${MMML_MM_PAIR_SOURCE:-jax}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/nh3_ch3cl}"
mkdir -p "${ARTIFACTS_DIR}"
export ARTIFACTS_DIR

# Default composition: ammonia + chloromethane (needs EXTRA_RTF for CH3CL)
export MMML_COMPOSITION="${MMML_COMPOSITION:-AMM1:1,CH3CL:1}"
export MMML_SPACING="${MMML_SPACING:-4.0}"
