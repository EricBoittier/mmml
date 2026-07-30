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

# Device selection. These examples are smoke/reproducibility runs, so they
# default to CPU — otherwise a stray JAX_PLATFORMS makes results depend on
# whichever node they land on. Ask for the GPUs with:
#
#     MMML_EXAMPLE_DEVICE=gpu bash examples/m/run_all.sh
#
# An explicitly pre-set JAX_PLATFORMS / MMML_MLPOT_DEVICE still wins over
# MMML_EXAMPLE_DEVICE, so per-variable overrides keep working.
MMML_EXAMPLE_DEVICE="$(printf '%s' "${MMML_EXAMPLE_DEVICE:-cpu}" | tr '[:upper:]' '[:lower:]')"
case "${MMML_EXAMPLE_DEVICE}" in
  cpu)
    _mmml_example_jax_platforms="cpu"
    _mmml_example_mlpot_device="cpu"
    ;;
  gpu | cuda)
    MMML_EXAMPLE_DEVICE="gpu"
    _mmml_example_jax_platforms="cuda"
    _mmml_example_mlpot_device="gpu"
    ;;
  *)
    echo "examples/m: MMML_EXAMPLE_DEVICE must be 'cpu' or 'gpu' (got '${MMML_EXAMPLE_DEVICE}')" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
export MMML_EXAMPLE_DEVICE

export JAX_PLATFORMS="${JAX_PLATFORMS:-${_mmml_example_jax_platforms}}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export MMML_MLPOT_DEVICE="${MMML_MLPOT_DEVICE:-${_mmml_example_mlpot_device}}"
export MMML_JAX_WARMUP_DEVICE="${MMML_JAX_WARMUP_DEVICE:-${_mmml_example_mlpot_device}}"
unset _mmml_example_jax_platforms _mmml_example_mlpot_device

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
  # Report the *effective* device: an explicit JAX_PLATFORMS / MMML_MLPOT_DEVICE
  # outranks MMML_EXAMPLE_DEVICE, and printing the request instead of the result
  # is exactly the kind of mismatch this banner is meant to expose.
  local _effective="cpu"
  case ":${JAX_PLATFORMS}:" in
    *cuda*|*gpu*|*rocm*) _effective="gpu" ;;
  esac
  printf '  device     : %s  (JAX_PLATFORMS=%s, MMML_MLPOT_DEVICE=%s)\n' \
    "${_effective}" "${JAX_PLATFORMS}" "${MMML_MLPOT_DEVICE}"
  if [[ "${_effective}" != "${MMML_EXAMPLE_DEVICE}" ]]; then
    printf '               (MMML_EXAMPLE_DEVICE=%s was overridden by an explicit\n' \
      "${MMML_EXAMPLE_DEVICE}"
    printf '                JAX_PLATFORMS / MMML_MLPOT_DEVICE in the environment)\n'
  fi
  if [[ "${_effective}" == "cpu" ]]; then
    printf '               (CPU by default — rerun with MMML_EXAMPLE_DEVICE=gpu to use the GPUs)\n'
  elif [[ "${MMML_EXAMPLE_SKIP_DEVICE_PROBE:-0}" != "1" ]]; then
    # Asking for the GPU and silently getting CPU (CPU-only jaxlib, or a
    # CUDA-12 build on an sm_120 card) is the failure this banner exists to
    # catch. One probe per pipeline; skip with MMML_EXAMPLE_SKIP_DEVICE_PROBE=1.
    local _backend
    _backend="$(cd "${REPO_ROOT}" && uv run python -c 'import jax; print(jax.default_backend())' 2>/dev/null | tail -n 1)"
    if [[ "${_backend}" == "gpu" || "${_backend}" == "cuda" ]]; then
      printf '               (JAX backend: %s)\n' "${_backend}"
    else
      printf '  WARNING: MMML_EXAMPLE_DEVICE=gpu but JAX reports backend=%s\n' \
        "${_backend:-unknown}" >&2
      printf '           Install the CUDA build:  uv sync --extra gpu\n' >&2
      printf '           (RTX 50xx / Blackwell needs the cuda13 build.)\n' >&2
    fi
  fi
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
