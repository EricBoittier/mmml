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
#     KARML_EXAMPLE_DEVICE=gpu bash examples/m/run_all.sh
#
# Precedence: an *explicit* KARML_EXAMPLE_DEVICE wins over an inherited
# JAX_PLATFORMS / KARML_MLPOT_DEVICE that implies a different device. A stale
# `export JAX_PLATFORMS=cpu` in a login profile used to silently downgrade a run
# that asked for the GPUs, which made KARML_EXAMPLE_DEVICE the one device knob
# that could not change the device. Setting only JAX_PLATFORMS /
# KARML_MLPOT_DEVICE still works as a per-variable override, and an inherited
# value that *agrees* with the request is kept verbatim (so
# `KARML_EXAMPLE_DEVICE=gpu JAX_PLATFORMS=cuda,cpu` keeps its cpu fallback).
#
# KARML_EXAMPLE_DEVICE_EXPLICIT is exported so nested `bash examples/m/0X_*.sh`
# steps do not mistake our own exported KARML_EXAMPLE_DEVICE for a user override.
# An already-set marker always wins, including "0": we export KARML_EXAMPLE_DEVICE
# unconditionally, so a nested step cannot tell a user request from our own
# default by looking at KARML_EXAMPLE_DEVICE alone.
if [[ -z "${KARML_EXAMPLE_DEVICE_EXPLICIT:-}" ]]; then
  if [[ -n "${KARML_EXAMPLE_DEVICE:-}" ]]; then
    KARML_EXAMPLE_DEVICE_EXPLICIT=1
  else
    KARML_EXAMPLE_DEVICE_EXPLICIT=0
  fi
fi
export KARML_EXAMPLE_DEVICE_EXPLICIT
KARML_EXAMPLE_DEVICE="$(printf '%s' "${KARML_EXAMPLE_DEVICE:-cpu}" | tr '[:upper:]' '[:lower:]')"
case "${KARML_EXAMPLE_DEVICE}" in
  cpu)
    _karml_example_jax_platforms="cpu"
    _karml_example_mlpot_device="cpu"
    ;;
  gpu | cuda)
    KARML_EXAMPLE_DEVICE="gpu"
    _karml_example_jax_platforms="cuda"
    _karml_example_mlpot_device="gpu"
    ;;
  *)
    echo "examples/m: KARML_EXAMPLE_DEVICE must be 'cpu' or 'gpu' (got '${KARML_EXAMPLE_DEVICE}')" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
export KARML_EXAMPLE_DEVICE

# Records every inherited var we discarded, so the banner can name the polluted
# environment instead of silently papering over it.
KARML_EXAMPLE_DEVICE_FORCED=""

# Clear an inherited value that contradicts an explicit KARML_EXAMPLE_DEVICE; the
# `${VAR:-default}` expansions below then fill in the requested device.
_karml_drop_conflicting_device_var() {
  # $1 = var name, $2 = device the inherited value implies ("" when unset)
  [[ "${KARML_EXAMPLE_DEVICE_EXPLICIT}" == "1" ]] || return 0
  [[ -n "${2}" && "${2}" != "${KARML_EXAMPLE_DEVICE}" ]] || return 0
  KARML_EXAMPLE_DEVICE_FORCED="${KARML_EXAMPLE_DEVICE_FORCED:+${KARML_EXAMPLE_DEVICE_FORCED} }${1}=${!1}"
  unset "${1}"
}

if [[ -n "${JAX_PLATFORMS:-}" ]]; then
  case ":${JAX_PLATFORMS}:" in
    *cuda*|*gpu*|*rocm*) _karml_inherited_platforms_device="gpu" ;;
    *) _karml_inherited_platforms_device="cpu" ;;
  esac
else
  _karml_inherited_platforms_device=""
fi
_karml_drop_conflicting_device_var JAX_PLATFORMS "${_karml_inherited_platforms_device}"
_karml_drop_conflicting_device_var KARML_MLPOT_DEVICE "${KARML_MLPOT_DEVICE:-}"
_karml_drop_conflicting_device_var KARML_JAX_WARMUP_DEVICE "${KARML_JAX_WARMUP_DEVICE:-}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-${_karml_example_jax_platforms}}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export KARML_MLPOT_DEVICE="${KARML_MLPOT_DEVICE:-${_karml_example_mlpot_device}}"
export KARML_JAX_WARMUP_DEVICE="${KARML_JAX_WARMUP_DEVICE:-${_karml_example_mlpot_device}}"
export KARML_EXAMPLE_DEVICE_FORCED
unset _karml_example_jax_platforms _karml_example_mlpot_device
unset _karml_inherited_platforms_device

# Checkpoint + dataset from commit 30eb7a01f7fcf1d42a795f188526a80e547110fd
#
# A pre-set KARML_CKPT wins on purpose (so a run can be pointed at another
# model), but that override is easy to forget about — an KARML_CKPT left in a
# login profile silently evaluates a different checkpoint than the example
# claims. Record where the value came from and surface it in the banner.
if [[ -n "${KARML_CKPT:-}" ]]; then
  KARML_CKPT_SOURCE="environment (pre-set KARML_CKPT)"
else
  KARML_CKPT="${EXAMPLE_DIR}/model_ext.json"
  KARML_CKPT_SOURCE="examples/m default"
fi
export KARML_CKPT KARML_CKPT_SOURCE

if [[ -n "${KARML_DATA:-}" ]]; then
  KARML_DATA_SOURCE="environment (pre-set KARML_DATA)"
else
  KARML_DATA="${EXAMPLE_DIR}/nh3_ch3cl_filtered.npz"
  KARML_DATA_SOURCE="examples/m default"
fi
export KARML_DATA KARML_DATA_SOURCE

# Print the resolved inputs once per pipeline. The guard is exported so nested
# `bash examples/m/0X_*.sh` steps inherit it and do not repeat the banner.
karml_example_env_banner() {
  if [[ "${KARML_EXAMPLE_ENV_BANNER_SHOWN:-0}" == "1" ]]; then
    return 0
  fi
  export KARML_EXAMPLE_ENV_BANNER_SHOWN=1
  printf 'examples/m inputs\n'
  # Report the *effective* device, never the request: a device request that did
  # not take effect is exactly the mismatch this banner exists to expose.
  local _effective="cpu"
  case ":${JAX_PLATFORMS}:" in
    *cuda*|*gpu*|*rocm*) _effective="gpu" ;;
  esac
  printf '  device     : %s  (JAX_PLATFORMS=%s, KARML_MLPOT_DEVICE=%s)\n' \
    "${_effective}" "${JAX_PLATFORMS}" "${KARML_MLPOT_DEVICE}"
  if [[ -n "${KARML_EXAMPLE_DEVICE_FORCED:-}" ]]; then
    printf '               (KARML_EXAMPLE_DEVICE=%s overrode inherited %s —\n' \
      "${KARML_EXAMPLE_DEVICE}" "${KARML_EXAMPLE_DEVICE_FORCED}"
    printf '                probably a stale export in a login profile; unset it there)\n'
  fi
  # Only reachable without an explicit KARML_EXAMPLE_DEVICE: when it is explicit
  # the forcing above guarantees JAX_PLATFORMS agrees with it, so a surviving
  # mismatch means only the per-variable knobs were set. Informational, not a
  # warning — nothing was ignored. An explicit request that cannot be honoured
  # is caught by the backend probe below instead.
  if [[ "${_effective}" != "${KARML_EXAMPLE_DEVICE}" ]]; then
    printf '               (KARML_EXAMPLE_DEVICE defaults to %s; an explicit\n' \
      "${KARML_EXAMPLE_DEVICE}"
    printf '                JAX_PLATFORMS / KARML_MLPOT_DEVICE selected %s instead)\n' \
      "${_effective}"
  fi
  if [[ "${_effective}" == "cpu" ]]; then
    if [[ "${KARML_EXAMPLE_DEVICE_EXPLICIT:-0}" != "1" ]]; then
      printf '               (CPU by default — rerun with KARML_EXAMPLE_DEVICE=gpu to use the GPUs)\n'
    fi
  elif [[ "${KARML_EXAMPLE_SKIP_DEVICE_PROBE:-0}" != "1" ]]; then
    # Asking for the GPU and silently getting CPU (CPU-only jaxlib, or a
    # CUDA-12 build on an sm_120 card) is the failure this banner exists to
    # catch. One probe per pipeline; skip with KARML_EXAMPLE_SKIP_DEVICE_PROBE=1.
    local _backend
    _backend="$(cd "${REPO_ROOT}" && uv run python -c 'import jax; print(jax.default_backend())' 2>/dev/null | tail -n 1)"
    if [[ "${_backend}" == "gpu" || "${_backend}" == "cuda" ]]; then
      printf '               (JAX backend: %s)\n' "${_backend}"
    else
      # A GPU request that silently lands on CPU wastes the entire pipeline, so
      # this goes to stderr and says so plainly rather than as a parenthetical.
      printf '  WARNING: GPU was requested but JAX reports backend=%s — this run\n' \
        "${_backend:-unknown}" >&2
      printf '           will execute on CPU (expect ~10-100x slower).\n' >&2
      printf '           Install the CUDA build:  uv sync --extra gpu\n' >&2
      printf '           (RTX 50xx / Blackwell needs the cuda13 build.)\n' >&2
      printf '           Set KARML_EXAMPLE_DEVICE=cpu to accept CPU and silence this.\n' >&2
    fi
  fi
  printf '  checkpoint : %s\n' "${KARML_CKPT}"
  printf '               (%s)\n' "${KARML_CKPT_SOURCE}"
  printf '  dataset    : %s\n' "${KARML_DATA}"
  printf '               (%s)\n' "${KARML_DATA_SOURCE}"
  if [[ ! -e "${KARML_CKPT}" ]]; then
    printf '  WARNING: checkpoint path does not exist\n' >&2
  elif [[ -d "${KARML_CKPT}" ]]; then
    printf '  note: checkpoint is a directory; the newest epoch-* run inside it is used\n'
  fi
  if [[ "${KARML_CKPT_SOURCE}" == environment* ]]; then
    printf '  note: KARML_CKPT came from the environment, not from examples/m.\n'
    printf '        Run `unset KARML_CKPT` to use %s/model_ext.json.\n' "${EXAMPLE_DIR}"
  fi
  printf '\n'
}
export KARML_CGENFF_EXTRA_RTF="${KARML_CGENFF_EXTRA_RTF:-${EXAMPLE_DIR}/top_ch3cl.rtf}"
export KARML_CGENFF_EXTRA_PRM="${KARML_CGENFF_EXTRA_PRM:-${EXAMPLE_DIR}/par_ch3cl.prm}"
# Vacuum/all-ML jax_mic empties CHARMM nonbond lists — use JAX MM pairs.
export KARML_MM_PAIR_SOURCE="${KARML_MM_PAIR_SOURCE:-jax}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/nh3_ch3cl}"
mkdir -p "${ARTIFACTS_DIR}"
export ARTIFACTS_DIR

# Default composition: ammonia + chloromethane (needs EXTRA_RTF for CH3CL)
export KARML_COMPOSITION="${KARML_COMPOSITION:-AMM1:1,CH3CL:1}"
export KARML_SPACING="${KARML_SPACING:-4.0}"
