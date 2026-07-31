# Source from repo root:  source examples/lj_scales/_env.sh
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

EXAMPLE_DIR="${REPO_ROOT}/examples/lj_scales"
export EXAMPLE_DIR

# Device. Steps 00-04 are tiny and run fine on CPU, so that is the default and
# results do not depend on which node you land on. Step 05 (real training) and
# step 07 (condensed-phase MD, thousands of hybrid force evaluations) want a GPU:
#
#     LJ_DEVICE=gpu bash examples/lj_scales/05_train.sh
#     LJ_DEVICE=gpu bash examples/lj_scales/07_deploy_md.sh
#
# An explicitly pre-set JAX_PLATFORMS / MMML_MLPOT_DEVICE is honoured only when
# LJ_DEVICE was NOT given: a stale `export JAX_PLATFORMS=cpu` in a login profile
# must not silently downgrade a run that asked for the GPU. Same precedence rule
# as examples/m/_env.sh.
if [[ -n "${LJ_DEVICE:-}" ]]; then
  case "$(printf '%s' "${LJ_DEVICE}" | tr '[:upper:]' '[:lower:]')" in
    cpu) _lj_platforms="cpu"; _lj_mlpot="cpu" ;;
    gpu|cuda) _lj_platforms="cuda"; _lj_mlpot="gpu" ;;
    *) echo "examples/lj_scales: LJ_DEVICE must be 'cpu' or 'gpu' (got '${LJ_DEVICE}')" >&2
       return 1 2>/dev/null || exit 1 ;;
  esac
  export JAX_PLATFORMS="${_lj_platforms}"
  export MMML_MLPOT_DEVICE="${_lj_mlpot}"
  export MMML_JAX_WARMUP_DEVICE="${_lj_mlpot}"
  unset _lj_platforms _lj_mlpot
else
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
  export MMML_MLPOT_DEVICE="${MMML_MLPOT_DEVICE:-cpu}"
  export MMML_JAX_WARMUP_DEVICE="${MMML_JAX_WARMUP_DEVICE:-cpu}"
  # Those three inherit independently, so they can end up disagreeing: sourcing
  # examples/m/_env.sh earlier in the same shell leaves MMML_MLPOT_DEVICE=gpu
  # behind while JAX_PLATFORMS falls back to cpu here. JAX_PLATFORMS is the hard
  # gate — at cpu, JAX never enumerates the GPU — so the gpu half is a lie that
  # still reads as a GPU run downstream. Make the pair agree, loudly.
  case ":${JAX_PLATFORMS}:" in
    *cuda*|*gpu*|*rocm*) _lj_eff="gpu" ;;
    *) _lj_eff="cpu" ;;
  esac
  if [[ "${MMML_MLPOT_DEVICE}" != "${_lj_eff}" ]]; then
    printf 'examples/lj_scales: inherited MMML_MLPOT_DEVICE=%s conflicts with JAX_PLATFORMS=%s; forcing %s (use LJ_DEVICE=gpu to run on the GPU)\n' \
      "${MMML_MLPOT_DEVICE}" "${JAX_PLATFORMS}" "${_lj_eff}" >&2
    export MMML_MLPOT_DEVICE="${_lj_eff}"
    export MMML_JAX_WARMUP_DEVICE="${_lj_eff}"
  fi
  unset _lj_eff
fi
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

# Input QM data. MUST be PSF-ordered — 02_inspect_dataset.py checks this, and
# getting it wrong mis-assigns CGenFF types silently rather than crashing.
export LJ_DATASET="${LJ_DATASET:-${REPO_ROOT}/examples/dcm_mp2_psf_order.npz}"

# Outputs. Namespaced, and deliberately NOT inherited from a bare
# ARTIFACTS_DIR: examples/m, examples/acetone_crystal and others export that
# same generic name from their own _env.sh, so sourcing one of them earlier in
# the shell used to redirect this ladder's dataset and checkpoints into that
# study's folder without a word. This ladder does not export ARTIFACTS_DIR
# either, so it cannot capture those examples in return.
_LJ_INHERITED_ARTIFACTS_DIR="${ARTIFACTS_DIR:-}"
LJ_ARTIFACTS_DIR="${LJ_ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/lj_scales}"
mkdir -p "${LJ_ARTIFACTS_DIR}"
export LJ_ARTIFACTS_DIR

export LJ_ENRICHED="${LJ_ENRICHED:-${LJ_ARTIFACTS_DIR}/dataset_cgenff.npz}"
export LJ_CKPT_DIR="${LJ_CKPT_DIR:-${LJ_ARTIFACTS_DIR}/ckpts}"
export LJ_TAG="${LJ_TAG:-hybrid_mm_fixed_lj_scales}"
export LJ_EPOCHS="${LJ_EPOCHS:-500}"
export LJ_NTRAIN="${LJ_NTRAIN:-8000}"
export LJ_NVALID="${LJ_NVALID:-1000}"

# Newest file matching the given `find` expression under a directory, by mtime.
# Empty (exit 0) when the directory or the match is missing.
#
# `find ... | head -1` is wrong here and bit us once: runs accumulate under
# LJ_CKPT_DIR, traversal order is arbitrary, and a failed earlier run left a
# hybrid_mm.json that step 07 then tried to deploy.
lj_newest_file() {
  local dir="${1}"
  shift
  [[ -d "${dir}" ]] || return 0
  find "${dir}" "$@" -printf '%T@\t%p\n' 2>/dev/null | sort -rn | head -1 | cut -f2-
}

lj_scales_banner() {
  [[ "${LJ_BANNER_SHOWN:-0}" == "1" ]] && return 0
  export LJ_BANNER_SHOWN=1
  local _eff="cpu"
  case ":${JAX_PLATFORMS}:" in *cuda*|*gpu*|*rocm*) _eff="gpu" ;; esac
  printf 'examples/lj_scales inputs\n'
  printf '  device    : %s  (JAX_PLATFORMS=%s, MMML_MLPOT_DEVICE=%s)\n' \
    "${_eff}" "${JAX_PLATFORMS}" "${MMML_MLPOT_DEVICE}"
  [[ "${_eff}" == "cpu" ]] && \
    printf '              (CPU by default — LJ_DEVICE=gpu for steps 05 and 07)\n'
  printf '  dataset   : %s\n' "${LJ_DATASET}"
  [[ -f "${LJ_DATASET}" ]] || printf '              WARNING: does not exist\n'
  printf '  artifacts : %s\n' "${LJ_ARTIFACTS_DIR}"
  if [[ -n "${_LJ_INHERITED_ARTIFACTS_DIR}" \
        && "${_LJ_INHERITED_ARTIFACTS_DIR}" != "${LJ_ARTIFACTS_DIR}" ]]; then
    printf '              (ignoring inherited ARTIFACTS_DIR=%s —\n' \
      "${_LJ_INHERITED_ARTIFACTS_DIR}"
    printf '               export LJ_ARTIFACTS_DIR to redirect this ladder)\n'
  fi
  printf '\n'
}
