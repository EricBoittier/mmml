# Source from repo root:  source examples/dcm_crystal/_env.sh
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

EXAMPLE_DIR="${REPO_ROOT}/examples/dcm_crystal"
export EXAMPLE_DIR

# A 20-atom unit cell. Every step is seconds of numpy plus a few JAX reductions,
# so CPU is the default and there is no MD anywhere in this ladder.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

# Which deposited structure to work from: pbcn_133gpa or pbcn_163gpa.
# The 1.33 GPa point is the default because it is the less compressed of the
# two, and so the shorter extrapolation when step 05 relaxes to zero pressure.
export DCM_PHASE="${DCM_PHASE:-pbcn_133gpa}"

# Real-space cutoff (A) for the LJ lattice sum and the Ewald split. Step 04
# demonstrates convergence rather than asking you to take 12 A on faith.
export DCM_CUTOFF="${DCM_CUTOFF:-12.0}"

# Temperature for the -2RT term in dH_sub. The default is the melting point the
# experimental dH_fus was measured at, which is where the reference cycle is
# anchored; there is no crystal above it.
export DCM_TEMPERATURE="${DCM_TEMPERATURE:-178.2}"

# Optional: a hybrid_mm.json carrying learned per-type LJ scales, so step 05 can
# re-evaluate the crystal under trained parameters. See examples/lj_scales.
export DCM_SCALES="${DCM_SCALES:-}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/dcm_crystal}"
mkdir -p "${ARTIFACTS_DIR}"
export ARTIFACTS_DIR

dcm_crystal_banner() {
  [[ "${DCM_BANNER_SHOWN:-0}" == "1" ]] && return 0
  export DCM_BANNER_SHOWN=1
  printf 'examples/dcm_crystal inputs\n'
  printf '  phase       : %s\n' "${DCM_PHASE}"
  printf '  cutoff      : %s A\n' "${DCM_CUTOFF}"
  printf '  temperature : %s K\n' "${DCM_TEMPERATURE}"
  printf '  artifacts   : %s\n' "${ARTIFACTS_DIR}"
  if [[ -n "${DCM_SCALES}" ]]; then
    printf '  LJ scales   : %s\n' "${DCM_SCALES}"
    [[ -f "${DCM_SCALES}" ]] || printf '                WARNING: does not exist\n'
  fi
  printf '\n'
}
