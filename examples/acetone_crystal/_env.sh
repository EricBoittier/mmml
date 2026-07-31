# Source from repo root:  source examples/acetone_crystal/_env.sh
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

EXAMPLE_DIR="${REPO_ROOT}/examples/acetone_crystal"
export EXAMPLE_DIR

# Every step here is a few seconds of numpy and a handful of JAX reductions on a
# 160-atom cell, so CPU is the default and nothing depends on which node you
# land on. There is no MD in this ladder.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

# Which published structure to work with. The five phases of Allan et al. are
# pbca_5k, pbca_110k, pbca_150k, cmcm_160k, cmcm_15kbar.
#
# pbca_150k is the default: it is the stable low-temperature phase, refined from
# single-crystal X-ray data with ordered hydrogens.
export ACO_PHASE="${ACO_PHASE:-pbca_150k}"

# Real-space cutoff (A) for the LJ lattice sum and the Ewald split. 12 A is
# converged to under 0.01 kcal/mol per molecule; step 04 demonstrates that
# rather than asking you to take it on faith.
export ACO_CUTOFF="${ACO_CUTOFF:-12.0}"

# Supercell repeats for step 03. The default keeps the unit cell as deposited,
# which is all the lattice energy needs; a supercell is only for visualisation
# or for handing coordinates to something else.
export ACO_SUPERCELL="${ACO_SUPERCELL:-1,1,1}"

# Optional: a hybrid_mm.json carrying learned per-type LJ scales, so step 05 can
# re-evaluate the crystal under trained parameters. See examples/lj_scales.
export ACO_SCALES="${ACO_SCALES:-}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/acetone_crystal}"
mkdir -p "${ARTIFACTS_DIR}"
export ARTIFACTS_DIR

aco_crystal_banner() {
  [[ "${ACO_BANNER_SHOWN:-0}" == "1" ]] && return 0
  export ACO_BANNER_SHOWN=1
  printf 'examples/acetone_crystal inputs\n'
  printf '  phase     : %s\n' "${ACO_PHASE}"
  printf '  cutoff    : %s A\n' "${ACO_CUTOFF}"
  printf '  artifacts : %s\n' "${ARTIFACTS_DIR}"
  if [[ -n "${ACO_SCALES}" ]]; then
    printf '  LJ scales : %s\n' "${ACO_SCALES}"
    [[ -f "${ACO_SCALES}" ]] || printf '              WARNING: does not exist\n'
  fi
  printf '\n'
}
