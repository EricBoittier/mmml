# Source from repo root:  source examples/md_system_from_pdb/_env.sh
# Reuses the CPU MD env (JAX CPU, bundled DESdimers checkpoint).

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
else
  _ENV_DIR="$(cd "$(dirname "$0")" && pwd)"
  REPO_ROOT="$(cd "${_ENV_DIR}/../.." && pwd)"
fi
export REPO_ROOT

# shellcheck source=/dev/null
source "${REPO_ROOT}/examples/md_cpu/_env.sh"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/md_system_from_pdb}"
mkdir -p "${ARTIFACTS_DIR}"
export ARTIFACTS_DIR

# CGenFF-named acetone monomer (single residue). Safe for --from-pdb smoke
# and as a Packmol template via --composition "${PDB}:N".
PDB_MONOMER="${PDB_MONOMER:-${REPO_ROOT}/mmml/generate/sample/pdb/aco_monomer.pdb}"
export PDB_MONOMER

# Optional: certified liquid-box (PSF/CRD). Override if you have a local box.
CERTIFIED_BOX_DIR="${CERTIFIED_BOX_DIR:-}"
if [[ -z "${CERTIFIED_BOX_DIR}" ]]; then
  _TUT="${REPO_ROOT}/../mmml_tutorial/example_systems/acodcm/boxes/dcm206"
  if [[ -f "${_TUT}/model.psf" && -f "${_TUT}/model.crd" ]]; then
    CERTIFIED_BOX_DIR="${_TUT}"
  fi
fi
export CERTIFIED_BOX_DIR
