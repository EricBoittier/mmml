#!/usr/bin/env bash
# Shared configuration for the liquid DCM / ACO density sweep.
#
# Source this from the numbered driver scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# Every script honours:
#   DRY_RUN=1   print commands instead of running them
#   BOX_SIZE    cubic cell side in Å (default 28)
#   SOLVENTS    space-separated residues     (default "DCM ACO")
#   FRACTIONS   space-separated ρ/ρ_bulk     (default "0.50 0.75 1.00")
#   OUT_ROOT    output root (default artifacts/liquid_density_sweep)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BOX_SIZE="${BOX_SIZE:-28}"
SOLVENTS="${SOLVENTS:-DCM ACO}"
FRACTIONS="${FRACTIONS:-0.50 0.75 1.00}"
OUT_ROOT="${OUT_ROOT:-${WORKFLOW_DIR}/artifacts}"

# Dynamics extent. Deliberately short so the matrix is a *validation* sweep;
# raise PS_PROD for production.
TEMPERATURE="${TEMPERATURE:-300}"
DT_FS="${DT_FS:-0.25}"
PS_PROD="${PS_PROD:-5}"

# Experimental bulk liquid densities at ~298 K, from
# mmml/interfaces/pycharmmInterface/mlpot/box_sizing.py::SOLVENT_BULK_PROPS.
declare -A RHO_BULK=( [DCM]=1.326 [ACO]=0.784 )
declare -A N_ATOMS=(  [DCM]=5     [ACO]=10    )

# Monomer counts per (solvent, box side, fraction), computed with
# box_sizing.n_molecules_for_target_density_in_fixed_box. Regenerate with:
#   scripts/print_density_table.py
declare -A N_MONOMERS=(
  [DCM_28_0.50]=103  [DCM_28_0.75]=155  [DCM_28_1.00]=206
  [ACO_28_0.50]=89   [ACO_28_0.75]=134  [ACO_28_1.00]=178
  [DCM_32_0.50]=154  [DCM_32_0.75]=231  [DCM_32_1.00]=308
  [ACO_32_0.50]=133  [ACO_32_0.75]=200  [ACO_32_1.00]=266
  [DCM_36_0.50]=219  [DCM_36_0.75]=329  [DCM_36_1.00]=439
  [ACO_36_0.50]=190  [ACO_36_0.75]=284  [ACO_36_1.00]=379
)

n_monomers_for() {  # solvent fraction -> count
  local key="${1}_${BOX_SIZE}_${2}"
  local n="${N_MONOMERS[$key]:-}"
  if [[ -z "$n" ]]; then
    echo "no monomer count for ${key}; add it to N_MONOMERS or use" \
         "--bulk-density-fraction (see README)" >&2
    return 1
  fi
  printf '%s' "$n"
}

# Tag used for output directories: dcm_f050_l28
cell_tag() {  # solvent fraction
  printf '%s_f%s_l%s' "$(echo "$1" | tr '[:upper:]' '[:lower:]')" \
    "$(echo "$2" | tr -d '.')" "$BOX_SIZE"
}

run_cmd() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '  ['
    printf ' %q' "$@"
    printf ' ]\n'
  else
    "$@"
  fi
}

# MLpot runs need the MPI-linked libcharmm launcher; plain `mmml` is fine for
# MM-only and for the ASE / JAX-MD backends.
MPIRUN_WRAPPER="${MPIRUN_WRAPPER:-${REPO_ROOT}/scripts/mmml-charmm-mpirun.sh}"

require_checkpoint() {
  if [[ -z "${MMML_CKPT:-}" ]]; then
    echo "MMML_CKPT is not set — export your PhysNet/SpookyNet checkpoint:" >&2
    echo "  export MMML_CKPT=/path/to/DESdimers_params.json" >&2
    return 1
  fi
}

banner() { printf '\n=== %s ===\n' "$*"; }
