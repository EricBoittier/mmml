#!/usr/bin/env bash
# Long-range Coulomb smokes: all lr_solver options (TIP3 full matrix + ACN/DMSO subsets).
# Optional libraries (jax-pme, nvalchemiops, ScaFaCoS) are skipped when missing.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

RUN_PYCHARMM="${RUN_PYCHARMM:-1}"
RUN_TIP3_FULL="${RUN_TIP3_FULL:-1}"
RUN_ACN="${RUN_ACN:-1}"
RUN_DMSO="${RUN_DMSO:-1}"
# Core jobs always attempted; optional LR backends gated below.
CORE_JOBS="${CORE_JOBS:-mic_ase mic_jaxmd mic_pycharmm ewald_ase ewald_jaxmd ewald_pycharmm ewald_omit_self_ase ewald_omit_self_jaxmd ewald_omit_self_pycharmm}"
JAX_PME_JOBS="${JAX_PME_JOBS:-jax_pme_ewald_ase jax_pme_ewald_jaxmd jax_pme_ewald_pycharmm jax_pme_pme_pycharmm jax_pme_p3m_pycharmm}"
PE_CORE_JOBS="${PE_CORE_JOBS:-pe_ewald_pycharmm pe_ewald_coulomb_only_pycharmm}"
PE_OPT_JOBS="${PE_OPT_JOBS:-pe_nvalchemiops_pycharmm pe_scafacos_pycharmm}"

has_pycharmm=0
if uv run python -c "import pycharmm" >/dev/null 2>&1; then
  has_pycharmm=1
fi

has_jax_pme=0
if uv run python -c "import jax_pme" >/dev/null 2>&1; then
  has_jax_pme=1
fi

has_nval=0
if uv run python -c "import nvalchemiops" >/dev/null 2>&1; then
  has_nval=1
fi

has_scafacos=0
if [[ -n "${SCAFACOS_LIB:-}" && -e "${SCAFACOS_LIB}" ]]; then
  has_scafacos=1
fi

run_job() {
  local cfg="$1"
  local job="$2"
  local need_charmm="${3:-0}"
  if [[ "${need_charmm}" == "1" && "${has_pycharmm}" != "1" ]]; then
    echo "SKIP (no PyCHARMM): ${cfg} --job-id ${job}"
    return 0
  fi
  if [[ "${need_charmm}" == "1" && "${RUN_PYCHARMM}" != "1" ]]; then
    echo "SKIP (RUN_PYCHARMM=0): ${cfg} --job-id ${job}"
    return 0
  fi
  echo "=== ${cfg} --job-id ${job} ==="
  if ! uv run mmml md-system --config "${cfg}" --job-id "${job}"; then
    echo "FAIL: ${job} (continuing)"
    return 1
  fi
  return 0
}

failed=0

run_tip3_matrix() {
  local cfg="examples/m/yaml/ewald_all_tip3.yaml"
  local j
  for j in ${CORE_JOBS}; do
    need=0
    [[ "${j}" == *pycharmm* ]] && need=1
    run_job "${cfg}" "${j}" "${need}" || failed=1
  done
  if [[ "${has_jax_pme}" == "1" ]]; then
    for j in ${JAX_PME_JOBS}; do
      need=0
      [[ "${j}" == *pycharmm* ]] && need=1
      run_job "${cfg}" "${j}" "${need}" || failed=1
    done
  else
    echo "SKIP jax-pme jobs (import jax_pme failed)"
  fi
  for j in ${PE_CORE_JOBS}; do
    run_job "${cfg}" "${j}" 1 || failed=1
  done
  if [[ "${has_nval}" == "1" ]]; then
    run_job "${cfg}" pe_nvalchemiops_pycharmm 1 || failed=1
  else
    echo "SKIP pe_nvalchemiops_pycharmm (nvalchemiops not importable)"
  fi
  if [[ "${has_scafacos}" == "1" ]]; then
    run_job "${cfg}" pe_scafacos_pycharmm 1 || failed=1
  else
    echo "SKIP pe_scafacos_pycharmm (set SCAFACOS_LIB to libfcs.so)"
  fi
}

run_solvent_subset() {
  local cfg="$1"
  local j
  for j in mic_pycharmm ewald_ase ewald_jaxmd ewald_pycharmm ewald_omit_self_pycharmm pe_ewald_pycharmm; do
    need=0
    [[ "${j}" == *pycharmm* ]] && need=1
    run_job "${cfg}" "${j}" "${need}" || failed=1
  done
  if [[ "${has_jax_pme}" == "1" ]]; then
    run_job "${cfg}" jax_pme_ewald_pycharmm 1 || failed=1
  else
    echo "SKIP jax_pme_ewald_pycharmm on $(basename "${cfg}")"
  fi
}

if [[ "${RUN_TIP3_FULL}" == "1" ]]; then
  run_tip3_matrix
fi
if [[ "${RUN_ACN}" == "1" ]]; then
  run_solvent_subset examples/m/yaml/ewald_all_acn.yaml
fi
if [[ "${RUN_DMSO}" == "1" ]]; then
  run_solvent_subset examples/m/yaml/ewald_all_dmso.yaml
fi

if [[ "${failed}" != "0" ]]; then
  echo "FAIL: one or more ewald smokes failed (see above)"
  exit 1
fi
echo "PASS: ewald LR smokes (artifacts under ${ARTIFACTS_DIR}/ewald_all_*)"
