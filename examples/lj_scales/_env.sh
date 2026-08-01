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

# PyCHARMM's libcharmm.so links OpenCL. The uv venv often lacks libOpenCL.so.1
# even when a conda env on the same machine has it — prepend the first hit.
_lj_prepend_opencl_lib() {
  local cand
  for cand in \
    "${CONDA_PREFIX:-}/lib" \
    "${CONDA_PREFIX:-}/targets/x86_64-linux/lib" \
    "${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}/envs/mmml-full/targets/x86_64-linux/lib" \
    "${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}/envs/mmml-full/lib" \
    "/mmhome/boittier/home/micromamba/envs/mmml-full/targets/x86_64-linux/lib" \
    "/mmhome/boittier/home/micromamba/envs/mmml-full/lib"
  do
    if [[ -n "${cand}" && -e "${cand}/libOpenCL.so.1" ]]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":${cand}:"*) ;;
        *) export LD_LIBRARY_PATH="${cand}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
      esac
      return 0
    fi
  done
  return 0
}
_lj_prepend_opencl_lib
unset -f _lj_prepend_opencl_lib

# Input QM data. MUST be PSF-ordered — 02_inspect_dataset.py checks this, and
# getting it wrong mis-assigns CGenFF types silently rather than crashing.
#
# LJ_JOINT=1 switches the ladder onto the ACO+DCM RI-MP2 joint path (steps
# 08–11 + multi-backend liquid campaign). Defaults below stay DCM-only unless
# that flag is set, so existing DCM runs are undisturbed.
export LJ_JOINT="${LJ_JOINT:-0}"

# LJ_DES=1 switches onto the SO3LR/DES dimer set (step 12). Much broader
# chemistry than the ACO/DCM paths -- 83 CGenFF residues and 90 LJ types are
# reachable -- but the tail is thin, so the ladder trains on the best-sampled
# LJ_DES_TOP_RESIDUES of them by default. See docs/des-so3lr-dimers.md.
export LJ_DES="${LJ_DES:-0}"
if [[ "${LJ_DES}" == "1" && "${LJ_JOINT}" == "1" ]]; then
  echo "examples/lj_scales: LJ_DES=1 and LJ_JOINT=1 are mutually exclusive" >&2
  return 1 2>/dev/null || exit 1
fi

# Outputs. Namespaced, and deliberately NOT inherited from a bare
# ARTIFACTS_DIR: examples/m, examples/acetone_crystal and others export that
# same generic name from their own _env.sh, so sourcing one of them earlier in
# the shell used to redirect this ladder's dataset and checkpoints into that
# study's folder without a word. This ladder does not export ARTIFACTS_DIR
# either, so it cannot capture those examples in return.
_LJ_INHERITED_ARTIFACTS_DIR="${ARTIFACTS_DIR:-}"
if [[ "${LJ_JOINT}" == "1" ]]; then
  LJ_ARTIFACTS_DIR="${LJ_ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/lj_scales_joint}"
elif [[ "${LJ_DES}" == "1" ]]; then
  LJ_ARTIFACTS_DIR="${LJ_ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/lj_scales_des}"
else
  LJ_ARTIFACTS_DIR="${LJ_ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/lj_scales}"
fi
mkdir -p "${LJ_ARTIFACTS_DIR}"
export LJ_ARTIFACTS_DIR

if [[ "${LJ_DES}" == "1" ]]; then
  # Source HDF5 lives on scicore; override LJ_DES_H5 elsewhere.
  export LJ_DES_H5="${LJ_DES_H5:-${HOME}/qcell/qcell_dimers.h5}"
  # 34 is the measured DES dimer maximum -- not the ladder's usual 20.
  export LJ_DES_PAD="${LJ_DES_PAD:-34}"
  # Residue cut. 50 keeps 79% of typeable frames and 69 LJ types while holding
  # every one of them above ~1,400 frames -- the same floor as a 40-residue cut,
  # for 10 points more data and 12 more types. At 60 the floor drops to ~1,100
  # and by 94 (everything) to 140, which is where thin types start drifting.
  export LJ_DES_TOP_RESIDUES="${LJ_DES_TOP_RESIDUES:-50}"
  # Net charge filter for the h5 -> NPZ step. Neutral-only by default, matching
  # ~/trainDES/train.py. The ion residues merged from toppar_water_ions.str
  # (CLA/SOD/POT/LIT/CAL/MG) only ever appear in net-charged dimers, so
  # LJ_DES_ALL_CHARGES=1 is what actually admits them -- a separate decision
  # from having the templates available.
  export LJ_DES_ALL_CHARGES="${LJ_DES_ALL_CHARGES:-0}"
  export LJ_DES_RAW="${LJ_DES_RAW:-${LJ_ARTIFACTS_DIR}/des_dimers_raw.npz}"
  export LJ_DES_ALL="${LJ_DES_ALL:-${LJ_ARTIFACTS_DIR}/des_dimers_cgenff_all.npz}"
  export LJ_DATASET="${LJ_DATASET:-${LJ_DES_RAW}}"
  export LJ_ENRICHED="${LJ_ENRICHED:-${LJ_ARTIFACTS_DIR}/des_dimers_cgenff_top${LJ_DES_TOP_RESIDUES}.npz}"
  export LJ_TAG="${LJ_TAG:-hybrid_mm_fixed_lj_scales_des}"
elif [[ "${LJ_JOINT}" == "1" ]]; then
  export LJ_DATASET="${LJ_DATASET:-${LJ_ARTIFACTS_DIR}/joint_rimp2_merged.npz}"
  export LJ_ENRICHED="${LJ_ENRICHED:-${LJ_ARTIFACTS_DIR}/joint_rimp2_cgenff.npz}"
  export LJ_TAG="${LJ_TAG:-hybrid_mm_fixed_lj_scales_aco_dcm}"
else
  export LJ_DATASET="${LJ_DATASET:-${REPO_ROOT}/examples/dcm_mp2_psf_order.npz}"
  export LJ_ENRICHED="${LJ_ENRICHED:-${LJ_ARTIFACTS_DIR}/dataset_cgenff.npz}"
  export LJ_TAG="${LJ_TAG:-hybrid_mm_fixed_lj_scales}"
fi
export LJ_CKPT_DIR="${LJ_CKPT_DIR:-${LJ_ARTIFACTS_DIR}/ckpts}"
export LJ_EPOCHS="${LJ_EPOCHS:-500}"
export LJ_NTRAIN="${LJ_NTRAIN:-8000}"
export LJ_NVALID="${LJ_NVALID:-1000}"

# --- Joint geometry / ORCA / liquid-box (LJ_JOINT=1) ------------------------
# Source monomers with res_name + CGenFF (not the DCM-only PSF NPZ).
export LJ_GEOM_SOURCE="${LJ_GEOM_SOURCE:-${REPO_ROOT}/examples/mp2_nms15_train.npz}"
export LJ_JOINT_GEOMS="${LJ_JOINT_GEOMS:-${LJ_ARTIFACTS_DIR}/joint_geoms.npz}"
export LJ_ORCA_RUN_DIR="${LJ_ORCA_RUN_DIR:-${LJ_ARTIFACTS_DIR}/orca_rimp2}"
# collect_orca_array writes ${base}_{train,valid,test}.npz; base is this path
# without .npz (we pass --out with .npz and use the train split / merge).
export LJ_JOINT_LABELED_BASE="${LJ_JOINT_LABELED_BASE:-${LJ_ARTIFACTS_DIR}/joint_rimp2}"
export LJ_JOINT_LABELED="${LJ_JOINT_LABELED:-${LJ_JOINT_LABELED_BASE}_train.npz}"
export LJ_JOINT_MERGED="${LJ_JOINT_MERGED:-${LJ_ARTIFACTS_DIR}/joint_rimp2_merged.npz}"
export LJ_PAD_TO="${LJ_PAD_TO:-20}"

# Thermal NMS (required >= 2). Not isotropic noise.
export LJ_NMS_CONFORMERS="${LJ_NMS_CONFORMERS:-64}"
export LJ_NMS_TEMPERATURE="${LJ_NMS_TEMPERATURE:-300}"
export LJ_NMS_FREQ_MIN="${LJ_NMS_FREQ_MIN:-200}"
export LJ_N_DIRECTIONS="${LJ_N_DIRECTIONS:-6}"
export LJ_N_ORIENTATIONS="${LJ_N_ORIENTATIONS:-12}"
export LJ_N_R="${LJ_N_R:-20}"
export LJ_R_MIN="${LJ_R_MIN:-2.8}"
export LJ_R_MAX="${LJ_R_MAX:-10.0}"
export LJ_R_DENSE_TO="${LJ_R_DENSE_TO:-6.0}"
export LJ_GEOM_SEED="${LJ_GEOM_SEED:-0}"

export LJ_ORCA_MODE="${LJ_ORCA_MODE:-submit}"
export LJ_ORCA_KEYWORDS="${LJ_ORCA_KEYWORDS:-RI-MP2 def2-TZVP def2-TZVP/C def2/J RIJCOSX TightSCF EnGrad}"
export LJ_ORCA_NPROCS="${LJ_ORCA_NPROCS:-16}"
export LJ_ORCA_MAXCORE="${LJ_ORCA_MAXCORE:-3000}"
export LJ_ORCA_CHUNK="${LJ_ORCA_CHUNK:-40}"
export LJ_ORCA_THROTTLE="${LJ_ORCA_THROTTLE:-128}"
export LJ_ORCA_WALLTIME="${LJ_ORCA_WALLTIME:-24:00:00}"
export LJ_ORCA_PARTITION="${LJ_ORCA_PARTITION:-long}"
export LJ_ORCA_MODULE="${LJ_ORCA_MODULE:-orca/orca-openmpi-6.1.0}"

# Liquid boxes (pure ACO + pure DCM). Smoke sizes by default; raise N / fraction
# for production.
export LJ_BOX_SIZE="${LJ_BOX_SIZE:-28}"
export LJ_BULK_DENSITY_FRACTION="${LJ_BULK_DENSITY_FRACTION:-0.5}"
export LJ_DCM_DENSITY="${LJ_DCM_DENSITY:-1.326}"
export LJ_ACO_DENSITY="${LJ_ACO_DENSITY:-0.784}"
export LJ_BOX_DCM_DIR="${LJ_BOX_DCM_DIR:-${LJ_ARTIFACTS_DIR}/boxes/dcm}"
export LJ_BOX_ACO_DIR="${LJ_BOX_ACO_DIR:-${LJ_ARTIFACTS_DIR}/boxes/aco}"

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
  printf '  joint     : %s\n' "${LJ_JOINT}"
  if [[ "${LJ_DES}" == "1" ]]; then
    printf '  des       : 1  (h5 %s)\n' "${LJ_DES_H5}"
    [[ -f "${LJ_DES_H5}" ]] || printf '              WARNING: h5 does not exist\n'
    printf '              pad %s, top %s residues\n' \
      "${LJ_DES_PAD}" "${LJ_DES_TOP_RESIDUES}"
  fi
  printf '  dataset   : %s\n' "${LJ_DATASET}"
  [[ -f "${LJ_DATASET}" ]] || printf '              WARNING: does not exist\n'
  printf '  artifacts : %s\n' "${LJ_ARTIFACTS_DIR}"
  if [[ -n "${_LJ_INHERITED_ARTIFACTS_DIR}" \
        && "${_LJ_INHERITED_ARTIFACTS_DIR}" != "${LJ_ARTIFACTS_DIR}" ]]; then
    printf '              (ignoring inherited ARTIFACTS_DIR=%s —\n' \
      "${_LJ_INHERITED_ARTIFACTS_DIR}"
    printf '               export LJ_ARTIFACTS_DIR to redirect this ladder)\n'
  fi
  if [[ "${LJ_JOINT}" == "1" ]]; then
    printf '  geom src  : %s\n' "${LJ_GEOM_SOURCE}"
    printf '  nms       : %s conf @ %s K (freq_min %s cm^-1)\n' \
      "${LJ_NMS_CONFORMERS}" "${LJ_NMS_TEMPERATURE}" "${LJ_NMS_FREQ_MIN}"
  fi
  printf '\n'
}
