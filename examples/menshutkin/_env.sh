# Menshutkin NH3 + CH3Cl reactive MLMM campaign.
#   source examples/menshutkin/_env.sh
#
# Host notes
# ----------
# Login node: no GPU (cuInit fails), and libcharmm.so needs libOpenCL.so.1 which
#   is not installed there -- MMML_OPENCL_STUB supplies a no-op loader so
#   PyCHARMM imports. Use it for smoke runs only.
# gpu09:      2x RTX 5090 and a real libOpenCL; run production here via
#             `ssh gpu09`. GPU 0 is often occupied by another user, so
#             CUDA_VISIBLE_DEVICES defaults to 1.

if [[ -n "${BASH_VERSION:-}" ]]; then
  _MENSH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  _MENSH_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
REPO_ROOT="$(cd "${_MENSH_DIR}/../.." && pwd)"
export REPO_ROOT
export MENSH_DIR="${_MENSH_DIR}"

# Non-interactive ssh (how the GPU host is driven) does not read the profile
# that puts uv on PATH.
case ":${PATH}:" in
  *":${HOME}/.local/bin:"*) ;;
  *) export PATH="${HOME}/.local/bin:${PATH}" ;;
esac

export CHARMM_LIB_DIR="${CHARMM_LIB_DIR:-${REPO_ROOT}/setup/charmm/lib}"

# No-op OpenCL loader for hosts without one (login node). Harmless where a real
# libOpenCL exists because it is appended, not prepended.
MMML_OPENCL_STUB="${MMML_OPENCL_STUB:-${HOME}/.local/opencl-stub}"
if [[ ! -e /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 && -d "${MMML_OPENCL_STUB}" ]]; then
  export LD_LIBRARY_PATH="${MMML_OPENCL_STUB}:${LD_LIBRARY_PATH:-}"
fi

# GPU when one is visible, CPU otherwise -- unless MENSH_DEVICE says otherwise.
#
# MENSH_DEVICE=cpu is the way to run on a GPU HOST without touching the GPUs.
# It has to exist because the auto-detect below `unset JAX_PLATFORMS`, so on
# gpu08/gpu09 an exported JAX_PLATFORMS=cpu was silently discarded and the job
# landed on CUDA_VISIBLE_DEVICES=1 -- i.e. straight onto whichever GPU the
# production umbrella runs are using. That is the opposite of what anyone asking
# for CPU wants, and it is invisible until the run either OOMs or quietly
# competes with production for the device.
case "${MENSH_DEVICE:-auto}" in
  cpu)
    export JAX_PLATFORMS=cpu
    unset CUDA_VISIBLE_DEVICES
    # JAX_PLATFORMS alone is NOT enough for anything under `mmml`. The CLI calls
    # apply_mlpot_jax_platform_env(), which treats JAX_PLATFORMS=cpu as a stale
    # login-node export and REWRITES it to put CUDA first whenever
    # MMML_MLPOT_DEVICE is unset (it defaults to gpu). The job then dies with
    # "no supported devices found for platform CUDA" -- or worse, succeeds and
    # silently runs on a production GPU. This is the variable that actually
    # decides, so set it here.
    export MMML_MLPOT_DEVICE=cpu
    ;;
  gpu)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
    unset JAX_PLATFORMS
    ;;
  *)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
      unset JAX_PLATFORMS
    else
      export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
    fi
    ;;
esac
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

# The 128-feature PhysNet trained on the extended set. Unlike examples/m/kl.json
# its training data covers the transition-state window xi in [-0.75, +0.5];
# kl.json has ~4 frames there, so any barrier from it is meaningless.
# PRODUCTION CHECKPOINT. Was model_ext.json (cutoff 8) until 2026-08-02, while
# run_solvated_production.sh overrode it to epoch-1436 (cutoff 14) -- so every
# solvated PMF used the long-cutoff model and the GAS PMF silently used the
# short one. They are not comparable: cutoff 8 is 7x worse on held-out energies
# (0.489 vs 0.067 kcal/mol MAE) and degrades 2.3x specifically TOWARD PRODUCTS,
# where the C-Cl pair approaches its radial cutoff -- i.e. exactly where the gas
# transition state sits. Defaulting to the production checkpoint here removes
# the trap; run_solvated_production.sh's override is now redundant but harmless.
export MENSH_CKPT="${MENSH_CKPT:-${REPO_ROOT}/ckpts/menshutkin_longrange/longrange-c2398efc-a6f3-478f-8a3b-af6c66cda0fc/epoch-1436}"
# 2500-frame reaction-coordinate scan, uniform over xi = -2.2 .. 4.1.
export MENSH_SCAN="${MENSH_SCAN:-${REPO_ROOT}/examples/m/scan_nh3_ch3cl.npz}"
export MENSH_ARTIFACTS="${MENSH_ARTIFACTS:-${REPO_ROOT}/artifacts/menshutkin}"
mkdir -p "${MENSH_ARTIFACTS}"

# Append residues missing from stock CGenFF: MECL (chloromethane, 4-char
# PDB-legal name -- see top_mecl.rtf) and CHEX
# (cyclohexane -- only exists in top_all35_ethers.rtf with legacy ether types,
# so examples/menshutkin/top_chex.rtf re-types it for CGenFF). Colon-separated.
export MMML_CGENFF_EXTRA_RTF="${MMML_CGENFF_EXTRA_RTF:-${MENSH_DIR}/top_mecl.rtf:${MENSH_DIR}/top_chex.rtf}"
export MMML_CGENFF_EXTRA_PRM="${MMML_CGENFF_EXTRA_PRM:-${REPO_ROOT}/examples/m/par_ch3cl.prm}"

# --- Solvents (Turan, Brickel & Meuwly, JPCB 126, 1951 (2022)) ---------------
# name:CGenFF residue:density(kg/m3):box side(A)
# Densities are experimental values at 298 K; box sides follow the paper.
export MENSH_SOLVENTS="${MENSH_SOLVENTS:-water:TIP3:997:30 methanol:MEOH:792:25 acetonitrile:ACN:786:28 benzene:BENZ:874:27 cyclohexane:CHEX:774:30}"

# --- Reaction coordinate ----------------------------------------------------
# Seeds written by 01_seed_windows.py use the canonical order Cl=0, N=1, C=2,
# then the 6 hydrogens, so the CV indices below are stable regardless of which
# source NPZ the seeds came from.
export MENSH_CV_DIFFERENCE="2,0,2,1"   # xi = r(C-Cl) - r(C-N)
# Turan et al. (JPCB 2022, 126, 1951) protocol: xi in [-1.3, 1.6], 0.1 A spacing.
export MENSH_XI_MIN="${MENSH_XI_MIN:--1.3}"
export MENSH_XI_MAX="${MENSH_XI_MAX:-1.6}"
export MENSH_N_WINDOWS="${MENSH_N_WINDOWS:-30}"
# Turan k = 150 kcal/mol/A^2; the sampler works in eV/A^2.
export MENSH_K_EV="${MENSH_K_EV:-6.505}"
export MENSH_TEMPERATURE="${MENSH_TEMPERATURE:-300}"
