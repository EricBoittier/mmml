#!/usr/bin/env bash
# DCM:9 free-space stability: CHARMM MM pre-min → MLpot SD → heating.
#
# Based on the validated DCM:2 smoke (no cons_fix, no bonded-mm-mini). See
# docs/agent-handoff-pycharmm-9mer.md for pass/fail checks and VMD notes.
#
# Packmol sphere radius (initial guess, same scaling as run_dcm90_free_nvt.sh):
#   R ≈ 18 * (9/60)^(1/3) ≈ 9.6 Å  (override with PACKMOL_R=...)
#
# Prerequisites:
#   - Launch under OpenMPI (libcharmm is MPI-linked on GPU nodes):
#       ./scripts/run_dcm9_stability.sh
#   - JAX CUDA 13 needs cuDNN >= 9.10.1 (cluster module cudnn/9.4 breaks GPU init):
#       uv sync --extra gpu
#       module unload cudnn    # if a old cuDNN module is loaded
#   - Checkpoint for DCM PhysNet:
#       export MMML_CKPT=/path/to/dcm1-.../ckpts/dcm1-...
#   - packmol on PATH (first run only if cache miss); rebuild libcharmm if you scale cluster size up
#   - Packmol placement cached under <output-dir>/.packmol_cache (or MMML_PACKMOL_CACHE).
#     Use --rebuild-packmol to force repack; --save-run-state for Orbax/NPZ geometry + metadata.
#
# Defaults: one GPU (CUDA_VISIBLE_DEVICES=0, --ml-gpu-count 1) to avoid multi-GPU
# cuDNN/cuBLASLt issues on shared nodes.
#
# Examples:
#   MMML_CKPT=$HOME/mmml_tutorial/acodcm/ckpts/dcm1-... ./scripts/run_dcm9_stability.sh
#   PS_HEAT=20 MD_STAGES=mini,heat,equi ./scripts/run_dcm9_stability.sh
#   CHARMM_MM_PRETREAT=1 MD_STAGES=mini,heat ./scripts/run_dcm9_stability.sh
#     # CGENFF min + 2000-step CHARMM heat before MLpot (outputs charmm_mm_heat_dcm_9.*)
#   ENABLE_FB=1 FB_RAD=14 ./scripts/run_dcm9_stability.sh
#   ./scripts/run_dcm9_stability.sh --ps-heat 30 --heat-ihtfrq 100
#   # softer ramp (defaults): 0 K -> 240 K over 20 ps, scale at ihtfrq
#
# After run, confirm in the log (non-quiet):
#   HEAT complete: restart_step=~40000, dcd_frames=~81  (new validation)
#   cons_fix: no monomers constrained
#   MLpot USER active before staged dynamics
#   Removed prior DCD          (or pull latest repo; old builds say Rescued)
# If heat stops early (~1–3k steps): grep 'TOLERANCE\\|echeck' in the log.
#   DCM:9 auto-loosens echeck (9 monomers); override with --no-echeck if needed.
#   Heat: 0 K -> 240 K; iasors=0 scaling (see mmml/.../mlpot/COMP_AND_HEATING.md).
#   COMP is OFF by default — do not enable --heat-comp-damp unless testing COMP.
#   H-on-C overlap in early DCD frames: X-H not constrained (no SHAKE); verify mini + frames 0-2.
#
# VMD:
#   vmd artifacts/pycharmm_mlpot/dcm9_stability/cluster_for_vmd_dcm_9.psf
#   # trajectory: heat_dcm_9.dcd

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Prefer pip cuDNN before mpirun (mmml-charmm-mpirun.sh also does this after MPI setup).
# shellcheck source=scripts/setup_jax_cuda_env.sh
source "$REPO_ROOT/scripts/setup_jax_cuda_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -z "${MMML_CKPT:-}" ]]; then
  echo "Set MMML_CKPT to your DCM PhysNet checkpoint directory." >&2
  echo "  Example: export MMML_CKPT=\$HOME/mmml_tutorial/acodcm/ckpts/dcm1-..." >&2
  exit 1
fi

N_MOLECULES=9
# Same (N/60)^(1/3) scaling as DCM:90 → 21 Å at N=90.
PACKMOL_R="${PACKMOL_R:-$(
  python3 -c "print(round(18 * (${N_MOLECULES}/60) ** (1/3), 1))"
)}"
OUT_DIR="${OUT_DIR:-artifacts/pycharmm_mlpot/dcm9_stability}"
MD_STAGES="${MD_STAGES:-mini,heat}"
PS_HEAT="${PS_HEAT:-20.0}"
HEAT_FIRSTT="${HEAT_FIRSTT:-0}"
HEAT_FINALT="${HEAT_FINALT:-240}"
HEAT_IHTFRQ="${HEAT_IHTFRQ:-100}"
MINI_NSTEP="${MINI_NSTEP:-150}"
DYN_NPRINT="${DYN_NPRINT:-500}"
DCD_NSAVC="${DCD_NSAVC:-500}"

PRETREAT_ARGS=()
if [[ "${CHARMM_MM_PRETREAT:-0}" == "1" ]]; then
  PRETREAT_ARGS=(
    --charmm-mm-pretreat
    --charmm-mm-pretreat-heat-nstep "${CHARMM_MM_PRETREAT_HEAT_NSTEP:-2000}"
  )
fi

MPIRUN="${MMML_MPIRUN_WRAPPER:-$REPO_ROOT/scripts/mmml-charmm-mpirun.sh}"

FB_ARGS=()
if [[ "${ENABLE_FB:-0}" == "1" ]]; then
  FB_RAD="${FB_RAD:-14.0}"
  FB_ARGS=(
    --flat-bottom-radius "$FB_RAD"
    --flat-bottom-selection "TYPE C"
    --flat-bottom-k 0.00001
  )
fi

exec "$MPIRUN" md-system \
  --setup free_nvt \
  --backend pycharmm \
  --composition "DCM:${N_MOLECULES}" \
  --output-dir "$OUT_DIR" \
  --job-name dcm9_stability \
  --checkpoint "$MMML_CKPT" \
  --md-stages "$MD_STAGES" \
  --spacing 5.0 \
  --packmol-sphere \
  --packmol-radius "$PACKMOL_R" \
  --packmol-tolerance 1.0 \
  --mini-nstep "$MINI_NSTEP" \
  --ps-heat "$PS_HEAT" \
  --heat-firstt "$HEAT_FIRSTT" \
  --heat-finalt "$HEAT_FINALT" \
  --heat-ihtfrq "$HEAT_IHTFRQ" \
  --dyn-nprint "$DYN_NPRINT" \
  --dyn-iprfrq 2000 \
  --dcd-nsavc "$DCD_NSAVC" \
  --dynamics-overlap-action off \
  --dt-fs 0.25 \
  --temperature 300.0 \
  --mm-switch-on 7.0 \
  --mm-switch-width 5.0 \
  --ml-switch-width 0.1 \
  --charmm-sd-steps 25 \
  --charmm-abnr-steps 100 \
  --ml-gpu-count 1 \
  --skip-energy-show \
  --seed 123 \
  "${PRETREAT_ARGS[@]}" \
  "${FB_ARGS[@]}" \
  "$@"
