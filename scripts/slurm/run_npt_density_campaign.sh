#!/bin/bash
# NpT at 1 atm on the certified liquid boxes, then equilibrated density.
#
# Why NpT and not the box we built: a Packmol box is constructed AT a target
# density, so its density is an input. Only letting the box find its own volume
# at fixed pressure turns density into a measurement of the potential. This is
# the first quantity in this project that sits outside the training loss.
#
# Ammonia runs at 240 K, not 298 K: NH3 boils at 239.8 K, so a 298 K box is a
# gas and its "liquid density" is meaningless.
#
#   SPECIES=tip3 bash scripts/slurm/run_npt_density_campaign.sh
#
# Env:
#   SPECIES      tip3 | meoh | amm1        (default: tip3)
#   CKPT         PhysNet checkpoint         (default: best measured DES fit)
#   LJ_SCALES    hybrid_mm.json sidecar     (default: none -> unscaled CGenFF LJ)
#   EQ_PS/PROD_PS  equilibration/production picoseconds
#SBATCH --job-name=npt-density
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mmhome/boittier/home/mmml/artifacts/npt_density/logs/slurm-%j.out
#SBATCH --error=/mmhome/boittier/home/mmml/artifacts/npt_density/logs/slurm-%j.err

set -euo pipefail

REPO="${MMML_REPO:-$HOME/mmml}"
cd "$REPO"

SPECIES="${SPECIES:-tip3}"
BOX="$REPO/artifacts/lj_scales_des_validation/boxes/${SPECIES}"
OUT="$REPO/artifacts/npt_density/${SPECIES}"
EQ_PS="${EQ_PS:-100}"
PROD_PS="${PROD_PS:-500}"

# Temperature must match the box it was built at, and the reference it will be
# compared against. Getting this wrong silently compares a 298 K run to a
# 239.8 K reference.
case "$SPECIES" in
  tip3) TEMP=298.0; RESI=TIP3 ;;
  meoh) TEMP=298.0; RESI=MEOH ;;
  amm1) TEMP=240.0; RESI=AMM1 ;;
  *) echo "ERROR: unknown SPECIES '$SPECIES' (tip3|meoh|amm1)" >&2; exit 2 ;;
esac

CKPT="${CKPT:-$REPO/artifacts/lj_scales_des/ckpts/hybrid_mm_fixed_lj_scales_des-72960237-bc4c-4410-b3f2-f80fb25bc0f1/epoch-25}"
LJ_SCALES="${LJ_SCALES:-}"

source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export JAX_PLATFORMS=cuda
export MMML_MLPOT_DEVICE=gpu
# Compute nodes have no outbound network; uv would otherwise re-resolve the
# editable install against pypi.org and die on a retry timeout.
export UV_NO_SYNC=1
export UV_OFFLINE=1
mkdir -p "$OUT" "$REPO/artifacts/npt_density/logs"

[[ -f "$BOX/model.psf" ]] || { echo "ERROR: missing $BOX/model.psf - build the box first" >&2; exit 2; }
[[ -f "$BOX/model.crd" || -f "$BOX/model.pdb" ]] || { echo "ERROR: missing $BOX coordinates" >&2; exit 2; }
[[ -e "$CKPT" ]] || { echo "ERROR: missing checkpoint $CKPT" >&2; exit 2; }

echo "=== NpT density: $RESI at $TEMP K, 1 atm ==="
echo "box:        $BOX"
echo "checkpoint: $CKPT"
echo "lj scales:  ${LJ_SCALES:-<none: unscaled CGenFF LJ>}"
echo "eq/prod:    ${EQ_PS} / ${PROD_PS} ps"
python -c "import jax; print('JAX devices:', jax.devices())"

# lr_solver must stay mic: the trained LJ scales only enter the switched-MM
# pair loop, which is off under ewald/nvalchemiops_pme.
# There is no --ensemble flag: the ensemble is the --setup preset, and
# --n-equil is lambda_ti-only. Equilibration is therefore handled by running
# TOTAL_PS and discarding the leading fraction in the analysis below, which is
# also the only place drift is actually tested.
TOTAL_PS=$(python -c "print(float('$EQ_PS') + float('$PROD_PS'))")
DISCARD=$(python -c "print(round(float('$EQ_PS')/(float('$EQ_PS')+float('$PROD_PS')), 4))")

uv run mmml md-system \
  --setup pbc_npt \
  --backend jaxmd \
  --pressure 1.0 \
  --temperature "$TEMP" \
  --from-psf "$BOX/model.psf" \
  --from-crd "$BOX/model.crd" \
  --composition "${RESI}:1" \
  --checkpoint "$CKPT" \
  ${LJ_SCALES:+--mm-lj-scales-file "$LJ_SCALES"} \
  --mm-nonbond-mode jax_mic \
  --mm-charge-mode fixed \
  --include-mm \
  --ps "$TOTAL_PS" \
  --output-dir "$OUT"

TRAJ=$(ls -t "$OUT"/*.h5 2>/dev/null | head -1)
[[ -n "$TRAJ" ]] || { echo "ERROR: md-system wrote no HDF5 under $OUT" >&2; exit 3; }

uv run python scripts/analyze_npt_density.py \
  --traj "$TRAJ" \
  --species "$RESI" \
  --temperature "$TEMP" \
  --discard-frac "$DISCARD" \
  -o "$OUT/density.json"
