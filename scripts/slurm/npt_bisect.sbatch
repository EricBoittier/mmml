#!/bin/bash
#SBATCH --job-name=npt-bisect
#SBATCH --partition=a100
#SBATCH --qos=a100-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/npt_argon_water/bisect/%x-%A_%a.out
#SBATCH --error=artifacts/npt_argon_water/bisect/%x-%A_%a.err
#
# Bisect the "energy blow-up at step 100" that killed every run of the NpT
# campaign -- argon and water, trained scales and unit control alike, at
# saturation pressure and at 1 atm. Because the control fails identically, the
# fault is upstream of the LJ scales.
#
# Two independent variables, 2 ps each so the whole matrix is minutes:
#
#   ensemble   nvt (no barostat) vs npt   -> is it the Nose-Hoover barostat?
#   prep       "campaign" vs "mine"       -> is it the pre-minimisation?
#
# "mine" is what the failing runs used: --calculator-pre-minimize plus
# --charmm-pre-minimize and 200 FIRE steps. Its fmax was DIVERGING -- 222 ->
# 376 eV/A on water over 20 steps -- so MD was handed an already-broken state.
# "campaign" is the flag set from scripts/slurm/dense_dt_campaign/run_one.sh on
# origin/cursor/denser-box-dt-campaign-72a7, which is the only jax-md MD
# configuration in this repo known to have run: it skips the ASE/CHARMM
# pre-minimisation entirely and relies on a short jax-md FIRE instead, with a
# larger neighbour-list skin and a fixed rebuild interval.
#
#   idx 0  nvt  campaign      <- if this is stable, the force path is fine
#   idx 1  npt  campaign      <- vs 0 isolates the barostat
#   idx 2  nvt  mine          <- vs 0 isolates the pre-minimisation
#   idx 3  npt  mine          <- reproduces the failure
set -uo pipefail

REPO="${MMML_REPO:-$HOME/mmml_npt}"
cd "$REPO" || exit 1
export MMML_SCICORE_CMAKE="${MMML_SCICORE_CMAKE:-CMake/3.31.8-GCCcore-14.3.0}"
export MMML_SCICORE_TOOLCHAIN="${MMML_SCICORE_TOOLCHAIN:-foss/2025a}"
# shellcheck disable=SC1091
source scripts/scicore_env.sh
if [[ ! -f "$REPO/setup/charmm/libcharmm.so" && -f "$HOME/mmml/setup/charmm/libcharmm.so" ]]; then
  export CHARMM_LIB_DIR="$HOME/mmml/setup/charmm" CHARMM_HOME="$HOME/mmml/setup/charmm"
fi
source .venv/bin/activate
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export UV_NO_SYNC=1 UV_OFFLINE=1

OUT_ROOT="$REPO/artifacts/npt_argon_water"
BISECT="$OUT_ROOT/bisect"
mkdir -p "$BISECT"

CKPT="${CKPT:-$REPO/artifacts/lj_scales_des_production/seed_42/ckpts/hybrid_mm_scaled_lj_des_full_seed42-5c75054c-bab3-4b9f-99bd-79d5b7fd7307}"
BOX="$OUT_ROOT/boxes/tip3_298k"
PS="${PS:-2}"

CASES=(
  "nvt campaign"
  "npt campaign"
  "nvt mine"
  "npt mine"
)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
read -r ens prep <<<"${CASES[$IDX]}"
tag="bisect_${ens}_${prep}"
out="$BISECT/$tag"
rm -rf "$out"; mkdir -p "$out"

case "$ens" in
  nvt) setup=pbc_nvt ;;
  npt) setup=pbc_npt ;;
esac

# The control arm's scales: this is a stability question, not a scales question.
sidecar="$OUT_ROOT/scales/scales_unit.json"

read -r nmol side < <(uv run python -c "
import json,sys
b=json.load(open(sys.argv[1]))
print(int(b['n_molecules']), float(b['box_side_A']))" "$BOX/box.json") || exit 5

common=(
  --backend jaxmd --setup "$setup"
  --composition "TIP3:${nmol}"
  --from-psf "$BOX/model.psf" --from-crd "$BOX/model.crd"
  --skip-cluster-build --box-size "$side"
  --checkpoint "$CKPT/epoch-25"
  --mm-lj-scales-file "$sidecar"
  --temperature 298.15 --pressure 1.0
  --ps "$PS" --dt-fs 0.25
  --output-dir "$out"
  --include-mm --do-ml --do-ml-dimer
  --seed 123
)

if [[ "$prep" == "campaign" ]]; then
  # dense_dt_campaign/run_one.sh: no ASE/CHARMM pre-minimisation at all.
  prep_flags=(
    --no-calculator-pre-minimize --no-charmm-pre-minimize
    --fire-min-steps 0
    --jaxmd-minimize-steps 50 --jaxmd-pbc-minimize-steps 50
    --no-handoff-quality-gate
    --mm-charge-mode fixed
    --jax-md-skin-distance 0.5 --jax-md-update-interval 40
    --steps-per-recording "${REC:-100}"
  )
else
  # What the failing campaign used.
  prep_flags=(
    --calculator-pre-minimize --charmm-pre-minimize
    --fire-min-steps 200 --pre-min-steps 50 --pre-min-fmax 0.1
    --jaxmd-minimize-steps 1000 --jaxmd-pbc-minimize-steps 1000
    --no-handoff-quality-gate
    --jax-md-skin-distance 0.25 --jax-md-update-interval 0
    --steps-per-recording "${REC:-100}"
  )
fi

echo "=== [$IDX] $tag : ensemble=$ens prep=$prep  ${nmol} TIP3, ${side} A, ${PS} ps"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1

uv run mmml md-system "${common[@]}" "${prep_flags[@]}" > "$out/md.log" 2>&1
rc=$?

echo "--- exit $rc"
grep -hoE "energy blow-up at step [0-9]+ \(E_pot=[-0-9.eE+]+\)" "$out/md.log" | head -1
grep -hoE "frames +[0-9]+|status +[a-z]+" "$out/md.log" | tail -2
tail -3 "$out/md.log"
exit $rc
