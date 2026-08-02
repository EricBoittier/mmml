#!/bin/bash
#SBATCH --job-name=npt-lj
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/npt_argon_water/logs/%x-%A_%a.out
#SBATCH --error=artifacts/npt_argon_water/logs/%x-%A_%a.err
#
# One array task = one (state point, scale mode) NpT run + its density analysis.
#
# The array index maps onto RUNS[] below. Modes alternate, so index 2k is the
# TRAINED arm and 2k+1 is the UNIT control for the same state point -- keep them
# adjacent so a partial array still yields matched pairs rather than an
# unbalanced set that cannot be compared.
#
# Submit only the indices whose box is certified, e.g. water alone:
#   sbatch --array=8-9 scripts/slurm/npt_campaign.sbatch
#
# ARGON PRESSURES ARE NOT 1 atm. Argon boils at 87.28 K, so at 90 K / 1 atm it
# is a vapour; each temperature runs at its own saturation pressure. 150 K is
# deliberately absent (Tr = 0.994 -- diverging compressibility and correlation
# length; see run_npt_argon_water_campaign.sh).
set -uo pipefail

REPO="${MMML_REPO:-$HOME/mmml}"
cd "$REPO" || exit 1
source .venv/bin/activate 2>/dev/null
# Compute nodes have no outbound network; uv would otherwise try pypi and hang.
export UV_NO_SYNC=1 UV_OFFLINE=1

_NG="$REPO/mmml/data/charmm"
export MMML_CGENFF_EXTRA_RTF="$_NG/top_noble_gases_literature.rtf"
export MMML_CGENFF_EXTRA_PRM="$_NG/par_noble_gases_literature.prm"

OUT_ROOT="$REPO/artifacts/npt_argon_water"
BOX_ROOT="$OUT_ROOT/boxes"
SCALES="$OUT_ROOT/scales"
mkdir -p "$OUT_ROOT/logs" "$SCALES"

CKPT="${CKPT:-$REPO/artifacts/lj_scales_des_production/seed_42/ckpts/hybrid_mm_scaled_lj_des_full_seed42-5c75054c-bab3-4b9f-99bd-79d5b7fd7307}"
EPOCH="${EPOCH:-25}"
EQ_PS="${EQ_PS:-100}"
PROD_PS="${PROD_PS:-400}"
DISCARD="${DISCARD:-0.4}"

# tag  resid  T(K)  P(atm)  rho_ref  mode
RUNS=(
  "ar1_90k   AR1   90.0    1.3176   1.37860  trained"
  "ar1_90k   AR1   90.0    1.3176   1.37860  unit"
  "ar1_120k  AR1  120.0   11.9714   1.16280  trained"
  "ar1_120k  AR1  120.0   11.9714   1.16280  unit"
  "ar1_130k  AR1  130.0   19.9901   1.06810  trained"
  "ar1_130k  AR1  130.0   19.9901   1.06810  unit"
  "ar1_140k  AR1  140.0   31.2678   0.94371  trained"
  "ar1_140k  AR1  140.0   31.2678   0.94371  unit"
  "tip3_298k TIP3 298.15   1.0000   0.99705  trained"
  "tip3_298k TIP3 298.15   1.0000   0.99705  unit"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#RUNS[@]} )); then
  echo "array index $IDX out of range (0..$(( ${#RUNS[@]} - 1 )))" >&2
  exit 2
fi
read -r tag resid temp press rho mode <<<"${RUNS[$IDX]}"

box="$BOX_ROOT/$tag"
if [[ ! -f "$box/box.json" ]]; then
  echo "ABORT: box $tag is not certified ($box/box.json missing)." >&2
  echo "Refusing to run MD on an uncertified box -- a bad pack is invisible in" >&2
  echo "a density number because Packmol reports the density it was asked for." >&2
  exit 3
fi

# The sidecar carries the fitted scales; the training hybrid_mm.json does NOT
# (config only), so it cannot be passed to --mm-lj-scales-file directly.
sidecar="$SCALES/scales_$mode.json"
if [[ ! -f "$sidecar" ]]; then
  echo "=== exporting $mode scales"
  uv run python scripts/export_mm_lj_scales.py \
    --checkpoint "$CKPT/epoch-$EPOCH" --hybrid-mm "$CKPT/hybrid_mm.json" \
    --mode "$mode" -o "$sidecar" || exit 4
fi

out="$OUT_ROOT/runs/${tag}_${mode}"
mkdir -p "$out"
echo "=== [$IDX] $tag / $mode : T=$temp K, P=$press atm, $((EQ_PS+PROD_PS)) ps"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1

uv run mmml md-system \
  --setup pbc_npt --backend jaxmd \
  --composition "${resid}:1" \
  --checkpoint "$CKPT/epoch-$EPOCH" \
  --mm-lj-scales-file "$sidecar" \
  --temperature "$temp" --pressure "$press" \
  --ps "$((EQ_PS + PROD_PS))" \
  --output-dir "$out" || { echo "md-system failed for $tag/$mode" >&2; exit 5; }

traj=$(find "$out" -name "*.h5" | head -1)
if [[ -z "$traj" ]]; then
  echo "no trajectory written for $tag/$mode" >&2
  exit 6
fi

uv run python scripts/analyze_npt_density.py \
  --traj "$traj" --species "$resid" --discard-frac "$DISCARD" \
  --reference "$rho" --temperature "$temp" \
  -o "$out/density.json"
