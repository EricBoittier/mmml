#!/usr/bin/env bash
#SBATCH --job-name=ar-mmonly-npt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu08,gpu09,gpu10
#SBATCH --output=artifacts/npt_argon_water/logs/ar-mmonly-npt-%j.out
#SBATCH --error=artifacts/npt_argon_water/logs/ar-mmonly-npt-%j.err
#
# MM-only argon NpT pressure smoke (pure LJ, no ML).
# Certified AR1:500 box at 90 K / saturation P; unit LJ scales.
# jaxmd-unified so trajectory.npz carries P / P_kin / P_vir.
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"
source .venv/bin/activate
export PATH="${HOME}/.local/bin:${PATH}"
export UV_NO_SYNC="${UV_NO_SYNC:-1}"
export UV_OFFLINE="${UV_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export LJ_DEVICE=gpu
export JAX_PLATFORMS=cuda
export MMML_MLPOT_DEVICE=gpu
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

_NG="$ROOT/mmml/data/charmm"
export MMML_CGENFF_EXTRA_RTF="${MMML_CGENFF_EXTRA_RTF:+$MMML_CGENFF_EXTRA_RTF:}$_NG/top_noble_gases_literature.rtf"
export MMML_CGENFF_EXTRA_PRM="${MMML_CGENFF_EXTRA_PRM:+$MMML_CGENFF_EXTRA_PRM:}$_NG/par_noble_gases_literature.prm"

_CH="$ROOT/setup/charmm"
_OCL="${HOME}/micromamba/pkgs/ocl-icd-2.3.3-hb9d3cd8_0/lib"
if [[ -f "$_CH/libcharmm.so" ]]; then
  export CHARMM_LIB_DIR="$_CH" CHARMM_HOME="$_CH"
  export LD_LIBRARY_PATH="${_OCL}:/usr/lib64/openmpi/lib/:${LD_LIBRARY_PATH:-}"
fi

BOX="${AR_BOX:-artifacts/npt_argon_water/boxes/ar1_90k}"
SCALES="${AR_SCALES:-artifacts/npt_argon_water/scales/scales_unit.json}"
OUT="${AR_OUT:-artifacts/npt_argon_water/runs/ar1_90k_mmonly_unit}"
PS="${AR_PS:-20}"
DT_FS="${AR_DT_FS:-1.0}"
# NIST sat. P at 90 K is 1.3176 atm → bar for jaxmd-unified
P_ATM="${AR_P_ATM:-1.3176}"
T_K="${AR_T_K:-90.0}"
P_BAR="$(python -c "print(float('$P_ATM') * 1.01325)")"

mkdir -p "$OUT" artifacts/npt_argon_water/logs docs/images/npt_argon_water
read -r NMOL SIDE < <(python -c "
import json
b=json.load(open('$BOX/box.json'))
print(int(b['n_molecules']), float(b['box_side_A']))
")

echo "=== argon MM-only NPT $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "box=$BOX nmol=$NMOL L=$SIDE A  T=$T_K K  P=${P_ATM} atm (${P_BAR} bar)"
echo "scales=$SCALES out=$OUT ps=$PS dt_fs=$DT_FS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true

# Pure CGenFF LJ (--ff cgenff, no checkpoint, no ML). Load certified geometry.
uv run mmml md-system \
  --backend jaxmd \
  --jaxmd-unified \
  --ff cgenff \
  --setup pbc_npt \
  --from-pdb "$BOX/model.pdb" \
  --composition "AR1:${NMOL}" \
  --box-size "$SIDE" \
  --temperature "$T_K" \
  --pressure "$P_BAR" \
  --ps "$PS" \
  --dt-fs "$DT_FS" \
  --include-mm \
  --no-do-ml \
  --no-do-ml-dimer \
  --mm-lj-scales-file "$SCALES" \
  --jaxmd-minimize-steps 200 \
  --seed 90 \
  --quiet \
  --output-dir "$OUT" \
  >"$OUT/md.log" 2>&1

echo "=== plot pressure"
uv run python scripts/plot_npt_pressure.py \
  --traj "$OUT/trajectory.npz" \
  --out docs/images/npt_argon_water/ar1_90k_mmonly_pressure.png \
  --title "AR1 90 K MM-only (unit LJ scales), P_sat=${P_ATM} atm" \
  --target-atm "$P_ATM" \
  --dt-fs "$DT_FS"

# Density vs NIST liquid sat. reference (1.37860 g/cm3)
if [[ -f "$OUT/trajectory.npz" ]]; then
  uv run python scripts/analyze_npt_density.py \
    --traj "$OUT/trajectory.npz" --species AR1 --discard-frac 0.3 \
    --reference 1.37860 --temperature "$T_K" \
    -o "$OUT/density.json" 2>&1 | tee -a "$OUT/md.log" || true
fi

echo "=== done $(date -Is)"
ls -la "$OUT/trajectory.npz" docs/images/npt_argon_water/ar1_90k_mmonly_pressure.png
python - <<PY
import json, numpy as np
from pathlib import Path
z = np.load("$OUT/trajectory.npz", allow_pickle=True)
print("keys", sorted(z.files))
print("P mean/std", float(np.mean(z["pressures_bar"])), float(np.std(z["pressures_bar"])))
if "pressures_kin_bar" in z.files:
    print("Pkin mean", float(np.mean(z["pressures_kin_bar"])), "Pvir mean", float(np.mean(z["pressures_vir_bar"])))
print("target", float(z["target_pressure_bar"]) if "target_pressure_bar" in z.files else None)
d = Path("$OUT/density.json")
if d.is_file():
    print(json.dumps(json.loads(d.read_text()), indent=2)[:600])
PY
