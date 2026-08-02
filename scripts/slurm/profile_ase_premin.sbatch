#!/bin/bash
#SBATCH --job-name=ase-prof
#SBATCH --partition=a100
#SBATCH --qos=a100-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=artifacts/npt_argon_water/prof/%x-%j.out
#SBATCH --error=artifacts/npt_argon_water/prof/%x-%j.err
#
# MEASURE why the ASE pre-minimisation is slow. Do not guess at this -- I already
# guessed once (that a host numpy array was disabling the neighbour-list Verlet
# skin cache, because `update_mm_pairs` gates on
# `hasattr(positions, "__dlpack_device__")`) and it was WRONG: numpy >= 1.23
# implements DLPack, so a plain np.ndarray satisfies that check and reports
# device type kDLCPU exactly as a CPU JAX array does. A mutation test caught it.
#
# Observed cost to explain: ~6 s per FIRE step, 2,196-atom water box, so a
# 200-step pre-min burned ~20 minutes before MD even started.
#
# This runs a short ASE pre-minimisation under cProfile and additionally dumps
# the pair-list counters, which distinguish the two candidate explanations
# directly:
#   reused ~ 0 and cpu_rebuilds ~ calls  -> neighbour list rebuilt every step
#   reused ~ calls                       -> the cost is inside the model itself
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
PROF="$OUT_ROOT/prof"
mkdir -p "$PROF"

CKPT="${CKPT:-$REPO/artifacts/lj_scales_des_production/seed_42/ckpts/hybrid_mm_scaled_lj_des_full_seed42-5c75054c-bab3-4b9f-99bd-79d5b7fd7307}"
BOX="$OUT_ROOT/boxes/tip3_298k"
STEPS="${STEPS:-20}"

read -r nmol side < <(uv run python -c "
import json,sys
b=json.load(open(sys.argv[1]))
print(int(b['n_molecules']), float(b['box_side_A']))" "$BOX/box.json") || exit 5

echo "=== profiling ASE pre-min: $nmol TIP3, ${side} A, $STEPS FIRE steps"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1

# --ps 0 so nothing beyond setup + pre-minimisation runs.
PYTHONPROFILEIMPORTTIME=0 uv run python -X importtime -c "
import cProfile, pstats, io, sys, runpy
sys.argv = ['mmml', 'md-system',
  '--backend','jaxmd','--setup','pbc_nvt',
  '--composition','TIP3:${nmol}',
  '--from-psf','${BOX}/model.psf','--from-crd','${BOX}/model.crd',
  '--skip-cluster-build','--box-size','${side}',
  '--checkpoint','${CKPT}/epoch-25',
  '--temperature','298.15','--ps','0.05','--dt-fs','0.25',
  '--output-dir','${PROF}/run',
  '--include-mm','--do-ml','--do-ml-dimer',
  '--calculator-pre-minimize','--charmm-pre-minimize',
  '--fire-min-steps','${STEPS}','--pre-min-steps','10',
  '--no-handoff-quality-gate','--seed','123',
]
pr = cProfile.Profile()
pr.enable()
try:
    runpy.run_module('mmml.cli.__main__', run_name='__main__')
except SystemExit:
    pass
finally:
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(45)
    print('==== CUMULATIVE TOP 45 ====')
    print(s.getvalue())
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats('tottime').print_stats(30)
    print('==== TOTTIME TOP 30 ====')
    print(s2.getvalue())
" > "$PROF/profile.txt" 2>&1
rc=$?

echo "--- exit $rc"
echo "=== pair-list counters (if reported) ==="
grep -iE "pair.*(reuse|rebuild|calls)|nbr|neighbou?r" "$PROF/profile.txt" 2>/dev/null | head -10
echo "=== profile head ==="
sed -n '/CUMULATIVE TOP/,/^$/p' "$PROF/profile.txt" 2>/dev/null | head -30
exit $rc
