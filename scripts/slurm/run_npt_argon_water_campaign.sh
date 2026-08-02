#!/bin/bash
# NpT density validation: argon along its saturation curve + water at ambient,
# each with the TRAINED LJ scales and with UNIT scales as a control.
#
# Why a control arm. Several LJ types come out of the fit pinned at their
# bounds: argon, helium and neon sit on BOTH floors (sigma 0.80 / eps 0.25 under
# the production prior, 0.60 / 0.05 under the wide one), krypton and xenon on
# both ceilings. Argon's MM dispersion is therefore essentially switched off,
# and a wrong density with the trained scales alone could not be attributed --
# it might equally be the literature noble-gas parameters, which were never
# fitted alongside CGenFF. Running the same boxes with all scales = 1.0
# separates the two. Argon is single-site, so the control is cheap.
#
# Why argon at all. It is PURE Lennard-Jones -- one site, no charges, no
# intramolecular terms -- so a wrong sigma/eps shows up in the density with
# nothing else to blame. Water confounds LJ with electrostatics, geometry and
# the ML term. Argon is also ~10x sparser than water in atoms/A^3 near Tc.
#
# PRESSURE IS NOT 1 atm FOR ARGON. Its normal boiling point is 87.28 K, so at
# 90 K / 1 atm argon is a vapour. Each temperature runs at its own saturation
# pressure. Getting this wrong produces a gas and a meaningless density.
#
# Reference densities are NIST Chemistry WebBook saturation data (argon
# C7440371, water C7732185); see mmml/data/reference_state_points.py, which
# refuses to store a density that was not verified against a cited source.
set -uo pipefail

REPO="${MMML_REPO:-$HOME/mmml}"
cd "$REPO" || exit 1
source .venv/bin/activate 2>/dev/null
# Compute nodes have no outbound network; without these uv tries pypi and hangs.
export UV_NO_SYNC=1 UV_OFFLINE=1

# CHARMM ships no residue for Ar/Kr/Xe, so AR1 is unknown to the box builder
# without this: `mmml liquid-box --composition AR1:1` dies with
#   ValueError: Unknown CGenFF residue 'AR1'
# The .str in the same directory is a *stream* file (it carries its own
# `read rtf card append` directives) and cannot be used here -- these two are
# the split-out plain RTF/PRM that `read.rtf(append=True)` can consume.
_NG="$REPO/mmml/data/charmm"
export MMML_CGENFF_EXTRA_RTF="${MMML_CGENFF_EXTRA_RTF:+$MMML_CGENFF_EXTRA_RTF:}$_NG/top_noble_gases_literature.rtf"
export MMML_CGENFF_EXTRA_PRM="${MMML_CGENFF_EXTRA_PRM:+$MMML_CGENFF_EXTRA_PRM:}$_NG/par_noble_gases_literature.prm"

# pc-studix PyCHARMM environment. libcharmm.so links OpenCL and the repo venv
# does not carry libOpenCL.so.1, so without this every `import pycharmm` dies
# with "OSError: libOpenCL.so.1: cannot open shared object file" and every box
# build fails. Same block as build_des_validation_boxes.sh.
_CH="$REPO/setup/charmm"
_OCL="$HOME/micromamba/pkgs/ocl-icd-2.3.3-hb9d3cd8_0/lib"
if [[ -f "$_CH/libcharmm.so" ]]; then
  export CHARMM_LIB_DIR="$_CH" CHARMM_HOME="$_CH"
  export LD_LIBRARY_PATH="$_OCL:/usr/lib64/openmpi/lib/:${LD_LIBRARY_PATH:-}"
fi

OUT_ROOT="${OUT_ROOT:-$REPO/artifacts/npt_argon_water}"
BOX_ROOT="$OUT_ROOT/boxes"
SCALES="$OUT_ROOT/scales"
mkdir -p "$BOX_ROOT" "$SCALES"

# Which fit to validate. Production prior (sigma 0.8-1.2, eps 0.25-4).
CKPT="${CKPT:-$REPO/artifacts/lj_scales_des_production/seed_42/ckpts/hybrid_mm_scaled_lj_des_full_seed42-5c75054c-bab3-4b9f-99bd-79d5b7fd7307}"
EPOCH="${EPOCH:-25}"

EQ_PS="${EQ_PS:-100}"
PROD_PS="${PROD_PS:-400}"
DISCARD="${DISCARD:-0.4}"

# resid  T(K)  P(atm)   rho_ref(g/cm3)   N_atoms
#   argon: saturation pressures converted from NIST bar (P_bar / 1.01325)
#   water: ambient
#
# The sparsest saturated point in the NIST table, 150 K, is deliberately NOT
# here. Argon's critical temperature is 150.86 K, so 150 K is Tr = 0.994: the
# isothermal compressibility diverges (the barostat volume wanders without
# bound), the correlation length outgrows a 36 A box, and critical slowing down
# puts equilibration far beyond 500 ps. It would produce a number, and the
# number would be meaningless. 130 K (Tr = 0.862) is the sparse end instead, and
# 140 K (Tr = 0.928) is already near enough to Tc that its error bar will be the
# largest of the four -- judge it accordingly.
STATES=(
  "AR1  90.0   1.3176   1.37860   500"
  "AR1 120.0  11.9714   1.16280   500"
  "AR1 130.0  19.9901   1.06810   500"
  "AR1 140.0  31.2678   0.94371   500"
  "TIP3 298.15 1.0000   0.99705   732"
)

# ---------------------------------------------------------------- scales
echo "=== exporting LJ scale sidecars from $CKPT/epoch-$EPOCH"
for mode in trained unit; do
  uv run python scripts/export_mm_lj_scales.py \
    --checkpoint "$CKPT/epoch-$EPOCH" \
    --hybrid-mm "$CKPT/hybrid_mm.json" \
    --mode "$mode" \
    -o "$SCALES/scales_$mode.json" || { echo "FAILED: export $mode" >&2; exit 1; }
done

# ---------------------------------------------------------------- boxes
build_box() {
  local resid="$1" temp="$2" rho="$3" n="$4"
  local tag="${resid,,}_${temp%.*}k"
  local out="$BOX_ROOT/$tag"
  [[ -f "$out/box.json" ]] && { echo "  box $tag already certified"; return 0; }
  # Cubic side that puts n atoms at the reference density, so the barostat
  # starts near the answer and only has to relax, not to find the phase.
  local L
  L=$(uv run python -c "
import sys
rho,n=float(sys.argv[1]),int(sys.argv[2])
M={'AR1':39.948,'TIP3':18.015}[sys.argv[3]]
nat={'AR1':1,'TIP3':3}[sys.argv[3]]
nmol=n/nat
V=nmol*M/(rho*6.02214076e23)*1e24
print(f'{V**(1/3):.3f}')" "$rho" "$n" "$resid")
  echo "  building $tag: $n atoms, L=$L A, rho_target=$rho"
  mkdir -p "$out"
  # Box building is Packmol + CHARMM only -- no ML model is evaluated -- so it
  # is forced onto CPU. Left to autodetect, JAX grabs (or fails to grab) a GPU
  # and the CUDA init error is pure noise in the build log.
  JAX_PLATFORMS=cpu uv run mmml liquid-box \
    --composition "${resid}:1" --box-auto count --box-size "$L" \
    --target-density-g-cm3 "$rho" --temperature "$temp" \
    --output-dir "$out" || { echo "  FAILED box: $tag" >&2; return 1; }
}

echo
echo "=== building boxes"
for s in "${STATES[@]}"; do
  read -r resid temp press rho n <<<"$s"
  build_box "$resid" "$temp" "$rho" "$n"
done

# ---------------------------------------------------------------- NpT
run_npt() {
  local resid="$1" temp="$2" press="$3" rho="$4" mode="$5"
  local tag="${resid,,}_${temp%.*}k"
  local box="$BOX_ROOT/$tag"
  local out="$OUT_ROOT/runs/${tag}_${mode}"
  [[ -f "$box/box.json" ]] || { echo "  SKIP $tag/$mode: box not certified" >&2; return 1; }
  mkdir -p "$out"
  echo "  NpT $tag/$mode: T=$temp K P=$press atm, $((EQ_PS+PROD_PS)) ps"
  uv run mmml md-system \
    --setup pbc_npt --backend jaxmd \
    --composition "${resid}:1" \
    --checkpoint "$CKPT/epoch-$EPOCH" \
    --mm-lj-scales-file "$SCALES/scales_$mode.json" \
    --temperature "$temp" --pressure "$press" \
    --ps "$((EQ_PS + PROD_PS))" \
    --output-dir "$out" \
    > "$out/md.log" 2>&1 || { echo "  FAILED NpT: $tag/$mode (see $out/md.log)" >&2; return 1; }

  local traj
  traj=$(find "$out" -name "*.h5" | head -1)
  [[ -n "$traj" ]] || { echo "  no trajectory for $tag/$mode" >&2; return 1; }
  uv run python scripts/analyze_npt_density.py \
    --traj "$traj" --species "$resid" --discard-frac "$DISCARD" \
    --reference "$rho" --temperature "$temp" \
    -o "$out/density.json" 2>&1 | tail -5
}

if [[ "${BOXES_ONLY:-0}" == "1" ]]; then
  echo
  echo "=== BOXES_ONLY=1: stopping before the NpT runs."
  echo "    Box building is CPU-only (packmol + CHARMM) and runs anywhere;"
  echo "    the NpT runs need a GPU -- pc-studix has none, so they go to scicore."
  for s in "${STATES[@]}"; do
    read -r resid temp press rho n <<<"$s"
    tag="${resid,,}_${temp%.*}k"
    printf "  %-12s " "$tag"
    [[ -f "$BOX_ROOT/$tag/box.json" ]] && echo "certified" || echo "NOT certified"
  done
  exit 0
fi

echo
echo "=== NpT runs"
rc=0
for s in "${STATES[@]}"; do
  read -r resid temp press rho n <<<"$s"
  for mode in trained unit; do
    run_npt "$resid" "$temp" "$press" "$rho" "$mode" || rc=1
  done
done

echo
echo "=== summary ==="
for s in "${STATES[@]}"; do
  read -r resid temp press rho n <<<"$s"
  tag="${resid,,}_${temp%.*}k"
  for mode in trained unit; do
    f="$OUT_ROOT/runs/${tag}_${mode}/density.json"
    printf "  %-14s %-8s " "$tag" "$mode"
    if [[ -f "$f" ]]; then
      uv run python -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('rho=%.4f +/- %.4f  ref=%.4f  dev=%+.1f%%  %s' % (
  d['density_g_cm3'], d.get('sem',float('nan')), d['reference_g_cm3'],
  100*(d['density_g_cm3']/d['reference_g_cm3']-1), d.get('status','')))" "$f"
    else
      echo "(no result)"
    fi
  done
done
exit $rc
