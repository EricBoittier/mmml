#!/bin/bash
# Liquid boxes for validating the DES-trained LJ scales on a property OUTSIDE
# the training loss (density), per docs/des-lj-scales-handoff.md.
#
# Species chosen because they dominate the fit AND have liquid-phase reference
# densities: TIP3 63,159 frames / MEOH 6,660 / AMM1 12,762.
#
# TEMPERATURES ARE NOT ALL 298 K. Ammonia boils at 239.8 K, so a 298 K "liquid"
# box of AMM1 is a gas and its density is meaningless as a validation target.
# It is built and equilibrated at 240 K against rho = 0.6819 g/cm3 instead.
#
#   TIP3  298 K   0.9970 g/cm3   (water)
#   MEOH  298 K   0.7918 g/cm3   (methanol)
#   AMM1  240 K   0.6819 g/cm3   (ammonia, just below its boiling point)
#
# CPU-only (Packmol + PyCHARMM minimisation); no GPU needed, so this runs in
# parallel with the queued GPU fits.
set -uo pipefail

REPO="${MMML_REPO:-$HOME/mmml}"
cd "$REPO" || exit 1
source .venv/bin/activate 2>/dev/null
export UV_NO_SYNC=1 UV_OFFLINE=1
export JAX_PLATFORMS=cpu

# pcstudix PyCHARMM environment. libcharmm.so links OpenCL, and the repo venv
# does not carry libOpenCL.so.1 - without this every `import pycharmm` dies with
#   OSError: libOpenCL.so.1: cannot open shared object file
_CH="$REPO/setup/charmm"
_OCL="$HOME/micromamba/pkgs/ocl-icd-2.3.3-hb9d3cd8_0/lib"
export CHARMM_LIB_DIR="$_CH" CHARMM_HOME="$_CH"
export LD_LIBRARY_PATH="$_OCL:/usr/lib64/openmpi/lib/:${LD_LIBRARY_PATH:-}"

if ! uv run python -c "import pycharmm" >/dev/null 2>&1; then
  echo "ERROR: PyCHARMM not importable - cannot certify liquid boxes." >&2
  exit 1
fi

L="${BOX_SIZE:-28}"
OUT_ROOT="$REPO/artifacts/lj_scales_des_validation/boxes"
mkdir -p "$OUT_ROOT"

build() {
  local resid="$1" temp="$2" rho="$3"
  local out="$OUT_ROOT/${resid,,}"
  echo "=== ${resid}  L=${L} A  T=${temp} K  rho_target=${rho} g/cm3 -> ${out}"
  mkdir -p "$out"
  uv run mmml liquid-box \
    --composition "${resid}:1" \
    --box-auto count \
    --box-size "$L" \
    --target-density-g-cm3 "$rho" \
    --temperature "$temp" \
    --output-dir "$out" || { echo "  FAILED: $resid" >&2; return 1; }

  if [[ ! -f "$out/model.psf" || ! -f "$out/model.crd" ]]; then
    echo "  ERROR: $out missing model.psf / model.crd" >&2
    return 1
  fi
  [[ -f "$out/box.json" ]] && uv run python -c "
import json,sys
b=json.load(open(sys.argv[1]))
print(f\"  status={b.get('status')} N={b.get('n_molecules')} \"
      f\"L={b.get('box_side_A')} rho={b.get('density_g_cm3')}\")
" "$out/box.json"
  return 0
}

rc=0
build TIP3 298 0.9970 || rc=1
build MEOH 298 0.7918 || rc=1
build AMM1 240 0.6819 || rc=1

echo
echo "=== summary ==="
for d in "$OUT_ROOT"/*/; do
  n=$(basename "$d")
  if [[ -f "$d/model.psf" && -f "$d/model.crd" ]]; then echo "  OK   $n"; else echo "  MISS $n"; fi
done
exit $rc
