#!/usr/bin/env bash
# ACEM dihedral ic-scan: PhysNet vs KerNN (GT) vs KerNN (distill).
#
# Run from anywhere; paths default to ~/abirh layout used for ACEM training.
#
#   bash examples/kernnn/acem_dihedral_scan_compare.sh
#
# Env overrides:
#   ACEM_ROOT, PHYSNET_ACEM, KERNN_GT, KERNN_DISTILL, OUT, NPZ_REF, N_POINTS
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACEM_ROOT="${ACEM_ROOT:-$HOME/abirh}"
PHYSNET_ACEM="${PHYSNET_ACEM:-$ACEM_ROOT/params_acem1_2026-07-29_04-54-20.json}"
KERNN_GT="${KERNN_GT:-$ACEM_ROOT/artifacts/kernnn/acem_gt/best.json}"
KERNN_DISTILL="${KERNN_DISTILL:-$ACEM_ROOT/artifacts/kernnn/acem_distill/best.json}"
NPZ_REF="${NPZ_REF:-$ACEM_ROOT/splits/acem/energies_forces_dipoles_train.npz}"
OUT="${OUT:-$ACEM_ROOT/artifacts/ic_scan/acem_dihedrals}"
CFG_DIR="$OUT/configs"
EXAMPLE_CFG="$REPO/examples/ic_scan/acem_dihedrals.yaml"

mkdir -p "$OUT" "$CFG_DIR"

STRUCTURE="$REPO/examples/ic_scan/acem.xyz"
if [[ -f "$NPZ_REF" ]]; then
  STRUCTURE="$OUT/acem_ref.xyz"
  echo "=== structure from NPZ (matches training atom order) ==="
  uv run python "$REPO/examples/ic_scan/structure_from_npz.py" \
    --npz "$NPZ_REF" --out "$STRUCTURE"
else
  echo "WARNING: NPZ not found ($NPZ_REF); using bundled CGenFF acem.xyz"
fi

_write_cfg() {
  local name="$1" calculator="$2" checkpoint="$3"
  local dest="$CFG_DIR/${name}.yaml"
  # Keep DoFs from the example; swap calculator / checkpoint / structure.
  uv run python - <<PY
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path("$EXAMPLE_CFG").read_text())
cfg["structure"] = "$STRUCTURE"
cfg["calculator"] = "$calculator"
cfg["checkpoint"] = "$checkpoint"
cfg["evaluate"] = "energy"
Path("$dest").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote $dest")
PY
}

_write_cfg physnet physnet "$PHYSNET_ACEM"
_write_cfg kernnn_gt kernnn "$KERNN_GT"
_write_cfg kernnn_distill kernnn "$KERNN_DISTILL"

for name in physnet kernnn_gt kernnn_distill; do
  echo "=== ic-scan $name ==="
  uv run mmml ic-scan \
    --config "$CFG_DIR/${name}.yaml" \
    --output "$OUT/$name" \
    --overwrite
done

echo "=== compare plots ==="
uv run python - <<PY
from pathlib import Path
from mmml.ic_scan.result import ScanResult
from mmml.ic_scan.plotting import plot_model_comparison

root = Path("$OUT")
series = {
    "PhysNet": ScanResult.read(root / "physnet"),
    "KerNN GT": ScanResult.read(root / "kernnn_gt"),
    "KerNN distill": ScanResult.read(root / "kernnn_distill"),
}
paths = plot_model_comparison(series, root / "compare")
for p in paths:
    print(p)
print("done →", root / "compare")
PY
