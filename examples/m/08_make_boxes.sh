#!/usr/bin/env bash
# Export NPZ dimer → CGenFF PDB, then mmml make-box with ACN / TIP3 / DMSO.
# Needs Packmol + PyCHARMM (same as other make-box / md-system builds).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

SOLUTE_PDB="${SOLUTE_PDB:-${ARTIFACTS_DIR}/solute_amm1_ch3cl.pdb}"
BOX_SIZE="${BOX_SIZE:-20.0}"
N_SOLVENT="${N_SOLVENT:-12}"
FRAME="${FRAME:-}"

echo "=== 07 export solute PDB ==="
EXPORT_ARGS=(uv run python examples/m/07_export_solute_pdb.py -o "${SOLUTE_PDB}")
if [[ -n "${FRAME}" ]]; then
  EXPORT_ARGS+=(--frame "${FRAME}")
fi
"${EXPORT_ARGS[@]}"

# Densities (kg/m³) for make-box --density when sizing from density; smoke uses --n.
declare -A DENSITY=(
  [ACN]=786
  [TIP3]=1000
  [DMSO]=1100
)

has_pycharmm=0
if uv run python -c "import pycharmm" >/dev/null 2>&1; then
  has_pycharmm=1
fi
if [[ "${has_pycharmm}" != "1" ]]; then
  echo "SKIP: PyCHARMM not importable — cannot run make-box"
  echo "      Solute PDB is ready at ${SOLUTE_PDB}"
  exit 0
fi

make_one() {
  local solvent="$1"
  local tag
  tag="$(echo "${solvent}" | tr '[:upper:]' '[:lower:]')"
  local work="${ARTIFACTS_DIR}/make_box_work_${tag}"
  local out="${ARTIFACTS_DIR}/boxes/${tag}"
  rm -rf "${work}"
  mkdir -p "${work}" "${out}"
  (
    cd "${work}"
    echo "=== make-box --solvent ${solvent} (n=${N_SOLVENT}, L=${BOX_SIZE}) ==="
    uv run mmml make-box \
      --pdb "${SOLUTE_PDB}" \
      --res "nh3ch3cl_${tag}" \
      --n "${N_SOLVENT}" \
      --box-size "${BOX_SIZE}" \
      --solvent "${solvent}" \
      --density "${DENSITY[${solvent}]}"
    # Collect Packmol + CHARMM products
    cp -f "pdb/init-${tag}box.pdb" "${out}/packmol_${tag}box.pdb" 2>/dev/null || true
    cp -f "pdb/init-nh3ch3cl_${tag}.pdb" "${out}/model.pdb"
    cp -f "psf/system-nh3ch3cl_${tag}.psf" "${out}/model.psf"
    # Sibling box.json so md-system --from-pdb can resolve L without --box-size.
    uv run python - <<PY
import json
from pathlib import Path
out = Path("${out}")
side = float("${BOX_SIZE}")
(out / "box.json").write_text(
    json.dumps({"box_size": side, "side_length_A": side, "solvent": "${solvent}", "n_solvent": int("${N_SOLVENT}")}, indent=2),
    encoding="utf-8",
)
print(f"Wrote {out / 'box.json'}")
PY
  )
  echo "PASS: ${solvent} box -> ${out}"
}

make_one ACN
make_one TIP3
make_one DMSO

echo "PASS: make-box ACN/TIP3/DMSO under ${ARTIFACTS_DIR}/boxes/"
