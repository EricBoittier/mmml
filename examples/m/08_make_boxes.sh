#!/usr/bin/env bash
# Export NPZ dimer → CGenFF PDB, then mmml make-box with ACN / TIP3 / DMSO.
# Needs Packmol + PyCHARMM (same as other make-box / md-system builds).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

SOLUTE_PDB="${SOLUTE_PDB:-${ARTIFACTS_DIR}/solute_amm1_ch3cl.pdb}"
BOX_SIZE="${BOX_SIZE:-30.0}"
N_SOLVENT="${N_SOLVENT:-12}"
FRAME="${FRAME:-}"

if [[ "${SKIP_SOLUTE_EXPORT:-0}" == "1" ]]; then
  echo "=== reuse solute PDB ${SOLUTE_PDB} (SKIP_SOLUTE_EXPORT=1) ==="
  if [[ ! -f "${SOLUTE_PDB}" ]]; then
    echo "FAIL: missing ${SOLUTE_PDB}" >&2
    exit 1
  fi
else
  echo "=== 07 export solute PDB ==="
  EXPORT_ARGS=(uv run python examples/m/07_export_solute_pdb.py -o "${SOLUTE_PDB}")
  if [[ -n "${FRAME}" ]]; then
    EXPORT_ARGS+=(--frame "${FRAME}")
  fi
  "${EXPORT_ARGS[@]}"
fi

# Optional: USE_DENSITY=1 sizes N from bulk density (overrides --n). Smoke keeps --n.
declare -A DENSITY=(
  [ACN]=786
  [TIP3]=1000
  [DMSO]=1100
)
USE_DENSITY="${USE_DENSITY:-0}"

# Packmol packing knobs. N from --density is sized to the cubic cell volume L³,
# so PACKMOL_REGION must be 'box' — packing the inscribed sphere instead only
# offers pi/6 = 52% of that volume and Packmol exits 173 ("failed to converge").
# FILL_FRACTION leaves headroom below ideal bulk density; lower it (or lower
# PACKMOL_TOLERANCE) if a solvent still fails to converge.
PACKMOL_REGION="${PACKMOL_REGION:-box}"
PACKMOL_TOLERANCE="${PACKMOL_TOLERANCE:-2.0}"
PACKMOL_NLOOP="${PACKMOL_NLOOP:-200}"
FILL_FRACTION="${FILL_FRACTION:-0.98}"

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
    cmd=(
      uv run mmml make-box
      --pdb "${SOLUTE_PDB}"
      --res "nh3ch3cl_${tag}"
      --box-size "${BOX_SIZE}"
      --solvent "${solvent}"
      --packmol-region "${PACKMOL_REGION}"
      --packmol-tolerance "${PACKMOL_TOLERANCE}"
      --packmol-nloop "${PACKMOL_NLOOP}"
      --fill-fraction "${FILL_FRACTION}"
    )
    if [[ "${USE_DENSITY}" == "1" ]]; then
      cmd+=(--density "${DENSITY[${solvent}]}")
    else
      cmd+=(--n "${N_SOLVENT}")
    fi
    "${cmd[@]}"
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
solvent = "${solvent}"
# Count the solvent residues Packmol actually placed: with --density the count
# is clamped to the cell capacity, so N_SOLVENT is not authoritative.
n_solvent = int("${N_SOLVENT}")
packed = out / "packmol_$(echo "${solvent}" | tr '[:upper:]' '[:lower:]')box.pdb"
if packed.is_file():
    seen = set()
    for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[17:21].strip() == solvent:
            # PDB cols 23-26 (0-index 22:26) = residue sequence number
            seen.add(line[22:26])
    if seen:
        n_solvent = len(seen)
(out / "box.json").write_text(
    json.dumps(
        {
            # Canonical key for md-system / liquid-box handoff.
            "box_side_A": side,
            # Aliases kept for hybrid umbrella + older readers.
            "box_size": side,
            "side_length_A": side,
            "solvent": solvent,
            "n_solvent": n_solvent,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"Wrote {out / 'box.json'} (n_solvent={n_solvent})")
PY
  )
  echo "PASS: ${solvent} box -> ${out}"
}

# SOLVENT_ONLY=tip3|acn|dmso (or TIP3|ACN|DMSO) builds a single solvent box.
_solvents=(ACN TIP3 DMSO)
if [[ -n "${SOLVENT_ONLY:-}" ]]; then
  case "$(echo "${SOLVENT_ONLY}" | tr '[:upper:]' '[:lower:]')" in
    acn) _solvents=(ACN) ;;
    tip3) _solvents=(TIP3) ;;
    dmso) _solvents=(DMSO) ;;
    *)
      echo "FAIL: SOLVENT_ONLY=${SOLVENT_ONLY} (expected tip3|acn|dmso)" >&2
      exit 1
      ;;
  esac
fi
for _s in "${_solvents[@]}"; do
  make_one "${_s}"
done

echo "PASS: make-box ${_solvents[*]} under ${ARTIFACTS_DIR}/boxes/"
