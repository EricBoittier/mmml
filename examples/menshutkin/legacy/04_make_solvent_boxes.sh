#!/usr/bin/env bash
# Solvate the NH3 + CH3Cl solute in the five Turan et al. solvents.
#
# Turan, Brickel & Meuwly (J. Phys. Chem. B 126, 1951 (2022)) used water,
# methanol, acetonitrile, benzene and cyclohexane -- a ladder from strongly
# polar protic to essentially non-polar, which is what makes the barrier vary by
# ~16 kcal/mol across the set. Box sides follow the paper; solvent counts come
# from experimental densities at 298 K rather than being hand-set, so the boxes
# start near the right density instead of relying on equilibration to fix it.
#
#   source examples/menshutkin/_env.sh
#   bash examples/menshutkin/04_make_solvent_boxes.sh
#   SMOKE=1 bash examples/menshutkin/04_make_solvent_boxes.sh   # 12 molecules, L=20
#   SOLVENTS="water:TIP3:997:30" bash examples/menshutkin/04_make_solvent_boxes.sh
#
# Needs Packmol + PyCHARMM. Outputs per solvent:
#   $MENSH_ARTIFACTS/boxes/<name>/{model.pdb,model.psf,box.json}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/menshutkin/_env.sh"
cd "${ROOT}"

BOXES="${MENSH_ARTIFACTS}/boxes"
SOLUTE_PDB="${SOLUTE_PDB:-${MENSH_ARTIFACTS}/solute_amm1_mecl.pdb}"
SOLVENTS="${SOLVENTS:-${MENSH_SOLVENTS}}"

# Seed the solute at the transition state. The umbrella windows re-seed the
# solute anyway, but building the box around a TS-like geometry means the
# solvent shell is equilibrated around the charge distribution that matters,
# rather than around a neutral reactant pair that then has to reorganise.
SOLUTE_XI="${SOLUTE_XI:-0.0}"

echo "=== solute PDB (xi = ${SOLUTE_XI}) ==="
uv run python examples/menshutkin/05_export_solute.py --xi "${SOLUTE_XI}" -o "${SOLUTE_PDB}"

if ! uv run python -c "import pycharmm" >/dev/null 2>&1; then
  echo "SKIP: PyCHARMM not importable; solute PDB is ready at ${SOLUTE_PDB}"
  exit 0
fi

mkdir -p "${BOXES}"
for entry in ${SOLVENTS}; do
  IFS=: read -r name resi density side <<<"${entry}"
  work="${MENSH_ARTIFACTS}/make_box_work_${name}"
  out="${BOXES}/${name}"
  rm -rf "${work}"
  mkdir -p "${work}" "${out}"

  echo
  echo "=== ${name} (RESI ${resi}, rho=${density} kg/m3, L=${side} A) ==="
  (
    cd "${work}"
    cmd=(uv run mmml make-box
         --pdb "${SOLUTE_PDB}"
         --res "mensh_${name}"
         --solvent "${resi}")
    if [[ "${SMOKE:-0}" == "1" ]]; then
      cmd+=(--box-size 20.0 --n 12)
    else
      cmd+=(--box-size "${side}" --density "${density}")
    fi
    "${cmd[@]}"

    cp -f "pdb/init-mensh_${name}.pdb" "${out}/model.pdb"
    cp -f "psf/system-mensh_${name}.psf" "${out}/model.psf"
  )

  side_used="${side}"
  [[ "${SMOKE:-0}" == "1" ]] && side_used=20.0
  uv run python - "${out}" "${name}" "${resi}" "${density}" "${side_used}" <<'PY'
import json, sys
from pathlib import Path

out, name, resi, density, side = sys.argv[1:6]
out = Path(out)
pdb = (out / "model.pdb").read_text().splitlines()
atoms = [ln for ln in pdb if ln.startswith(("ATOM", "HETATM"))]
resids = {ln[22:27].strip() + ln[17:21].strip() for ln in atoms}
n_solvent = sum(1 for r in resids if resi.upper() in r.upper())
(out / "box.json").write_text(json.dumps({
    "solvent": name,
    "residue": resi,
    "density_kg_m3": float(density),
    "box_size": float(side),
    "side_length_A": float(side),
    "n_atoms_total": len(atoms),
    "n_solvent_residues": n_solvent,
}, indent=2) + "\n")
print(f"  {len(atoms)} atoms, ~{n_solvent} {resi} residues -> {out/'box.json'}")
PY
  echo "PASS: ${name} -> ${out}"
done

echo
echo "All boxes under ${BOXES}"
