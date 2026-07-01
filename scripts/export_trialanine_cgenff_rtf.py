"""Build ``RESI TRIALANINE`` for the bundled CGENFF RTF (ACE–ALA×3–CT3).

Run under PyCHARMM::

    ./scripts/mmml-charmm-mpirun.sh python scripts/export_trialanine_cgenff_rtf.py

Writes ``mmml/data/charmm/top_trialanine_cgenff.rtf`` and appends a line to
``CGENFF.RES``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_RTF = REPO / "mmml" / "data" / "charmm" / "top_trialanine_cgenff.rtf"
OUT_RES = REPO / "mmml" / "data" / "charmm" / "CGENFF.RES"
RESI_NAME = "TRIA"  # ACE–ALA×3–CT3 tri-alanine (CHARMM sequence names ≤ 4 characters)

# Protein → CGENFF atom-type map (bonded params live in ``par_all36_cgenff.prm``).
_PROTEIN_TO_CGENFF = {
    "H": "HGP1",
    "HC": "HGA3",
    "HA": "HGA1",
    "HP": "HGP1",
    "HB1": "HGA1",
    "HB2": "HGA2",
    "HB3": "HGA3",
    "HN": "HGP1",
    "HNT": "HGP1",
    "HT1": "HGA3",
    "HT2": "HGA3",
    "HT3": "HGA3",
    "HY1": "HGA3",
    "HY2": "HGA3",
    "HY3": "HGA3",
    "HA1": "HGA1",
    "HA2": "HGA2",
    "HA3": "HGA3",
    "C": "CG2O1",
    "CA": "CG311",
    "CAT": "CG331",
    "CAY": "CG331",
    "CB": "CG331",
    "CY": "CG2O1",
    "CT": "CG331",
    "CT1": "CG311",
    "CT2": "CG321",
    "CT3": "CG331",
    "CC": "CG2O1",
    "CD": "CG2O1",
    "N": "NG2S1",
    "NT": "NG2S1",
    "NH1": "NG2S1",
    "NH2": "NG2S1",
    "NH3": "NG3P3",
    "O": "OG2D1",
    "OY": "OG2D1",
    "OC": "OG2D2",
    "OH1": "OG301",
}


def _map_atype(protein_type: str) -> str:
    key = protein_type.strip().upper()
    if key in _PROTEIN_TO_CGENFF:
        return _PROTEIN_TO_CGENFF[key]
    raise KeyError(f"No CGENFF mapping for protein atom type {protein_type!r}")


def _protein_toppar_paths() -> tuple[Path, Path]:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CHARMM_HOME

    base = Path(CHARMM_HOME) / "toppar"
    rtf = base / "top_all36_prot.rtf"
    prm = base / "par_all36m_prot.prm"
    if not prm.is_file():
        prm = base / "par_all36_prot.prm"
    if not rtf.is_file() or not prm.is_file():
        raise FileNotFoundError(f"Protein toppar not found under {base}")
    return rtf, prm


def _build_minimized_trialanine() -> None:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded

    ensure_pycharmm_loaded()
    from mmml.interfaces.pycharmmInterface import setupRes
    import pycharmm.read as read
    import pycharmm.generate as generate
    import pycharmm.lingo as lingo
    from mmml.interfaces.pycharmmInterface.import_pycharmm import reset_block
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    rtf, prm = _protein_toppar_paths()
    lingo.charmm_script("DELETE ATOM SELE ALL END")
    reset_block()
    with charmm_relaxed_bomlev():
        read.rtf(str(rtf))
        read.prm(str(prm))
    read.sequence_string("ALA ALA ALA")
    generate.new_segment(
        seg_name="TRIA",
        first_patch="ACE",
        last_patch="CT3",
        setup_ic=True,
    )
    setupRes.generate_coordinates(skip_energy_show=True, validate=True)


def _atom_names_in_psf_order() -> list[str]:
    from collections import Counter
    from pathlib import Path

    pdb_path = Path("pdb/initial.pdb")
    if not pdb_path.is_file():
        raise FileNotFoundError(f"Expected {pdb_path} after setupRes.generate_coordinates")
    raw_names: list[str] = []
    resnums: list[int] = []
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        raw_names.append(line[12:16].strip())
        resnums.append(int(line[22:26]))
    if not raw_names:
        raise RuntimeError(f"No ATOM records in {pdb_path}")

    duplicates = {name for name, count in Counter(raw_names).items() if count > 1}
    unique: list[str] = []
    used: set[str] = set()
    for name, rnum in zip(raw_names, resnums):
        if name not in duplicates:
            uname = name
        else:
            suffix = str(rnum)
            uname = f"{name[: 4 - len(suffix)]}{suffix}"[:4]
        base = uname
        nudge = 1
        while uname in used:
            suffix = f"{rnum}{nudge}"
            uname = f"{base[: 4 - len(suffix)]}{suffix}"[:4]
            nudge += 1
        used.add(uname)
        unique.append(uname)
    return unique


def _format_rtf_block() -> str:
    import pycharmm.psf as psf

    n = int(psf.get_natom())
    atypes = psf.get_atype()
    charges = psf.get_charges()
    names = _atom_names_in_psf_order()
    if len(names) != n:
        raise RuntimeError(f"Atom name count {len(names)} != PSF natom {n}")

    ib, jb = psf.get_ib_jb()
    bonds: list[tuple[int, int]] = []
    for i, j in zip(ib, jb):
        a, b = int(i), int(j)
        if a > b:
            a, b = b, a
        bonds.append((a, b))
    bonds = sorted(set(bonds))

    lines: list[str] = [
        "* TRIALANINE — ACE–ALA×3–CT3 capped tri-alanine (CGENFF atom types)",
        "* Regenerate: ./scripts/mmml-charmm-mpirun.sh python scripts/export_trialanine_cgenff_rtf.py",
        "",
        f"RESI {RESI_NAME:<8}  0.00 ! C12H22N4O5, ACE–ALA×3–CT3 (TRIALANINE)",
    ]

    for idx in range(n):
        name = str(names[idx]).strip()
        cg_type = _map_atype(str(atypes[idx]))
        charge = float(charges[idx])
        if idx == 0 or idx % 8 == 0:
            lines.append("GROUP")
        lines.append(f"ATOM {name:<4} {cg_type:<6} {charge:7.2f}")

    bond_chunks: list[str] = []
    for a, b in bonds:
        bond_chunks.extend([names[a - 1], names[b - 1]])
    for i in range(0, len(bond_chunks), 8):
        chunk = bond_chunks[i : i + 8]
        lines.append("BOND " + " ".join(f"{tok:<4}" for tok in chunk))

    lines.append("PATC FIRS NONE LAST NONE")
    lines.append("")
    return "\n".join(lines) + "\n"


def _ensure_cgenff_res_entry() -> None:
    entry = f"RESI {RESI_NAME:<8}  0.00 ! ACE–ALA×3–CT3 tri-alanine (TRIALANINE bundle)"
    text = OUT_RES.read_text(encoding="utf-8")
    if f"RESI {RESI_NAME}" in text or "RESI TRIALAN" in text or "RESI TRIALANINE" in text:
        return
    lines = text.splitlines()
    insert_at = next(
        (i for i, ln in enumerate(lines) if ln.startswith("RESI TIP3")),
        len(lines),
    )
    lines.insert(insert_at, entry)
    OUT_RES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import tempfile

    workdir = Path(tempfile.mkdtemp(prefix="trialanine_rtf_"))
    try:
        os.chdir(workdir)
        _build_minimized_trialanine()
        OUT_RTF.parent.mkdir(parents=True, exist_ok=True)
        OUT_RTF.write_text(_format_rtf_block(), encoding="utf-8")
        _ensure_cgenff_res_entry()
        print(f"wrote {OUT_RTF.relative_to(REPO)} ({OUT_RTF.stat().st_size} bytes)")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
