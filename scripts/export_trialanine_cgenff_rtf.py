"""Export RESI TRIALANINE from a minimized ACE–ALA×3–CT3 build (one-time maintainer script)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_RTF = REPO / "mmml" / "data" / "charmm" / "top_trialanine_cgenff.rtf"
OUT_RES_LINE = REPO / "mmml" / "data" / "charmm" / "CGENFF.RES"


def _export_trialanine_rtf(workdir: Path) -> Path:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded

    ensure_pycharmm_loaded()
    from mmml.interfaces.pycharmmInterface import setupRes
    import pycharmm.read as read
    import pycharmm.generate as generate
    import pycharmm.lingo as lingo
    from mmml.interfaces.pycharmmInterface.import_pycharmm import reset_block
    from mmml.interfaces.pycharmmInterface.charmm_levels import (
        charmm_relaxed_bomlev,
        run_charmm_script_quiet,
    )
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import protein_toppar_paths

    os.chdir(workdir)
    rtf, prm = protein_toppar_paths()
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
    run_charmm_script_quiet("WRIT RTF CARD NAME TRIALANINE SELE ALL END")
    candidates = sorted(workdir.glob("*.rtf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"No RTF written under {workdir}")
    return candidates[0]


def _normalize_exported_rtf(src: Path) -> str:
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = [
        "* TRIALANINE — ACE–ALA×3–CT3 capped tri-alanine (exported for CGENFF bundle)",
        "* Regenerate: ./scripts/mmml-charmm-mpirun.sh python scripts/export_trialanine_cgenff_rtf.py",
        "",
    ]
    in_resi = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("RESI"):
            in_resi = stripped.startswith("RESI TRIALANINE") or stripped.startswith("RESI TRIA")
            if in_resi and stripped.startswith("RESI TRIA"):
                line = line.replace("RESI TRIA", "RESI TRIALANINE", 1)
        if in_resi:
            out.append(line.rstrip())
        if in_resi and stripped == "END" and "READ" not in stripped:
            break
    if not any(l.startswith("RESI TRIALANINE") for l in out):
        raise RuntimeError(f"RESI TRIALANINE not found in {src}")
    if not out[-1].strip().endswith("END"):
        out.append("END")
    return "\n".join(out) + "\n"


def _ensure_cgenff_res_entry() -> None:
    text = OUT_RES_LINE.read_text(encoding="utf-8")
    entry = "RESI TRIALANINE     0.00 ! ACE–ALA×3–CT3 capped tri-alanine (mmml bundle)"
    if "RESI TRIALANINE" in text:
        return
    lines = text.splitlines()
    insert_at = next(
        (i for i, ln in enumerate(lines) if ln.startswith("RESI TIP3")),
        len(lines),
    )
    lines.insert(insert_at, entry)
    OUT_RES_LINE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import tempfile

    workdir = Path(tempfile.mkdtemp(prefix="trialanine_rtf_"))
    try:
        exported = _export_trialanine_rtf(workdir)
        OUT_RTF.parent.mkdir(parents=True, exist_ok=True)
        OUT_RTF.write_text(_normalize_exported_rtf(exported), encoding="utf-8")
        _ensure_cgenff_res_entry()
        print(f"wrote {OUT_RTF.relative_to(REPO)} ({OUT_RTF.stat().st_size} bytes)")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
