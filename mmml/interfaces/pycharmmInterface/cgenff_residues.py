"""Parse and display CGENFF residue names from the bundled RTF topology."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_RESI_LINE = re.compile(r"^RESI\s+(\S+)\s+(\S+)")


@dataclass(frozen=True, slots=True)
class CgenffResidue:
    name: str
    charge: str
    comment: str


def default_cgenff_rtf_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "charmm" / "top_all36_cgenff.rtf"


def parse_cgenff_residue_line(line: str) -> CgenffResidue | None:
    """Parse one ``RESI`` record from a CHARMM RTF file."""
    stripped = line.rstrip("\n")
    if not stripped.startswith("RESI"):
        return None
    comment = ""
    head = stripped
    if "!" in stripped:
        head, comment = stripped.split("!", 1)
        comment = comment.strip()
    match = _RESI_LINE.match(head.strip())
    if match is None:
        return None
    return CgenffResidue(name=match.group(1), charge=match.group(2), comment=comment)


def parse_cgenff_residues(rtf_path: Path | str | None = None) -> list[CgenffResidue]:
    """Return all ``RESI`` entries from ``top_all36_cgenff.rtf`` (sorted by name)."""
    path = Path(rtf_path) if rtf_path is not None else default_cgenff_rtf_path()
    residues: list[CgenffResidue] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            residue = parse_cgenff_residue_line(line)
            if residue is not None:
                residues.append(residue)
    residues.sort(key=lambda item: item.name.upper())
    return residues


def format_cgenff_residue_list(
    residues: list[CgenffResidue],
    *,
    rtf_path: Path | str | None = None,
) -> str:
    """Format residues as a fixed-width table for terminal or pager output."""
    path = Path(rtf_path) if rtf_path is not None else default_cgenff_rtf_path()
    if not residues:
        return f"No RESI records found in {path}\n"

    name_w = max(len("RESIDUE"), max(len(r.name) for r in residues))
    charge_w = max(len("CHARGE"), max(len(r.charge) for r in residues))
    lines = [
        f"CGENFF residues in {path}",
        f"{len(residues)} residue templates (RESI records)",
        "",
        f"{'RESIDUE':<{name_w}}  {'CHARGE':>{charge_w}}  DESCRIPTION",
        f"{'-' * name_w}  {'-' * charge_w}  {'-' * 11}",
    ]
    for residue in residues:
        desc = residue.comment or "(no comment in RTF)"
        lines.append(
            f"{residue.name:<{name_w}}  {residue.charge:>{charge_w}}  {desc}"
        )
    lines.append("")
    lines.append("Usage: mmml make-res --res RESIDUE")
    return "\n".join(lines) + "\n"


def show_cgenff_residue_list(
    *,
    rtf_path: Path | str | None = None,
    pager: bool | None = None,
) -> None:
    """Print CGENFF residue names; open ``less`` when stdout is a TTY (unless disabled)."""
    path = Path(rtf_path) if rtf_path is not None else default_cgenff_rtf_path()
    text = format_cgenff_residue_list(parse_cgenff_residues(path), rtf_path=path)
    use_pager = pager
    if use_pager is None:
        use_pager = sys.stdout.isatty() and shutil.which("less") is not None
    if use_pager:
        subprocess.run(
            ["less", "-R", "-F"],
            input=text,
            text=True,
            check=False,
        )
    else:
        sys.stdout.write(text)
