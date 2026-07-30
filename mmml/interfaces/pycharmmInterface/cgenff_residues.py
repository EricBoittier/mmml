"""Parse and display CGENFF residue names from the bundled RTF topology."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_RESI_LINE = re.compile(r"^RESI\s+(\S+)\s+(\S+)")

# Common non-RESI spellings → CGenFF RESI names.
CGENFF_RESIDUE_ALIASES: dict[str, str] = {
    "WATER": "TIP3",
    "OCTANOL": "OCOH",
    "CH4": "METH",
    "METHANE": "METH",
}

# Colon- or comma-separated append RTF paths (extra RESI records + CHARMM append).
_EXTRA_RTF_ENV = "MMML_CGENFF_EXTRA_RTF"
# Colon- or comma-separated append PRM paths (bonded params for append residues).
_EXTRA_PRM_ENV = "MMML_CGENFF_EXTRA_PRM"


@dataclass(frozen=True, slots=True)
class CgenffResidue:
    name: str
    charge: str
    comment: str


def default_cgenff_rtf_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "charmm" / "top_all36_cgenff.rtf"


def _extra_paths_from_env(env_key: str, *, env: os._Environ | None = None) -> tuple[Path, ...]:
    environ = env if env is not None else os.environ
    raw = (environ.get(env_key) or "").strip()
    if not raw:
        return ()
    out: list[Path] = []
    for tok in re.split(r"[:,]", raw):
        part = tok.strip()
        if not part:
            continue
        path = Path(os.path.expandvars(part)).expanduser().resolve()
        if path.is_file():
            out.append(path)
    return tuple(out)


def extra_cgenff_rtf_paths(*, env: os._Environ | None = None) -> tuple[Path, ...]:
    """Append-topology RTF paths from ``MMML_CGENFF_EXTRA_RTF`` (``:`` / ``,`` separated)."""
    return _extra_paths_from_env(_EXTRA_RTF_ENV, env=env)


def extra_cgenff_prm_paths(*, env: os._Environ | None = None) -> tuple[Path, ...]:
    """Append-parameter PRM paths from ``MMML_CGENFF_EXTRA_PRM`` (``:`` / ``,`` separated)."""
    return _extra_paths_from_env(_EXTRA_PRM_ENV, env=env)


def normalize_cgenff_residue_name(name: str) -> str:
    """Return an uppercase CGenFF residue name, applying common aliases."""
    key = str(name).strip().upper()
    if not key:
        raise ValueError("residue name must not be empty")
    return CGENFF_RESIDUE_ALIASES.get(key, key)


@lru_cache(maxsize=8)
def cgenff_residue_name_set(
    rtf_path: str | None = None,
    extra_rtf_paths: tuple[str, ...] = (),
) -> frozenset[str]:
    """Uppercase RESI names from the bundled (or given) CGenFF RTF plus extras."""
    path = Path(rtf_path) if rtf_path is not None else default_cgenff_rtf_path()
    names = {r.name.upper() for r in parse_cgenff_residues(path)}
    for extra in extra_rtf_paths:
        names.update(r.name.upper() for r in parse_cgenff_residues(extra))
    return frozenset(names)


def is_cgenff_residue_name(name: str, *, rtf_path: Path | str | None = None) -> bool:
    """True if *name* (after alias normalization) is a RESI in the CGenFF RTF."""
    key = normalize_cgenff_residue_name(name)
    path = None if rtf_path is None else str(Path(rtf_path))
    extras = tuple(str(p) for p in extra_cgenff_rtf_paths())
    return key in cgenff_residue_name_set(path, extras)


def require_cgenff_residue_name(name: str, *, rtf_path: Path | str | None = None) -> str:
    """Normalize and validate a CGenFF residue name; raise ``ValueError`` if unknown."""
    key = normalize_cgenff_residue_name(name)
    if not is_cgenff_residue_name(key, rtf_path=rtf_path):
        raise ValueError(
            f"Unknown CGenFF residue {name!r} (normalized {key!r}). "
            "List valid names with: mmml make-res --list-residues "
            f"(or append via {_EXTRA_RTF_ENV})"
        )
    return key


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
