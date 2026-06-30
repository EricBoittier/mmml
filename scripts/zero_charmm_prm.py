#!/usr/bin/env python3
"""Zero bonded and nonbonded force constants in a CHARMM parameter (.prm) file.

Preserves atom types, equilibrium geometry (r0, theta0, dihedral phase/multiplicity),
masses, and comments. Intended for MLpot runs where CHARMM MM terms should contribute
no energy but topology lookup must remain intact.

Usage:
    python scripts/zero_charmm_prm.py \\
        mmml/data/charmm/par_all36_cgenff.prm \\
        mmml/data/charmm/zeroed_par_all36_cgenff.prm
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# CHARMM atom-type tokens (CGenFF / all36).
_ATOM = r"[A-Za-z0-9]+"

_BOND = re.compile(rf"^(\s*)({_ATOM})\s+({_ATOM})\s+([\d.-]+)\s+([\d.-]+)(\s*.*)$")
_ANGLE = re.compile(rf"^(\s*)({_ATOM})\s+({_ATOM})\s+({_ATOM})\s+([\d.-]+)\s+([\d.-]+)(\s*.*)$")
_DIHEDRAL = re.compile(
    rf"^(\s*)({_ATOM})\s+({_ATOM})\s+({_ATOM})\s+({_ATOM})\s+([\d.-]+)\s+(\d+)\s+([\d.-]+)(\s*.*)$"
)
_NONBONDED = re.compile(rf"^(\s*)({_ATOM})\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)(\s*.*)$")
_NBFIX = re.compile(rf"^(\s*)({_ATOM})\s+({_ATOM})\s+([\d.-]+)\s+([\d.-]+)(\s*.*)$")

_SECTION_HEADERS = frozenset(
    {
        "BONDS",
        "ANGLES",
        "DIHEDRALS",
        "IMPROPERS",
        "NONBONDED",
        "NBFIX",
        "HBOND",
        "END",
    }
)


def _section_from_line(line: str) -> str | None:
    token = line.strip().split()[0] if line.strip() else ""
    if token in _SECTION_HEADERS:
        return token
    return None


def zero_prm_line(
    line: str,
    section: str | None,
    *,
    skip_sections: frozenset[str] = frozenset(),
) -> str:
    """Return *line* with force constants zeroed for the active *section*."""
    if section in skip_sections:
        return line
    if section == "BONDS":
        m = _BOND.match(line)
        if m:
            lead, a1, a2, _kb, r0, tail = m.groups()
            return f"{lead}{a1}  {a2}    0.0       {r0}{tail}"
    elif section == "ANGLES":
        m = _ANGLE.match(line)
        if m:
            lead, a1, a2, a3, _k, theta0, tail = m.groups()
            return f"{lead}{a1}  {a2}  {a3}    0.0      {theta0}{tail}"
    elif section in ("DIHEDRALS", "IMPROPERS"):
        m = _DIHEDRAL.match(line)
        if m:
            lead, a1, a2, a3, a4, _vn, n, gamma, tail = m.groups()
            return f"{lead}{a1}  {a2}  {a3}  {a4}  0.0 {n}    {gamma}{tail}"
    elif section == "NONBONDED":
        m = _NONBONDED.match(line)
        if m:
            lead, atype, ignored, _eps, rmin, tail = m.groups()
            return f"{lead}{atype}     {ignored}        0.0     {rmin}{tail}"
    elif section == "NBFIX":
        m = _NBFIX.match(line)
        if m:
            lead, a1, a2, _eps, rmin, tail = m.groups()
            return f"{lead}{a1}  {a2}    0.0        {rmin}{tail}"
    return line


_OMIT_SECTIONS_BONDED_ONLY = frozenset({"NONBONDED", "NBFIX", "HBOND"})


def _nonbonded_atom_line(line: str) -> bool:
    return _NONBONDED.match(line) is not None


def bonded_only_prm_text(text: str, *, zero_constants: bool = True) -> str:
    """Keep BOND/ANGL/DIHE/IMPR sections only; omit NONBONDED/NBFIX/HBOND.

    When *zero_constants* is True, bonded force constants are zeroed (MLpot path).
    When False, lines are copied verbatim (append-safe bonded restore).
    """
    skip = frozenset({"NONBONDED", "NBFIX"}) if zero_constants else frozenset()
    section: str | None = None
    omit_section = False
    out: list[str] = []
    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\r\n")
        newline = raw[len(body) :]
        new_section = _section_from_line(body)
        if new_section is not None:
            if new_section == "END":
                section = None
                omit_section = False
                out.append(raw)
                continue
            section = new_section
            omit_section = section in _OMIT_SECTIONS_BONDED_ONLY
            if omit_section or section in ("NONBONDED", "HBOND"):
                continue
            out.append(raw)
            continue
        if omit_section:
            continue
        if section == "NONBONDED":
            continue
        if zero_constants:
            out.append(zero_prm_line(body, section, skip_sections=skip) + newline)
        else:
            out.append(raw)
    return "".join(out)


def zero_prm_text(text: str, *, bonded_only: bool = False) -> str:
    if bonded_only:
        return bonded_only_prm_text(text, zero_constants=True)
    skip = frozenset()
    section: str | None = None
    omit_section = False
    out: list[str] = []
    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\r\n")
        newline = raw[len(body) :]
        new_section = _section_from_line(body)
        if new_section is not None:
            if new_section == "END":
                section = None
                omit_section = False
                out.append(raw)
                continue
            section = new_section
            omit_section = False
            if section in ("NONBONDED", "HBOND"):
                continue
            out.append(raw)
            continue
        if omit_section:
            continue
        if section == "NONBONDED":
            stripped = body.strip()
            if not stripped:
                continue
            if stripped.startswith("!"):
                out.append(raw)
                continue
            if stripped.lower().startswith("cutnb") or "nbxmod" in stripped.lower():
                continue
            if not _nonbonded_atom_line(body):
                continue
        out.append(zero_prm_line(body, section, skip_sections=skip) + newline)
    return "".join(out)


def extract_bonded_prm_file(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8", errors="replace")
    header = (
        "*  BONDED COPY — BOND/ANGL/DIHE/IMPR only (append-safe; no NONBONDED/HBOND)\n"
        f"*  Source: {src.name}\n"
        f"*  Generated by: {Path(__file__).name}\n"
        "*  --------------------------------------------------------------------------  *\n"
    )
    bonded = header + bonded_only_prm_text(text, zero_constants=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(bonded, encoding="utf-8")


def zero_prm_file(src: Path, dst: Path, *, bonded_only: bool = False) -> None:
    text = src.read_text(encoding="utf-8", errors="replace")
    if bonded_only:
        header = (
            "*  ZEROED BONDED COPY — BOND/ANGL/DIHE/IMPR force constants set to 0.0\n"
            "*  NONBONDED / NBFIX left unchanged (periodic CHARMM VDW)\n"
            f"*  Source: {src.name}\n"
            f"*  Generated by: {Path(__file__).name}\n"
            "*  --------------------------------------------------------------------------  *\n"
        )
    else:
        header = (
            "*  ZEROED COPY — bonded/nonbond atom params zeroed (no NB control lines)\n"
            "*  Safe for READ PARAM APPEND (skips NONBONDED nbxmod / HBOND headers)\n"
            f"*  Source: {src.name}\n"
            f"*  Generated by: {Path(__file__).name}\n"
            "*  --------------------------------------------------------------------------  *\n"
        )
    zeroed = header + zero_prm_text(text, bonded_only=bonded_only)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(zeroed, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Input CHARMM .prm file")
    parser.add_argument("destination", type=Path, help="Output zeroed .prm file")
    parser.add_argument(
        "--bonded-only",
        action="store_true",
        help="Zero bonded terms only; omit NONBONDED/NBFIX/HBOND (append-safe)",
    )
    parser.add_argument(
        "--extract-bonded-only",
        action="store_true",
        help="Copy bonded sections only without zeroing (append-safe MM restore)",
    )
    args = parser.parse_args(argv)
    if not args.source.is_file():
        print(f"error: source not found: {args.source}", file=sys.stderr)
        return 1
    if args.extract_bonded_only and args.bonded_only:
        print("error: use only one of --bonded-only or --extract-bonded-only", file=sys.stderr)
        return 1
    if args.extract_bonded_only:
        extract_bonded_prm_file(args.source, args.destination)
    else:
        zero_prm_file(args.source, args.destination, bonded_only=args.bonded_only)
    print(f"Wrote {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
