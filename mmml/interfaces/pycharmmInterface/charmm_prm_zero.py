"""Zero CHARMM .prm force constants for MLpot PSF/param overlays (library)."""

from __future__ import annotations

import re
from pathlib import Path

# CHARMM atom-type tokens (CGenFF / all36).
_ATOM = r"[A-Za-z0-9]+"

_BOND = re.compile(rf"^(\s*)({_ATOM})\s+({_ATOM})\s+([\d.-]+)\s+([\d.-]+)(\s*.*)$")
_ANGLE = re.compile(rf"^(\s*)({_ATOM})\s+({_ATOM})\s+({_ATOM})\s+([\d.-]+)\s+([\d.-]+)(\s*.*)$")
_ANGLE_UB = re.compile(
    rf"^(\s*)({_ATOM})\s+({_ATOM})\s+({_ATOM})\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)(\s*.*)$"
)
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

_OMIT_SECTIONS_BONDED_ONLY = frozenset({"NONBONDED", "NBFIX", "HBOND"})


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
        m_ub = _ANGLE_UB.match(line)
        if m_ub:
            lead, a1, a2, a3, _k, theta0, _kub, rub, tail = m_ub.groups()
            return f"{lead}{a1}  {a2}  {a3}    0.0      {theta0}   0.0     {rub}{tail}"
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
            tail_parts = tail.split("!", 1)
            comment = ""
            if len(tail_parts) > 1:
                comment = " !" + tail_parts[1]
            tail_vals = tail_parts[0].split()
            if len(tail_vals) >= 3:
                p_14, _eps_14, rmin_14 = tail_vals[:3]
                tail = f"   {p_14}    0.0     {rmin_14}{comment}"
            return f"{lead}{atype}     {ignored}        0.0     {rmin}{tail}"
    elif section == "NBFIX":
        m = _NBFIX.match(line)
        if m:
            lead, a1, a2, _eps, rmin, tail = m.groups()
            return f"{lead}{a1}  {a2}    0.0        {rmin}{tail}"
    return line


def _nonbonded_atom_line(line: str) -> bool:
    return _NONBONDED.match(line) is not None


def bonded_only_prm_text(text: str, *, zero_constants: bool = True) -> str:
    """Keep bonded sections only; omit NONBONDED/NBFIX/HBOND (append-safe)."""
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


def nonbond_only_prm_text(text: str) -> str:
    """Append-safe marker for VDW removal with no NONBONDED/NBFIX records.

    VDW-bearing CHARMM parameter rows live in ``NONBONDED`` and ``NBFIX``.
    Re-emitting those rows, even with zero epsilon, still includes the VDW term
    in patched PRM files.  The runtime energy-policy path clears live CHARMM VDW
    tables separately, so the PRM patch itself must omit these sections.
    """
    section: str | None = None
    saw_nonbond = False

    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\r\n")
        new_section = _section_from_line(body)
        if new_section is not None:
            if new_section == "END":
                section = None
                continue
            section = new_section
            if section in _OMIT_SECTIONS_BONDED_ONLY:
                saw_nonbond = True
            continue
    if not saw_nonbond:
        return ""
    return "! MMML: VDW term removed from PRM patch\n"


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
            omit_section = section in _OMIT_SECTIONS_BONDED_ONLY
            if omit_section:
                continue
            out.append(raw)
            continue
        if omit_section:
            continue
        out.append(zero_prm_line(body, section, skip_sections=skip) + newline)
    return "".join(out)


def build_prm_policy_overlay_text(
    text: str,
    *,
    zero_bonded: bool = False,
    zero_nonbond: bool = False,
) -> str:
    """Build a READ PARAM APPEND overlay for selected zeroed sections."""
    if zero_bonded and zero_nonbond:
        return zero_prm_text(text, bonded_only=False)
    if zero_bonded:
        return bonded_only_prm_text(text, zero_constants=True)
    if zero_nonbond:
        return nonbond_only_prm_text(text)
    return ""


def write_prm_policy_overlay(
    src: Path,
    dst: Path,
    *,
    zero_bonded: bool = False,
    zero_nonbond: bool = False,
    note: str = "",
) -> Path:
    """Write an append-safe .prm overlay; return *dst*."""
    text = src.read_text(encoding="utf-8", errors="replace")
    body = build_prm_policy_overlay_text(
        text,
        zero_bonded=zero_bonded,
        zero_nonbond=zero_nonbond,
    )
    if not body.strip():
        raise ValueError("prm policy overlay is empty (no sections selected)")
    header = (
        "*  MMML energy-policy overlay (READ PARAM APPEND)\n"
        f"*  Source: {src.name}\n"
        f"*  zero_bonded={zero_bonded} zero_nonbond={zero_nonbond}\n"
    )
    if note:
        header += f"*  {note}\n"
    header += "*  --------------------------------------------------------------------------  *\n"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(header + body, encoding="utf-8")
    return dst


def zero_prm_file(src: Path, dst: Path, *, bonded_only: bool = False) -> None:
    text = src.read_text(encoding="utf-8", errors="replace")
    if bonded_only:
        header = (
            "*  ZEROED BONDED COPY — BOND/ANGL/DIHE/IMPR/UREY-b force constants set to 0.0\n"
            "*  NONBONDED / NBFIX / HBOND omitted (append-safe; lists not cleared)\n"
            f"*  Source: {src.name}\n"
            "*  --------------------------------------------------------------------------  *\n"
        )
    else:
        header = (
            "*  ZEROED COPY — bonded/nonbond atom params zeroed (no NB control lines)\n"
            "*  Safe for READ PARAM APPEND (skips NONBONDED nbxmod / HBOND headers)\n"
            f"*  Source: {src.name}\n"
            "*  --------------------------------------------------------------------------  *\n"
        )
    zeroed = header + zero_prm_text(text, bonded_only=bonded_only)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(zeroed, encoding="utf-8")
