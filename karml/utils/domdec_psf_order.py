"""Offline DOMDEC atom-order checks for CHARMM PSF files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PsfAtom:
    index: int
    segid: str
    resid: str
    resname: str
    atom_name: str
    atom_type: str


@dataclass(frozen=True, slots=True)
class DomdecOrderIssue:
    heavy_index: int
    heavy_name: str
    resid: str
    resname: str
    hydrogen_indices: tuple[int, ...]
    expected_indices: tuple[int, ...]

    def format(self) -> str:
        found = ",".join(str(i + 1) for i in self.hydrogen_indices)
        expected = ",".join(str(i + 1) for i in self.expected_indices)
        return (
            f"{self.resname}:{self.resid} heavy atom {self.heavy_name} "
            f"(PSF atom {self.heavy_index + 1}) has bonded hydrogens at [{found}], "
            f"expected adjacent [{expected}]"
        )


def _read_section_ints(lines: list[str], start: int, *, n_per_entry: int) -> tuple[list[int], int]:
    count = int(lines[start].split()[0])
    needed = count * n_per_entry
    values: list[int] = []
    idx = start + 1
    while len(values) < needed and idx < len(lines):
        stripped = lines[idx].strip()
        if stripped and "!" not in stripped:
            values.extend(int(x) for x in stripped.split())
        idx += 1
    if len(values) != needed:
        raise ValueError(f"PSF section expected {needed} integers, found {len(values)}")
    return values, idx


def read_psf_atoms_and_bonds(path: str | Path) -> tuple[list[PsfAtom], list[tuple[int, int]]]:
    psf_path = Path(path)
    lines = psf_path.read_text(encoding="utf-8", errors="replace").splitlines()
    atoms: list[PsfAtom] = []
    bonds: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "!NATOM" in line:
            natom = int(line.split()[0])
            atoms = []
            for j in range(i + 1, i + 1 + natom):
                parts = lines[j].split()
                if len(parts) < 6:
                    raise ValueError(f"Malformed PSF atom line: {lines[j]!r}")
                atoms.append(
                    PsfAtom(
                        index=int(parts[0]) - 1,
                        segid=parts[1],
                        resid=parts[2],
                        resname=parts[3],
                        atom_name=parts[4],
                        atom_type=parts[5],
                    )
                )
            i = i + 1 + natom
            continue
        if "!NBOND" in line:
            raw, i = _read_section_ints(lines, i, n_per_entry=2)
            bonds = [(raw[k] - 1, raw[k + 1] - 1) for k in range(0, len(raw), 2)]
            continue
        i += 1
    if not atoms:
        raise ValueError(f"No !NATOM section in {psf_path}")
    return atoms, bonds


def _is_hydrogen(atom: PsfAtom) -> bool:
    return atom.atom_name.upper().startswith("H") or atom.atom_type.upper().startswith("H")


def find_domdec_hydrogen_order_issues(path: str | Path) -> list[DomdecOrderIssue]:
    atoms, bonds = read_psf_atoms_and_bonds(path)
    bonded_h: dict[int, list[int]] = {}
    for a, b in bonds:
        atom_a = atoms[a]
        atom_b = atoms[b]
        if _is_hydrogen(atom_a) and not _is_hydrogen(atom_b):
            bonded_h.setdefault(b, []).append(a)
        elif _is_hydrogen(atom_b) and not _is_hydrogen(atom_a):
            bonded_h.setdefault(a, []).append(b)

    issues: list[DomdecOrderIssue] = []
    for heavy_idx, h_indices in sorted(bonded_h.items()):
        found = tuple(sorted(h_indices))
        expected = tuple(range(heavy_idx + 1, heavy_idx + 1 + len(found)))
        if found != expected:
            heavy = atoms[heavy_idx]
            issues.append(
                DomdecOrderIssue(
                    heavy_index=heavy_idx,
                    heavy_name=heavy.atom_name,
                    resid=heavy.resid,
                    resname=heavy.resname,
                    hydrogen_indices=found,
                    expected_indices=expected,
                )
            )
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("psf", type=Path)
    args = parser.parse_args(argv)
    issues = find_domdec_hydrogen_order_issues(args.psf)
    if issues:
        print(f"FAIL: {args.psf} is not DOMDEC hydrogen-order compatible")
        for issue in issues:
            print(f"  - {issue.format()}")
        return 1
    print(f"PASS: {args.psf} is DOMDEC hydrogen-order compatible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
