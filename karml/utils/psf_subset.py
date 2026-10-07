"""Select atoms from a CHARMM PSF and write a matching subset topology."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mmml.utils.domdec_psf_order import PsfAtom, read_psf_atoms_and_bonds

__all__ = [
    "parse_resname_list",
    "indices_for_resnames",
    "write_subset_psf",
    "copy_or_link_psf",
]


def parse_resname_list(raw: str | list[str] | None) -> list[str]:
    """Parse ``TRIA,TIP3`` / ``['TRIA','TIP3']`` into uppercased unique names."""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    else:
        parts = []
        for item in raw:
            parts.extend(str(item).replace(";", ",").split(","))
        parts = [p.strip() for p in parts]
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if not p:
            continue
        key = p.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def indices_for_resnames(
    psf_path: Path | str,
    resnames: list[str] | str,
) -> tuple[np.ndarray, list[PsfAtom]]:
    """Return 0-based atom indices (and atoms) matching any of ``resnames``."""
    wanted = set(parse_resname_list(resnames))
    if not wanted:
        raise ValueError("resnames selection is empty")
    atoms, _bonds = read_psf_atoms_and_bonds(psf_path)
    idxs = [a.index for a in atoms if a.resname.upper() in wanted]
    if not idxs:
        present = sorted({a.resname.upper() for a in atoms})
        raise ValueError(
            f"No atoms with resnames {sorted(wanted)} in {psf_path}; "
            f"present={present}"
        )
    selected = [atoms[i] for i in idxs]
    return np.asarray(idxs, dtype=np.int32), selected


def write_subset_psf(
    psf_in: Path | str,
    psf_out: Path | str,
    atom_indices: np.ndarray | list[int],
) -> Path:
    """Write a VMD/CHARMM-readable PSF containing only ``atom_indices``.

    Keeps original atom records (segid/resid/name/type/charge/mass when present)
    and remaps bonds that stay entirely inside the selection. Other PSF sections
    (angles, …) are omitted — enough for visualization with a matching DCD.
    """
    psf_in = Path(psf_in)
    psf_out = Path(psf_out)
    lines = psf_in.read_text(encoding="utf-8", errors="replace").splitlines()
    atoms, bonds = read_psf_atoms_and_bonds(psf_in)
    keep = np.asarray(atom_indices, dtype=np.int32).reshape(-1)
    if keep.size == 0:
        raise ValueError("atom_indices is empty")
    if int(keep.min()) < 0 or int(keep.max()) >= len(atoms):
        raise ValueError(
            f"atom_indices out of range for PSF with {len(atoms)} atoms"
        )
    keep_set = set(int(i) for i in keep.tolist())
    old_to_new = {int(old): new for new, old in enumerate(keep.tolist())}

    # Collect original NATOM record lines (preserve charge/mass columns).
    atom_lines: list[str] = []
    i = 0
    while i < len(lines):
        if "!NATOM" in lines[i]:
            natom = int(lines[i].split()[0])
            for j in range(i + 1, i + 1 + natom):
                parts = lines[j].split()
                idx0 = int(parts[0]) - 1
                if idx0 in keep_set:
                    atom_lines.append(lines[j])
            break
        i += 1
    if len(atom_lines) != len(keep):
        raise ValueError(
            f"PSF atom-line filter mismatch: kept {len(atom_lines)} lines "
            f"for {len(keep)} indices"
        )

    remapped_atoms: list[str] = []
    for new_i, line in enumerate(atom_lines):
        parts = line.split()
        parts[0] = str(new_i + 1)
        # Re-pad roughly like CHARMM: left index width 8, rest space-separated.
        remapped_atoms.append(
            f"{int(parts[0]):8d} {parts[1]:<4s} {parts[2]:>4s} {parts[3]:<4s} "
            f"{parts[4]:<4s} {parts[5]:<4s} "
            + (" ".join(parts[6:]) if len(parts) > 6 else "")
        )

    kept_bonds: list[tuple[int, int]] = []
    for a, b in bonds:
        if a in keep_set and b in keep_set:
            kept_bonds.append((old_to_new[a] + 1, old_to_new[b] + 1))

    psf_out.parent.mkdir(parents=True, exist_ok=True)
    out_lines = [
        "PSF",
        "",
        f"{len(remapped_atoms):8d} !NATOM",
        *remapped_atoms,
        "",
        f"{len(kept_bonds):8d} !NBOND: bonds",
    ]
    # CHARMM packs up to 4 bonds (8 ints) per line.
    flat: list[int] = []
    for a, b in kept_bonds:
        flat.extend([a, b])
    for k in range(0, len(flat), 8):
        chunk = flat[k : k + 8]
        out_lines.append("".join(f"{v:8d}" for v in chunk))
    out_lines.append("")
    psf_out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return psf_out


def copy_or_link_psf(psf_in: Path | str, psf_out: Path | str) -> Path:
    """Copy ``psf_in`` to ``psf_out`` (overwrite)."""
    import shutil

    psf_in = Path(psf_in)
    psf_out = Path(psf_out)
    psf_out.parent.mkdir(parents=True, exist_ok=True)
    if psf_in.resolve() != psf_out.resolve():
        shutil.copy2(psf_in, psf_out)
    return psf_out
