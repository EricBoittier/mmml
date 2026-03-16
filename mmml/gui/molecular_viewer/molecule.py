"""
Molecular structure and trajectory parsers.
Supports PDB, XYZ, and multi-frame XYZ trajectories.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Atom:
    """Single atom with element, coordinates, and optional metadata."""

    element: str
    x: float
    y: float
    z: float
    name: str = ""
    residue: str = ""
    fx: float | None = None
    fy: float | None = None
    fz: float | None = None

    @property
    def pos(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def force(self) -> tuple[float, float, float] | None:
        """Force vector (fx, fy, fz) if available."""
        if self.fx is not None and self.fy is not None and self.fz is not None:
            return (self.fx, self.fy, self.fz)
        return None


# Van der Waals radii (Å) for common elements
VDW_RADII: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98, "Fe": 1.80, "Cu": 1.40,
    "Zn": 1.39, "Ca": 1.97, "Mg": 1.73, "Na": 2.27, "K": 2.75,
}

# CPK colors (R, G, B) for common elements
CPK_COLORS: dict[str, tuple[float, float, float]] = {
    "H": (1.0, 1.0, 1.0), "C": (0.3, 0.3, 0.3), "N": (0.2, 0.2, 1.0),
    "O": (1.0, 0.2, 0.2), "S": (1.0, 0.8, 0.2), "P": (1.0, 0.5, 0.0),
    "F": (0.2, 1.0, 0.2), "Cl": (0.2, 1.0, 0.2), "Br": (0.6, 0.2, 0.2),
    "I": (0.6, 0.2, 0.6), "Fe": (0.8, 0.4, 0.2), "Cu": (0.8, 0.5, 0.2),
    "Zn": (0.5, 0.5, 0.5), "Ca": (0.2, 0.8, 0.2), "Mg": (0.2, 0.8, 0.2),
}


def _element_from_name(name: str) -> str:
    """Extract element symbol from atom name (e.g. CA -> C, FE -> Fe)."""
    if not name:
        return "?"
    # First letter uppercase, second lowercase for two-letter elements
    m = re.match(r"^([A-Za-z]{1,2})", name.strip())
    if m:
        s = m.group(1)
        if len(s) == 2:
            return s[0].upper() + s[1].lower()
        return s[0].upper()
    return "?"


def _cell_from_cryst1(line: str) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
    """Parse PDB CRYST1 record; return (a, b, c) Cartesian vectors or None."""
    try:
        a = float(line[6:15])
        b = float(line[15:24])
        c = float(line[24:33])
        alpha = math.radians(float(line[33:40]))
        beta = math.radians(float(line[40:47]))
        gamma = math.radians(float(line[47:54]))
    except (ValueError, IndexError):
        return None
    # Convert fractional (a,b,c,alpha,beta,gamma) to Cartesian cell vectors
    ax, ay, az = a, 0.0, 0.0
    bx = b * math.cos(gamma)
    by = b * math.sin(gamma)
    bz = 0.0
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma) if math.sin(gamma) > 1e-8 else 0.0
    cz = c * math.sqrt(1.0 - math.cos(beta) ** 2 - (cy / c) ** 2) if c > 1e-8 else 0.0
    return ((ax, ay, az), (bx, by, bz), (cx, cy, cz))


def load_pdb(path: str | Path) -> tuple[list[Atom], tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None]:
    """Parse PDB file. Returns (atoms, cell). Cell from CRYST1 if present."""
    path = Path(path)
    atoms: list[Atom] = []
    cell: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None
    with open(path) as f:
        for line in f:
            if line.startswith("CRYST1"):
                cell = _cell_from_cryst1(line)
            elif line.startswith(("ATOM  ", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    name = line[12:16].strip()
                    res = line[17:20].strip()
                    element = line[76:78].strip() or _element_from_name(name)
                    atoms.append(Atom(element=element, x=x, y=y, z=z, name=name, residue=res))
                except (ValueError, IndexError):
                    pass
    return atoms, cell


def _forces_col_from_properties(comment: str) -> int | None:
    """Parse Properties=... to find forces column start. Returns column index or None."""
    m = re.search(r'Properties=([^\s]+)', comment)
    if not m:
        return None
    parts = m.group(1).split(":")
    col = 0
    for i in range(0, len(parts) - 2, 3):
        name = parts[i].lower()
        count = int(parts[i + 2]) if i + 2 < len(parts) else 1
        if name == "forces" and count == 3:
            return col
        col += count
    return None


def _cell_from_lattice(lattice_str: str) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
    """Parse Lattice=\"ax ay az bx by bz cx cy cz\" (column-major). Returns (a, b, c) or None."""
    import re
    m = re.search(r'Lattice="([^"]+)"', lattice_str)
    if not m:
        return None
    try:
        vals = [float(x) for x in m.group(1).split()]
        if len(vals) >= 9:
            a = (vals[0], vals[1], vals[2])
            b = (vals[3], vals[4], vals[5])
            c = (vals[6], vals[7], vals[8])
            return (a, b, c)
    except (ValueError, IndexError):
        pass
    return None


def load_xyz(path: str | Path) -> tuple[list[Atom], tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None]:
    """Parse single-frame XYZ file. Returns (atoms, cell). Parses forces from extended XYZ when Properties=...forces:R:3."""
    path = Path(path)
    atoms: list[Atom] = []
    cell: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None
    with open(path) as f:
        n = int(f.readline().strip())
        comment = f.readline()
        cell = _cell_from_lattice(comment)
        forces_col = _forces_col_from_properties(comment)
        for _ in range(n):
            parts = f.readline().split()
            if len(parts) >= 4:
                elem = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                fx = fy = fz = None
                if forces_col is not None and len(parts) >= forces_col + 3:
                    try:
                        fx = float(parts[forces_col])
                        fy = float(parts[forces_col + 1])
                        fz = float(parts[forces_col + 2])
                    except (ValueError, IndexError):
                        pass
                atoms.append(Atom(element=elem, x=x, y=y, z=z, fx=fx, fy=fy, fz=fz))
    return atoms, cell


def load_xyz_trajectory(path: str | Path) -> tuple[list[list[Atom]], list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None]]:
    """Parse multi-frame XYZ file (trajectory). Returns (frames, cells)."""
    path = Path(path)
    frames: list[list[Atom]] = []
    cells: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None] = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                n = int(line.strip())
            except ValueError:
                continue
            comment = f.readline()
            cell = _cell_from_lattice(comment)
            forces_col = _forces_col_from_properties(comment)
            frame: list[Atom] = []
            for _ in range(n):
                parts = f.readline().split()
                if len(parts) >= 4:
                    elem = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    fx = fy = fz = None
                    if forces_col is not None and len(parts) >= forces_col + 3:
                        try:
                            fx = float(parts[forces_col])
                            fy = float(parts[forces_col + 1])
                            fz = float(parts[forces_col + 2])
                        except (ValueError, IndexError):
                            pass
                    frame.append(Atom(element=elem, x=x, y=y, z=z, fx=fx, fy=fy, fz=fz))
            if frame:
                frames.append(frame)
                cells.append(cell)
    return frames, cells


def load_traj(path: str | Path) -> tuple[list[list[Atom]], list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None], list[dict]]:
    """Load ASE trajectory (.traj). Requires ase package. Returns (frames, cells, metadata)."""
    try:
        from ase.io import read
    except ImportError as e:
        raise ImportError("Loading .traj files requires the ase package. Install with: pip install ase") from e
    path = Path(path)
    ase_atoms_list = read(str(path), index=":")
    if not isinstance(ase_atoms_list, list):
        ase_atoms_list = [ase_atoms_list]
    frames: list[list[Atom]] = []
    cells: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None] = []
    metadata: list[dict] = []
    for a in ase_atoms_list:
        symbols = a.get_chemical_symbols()
        positions = a.get_positions()
        try:
            forces = a.get_forces()
        except (RuntimeError, AttributeError):
            forces = None
        cell = None
        if a.cell.rank == 3:
            c = a.get_cell()
            cell = (
                tuple(c[0]),
                tuple(c[1]),
                tuple(c[2]),
            )
        frame: list[Atom] = []
        for i, (sym, pos) in enumerate(zip(symbols, positions)):
            fx = fy = fz = None
            if forces is not None and i < len(forces):
                fx, fy, fz = float(forces[i, 0]), float(forces[i, 1]), float(forces[i, 2])
            frame.append(Atom(element=sym, x=float(pos[0]), y=float(pos[1]), z=float(pos[2]), fx=fx, fy=fy, fz=fz))
        frames.append(frame)
        cells.append(cell)
        meta: dict = {}
        try:
            e = a.get_potential_energy()
            meta["energy"] = float(e)
        except (RuntimeError, AttributeError, KeyError):
            pass
        try:
            v = a.get_velocities()
            if v is not None:
                m = a.get_masses()
                ke = 0.5 * sum(m[i] * (v[i, 0] ** 2 + v[i, 1] ** 2 + v[i, 2] ** 2) for i in range(len(a)))
                meta["kinetic_energy"] = float(ke)
        except (RuntimeError, AttributeError, KeyError):
            pass
        if meta.get("energy") is not None and meta.get("kinetic_energy") is not None:
            meta["total_energy"] = meta["energy"] + meta["kinetic_energy"]
        metadata.append(meta)
    return frames, cells, metadata


def _parse_xyz_metadata(path: str | Path) -> list[dict]:
    """Parse Energy= etc. from extended XYZ comment lines. Returns list of metadata dicts."""
    path = Path(path)
    meta_list: list[dict] = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                n = int(line.strip())
            except ValueError:
                continue
            comment = f.readline()
            for _ in range(n):
                f.readline()
            meta: dict = {}
            for part in comment.split():
                if "=" in part:
                    k, _, v = part.partition("=")
                    k = k.strip().lower()
                    v = v.strip('"').strip()
                    try:
                        val = float(v)
                    except ValueError:
                        continue
                    if k in ("energy", "potentialenergy", "potential_energy"):
                        meta["energy"] = val
                    elif k in ("kineticenergy", "kinetic_energy"):
                        meta["kinetic_energy"] = val
                    elif k in ("totalenergy", "total_energy"):
                        meta["total_energy"] = val
            meta_list.append(meta)
    return meta_list if meta_list else []


def load_structure(path: str | Path) -> tuple[list[Atom] | list[list[Atom]], tuple | list | None, list[dict]]:
    """
    Auto-detect format and load. Returns (data, cell_or_cells, metadata):
    - Single: ([atoms], [cell], [meta])
    - Trajectory: (frames, cells, metadata)
    metadata: list of dicts with keys like 'energy', 'kinetic_energy', 'total_energy'
    """
    path = Path(path)
    suf = path.suffix.lower()
    empty_meta: list[dict] = []
    if suf == ".pdb":
        atoms, cell = load_pdb(path)
        return [atoms], [cell], [{}]
    if suf == ".xyz":
        with open(path) as f:
            first = f.readline().strip()
            try:
                n = int(first)
            except ValueError:
                atoms, cell = load_xyz(path)
                return [atoms], [cell], [{}]
            lines = f.readlines()
            frame_size = n + 2
            if len(lines) >= frame_size * 2:
                frames, cells = load_xyz_trajectory(path)
                meta = _parse_xyz_metadata(path)
                if len(meta) != len(frames):
                    meta = [{}] * len(frames)
                return frames, cells, meta
            atoms, cell = load_xyz(path)
            return [atoms], [cell], [{}]
    if suf == ".traj":
        frames, cells, meta = load_traj(path)
        return frames, cells, meta
    raise ValueError(f"Unsupported format: {suf}")


def compute_bonds(atoms: list[Atom], max_bond: float = 2.0) -> list[tuple[int, int]]:
    """
    Simple distance-based bond detection. Returns list of (i, j) atom indices.
    """
    bonds: list[tuple[int, int]] = []
    n = len(atoms)
    for i in range(n):
        ai = atoms[i]
        ri = VDW_RADII.get(ai.element, 1.7)
        for j in range(i + 1, n):
            aj = atoms[j]
            rj = VDW_RADII.get(aj.element, 1.7)
            dx = ai.x - aj.x
            dy = ai.y - aj.y
            dz = ai.z - aj.z
            d = (dx * dx + dy * dy + dz * dz) ** 0.5
            if d < (ri + rj) * 0.6 or d < max_bond:
                bonds.append((i, j))
    return bonds


def _bond_neighbors(bonds: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Map atom index -> list of bonded neighbor indices."""
    neighbors: dict[int, list[int]] = {}
    for i, j in bonds:
        neighbors.setdefault(i, []).append(j)
        neighbors.setdefault(j, []).append(i)
    return neighbors


def compute_angles(atoms: list[Atom], bonds: list[tuple[int, int]]) -> list[tuple[int, int, int, float]]:
    """
    Find angles A-B-C (B is vertex). Returns list of (i, j, k, angle_deg).
    """
    neighbors = _bond_neighbors(bonds)
    result: list[tuple[int, int, int, float]] = []
    seen: set[tuple[int, int, int]] = set()
    for j, nbrs in neighbors.items():
        for i in nbrs:
            for k in nbrs:
                if i >= k:
                    continue
                key = (i, j, k)
                if key in seen:
                    continue
                seen.add(key)
                ai, bj, ck = atoms[i], atoms[j], atoms[k]
                v1 = (ai.x - bj.x, ai.y - bj.y, ai.z - bj.z)
                v2 = (ck.x - bj.x, ck.y - bj.y, ck.z - bj.z)
                d1 = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) ** 0.5
                d2 = (v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2) ** 0.5
                if d1 < 1e-6 or d2 < 1e-6:
                    continue
                dot = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (d1 * d2)
                dot = max(-1.0, min(1.0, dot))
                angle_rad = math.acos(dot)
                angle_deg = math.degrees(angle_rad)
                result.append((i, j, k, angle_deg))
    return result


def compute_dihedrals(atoms: list[Atom], bonds: list[tuple[int, int]]) -> list[tuple[int, int, int, int, float]]:
    """
    Find dihedrals A-B-C-D (torsion around B-C). Returns list of (i, j, k, l, angle_deg).
    """
    neighbors = _bond_neighbors(bonds)
    result: list[tuple[int, int, int, int, float]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for i, j in bonds:
        for k in neighbors.get(j, []):
            if k == i:
                continue
            for l in neighbors.get(k, []):
                if l == j or l == i:
                    continue
                key = tuple(sorted([i, j, k, l]))
                if key in seen:
                    continue
                seen.add(key)
                a, b, c, d = atoms[i], atoms[j], atoms[k], atoms[l]
                b1 = (a.x - b.x, a.y - b.y, a.z - b.z)
                b2 = (c.x - b.x, c.y - b.y, c.z - b.z)
                b3 = (d.x - c.x, d.y - c.y, d.z - c.z)
                n1 = (
                    b1[1] * b2[2] - b1[2] * b2[1],
                    b1[2] * b2[0] - b1[0] * b2[2],
                    b1[0] * b2[1] - b1[1] * b2[0],
                )
                n2 = (
                    b2[1] * b3[2] - b2[2] * b3[1],
                    b2[2] * b3[0] - b2[0] * b3[2],
                    b2[0] * b3[1] - b2[1] * b3[0],
                )
                dn1 = (n1[0] ** 2 + n1[1] ** 2 + n1[2] ** 2) ** 0.5
                dn2 = (n2[0] ** 2 + n2[1] ** 2 + n2[2] ** 2) ** 0.5
                if dn1 < 1e-6 or dn2 < 1e-6:
                    continue
                dot = (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]) / (dn1 * dn2)
                dot = max(-1.0, min(1.0, dot))
                sign = 1.0 if (b2[0] * n1[0] + b2[1] * n1[1] + b2[2] * n1[2]) >= 0 else -1.0
                angle_rad = sign * math.acos(dot)
                angle_deg = math.degrees(angle_rad)
                result.append((i, j, k, l, angle_deg))
    return result


def center_and_scale(atoms: list[Atom], scale: float = 1.0) -> list[Atom]:
    """Center structure at origin and optionally scale."""
    if not atoms:
        return atoms
    cx = sum(a.x for a in atoms) / len(atoms)
    cy = sum(a.y for a in atoms) / len(atoms)
    cz = sum(a.z for a in atoms) / len(atoms)
    return [
        Atom(
            a.element,
            (a.x - cx) * scale,
            (a.y - cy) * scale,
            (a.z - cz) * scale,
            a.name,
            a.residue,
            a.fx,
            a.fy,
            a.fz,
        )
        for a in atoms
    ]
