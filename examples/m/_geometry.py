"""Load NH3–CH3Cl geometries from the bundled filtered NPZ."""

from __future__ import annotations

from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_NPZ = EXAMPLE_DIR / "nh3_ch3cl_filtered.npz"

# Dataset order: Cl, N, C, H×3(N), H×3(C). CGenFF residue atom orders differ.
_AMM1_ATOMS = ("N1", "H11", "H12", "H13")  # NPZ indices 1, 3, 4, 5
_CH3CL_ATOMS = ("C1", "CL1", "H11", "H12", "H13")  # NPZ indices 2, 0, 6, 7, 8
_AMM1_NPZ_IDX = (1, 3, 4, 5)
_CH3CL_NPZ_IDX = (2, 0, 6, 7, 8)


def load_dimer_frame(
    npz_path: Path | str | None = None,
    *,
    index: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Z, R)`` for one N=9 dimer frame (Å).

    Atom order in the dataset is ``Cl, N, C, H×3(N), H×3(C)`` — not monomer-blocked.
    For ML-only smokes treat the complex as a single 9-atom system.
    """
    path = Path(npz_path) if npz_path is not None else DEFAULT_NPZ
    data = np.load(path, allow_pickle=True)
    n = np.asarray(data["N"])
    dimer_idx = np.flatnonzero(n == 9)
    if dimer_idx.size == 0:
        raise ValueError(f"No N=9 frames in {path}")
    if index is None:
        rng = np.random.default_rng(seed)
        index = int(rng.choice(dimer_idx))
    else:
        index = int(index)
        if index not in set(int(i) for i in dimer_idx):
            # Allow absolute indices into the full array if they are dimers.
            if not (0 <= index < len(n) and int(n[index]) == 9):
                raise ValueError(f"index={index} is not an N=9 frame in {path}")
    z = np.asarray(data["Z"][index], dtype=np.int32)
    r = np.asarray(data["R"][index], dtype=np.float64)
    mask = z > 0
    return z[mask], r[mask]


# NPZ atom order (see comment above): Cl=0, N=1, C=2.
_NPZ_CL, _NPZ_N, _NPZ_C = 0, 1, 2


def frame_reaction_coord(r: np.ndarray) -> tuple[float, float, float]:
    """Return ``(xi, r_ClC, r_CN)`` for one N=9 frame; xi = r(Cl-C)/r(C-N)."""
    r_clc = float(np.linalg.norm(r[_NPZ_CL] - r[_NPZ_C]))
    r_cn = float(np.linalg.norm(r[_NPZ_C] - r[_NPZ_N]))
    return r_clc / r_cn, r_clc, r_cn


def find_frame_near_xi(
    target_xi: float,
    npz_path: Path | str | None = None,
) -> tuple[int, float, float, float]:
    """Absolute NPZ index of the N=9 frame whose xi is closest to ``target_xi``.

    Returns ``(index, xi, r_ClC, r_CN)``. Seeds ADUMB near the transition state
    (xi≈1); note the bundled dataset has *no* frames in xi∈[0.9,1.1], so the
    nearest available geometry is returned.
    """
    path = Path(npz_path) if npz_path is not None else DEFAULT_NPZ
    data = np.load(path, allow_pickle=True)
    n = np.asarray(data["N"])
    dimer_idx = np.flatnonzero(n == 9)
    if dimer_idx.size == 0:
        raise ValueError(f"No N=9 frames in {path}")
    r_all = np.asarray(data["R"])[dimer_idx]
    r_clc = np.linalg.norm(r_all[:, _NPZ_CL] - r_all[:, _NPZ_C], axis=1)
    r_cn = np.linalg.norm(r_all[:, _NPZ_C] - r_all[:, _NPZ_N], axis=1)
    xi = r_clc / r_cn
    best = int(np.argmin(np.abs(xi - float(target_xi))))
    idx = int(dimer_idx[best])
    return idx, float(xi[best]), float(r_clc[best]), float(r_cn[best])


def find_frame_near_rc(
    target_rcl: float,
    target_rcn: float,
    npz_path: Path | str | None = None,
) -> tuple[int, float, float, float]:
    """Absolute NPZ index of the N=9 frame nearest a 2D (r_ClC, r_CN) target (Å).

    Returns ``(index, xi, r_ClC, r_CN)``. Seeds the 2D ADUMB (r_cl, r_cn) map at a
    chosen point on the plane — e.g. the product basin (large r_ClC, small r_CN)
    so a window samples the broken-C-Cl region without crossing the barrier.
    """
    path = Path(npz_path) if npz_path is not None else DEFAULT_NPZ
    data = np.load(path, allow_pickle=True)
    n = np.asarray(data["N"])
    dimer_idx = np.flatnonzero(n == 9)
    if dimer_idx.size == 0:
        raise ValueError(f"No N=9 frames in {path}")
    r_all = np.asarray(data["R"])[dimer_idx]
    r_clc = np.linalg.norm(r_all[:, _NPZ_CL] - r_all[:, _NPZ_C], axis=1)
    r_cn = np.linalg.norm(r_all[:, _NPZ_C] - r_all[:, _NPZ_N], axis=1)
    d2 = (r_clc - float(target_rcl)) ** 2 + (r_cn - float(target_rcn)) ** 2
    best = int(np.argmin(d2))
    idx = int(dimer_idx[best])
    return idx, float(r_clc[best] / r_cn[best]), float(r_clc[best]), float(r_cn[best])


def write_evaluate_npz(
    out_path: Path | str,
    npz_path: Path | str | None = None,
    *,
    index: int | None = None,
    seed: int = 0,
) -> Path:
    """Write a single-frame NPZ for ``mmml md-system --evaluate-npz``."""
    z, r = load_dimer_frame(npz_path, index=index, seed=seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        positions=r.astype(np.float64),
        atomic_numbers=z.astype(np.int32),
        Z=z.astype(np.int32),
        R=r.astype(np.float64),
        N=np.array([len(z)], dtype=np.int32),
    )
    return out


def _pdb_atom_line(
    serial: int,
    name: str,
    resname: str,
    resid: int,
    xyz: np.ndarray,
    element: str,
) -> str:
    """Format one PDB ATOM record for CGenFF names (ASE + Packmol + CHARMM).

    Packmol ≥21 requires residue numbers strictly in columns 23–26, and
    ``make-box`` needs coordinates in columns 31–54 for ``ase.io.read``.

    Layout (1-based columns):
    - 13–16 atom name (leading space for ≤3-char names)
    - 17 altLoc (blank)
    - 18–21 4-char resname + blank chain, **or** 18–22 for 5-char (CH3CL)
    - 23–26 residue number
    - 31–54 x/y/z
    """
    x, y, z = (float(v) for v in xyz)
    aname = f" {name:<3s}" if len(name) <= 3 else f"{name:<4s}"
    alt = " "
    if len(resname) <= 3:
        mid = f"{resname:<3s}  "  # res + blank + blank chain
    elif len(resname) == 4:
        mid = f"{resname:<4s} "  # res + blank chain
    else:
        mid = f"{resname:<5s}"  # 5-char occupies chain column (Packmol-ok)
    prefix = f"ATOM  {serial:5d} {aname}{alt}{mid}{resid:4d}    "
    if len(prefix) != 30 or prefix[22:26] != f"{resid:4d}":
        raise ValueError(
            f"PDB column layout broken for {resname!r}/{name!r}: "
            f"len={len(prefix)} resid_field={prefix[22:26]!r}"
        )
    return (
        f"{prefix}{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
    )


def write_solute_pdb(
    out_path: Path | str,
    npz_path: Path | str | None = None,
    *,
    index: int | None = None,
    seed: int = 0,
    center: bool = True,
) -> Path:
    """Write a CGenFF-named AMM1+CH3CL PDB from one dimer frame (for ``make-box``).

    Residue order is AMM1 then CH3CL with standard CGenFF atom names so CHARMM
    ``READ SEQU PDB`` / Packmol solvation work. Requires
    ``MMML_CGENFF_EXTRA_RTF=examples/m/top_ch3cl.rtf`` for CH3CL.

    ``center`` (default True) translates the mass-weighted COM to the origin so the
    ``cons hmcm ... refx 0`` tether starts at ~0 energy (no t=0 yank).
    """
    z, r = load_dimer_frame(npz_path, index=index, seed=seed)
    if len(z) != 9:
        raise ValueError(f"expected 9 atoms after mask, got {len(z)}")
    # Sanity: expected elements in dataset order.
    expected = np.array([17, 7, 6, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    if not np.array_equal(z, expected):
        raise ValueError(
            f"unexpected Z order for dimer frame: {z.tolist()} (want {expected.tolist()})"
        )

    if center:
        from ase.data import atomic_masses

        masses = atomic_masses[z.astype(int)]
        com = (masses[:, None] * r).sum(axis=0) / masses.sum()
        r = r - com

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["REMARK  NH3-CH3Cl from examples/m NPZ (CGenFF AMM1 + CH3CL)"]
    serial = 1
    for name, idx in zip(_AMM1_ATOMS, _AMM1_NPZ_IDX, strict=True):
        elem = "N" if name.startswith("N") else "H"
        lines.append(
            _pdb_atom_line(serial, name, "AMM1", 1, r[idx], elem)
        )
        serial += 1
    lines.append("TER")
    for name, idx in zip(_CH3CL_ATOMS, _CH3CL_NPZ_IDX, strict=True):
        if name.startswith("CL"):
            elem = "Cl"
        elif name.startswith("C"):
            elem = "C"
        else:
            elem = "H"
        lines.append(
            _pdb_atom_line(serial, name, "CH3CL", 2, r[idx], elem)
        )
        serial += 1
    lines.append("END")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
