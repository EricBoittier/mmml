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
    """Format one CHARMM-friendly PDB ATOM record (left-aligned atom name)."""
    x, y, z = (float(v) for v in xyz)
    # CHARMM expects atom names left-justified in columns 13–16 for ≤3-char names.
    name_field = f"{name:<4}"[:4]
    return (
        f"ATOM  {serial:5d} {name_field} {resname:>3} A{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"
    )


def write_solute_pdb(
    out_path: Path | str,
    npz_path: Path | str | None = None,
    *,
    index: int | None = None,
    seed: int = 0,
) -> Path:
    """Write a CGenFF-named AMM1+CH3CL PDB from one dimer frame (for ``make-box``).

    Residue order is AMM1 then CH3CL with standard CGenFF atom names so CHARMM
    ``READ SEQU PDB`` / Packmol solvation work. Requires
    ``MMML_CGENFF_EXTRA_RTF=examples/m/top_ch3cl.rtf`` for CH3CL.
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
