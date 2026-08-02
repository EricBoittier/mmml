"""The reacting solute: atom layout, reaction-coordinate seeds, model loading.

Everything specific to *what is reacting* lives here, so swapping in a different
reaction means editing this file and nothing else. The solvent side is in
:mod:`solvent_models`, the box construction in :mod:`jaxmd_box`, and neither
knows anything about NH3 + CH3Cl.

Atom ordering
-------------
Two orders exist and they are not the same:

- **canonical** -- Cl, N, C, H(N)x3, H(C)x3. This is what the reaction-coordinate
  scan NPZ uses, and what ``01_seed_windows.py`` writes for the gas phase.
- **residue-grouped** -- AMM1 (N,H,H,H) then MECL (C,CL,H,H,H). This is what the
  solvated builder uses, because a CHARMM PSF is built by reading the sequence
  from a PDB and that requires each residue's atoms to be contiguous.

``CANONICAL_TO_GROUPED`` converts between them. The CV indices differ
accordingly, which is why both are stated explicitly below.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

__all__ = [
    "SOLUTE_N_ATOMS", "SOLUTE_Z", "IDX_N", "IDX_C", "IDX_CL",
    "CANONICAL_TO_GROUPED", "CV_CANONICAL", "CV_GROUPED",
    "SOLVENTS", "solute_geometry_at_xi", "load_model", "atomic_mass",
]

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent

SOLUTE_N_ATOMS = 9
#: Residue-grouped order: AMM1 (N,H,H,H) then MECL (C,CL,H,H,H).
SOLUTE_Z = np.array([7, 1, 1, 1, 6, 17, 1, 1, 1], dtype=np.int32)
#: CV atom indices in residue-grouped order.
IDX_N, IDX_C, IDX_CL = 0, 4, 5
#: canonical (Cl,N,C,H(N)x3,H(C)x3) index for each residue-grouped slot.
CANONICAL_TO_GROUPED = [1, 3, 4, 5, 2, 0, 6, 7, 8]
#: xi = r(C-Cl) - r(C-N) as ``--cv-difference`` flags, per ordering.
CV_CANONICAL = "2,0,2,1"
CV_GROUPED = "4,5,4,0"

#: Campaign solvents: name -> (CGenFF residue, density kg/m3 at 298 K, box side A).
#: Matches Turan, Brickel & Meuwly, J. Phys. Chem. B 126, 1951 (2022).
SOLVENTS = {
    "water": ("TIP3", 997.0, 30.0),
    "methanol": ("MEOH", 792.0, 25.0),
    "acetonitrile": ("ACN", 786.0, 28.0),
    "benzene": ("BENZ", 874.0, 27.0),
    "cyclohexane": ("CHEX", 774.0, 30.0),
}


def load_scan(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(Z, R, xi)`` for every 9-atom scan frame, in canonical order.

    The bundled sources disagree on atom order -- ``scan_nh3_ch3cl.npz`` stores
    (Cl, N, C) at indices (1, 5, 0) while ``nh3_ch3cl_filtered.npz`` uses
    (0, 1, 2) -- so each frame is reordered by element rather than trusted.
    """
    data = np.load(Path(path), allow_pickle=True)
    n = np.asarray(data["N"])
    keep = np.flatnonzero(n == 9)
    if keep.size == 0:
        raise ValueError(f"no 9-atom frames in {path}")
    z_all, r_all = np.asarray(data["Z"])[keep], np.asarray(data["R"])[keep]

    z_out = np.empty((len(keep), 9), dtype=np.int32)
    r_out = np.empty((len(keep), 9, 3), dtype=np.float64)
    for i, (z, r) in enumerate(zip(z_all, r_all, strict=True)):
        order = _canonical_order(z, r)
        z_out[i], r_out[i] = z[order], r[order]
    xi = np.linalg.norm(r_out[:, 0] - r_out[:, 2], axis=1) - np.linalg.norm(
        r_out[:, 2] - r_out[:, 1], axis=1
    )
    return z_out, r_out, xi


def _canonical_order(z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Permutation to Cl, N, C, H(N)x3, H(C)x3 for one frame.

    Hydrogens go to whichever heavy atom they sit closer to, which is
    unambiguous everywhere on this reaction path: a methyl hydrogen never
    approaches the nitrogen more closely than its own carbon.
    """
    z = np.asarray(z)
    (i_cl,), (i_n,), (i_c,) = (
        np.flatnonzero(z == 17), np.flatnonzero(z == 7), np.flatnonzero(z == 6)
    )
    h = np.flatnonzero(z == 1)
    d_n = np.linalg.norm(r[h] - r[i_n], axis=1)
    d_c = np.linalg.norm(r[h] - r[i_c], axis=1)
    on_n, on_c = h[d_n <= d_c], h[d_n > d_c]
    if len(on_n) != 3 or len(on_c) != 3:
        raise ValueError(
            f"expected 3 H on N and 3 on C, got {len(on_n)} and {len(on_c)}"
        )
    on_n = on_n[np.argsort(d_n[d_n <= d_c])]
    on_c = on_c[np.argsort(d_c[d_n > d_c])]
    return np.concatenate([[i_cl, i_n, i_c], on_n, on_c]).astype(np.int64)


def solute_geometry_at_xi(xi_target: float, scan: Path | str | None = None,
                          grouped: bool = True, verbose: bool = True) -> np.ndarray:
    """Scan geometry nearest ``xi_target``, in residue-grouped order by default."""
    scan = Path(scan or os.environ.get(
        "MENSH_SCAN", REPO_ROOT / "examples/m/scan_nh3_ch3cl.npz"))
    _z, r_all, xi = load_scan(scan)
    idx = int(np.argmin(np.abs(xi - float(xi_target))))
    if verbose:
        print(f"solute seed  scan frame {idx}, xi = {xi[idx]:+.3f} A "
              f"(target {float(xi_target):+.3f})")
    frame = r_all[idx]
    return frame[CANONICAL_TO_GROUPED] if grouped else frame


def load_model(checkpoint: Path | str | None = None):
    """Return ``(model, params)`` for the PhysNet checkpoint."""
    from mmml.interfaces.calculators.simple_inference import (
        create_calculator_from_checkpoint,
    )

    checkpoint = Path(checkpoint or os.environ.get(
        "MENSH_CKPT", REPO_ROOT / "model_ext.json"))
    calc = create_calculator_from_checkpoint(str(checkpoint))
    model = getattr(calc, "model", None) or calc._mmml_physnet_model
    params = getattr(calc, "params", None) or calc._mmml_physnet_params
    return model, params


def atomic_mass(z: int) -> float:
    from ase.data import atomic_masses

    return float(atomic_masses[int(z)])
