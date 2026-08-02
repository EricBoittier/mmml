#!/usr/bin/env python3
"""Generate a 2D dimer grid as a trajectory NPZ for ``md-system --evaluate-npz``.

Purpose: gate any further condensed-phase work on the dimer surface being sane.
In bulk water the DES-fitted hybrid released 99.3 kcal/mol per molecule as it
relaxed (docs/npt-campaign-status-2026-08-02.md), which is bond-breaking scale
and points at a spurious attractive well. A 1D distance scan can walk straight
past such a well if it sits at an orientation the scan does not visit, so the
grid here is two-dimensional:

    R      centre-of-mass separation of the two monomers
    theta  rotation of monomer B about the axis perpendicular to the
           separation vector, sweeping donor -> acceptor -> anti orientations

Evaluating the same grid with the ML dimer term on and off, and with the MM term
on and off, decomposes the surface and identifies which term supplies the well.

The monomer is rigid: this scans intermolecular geometry only, which is the part
the hybrid's dimer term is responsible for. Bond stretching would confound the
question with intramolecular ML behaviour.

Output NPZ carries the trajectory keys the evaluator expects (``R``, ``Z``,
``N``) plus ``grid_r`` / ``grid_theta`` / ``grid_shape`` so the surface can be
reshaped for plotting without recomputing the geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Rigid TIP3 geometry (the certified boxes use it): r(OH) = 0.9572 A, 104.52 deg.
TIP3_R_OH = 0.9572
TIP3_ANGLE_DEG = 104.52


def tip3_monomer() -> tuple[np.ndarray, np.ndarray]:
    """O at the origin, C2 axis along +z, molecule in the xz plane."""
    half = np.deg2rad(TIP3_ANGLE_DEG) / 2.0
    o = np.array([0.0, 0.0, 0.0])
    h1 = np.array([TIP3_R_OH * np.sin(half), 0.0, TIP3_R_OH * np.cos(half)])
    h2 = np.array([-TIP3_R_OH * np.sin(half), 0.0, TIP3_R_OH * np.cos(half)])
    return np.stack([o, h1, h2]), np.array([8, 1, 1], dtype=np.int32)


def rot_y(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def com(xyz: np.ndarray, z: np.ndarray) -> np.ndarray:
    # Mass-weighted; O dominates, but do it properly so R means what it says.
    masses = np.where(z == 8, 15.999, 1.008)
    return (xyz * masses[:, None]).sum(0) / masses.sum()


def build_grid(
    r_values: np.ndarray, theta_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mono, z1 = tip3_monomer()
    z = np.concatenate([z1, z1])
    frames = []
    for r in r_values:
        for th in theta_deg:
            a = mono.copy()
            b = mono @ rot_y(np.deg2rad(float(th))).T
            # Separate along +x by R between centres of mass.
            b = b - com(b, z1) + np.array([float(r), 0.0, 0.0]) + com(a, z1)
            frames.append(np.concatenate([a, b], axis=0))
    return np.asarray(frames, dtype=np.float64), z


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--r-min", type=float, default=2.5)
    ap.add_argument("--r-max", type=float, default=7.0)
    ap.add_argument("--n-r", type=int, default=25)
    ap.add_argument("--n-theta", type=int, default=24)
    a = ap.parse_args(argv)

    r_values = np.linspace(a.r_min, a.r_max, a.n_r)
    theta_deg = np.linspace(0.0, 360.0, a.n_theta, endpoint=False)
    frames, z = build_grid(r_values, theta_deg)
    n_frames, n_atoms = frames.shape[0], frames.shape[1]

    # Closest intermolecular contact PER FRAME. Short range is deliberately in
    # the grid -- a missing or mis-fitted repulsive wall shows up there as a
    # spurious attractive well, which is the whole point of the gate -- but
    # frames compressed past physical reach must be identifiable so a plot can
    # mark them rather than let one 10^6 kcal/mol point set the colour scale.
    contact = np.array(
        [
            float(np.linalg.norm(f[:3, None, :] - f[None, 3:, :], axis=-1).min())
            for f in frames
        ]
    )
    d_min = float(contact.min())
    if not np.isfinite(d_min) or d_min <= 0.3:
        raise SystemExit(
            f"grid contains a {d_min:.3f} A intermolecular contact: raise --r-min"
        )

    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        a.output,
        R=frames.astype(np.float32),
        Z=np.tile(z, (n_frames, 1)).astype(np.int32),
        N=np.full(n_frames, n_atoms, dtype=np.int32),
        # Placeholder reference energies: the evaluator takes multi-frame
        # geometries from --evaluate-reference-npz, which requires an E key.
        # These are NOT a reference and must never be read as one.
        E=np.zeros(n_frames, dtype=np.float64),
        grid_r=r_values.astype(np.float64),
        grid_theta=theta_deg.astype(np.float64),
        grid_shape=np.array([len(r_values), len(theta_deg)], dtype=np.int32),
        min_contact_A=contact.astype(np.float64),
        metadata=json.dumps(
            {
                "monomer": "TIP3 rigid",
                "r_definition": "mass-weighted COM separation (A)",
                "theta_definition": "rotation of monomer B about y (deg)",
                "min_intermolecular_distance_A": d_min,
            }
        ),
    )
    print(f"wrote {a.output}")
    print(f"  frames {n_frames}  ({len(r_values)} R x {len(theta_deg)} theta), "
          f"{n_atoms} atoms")
    print(f"  R    {r_values[0]:.2f} .. {r_values[-1]:.2f} A")
    print(f"  theta {theta_deg[0]:.0f} .. {theta_deg[-1]:.0f} deg")
    print(f"  closest intermolecular contact {d_min:.3f} A")
    for thr in (1.0, 1.4, 1.8):
        print(f"    frames with contact < {thr} A: {int((contact < thr).sum()):4d}"
              f" / {n_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
