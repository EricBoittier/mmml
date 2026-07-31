#!/usr/bin/env python3
"""Export gas φ/ψ scan frames as umbrella ``seed_mode=frames`` NPZ for one CV.

Picks the nearest scan column/row for each umbrella center along φ or ψ.
Writes ``R`` + ``Z`` so ``mmml umbrella-sample`` can load the file directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _centers(lo: float, hi: float, n: int) -> np.ndarray:
    return np.linspace(float(lo), float(hi), int(n), dtype=float)


def _periodic_abs(a: np.ndarray, x: float) -> np.ndarray:
    return np.abs(((a - x + 180.0) % 360.0) - 180.0)


def _load_z(gas: np.lib.npyio.NpzFile, traj: Path | None, n_atoms: int) -> np.ndarray:
    if "Z" in gas.files:
        z = np.asarray(gas["Z"], dtype=np.int32).reshape(-1)
        return z[:n_atoms]
    if traj is not None and traj.is_file():
        from ase.io import read

        atoms = read(str(traj), index=0)
        return np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)[:n_atoms]
    raise SystemExit(
        "Seed NPZ needs atomic numbers: pass --traj pointing at the gas "
        "ASE trajectory (e.g. phi_psi_pes.traj), or store Z in the gas NPZ."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gas-npz", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--cv", choices=("phi", "psi"), default="phi")
    p.add_argument("--xi-min", type=float, default=-180.0)
    p.add_argument("--xi-max", type=float, default=180.0)
    p.add_argument("--n-windows", type=int, default=13)
    p.add_argument(
        "--fixed-other",
        type=float,
        default=-60.0,
        help="Hold the other angle near this value when selecting seeds (°)",
    )
    p.add_argument(
        "--traj",
        type=Path,
        default=None,
        help="ASE traj for Z if missing from gas NPZ (default: sibling phi_psi_pes.traj)",
    )
    args = p.parse_args()

    gas_path = Path(args.gas_npz)
    gas = np.load(gas_path)
    phi_g = np.asarray(gas["phi_grid_deg"], dtype=float)
    psi_g = np.asarray(gas["psi_grid_deg"], dtype=float)
    pos = np.asarray(gas["positions_A"], dtype=float)
    n_atoms = int(pos.shape[-2])

    traj = args.traj
    if traj is None:
        cand = gas_path.parent / "phi_psi_pes.traj"
        traj = cand if cand.is_file() else None
    z = _load_z(gas, traj, n_atoms)

    centers = _centers(args.xi_min, args.xi_max, args.n_windows)
    frames = []
    used = []
    for xi0 in centers:
        if args.cv == "phi":
            i = int(np.argmin(_periodic_abs(phi_g, xi0)))
            j = int(np.argmin(_periodic_abs(psi_g, args.fixed_other)))
        else:
            i = int(np.argmin(_periodic_abs(phi_g, args.fixed_other)))
            j = int(np.argmin(_periodic_abs(psi_g, xi0)))
        frame = pos[i, j]
        if not np.all(np.isfinite(frame)):
            raise SystemExit(
                f"Non-finite seed at φ={phi_g[i]} ψ={psi_g[j]} (ξ₀={xi0}). "
                "Re-run the gas scan or pick a denser grid / different --fixed-other."
            )
        frames.append(frame)
        used.append((float(phi_g[i]), float(psi_g[j]), float(xi0)))

    r = np.stack(frames, axis=0)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        R=r,
        Z=z,
        xi0_deg=centers,
        seed_phi_psi_xi0=np.asarray(used, dtype=float),
    )
    print(f"Wrote {out}  shape={r.shape}  cv={args.cv}")
    for row in used:
        print(f"  seed φ={row[0]:+7.2f} ψ={row[1]:+7.2f} → ξ₀={row[2]:+7.2f}")


if __name__ == "__main__":
    main()
