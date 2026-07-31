#!/usr/bin/env python3
"""Export gas φ/ψ scan frames as umbrella ``seed_mode=frames`` NPZ for one CV.

Picks the nearest scan column/row for each umbrella center along φ or ψ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _centers(lo: float, hi: float, n: int) -> np.ndarray:
    return np.linspace(float(lo), float(hi), int(n), dtype=float)


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
    args = p.parse_args()

    gas = np.load(args.gas_npz)
    phi_g = np.asarray(gas["phi_grid_deg"], dtype=float)
    psi_g = np.asarray(gas["psi_grid_deg"], dtype=float)
    pos = np.asarray(gas["positions_A"], dtype=float)
    z = None
    # Z may be absent; umbrella loader can take Z from a sibling structure.
    if "Z" in gas.files:
        z = np.asarray(gas["Z"], dtype=np.int32)
        if z.ndim > 1:
            z = z.reshape(-1)[: pos.shape[-2]]

    centers = _centers(args.xi_min, args.xi_max, args.n_windows)
    frames = []
    used = []
    for xi0 in centers:
        if args.cv == "phi":
            i = int(np.argmin(np.abs(((phi_g - xi0 + 180) % 360) - 180)))
            j = int(np.argmin(np.abs(((psi_g - args.fixed_other + 180) % 360) - 180)))
            frames.append(pos[i, j])
            used.append((float(phi_g[i]), float(psi_g[j]), float(xi0)))
        else:
            i = int(np.argmin(np.abs(((phi_g - args.fixed_other + 180) % 360) - 180)))
            j = int(np.argmin(np.abs(((psi_g - xi0 + 180) % 360) - 180)))
            frames.append(pos[i, j])
            used.append((float(phi_g[i]), float(psi_g[j]), float(xi0)))

    r = np.stack(frames, axis=0)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"R": r, "xi0_deg": centers, "seed_phi_psi_xi0": np.asarray(used)}
    if z is not None:
        payload["Z"] = z
    else:
        # Fallback: all carbons — loader usually gets Z from checkpoint path /
        # companion structure; require user to pass a PDB if needed.
        pass
    np.savez(out, **payload)
    print(f"Wrote {out}  shape={r.shape}  cv={args.cv}")
    for row in used:
        print(f"  seed φ={row[0]:+7.2f} ψ={row[1]:+7.2f} → ξ₀={row[2]:+7.2f}")


if __name__ == "__main__":
    main()
