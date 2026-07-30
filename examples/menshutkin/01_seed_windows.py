#!/usr/bin/env python3
"""Pick one umbrella-window seed per xi target from the reaction-coordinate scan.

The packed umbrella sampler can seed windows by stretching the CV, but xi =
r(C-Cl) - r(C-N) does not determine a geometry: the methyl umbrella inverts
between reactant and product, and no rigid translation reproduces that. Seeding
from the scan instead gives every window a geometry that already lies on the
reaction path.

Output is written in a canonical atom order -- Cl, N, C, then the three N-bound
hydrogens and the three C-bound hydrogens -- so downstream CV indices
(``2,0,2,1`` for xi) do not depend on which NPZ the frames came from. The
bundled sources disagree: ``scan_nh3_ch3cl.npz`` stores (Cl, N, C) at indices
(1, 5, 0) while ``nh3_ch3cl_filtered.npz`` uses (0, 1, 2).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent

_Z_CL, _Z_N, _Z_C, _Z_H = 17, 7, 6, 1


def canonical_order(z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Index permutation putting one NH3+CH3Cl frame in Cl, N, C, H(N)x3, H(C)x3 order.

    Hydrogens are assigned to whichever heavy atom (N or C) they sit closer to,
    which is unambiguous for every geometry on this reaction path: a methyl
    hydrogen never approaches the nitrogen more closely than its own carbon.
    """
    z = np.asarray(z)
    (i_cl,), (i_n,), (i_c,) = (
        np.flatnonzero(z == _Z_CL),
        np.flatnonzero(z == _Z_N),
        np.flatnonzero(z == _Z_C),
    )
    hydrogens = np.flatnonzero(z == _Z_H)
    d_n = np.linalg.norm(r[hydrogens] - r[i_n], axis=1)
    d_c = np.linalg.norm(r[hydrogens] - r[i_c], axis=1)
    on_n = hydrogens[d_n <= d_c]
    on_c = hydrogens[d_n > d_c]
    if len(on_n) != 3 or len(on_c) != 3:
        raise ValueError(
            f"expected 3 H on N and 3 H on C, got {len(on_n)} and {len(on_c)}; "
            "the frame is probably mid-transfer or corrupt"
        )
    # Sort within each group by distance so equivalent hydrogens stay consistent.
    on_n = on_n[np.argsort(d_n[d_n <= d_c])]
    on_c = on_c[np.argsort(d_c[d_n > d_c])]
    return np.concatenate([[i_cl, i_n, i_c], on_n, on_c]).astype(np.int64)


def load_scan(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(Z, R, xi)`` for every 9-atom frame, in canonical atom order."""
    data = np.load(path, allow_pickle=True)
    n = np.asarray(data["N"])
    keep = np.flatnonzero(n == 9)
    if keep.size == 0:
        raise ValueError(f"no 9-atom frames in {path}")
    z_all = np.asarray(data["Z"])[keep]
    r_all = np.asarray(data["R"])[keep]

    z_out = np.empty((len(keep), 9), dtype=np.int32)
    r_out = np.empty((len(keep), 9, 3), dtype=np.float64)
    for i, (z, r) in enumerate(zip(z_all, r_all, strict=True)):
        order = canonical_order(z, r)
        z_out[i] = z[order]
        r_out[i] = r[order]
    # Canonical order: Cl=0, N=1, C=2.
    xi = np.linalg.norm(r_out[:, 0] - r_out[:, 2], axis=1) - np.linalg.norm(
        r_out[:, 2] - r_out[:, 1], axis=1
    )
    return z_out, r_out, xi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan",
        type=Path,
        default=Path(REPO_ROOT / "examples/m/scan_nh3_ch3cl.npz"),
        help="NPZ of scan geometries covering the reaction coordinate",
    )
    parser.add_argument("--xi-min", type=float, default=-1.3)
    parser.add_argument("--xi-max", type=float, default=1.6)
    parser.add_argument("--n-windows", type=int, default=30)
    parser.add_argument(
        "--max-xi-error",
        type=float,
        default=0.06,
        help=(
            "Fail if any window's nearest scan frame is farther than this from "
            "its target xi (A). A large gap means the scan does not cover that "
            "part of the reaction path and the window would start off-center."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Seed NPZ (default: $MENSH_ARTIFACTS/gas/window_seeds.npz)",
    )
    args = parser.parse_args()

    artifacts = Path(
        os.environ.get("MENSH_ARTIFACTS", REPO_ROOT / "artifacts/menshutkin")
    )
    out = (
        Path(args.output)
        if args.output is not None
        else artifacts / "gas" / "window_seeds.npz"
    )

    z, r, xi = load_scan(args.scan)
    targets = np.linspace(args.xi_min, args.xi_max, args.n_windows)

    chosen = np.empty(len(targets), dtype=np.int64)
    used: set[int] = set()
    for w, target in enumerate(targets):
        order = np.argsort(np.abs(xi - target))
        # Distinct frames per window: duplicated seeds start replicas correlated,
        # which inflates MBAR's effective sample count for those windows.
        pick = next((int(i) for i in order if int(i) not in used), int(order[0]))
        used.add(pick)
        chosen[w] = pick

    achieved = xi[chosen]
    error = np.abs(achieved - targets)
    worst = int(np.argmax(error))

    print(f"scan: {args.scan}  ({len(xi)} nine-atom frames, xi {xi.min():.2f} .. {xi.max():.2f})")
    print(f"windows: {len(targets)}  xi {args.xi_min} .. {args.xi_max}")
    for w, (t, a, idx) in enumerate(zip(targets, achieved, chosen, strict=True)):
        r_ccl = float(np.linalg.norm(r[idx, 0] - r[idx, 2]))
        r_cn = float(np.linalg.norm(r[idx, 2] - r[idx, 1]))
        print(
            f"  w{w:02d}  xi0={t:+.3f}  seed xi={a:+.3f} (dev {a - t:+.3f})  "
            f"frame={idx:5d}  r(C-Cl)={r_ccl:.3f}  r(C-N)={r_cn:.3f}"
        )
    print(f"worst |xi - xi0| = {error[worst]:.3f} A at window {worst}")

    if error[worst] > args.max_xi_error:
        print(
            f"FAIL: window {worst} seed is {error[worst]:.3f} A from its target "
            f"(limit {args.max_xi_error}); the scan does not cover xi="
            f"{targets[worst]:+.3f}",
            file=sys.stderr,
        )
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        R=r[chosen],
        Z=z[chosen][0],
        N=np.full(len(chosen), 9, dtype=np.int32),
        xi_target=targets,
        xi_seed=achieved,
        source_frame=chosen,
    )
    meta = {
        "scan": str(args.scan),
        "n_windows": int(len(targets)),
        "xi_min": float(args.xi_min),
        "xi_max": float(args.xi_max),
        "xi_target": targets.tolist(),
        "xi_seed": achieved.tolist(),
        "max_abs_error_A": float(error[worst]),
        "atom_order": "Cl, N, C, H(N)x3, H(C)x3",
        "cv": "xi = r(C-Cl) - r(C-N)  ->  --cv-difference 2,0,2,1",
    }
    (out.parent / "window_seeds.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nWrote {out}  ({len(chosen)} seeds, canonical order Cl,N,C,H...)")
    print(f"Wrote {out.parent / 'window_seeds.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
