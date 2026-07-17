#!/usr/bin/env python3
"""GFN2-xTB orientation scan on the same S² × SO(3) grid as the ML scan.

Writes ``rays.csv`` in the same schema as ``scan_dimer_orientations.py`` so the
hemisphere / atlas plotters can emit a flip-pair (``*_ML.png`` / ``*_xTB.png``).

    uv run python scripts/scan_orientations_xtb.py \\
        --monomer /Volumes/PortableSSD/DATA/acodcm/pdb/aco.pdb \\
        --n-directions 10 --n-orientations 24 \\
        --out /Volumes/PortableSSD/DATA/acodcm/orient_xtb
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

EV_TO_KCAL = 23.0605

# Module-level worker state (set once per process)
_Z: np.ndarray | None = None
_R1: np.ndarray | None = None
_E_MONO: float | None = None
_CALC = None


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )


def super_fibonacci(n: int) -> np.ndarray:
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    i = np.arange(n) + 0.5
    s = i / n
    t = s * n / phi
    d = 2.0 * np.pi * (t - np.floor(t))
    r = np.sqrt(s)
    R = np.sqrt(1.0 - s)
    t2 = i / psi
    a = 2.0 * np.pi * (t2 - np.floor(t2))
    return np.stack([r * np.sin(d), r * np.cos(d), R * np.sin(a), R * np.cos(a)], axis=1)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def find_minima(e: np.ndarray, prominence: float) -> list[int]:
    out = []
    for i in range(1, len(e) - 1):
        if e[i] < e[i - 1] and e[i] < e[i + 1]:
            left = np.max(e[:i])
            right = np.max(e[i + 1 :])
            if min(left, right) - e[i] >= prominence:
                out.append(i)
    return out


def _load_monomer(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".npz":
        raw = dict(np.load(path, allow_pickle=True))
        if "coords" in raw and "z" in raw:
            z = np.asarray(raw["z"], dtype=int)
            r = np.asarray(raw["coords"], dtype=float)
        else:
            raise SystemExit(f"unrecognised npz keys: {list(raw)}")
    else:
        atoms = read(str(path))
        z = atoms.get_atomic_numbers()
        r = atoms.get_positions()
    r = r - r.mean(axis=0)
    return z, r


def _worker_init(z: np.ndarray, r1: np.ndarray) -> None:
    global _Z, _R1, _E_MONO, _CALC
    from tblite.ase import TBLite

    _Z = np.asarray(z, dtype=int)
    _R1 = np.asarray(r1, dtype=float)
    _CALC = TBLite(method="GFN2-xTB", verbosity=0)
    m = Atoms(numbers=_Z, positions=_R1)
    m.calc = _CALC
    _E_MONO = float(m.get_potential_energy())


def _eval_ray(payload: tuple) -> dict:
    """Evaluate one ray: (ray, di, qi, direction, quat, rs, prominence_eV)."""
    from tblite.ase import TBLite

    global _Z, _R1, _E_MONO, _CALC
    ray, di, qi, direction, quat, rs, prominence = payload
    assert _Z is not None and _R1 is not None and _E_MONO is not None
    if _CALC is None:
        _CALC = TBLite(method="GFN2-xTB", verbosity=0)

    from ase.calculators.calculator import CalculationFailed

    Rb = _R1 @ quat_to_matrix(quat).T
    e = np.full(len(rs), np.nan, dtype=float)
    for i, r in enumerate(rs):
        ra = _R1 - 0.5 * r * direction
        rb = Rb + 0.5 * r * direction
        d = Atoms(numbers=np.concatenate([_Z, _Z]), positions=np.vstack([ra, rb]))
        try:
            d.calc = _CALC
            e[i] = float(d.get_potential_energy()) - 2.0 * _E_MONO
        except (CalculationFailed, RuntimeError):
            # SCF can fail at harsh clashes — drop the point; refresh calc state
            _CALC = TBLite(method="GFN2-xTB", verbosity=0)
            e[i] = np.nan

    ok = np.isfinite(e)
    if ok.sum() < 3:
        return {
            "ray": int(ray),
            "direction": int(di),
            "orientation": int(qi),
            "n_min": 0,
            "n_min_ml": 0,
            "n_min_wall": 0,
            "e_min_kcal": float("nan"),
            "r_at_min": float("nan"),
        }

    # subtract asymptote from the largest-r finite point
    e = e.copy()
    e[ok] = e[ok] - e[ok][np.argmax(rs[ok])]
    # find_minima needs a contiguous curve — fill tiny gaps by linear interp
    if not ok.all():
        idx = np.arange(len(e))
        e[~ok] = np.interp(idx[~ok], idx[ok], e[ok])

    mins = find_minima(e, prominence)
    if len(mins):
        i_best = mins[int(np.argmin(e[mins]))]
    else:
        i_best = int(np.argmin(e))
    return {
        "ray": int(ray),
        "direction": int(di),
        "orientation": int(qi),
        "n_min": len(mins),
        "n_min_ml": len(mins),  # schema compat: "spurious" = >1 minimum
        "n_min_wall": 0,
        "e_min_kcal": float(e[i_best] * EV_TO_KCAL),
        "r_at_min": float(rs[i_best]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--monomer", type=Path, required=True)
    p.add_argument("--n-directions", type=int, default=10)
    p.add_argument("--n-orientations", type=int, default=24)
    p.add_argument("--r-min", type=float, default=3.0)
    p.add_argument("--r-max", type=float, default=10.0)
    p.add_argument("--n-r", type=int, default=36)
    p.add_argument(
        "--prominence",
        type=float,
        default=0.0129,
        help="min well prominence (eV); default matches ML scan (kT @ 150 K)",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    z, r1 = _load_monomer(args.monomer)
    dirs = fibonacci_sphere(args.n_directions)
    quats = super_fibonacci(args.n_orientations)
    rs = np.linspace(args.r_min, args.r_max, args.n_r)
    n_rays = len(dirs) * len(quats)
    print(
        f"GFN2-xTB orientation scan: {len(dirs)} dirs x {len(quats)} oris = "
        f"{n_rays} rays x {len(rs)} r = {n_rays * len(rs)} evals  "
        f"(workers={args.workers})"
    )

    jobs = []
    for di, d in enumerate(dirs):
        for qi, q in enumerate(quats):
            ray = di * len(quats) + qi
            jobs.append((ray, di, qi, d, q, rs, args.prominence))

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(z, r1),
    ) as pool:
        futs = {pool.submit(_eval_ray, job): job[0] for job in jobs}
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            done += 1
            if done % 20 == 0 or done == n_rays:
                elapsed = time.time() - t0
                print(
                    f"  {done}/{n_rays} rays  ({elapsed:.0f}s, "
                    f"{elapsed / max(done, 1):.2f}s/ray)",
                    flush=True,
                )

    rows.sort(key=lambda r: r["ray"])
    csv_path = args.out / "rays.csv"
    with csv_path.open("w") as fh:
        fh.write("ray,direction,orientation,n_min,n_min_ml,n_min_wall,e_min_kcal,r_at_min\n")
        for r in rows:
            fh.write(
                f"{r['ray']},{r['direction']},{r['orientation']},"
                f"{r['n_min']},{r['n_min_ml']},{r['n_min_wall']},"
                f"{r['e_min_kcal']},{r['r_at_min']}\n"
            )

    n_spur = sum(1 for r in rows if r["n_min_ml"] > 1)
    summary = {
        "method": "GFN2-xTB",
        "monomer": str(args.monomer),
        "n_directions": args.n_directions,
        "n_orientations": args.n_orientations,
        "n_r": args.n_r,
        "r_min": args.r_min,
        "r_max": args.r_max,
        "prominence_eV": args.prominence,
        "n_rays": n_rays,
        "n_rays_spurious": n_spur,
        "frac_rays_spurious": n_spur / n_rays,
        "mean_min_kcal": float(np.mean([r["e_min_kcal"] for r in rows])),
        "deepest_kcal": float(np.min([r["e_min_kcal"] for r in rows])),
        "elapsed_s": time.time() - t0,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {csv_path}")
    print(
        f"spurious {n_spur}/{n_rays} ({100 * n_spur / n_rays:.1f}%)  "
        f"deepest {summary['deepest_kcal']:.2f} kcal/mol  "
        f"in {summary['elapsed_s']:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
