#!/usr/bin/env python3
"""Per-monomer COM displacement from a CHARMM DCD trajectory.

Pass/fail (heuristic):
  - No NaN/Inf coordinates
  - Cluster COM drift from t=0 (max displacement and mean MSD) below limits
  - Internal spread (monomer COMs vs instantaneous cluster COM) below limit
  - Optional: worst monomer max displacement vs median below ``--outlier-factor``

Uniform cluster blow-up (all monomers ~equal drift) fails the cluster MSD/drift checks;
a single escaping monomer still fails the outlier ratio.

Examples
--------
  python scripts/analyze_monomer_com_dcd.py \\
    --dcd workflows/dcm_nve_scaling/results/dcm_7_nve/nve_dcm_7.dcd \\
    --n-monomers 7 --atoms-per-monomer 5 -o com_analysis.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np


def monomer_offsets(n_atoms: int, n_monomers: int, atoms_per_monomer: int) -> np.ndarray:
    expected = int(n_monomers) * int(atoms_per_monomer)
    if int(n_atoms) != expected:
        raise ValueError(
            f"DCD atom count {n_atoms} != {n_monomers} * {atoms_per_monomer} = {expected}"
        )
    per = int(atoms_per_monomer)
    return np.arange(0, int(n_monomers) * per + 1, per, dtype=int)


def _cluster_com(com: np.ndarray) -> np.ndarray:
    """Cluster COM trajectory, shape (n_frames, 3)."""
    return com.mean(axis=1)


def analyze_com(
    positions: np.ndarray,
    offsets: np.ndarray,
    *,
    outlier_factor: float = 2.0,
    max_cluster_disp_limit_A: float = 50.0,
    max_mean_msd_cluster_limit_A2: float | None = None,
    max_internal_rmsd_limit_A: float = 15.0,
    check_outlier_ratio: bool = True,
) -> dict:
    """Compute COM trajectories and displacement / MSD metrics."""
    pos = np.asarray(positions, dtype=np.float64)
    if not np.all(np.isfinite(pos)):
        raise ValueError("DCD contains NaN or Inf coordinates")

    n_frames, _n_atoms, _ = pos.shape
    n_mol = int(len(offsets) - 1)
    com = np.zeros((n_frames, n_mol, 3), dtype=np.float64)
    for mi in range(n_mol):
        s, e = int(offsets[mi]), int(offsets[mi + 1])
        com[:, mi, :] = pos[:, s:e, :].mean(axis=1)

    com_cluster = _cluster_com(com)
    cluster_disp = np.linalg.norm(com_cluster - com_cluster[0:1], axis=1)
    max_cluster_com_disp_A = float(cluster_disp.max())
    mean_msd_cluster_A2 = float(np.mean(np.sum((com_cluster - com_cluster[0:1]) ** 2, axis=1)))

    if max_mean_msd_cluster_limit_A2 is None:
        max_mean_msd_cluster_limit_A2 = float(max_cluster_disp_limit_A) ** 2

    com_rel = com - com_cluster[:, np.newaxis, :]
    internal_rmsd = np.sqrt(np.mean(np.sum(com_rel**2, axis=2), axis=1))
    max_internal_rmsd_A = float(internal_rmsd.max())
    initial_internal = float(internal_rmsd[0]) if internal_rmsd[0] > 1e-8 else 1e-8
    internal_rmsd_growth = max_internal_rmsd_A / initial_internal

    com_disp = np.linalg.norm(com - com[0:1], axis=2)
    max_disp_per_monomer = com_disp.max(axis=0)
    worst_idx = int(np.argmax(max_disp_per_monomer))
    median = float(np.median(max_disp_per_monomer))
    worst = float(max_disp_per_monomer[worst_idx])
    if median > 1e-8:
        ratio = worst / median
    elif worst > 1e-8:
        ratio = float("inf")
    else:
        ratio = 1.0

    checks: list[tuple[str, bool]] = [
        ("cluster_com_disp", max_cluster_com_disp_A <= max_cluster_disp_limit_A),
        (
            "cluster_com_msd",
            mean_msd_cluster_A2 <= max_mean_msd_cluster_limit_A2,
        ),
        ("internal_rmsd", max_internal_rmsd_A <= max_internal_rmsd_limit_A),
    ]
    if check_outlier_ratio:
        checks.append(("outlier_ratio", ratio <= float(outlier_factor)))

    fail_reasons = [name for name, passed in checks if not passed]
    ok = not fail_reasons

    return {
        "com": com,
        "com_cluster": com_cluster,
        "com_disp": com_disp,
        "com_rel": com_rel,
        "max_disp_per_monomer": max_disp_per_monomer,
        "worst_monomer_0based": worst_idx,
        "worst_monomer_1based": worst_idx + 1,
        "median_max_disp_A": median,
        "worst_max_disp_A": worst,
        "outlier_ratio": ratio,
        "max_cluster_com_disp_A": max_cluster_com_disp_A,
        "mean_msd_cluster_A2": mean_msd_cluster_A2,
        "max_internal_rmsd_A": max_internal_rmsd_A,
        "internal_rmsd_growth": internal_rmsd_growth,
        "max_cluster_disp_limit_A": float(max_cluster_disp_limit_A),
        "max_mean_msd_cluster_limit_A2": float(max_mean_msd_cluster_limit_A2),
        "max_internal_rmsd_limit_A": float(max_internal_rmsd_limit_A),
        "fail_reasons": fail_reasons,
        "n_frames": n_frames,
        "n_monomers": n_mol,
        "ok": ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcd", type=Path, required=True)
    parser.add_argument("--n-monomers", type=int, required=True)
    parser.add_argument("--atoms-per-monomer", type=int, default=5)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--outlier-factor",
        type=float,
        default=2.0,
        help="Fail if max COM disp of worst monomer exceeds this × median",
    )
    parser.add_argument(
        "--max-cluster-disp-A",
        type=float,
        default=50.0,
        help="Fail if cluster COM drifts more than this from t=0 (Å)",
    )
    parser.add_argument(
        "--max-mean-msd-cluster-A2",
        type=float,
        default=None,
        help="Fail if time-averaged cluster COM MSD exceeds this (Å²); default: max_cluster_disp²",
    )
    parser.add_argument(
        "--max-internal-rmsd-A",
        type=float,
        default=15.0,
        help="Fail if max RMSD of monomer COMs about cluster COM exceeds this (Å)",
    )
    parser.add_argument(
        "--no-outlier-ratio",
        action="store_true",
        help="Skip worst-vs-median monomer outlier check",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit 0 after writing NPZ (record ok=false for downstream collect)",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    from mmml.utils.dcd_reader import read_dcd_trajectory

    pos, hdr = read_dcd_trajectory(args.dcd, max_frames=args.max_frames)
    offsets = monomer_offsets(
        pos.shape[1],
        args.n_monomers,
        args.atoms_per_monomer,
    )
    stats = analyze_com(
        pos,
        offsets,
        outlier_factor=args.outlier_factor,
        max_cluster_disp_limit_A=args.max_cluster_disp_A,
        max_mean_msd_cluster_limit_A2=args.max_mean_msd_cluster_A2,
        max_internal_rmsd_limit_A=args.max_internal_rmsd_A,
        check_outlier_ratio=not args.no_outlier_ratio,
    )
    stats["dcd_header"] = np.array([hdr], dtype=object)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        com=stats["com"],
        com_cluster=stats["com_cluster"],
        com_disp=stats["com_disp"],
        max_disp_per_monomer=stats["max_disp_per_monomer"],
        worst_monomer_0based=stats["worst_monomer_0based"],
        worst_monomer_1based=stats["worst_monomer_1based"],
        median_max_disp_A=stats["median_max_disp_A"],
        worst_max_disp_A=stats["worst_max_disp_A"],
        outlier_ratio=stats["outlier_ratio"],
        max_cluster_com_disp_A=stats["max_cluster_com_disp_A"],
        mean_msd_cluster_A2=stats["mean_msd_cluster_A2"],
        max_internal_rmsd_A=stats["max_internal_rmsd_A"],
        internal_rmsd_growth=stats["internal_rmsd_growth"],
        fail_reasons=np.array(stats["fail_reasons"], dtype=object),
        n_frames=stats["n_frames"],
        n_monomers=stats["n_monomers"],
        ok=stats["ok"],
    )

    status = "OK" if stats["ok"] else "FAIL"
    reasons = ",".join(stats["fail_reasons"]) if stats["fail_reasons"] else "-"
    print(
        f"{status}: N={stats['n_monomers']} frames={stats['n_frames']} "
        f"cluster_disp={stats['max_cluster_com_disp_A']:.4f} Å "
        f"cluster_MSD={stats['mean_msd_cluster_A2']:.4f} Å² "
        f"internal_rmsd={stats['max_internal_rmsd_A']:.4f} Å "
        f"worst monomer {stats['worst_monomer_1based']} "
        f"max_disp={stats['worst_max_disp_A']:.4f} Å "
        f"(median={stats['median_max_disp_A']:.4f}, ratio={stats['outlier_ratio']:.2f}) "
        f"[{reasons}]",
        flush=True,
    )
    if args.no_fail:
        return 0
    return 0 if stats["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
