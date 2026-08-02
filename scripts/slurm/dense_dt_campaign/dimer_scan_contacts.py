#!/usr/bin/env python3
"""Annotate orientation-scan CSVs with intermolecular dmin; filter clashes.

COM–COM ``r`` is not a steric coordinate for DCM: many rays at r≈3.5 Å still
have atom–atom contacts ≪ 2 Å. Those clash points dominate well metrics
(−30…−55 kcal/mol) and force envelopes. This module adds ``dmin_A`` and
clash-aware summaries using
:data:`mmml.analysis.dimer_scans.DEFAULT_ORIENT_MIN_CONTACT_A`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mmml.analysis.dimer_scans import (
    DEFAULT_ORIENT_MIN_CONTACT_A,
    intermolecular_min_distance,
)


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
        axis=1,
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
    return np.stack(
        [r * np.sin(d), r * np.cos(d), R * np.sin(a), R * np.cos(a)], axis=1
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_monomer(data: Path, n_mono: int = 5) -> tuple[np.ndarray, np.ndarray]:
    raw = dict(np.load(data, allow_pickle=True))
    Z1 = np.asarray(raw["Z"][0])[:n_mono]
    R1 = np.asarray(raw["R"][0])[:n_mono]
    R1 = R1 - R1.mean(axis=0)
    return R1, Z1


def dimer_positions(
    R1: np.ndarray, dvec: np.ndarray, quat: np.ndarray, r: float
) -> tuple[np.ndarray, np.ndarray]:
    Rb0 = R1 @ quat_to_matrix(quat).T
    Ra = R1 - 0.5 * r * dvec
    Rb = Rb0 + 0.5 * r * dvec
    return Ra, Rb


def annotate_dmin(
    df: pd.DataFrame,
    *,
    R1: np.ndarray,
    n_directions: int | None = None,
    n_orientations: int | None = None,
) -> pd.DataFrame:
    """Add ``dmin_A`` (cross-monomer atom–atom min) to an orient_components table."""
    out = df.copy()
    if n_directions is None:
        n_directions = int(out["direction"].max()) + 1
    if n_orientations is None:
        n_orientations = int(out["orientation"].max()) + 1
    dirs = fibonacci_sphere(n_directions)
    quats = super_fibonacci(n_orientations)
    # Cache unique (di, qi, r) — many rows share the same geometry key.
    cache: dict[tuple[int, int, float], float] = {}
    dmins = np.empty(len(out), dtype=np.float64)
    for i, row in enumerate(out.itertuples(index=False)):
        di = int(row.direction)
        qi = int(row.orientation)
        r = float(row.r_A)
        key = (di, qi, r)
        if key not in cache:
            Ra, Rb = dimer_positions(R1, dirs[di], quats[qi], r)
            cache[key] = intermolecular_min_distance(Ra, Rb)
        dmins[i] = cache[key]
    out["dmin_A"] = dmins
    out["contact_ok"] = out["dmin_A"] >= DEFAULT_ORIENT_MIN_CONTACT_A
    return out


def contact_filtered_metrics(
    df: pd.DataFrame,
    *,
    min_contact: float = DEFAULT_ORIENT_MIN_CONTACT_A,
    soft_r: float = 3.4,
    min_rays_for_mean: int | None = None,
) -> dict:
    """Well / mean-curve stats using only points with ``dmin_A >= min_contact``."""
    ok = df[df["dmin_A"] >= min_contact].copy()
    n_tot = int(len(df))
    n_ok = int(len(ok))
    n_rays_all = int(df["ray"].nunique()) if len(df) else 0
    if min_rays_for_mean is None:
        # Avoid quoting a "mean well" from the 2–6 rays that barely clear at
        # the shortest r; require ≥10% of the orientation set (min 8).
        min_rays_for_mean = max(8, int(np.ceil(0.10 * max(n_rays_all, 1))))
    if n_ok == 0:
        return {
            "min_contact_A": min_contact,
            "n_points_total": n_tot,
            "n_points_contact_ok": 0,
            "frac_points_contact_ok": 0.0,
            "min_rays_for_mean": min_rays_for_mean,
        }

    # Mean curve over contact-ok points only (r bins may have uneven n_rays).
    g = ok.groupby("r_A")["E_int_kcal"]
    mean_curve = g.mean()
    counts = g.count()
    reliable = counts >= min_rays_for_mean
    if reliable.any():
        mean_use = mean_curve[reliable]
        imin = int(mean_use.to_numpy().argmin())
        r_mean = float(mean_use.index[imin])
        e_mean = float(mean_use.iloc[imin])
    else:
        imin = int(mean_curve.to_numpy().argmin())
        r_mean = float(mean_curve.index[imin])
        e_mean = float(mean_curve.iloc[imin])

    # Per-ray wells among contact-ok points.
    ray_mins = []
    soft_mins = []
    for ray, sub in ok.groupby("ray"):
        i = int(sub["E_int_kcal"].argmin())
        row = sub.iloc[i]
        ray_mins.append(
            dict(
                ray=int(ray),
                r_A=float(row.r_A),
                E_int_kcal=float(row.E_int_kcal),
                dmin_A=float(row.dmin_A),
                direction=int(row.direction),
                orientation=int(row.orientation),
            )
        )
        soft = sub[sub["r_A"] >= soft_r]
        if len(soft):
            j = int(soft["E_int_kcal"].argmin())
            srow = soft.iloc[j]
            soft_mins.append(
                dict(
                    ray=int(ray),
                    r_A=float(srow.r_A),
                    E_int_kcal=float(srow.E_int_kcal),
                    dmin_A=float(srow.dmin_A),
                    direction=int(srow.direction),
                    orientation=int(srow.orientation),
                )
            )

    ray_e = np.array([x["E_int_kcal"] for x in ray_mins]) if ray_mins else np.array([])
    soft_e = np.array([x["E_int_kcal"] for x in soft_mins]) if soft_mins else np.array([])
    deepest_soft = min(soft_mins, key=lambda x: x["E_int_kcal"]) if soft_mins else None

    return {
        "min_contact_A": min_contact,
        "n_points_total": n_tot,
        "n_points_contact_ok": n_ok,
        "frac_points_contact_ok": n_ok / max(n_tot, 1),
        "n_rays_with_contact_ok": int(ok["ray"].nunique()),
        "mean_curve_min_kcal": e_mean,
        "r_at_mean_curve_min_A": r_mean,
        "n_rays_at_mean_curve_min": int(counts.loc[r_mean]),
        "min_rays_for_mean": min_rays_for_mean,
        "mean_of_ray_minima_kcal": float(ray_e.mean()) if ray_e.size else None,
        "median_of_ray_minima_kcal": float(np.median(ray_e)) if ray_e.size else None,
        "deepest_ray_min_kcal": float(ray_e.min()) if ray_e.size else None,
        "mean_soft_well_kcal": float(soft_e.mean()) if soft_e.size else None,
        "median_soft_well_kcal": float(np.median(soft_e)) if soft_e.size else None,
        "deepest_soft_kcal": float(soft_e.min()) if soft_e.size else None,
        "r_at_deepest_soft": (
            float(deepest_soft["r_A"]) if deepest_soft is not None else None
        ),
        "deepest_soft_ray": deepest_soft,
        "soft_wells": soft_mins,
        "ray_wells": ray_mins,
        "mean_curve": {
            "r_A": mean_curve.index.to_numpy().tolist(),
            "E_int_kcal_mean": mean_curve.to_numpy().tolist(),
            "n_rays": counts.to_numpy().astype(int).tolist(),
        },
    }


def annotate_csv(
    csv_path: Path,
    *,
    data: Path,
    out_csv: Path | None = None,
    summary_json: Path | None = None,
    min_contact: float = DEFAULT_ORIENT_MIN_CONTACT_A,
) -> tuple[pd.DataFrame, dict]:
    R1, _ = load_monomer(data)
    df = pd.read_csv(csv_path)
    df = annotate_dmin(df, R1=R1)
    metrics = contact_filtered_metrics(df, min_contact=min_contact)
    dest = out_csv or csv_path
    df.to_csv(dest, index=False)
    if summary_json is not None:
        # Drop bulky lists for the on-disk summary companion.
        slim = {
            k: v
            for k, v in metrics.items()
            if k not in ("soft_wells", "ray_wells", "mean_curve")
        }
        slim["deepest_soft"] = metrics.get("deepest_soft_ray")
        summary_json.write_text(json.dumps(slim, indent=2) + "\n")
    return df, metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/lj_scales/dense_dt_campaign/dimer_scans/orient_components.csv"),
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path("artifacts/lj_scales/dataset_cgenff.npz"),
    )
    p.add_argument("--min-contact", type=float, default=DEFAULT_ORIENT_MIN_CONTACT_A)
    p.add_argument(
        "--summary-json",
        type=Path,
        default=Path(
            "artifacts/lj_scales/dense_dt_campaign/dimer_scans/summary_contact_filtered.json"
        ),
    )
    args = p.parse_args()
    df, metrics = annotate_csv(
        args.csv,
        data=args.data,
        out_csv=args.csv,
        summary_json=args.summary_json,
        min_contact=args.min_contact,
    )
    print(
        f"annotated {args.csv}: {metrics['n_points_contact_ok']}/"
        f"{metrics['n_points_total']} points with dmin>={args.min_contact:g} Å"
    )
    print(
        f"  mean curve min {metrics.get('mean_curve_min_kcal'):.2f} kcal @ "
        f"{metrics.get('r_at_mean_curve_min_A'):.2f} Å "
        f"(n_rays={metrics.get('n_rays_at_mean_curve_min')})"
    )
    print(
        f"  soft well mean/median/deepest: "
        f"{metrics.get('mean_soft_well_kcal'):.2f} / "
        f"{metrics.get('median_soft_well_kcal'):.2f} / "
        f"{metrics.get('deepest_soft_kcal'):.2f} kcal/mol"
    )
    print(f"  -> {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
