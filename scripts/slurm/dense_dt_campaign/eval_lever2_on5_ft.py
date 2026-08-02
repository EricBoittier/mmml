#!/usr/bin/env python3
"""Contact-ok soft-well metrics for the lever-2 on=5 fine-tune checkpoint.

Scans the new FT ckpt at matching mm_switch_on=5 and compares to the archived
epoch222 deploy-only on=5 ablation CSV (if present).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "slurm" / "dense_dt_campaign"))

from ablate_overbind import run_scan  # noqa: E402

EV = 23.0605
DEFAULT_CKPT = ROOT / (
    "artifacts/lj_scales/ckpts/params_hybrid_mm_lever2_on5_ft_2026-08-02_18-29-24.json"
)
DEFAULT_SIDECAR = ROOT / (
    "artifacts/lj_scales/ckpts/"
    "hybrid_mm_lever2_on5_ft-06e6e3cd-1aa9-48f6-8b2a-f9427391caa8/hybrid_mm.json"
)
DEFAULT_DATA = ROOT / "artifacts/lj_scales/dataset_cgenff.npz"
DEFAULT_OUT = ROOT / "artifacts/lj_scales/dense_dt_campaign/overbind_ablation/lever2_on5_ft"
REF_ON5 = ROOT / (
    "artifacts/lj_scales/dense_dt_campaign/overbind_ablation/handoff_on5_w1p5_components.csv"
)
REF_BASE = ROOT / (
    "artifacts/lj_scales/dense_dt_campaign/dimer_scans/orient_components.csv"
)


def _soft_stats(
    csv: Path,
    *,
    min_contact: float = 2.0,
    dmin_lookup: pd.DataFrame | None = None,
) -> dict:
    df = pd.read_csv(csv)
    if "contact_ok" not in df.columns:
        df = df.copy()
        if "dmin_A" not in df.columns:
            if dmin_lookup is None:
                raise SystemExit(f"missing dmin_A/contact_ok in {csv}")
            # Geometry is model-independent: reuse dmin from a scan with same rays/r.
            key = dmin_lookup.set_index(["ray", "r_A"])["dmin_A"]
            df["dmin_A"] = [
                float(key.loc[(int(r.ray), float(r.r_A))])
                if (int(r.ray), float(r.r_A)) in key.index
                else float("nan")
                for r in df.itertuples()
            ]
        df["contact_ok"] = df["dmin_A"] >= min_contact
    ok = df[df["contact_ok"]]
    soft = []
    for _, sub in ok.groupby("ray"):
        s = sub[sub["r_A"] >= 3.4]
        if len(s):
            soft.append(float(s["E_int_kcal"].min()))
    soft_a = np.asarray(soft, dtype=float)
    g = ok.groupby("r_A")["E_int_kcal"]
    counts, means = g.count(), g.mean()
    keep = counts >= max(8, int(np.ceil(0.1 * max(df["ray"].nunique(), 1))))
    return {
        "soft_median": float(np.median(soft_a)) if soft_a.size else float("nan"),
        "soft_mean": float(soft_a.mean()) if soft_a.size else float("nan"),
        "soft_deepest": float(soft_a.min()) if soft_a.size else float("nan"),
        "mean_curve_min": float(means[keep].min()) if keep.any() else float(means.min()),
        "r_mean": float(means[keep].idxmin()) if keep.any() else float(means.idxmin()),
        "n_soft": int(soft_a.size),
        "frac_contact_ok": float(df["contact_ok"].mean()) if len(df) else 0.0,
    }


def main() -> int:
    ckpt = Path(os.environ.get("DDC_ON5_EVAL_CKPT", DEFAULT_CKPT))
    sidecar = Path(os.environ.get("DDC_ON5_EVAL_SIDECAR", DEFAULT_SIDECAR))
    data = Path(os.environ.get("DDC_ON5_EVAL_DATA", DEFAULT_DATA))
    out = Path(os.environ.get("DDC_ON5_EVAL_OUT", DEFAULT_OUT))
    out.mkdir(parents=True, exist_ok=True)

    print(f"ckpt={ckpt}")
    print(f"sidecar={sidecar}")
    print(f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')}")
    import jax

    print("JAX devices:", jax.devices())
    if not any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in jax.devices()):
        print("WARNING: no CudaDevice — eval will be slow on CPU", flush=True)

    summary = run_scan(
        checkpoint=ckpt,
        sidecar=sidecar,
        data=data,
        out_dir=out,
        tag="ft_on5_matched",
        es_off=False,
        mm_switch_on=5.0,
        ml_switch_width=1.5,
        mm_switch_width=5.0,
        n_directions=8,
        n_orientations=8,
        n_r=36,
        r_min=2.5,
        r_max=12.0,
        batch_size=64,
    )
    ft_csv = out / "ft_on5_matched_components.csv"
    ft_stats = _soft_stats(ft_csv)
    ft_df = pd.read_csv(ft_csv)

    compare = {
        "ft_on5_matched": ft_stats,
        "run_scan_summary": {
            k: summary.get(k)
            for k in (
                "median_soft_well_kcal",
                "mean_soft_well_kcal",
                "deepest_soft_kcal",
                "mean_curve_min_kcal",
                "frac_points_contact_ok",
                "mm_switch_on",
            )
            if k in summary
        },
    }
    if REF_ON5.is_file():
        compare["deploy_only_on5_epoch222"] = _soft_stats(REF_ON5, dmin_lookup=ft_df)
    if REF_BASE.is_file():
        try:
            compare["baseline_on8_epoch222"] = _soft_stats(REF_BASE)
        except SystemExit as e:
            compare["baseline_on8_epoch222_error"] = str(e)

    lit = [-5.0, -3.0]
    med = ft_stats["soft_median"]
    deep = ft_stats["soft_deepest"]
    mean_c = ft_stats["mean_curve_min"]
    compare["verdict"] = {
        "literature_kcal": lit,
        "soft_median_in_lit_window": bool(lit[0] <= med <= lit[1]) if med == med else False,
        "deploy_ready": bool(
            med == med
            and lit[0] - 1.0 <= med <= lit[1] + 0.5
            and deep > -15.0
            and mean_c > -8.0
        ),
        "note": (
            "FT matched train/deploy on=5. Target soft-well median ≈ −3…−5 kcal; "
            "reject if soft deepest ≲ −15 or mean-curve ≲ −8 (overbind tail)."
        ),
    }
    (out / "contact_ok_ft_compare.json").write_text(json.dumps(compare, indent=2) + "\n")
    print(json.dumps(compare, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
