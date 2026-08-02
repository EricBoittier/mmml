#!/usr/bin/env python3
"""Contact-ok soft-well sweep over distill FT Orbax epochs (+ final portable JSON).

Picks the epoch with soft median closest to lit (−4 kcal) subject to deepest ≳ −15.
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

LIT_TARGET = -4.0
LIT_LO, LIT_HI = -5.0, -3.0
DEEPEST_FLOOR = -15.0
MEAN_CURVE_FLOOR = -8.0

DEFAULT_RUN = ROOT / (
    "artifacts/lj_scales/ckpts/"
    # filled at runtime by newest hybrid_mm_lever2_on5_distill-* dir
)
DEFAULT_OUT = ROOT / (
    "artifacts/lj_scales/dense_dt_campaign/overbind_ablation/lever2_on5_distill"
)
DEFAULT_DATA = ROOT / "artifacts/lj_scales/dataset_cgenff.npz"
DEFAULT_SIDECAR_FALLBACK = ROOT / (
    "artifacts/lj_scales/ckpts/"
    "hybrid_mm_fixed_lj_scales-4d68132d-c686-4ded-9887-efc16d5b2638/hybrid_mm.json"
)
EPOCHS_DEFAULT = (1, 3, 5, 8, 10, 12, 15)


def _newest_run(ckpt_dir: Path, tag: str) -> Path:
    cands = sorted(ckpt_dir.glob(f"{tag}-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise SystemExit(f"no run dirs matching {tag}-* under {ckpt_dir}")
    return cands[0]


def _soft_stats(csv: Path, *, min_contact: float = 2.0) -> dict:
    df = pd.read_csv(csv)
    if "contact_ok" not in df.columns:
        df = df.copy()
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
        "soft_p10": float(np.percentile(soft_a, 10)) if soft_a.size else float("nan"),
        "mean_curve_min": float(means[keep].min()) if keep.any() else float(means.min()),
        "r_mean": float(means[keep].idxmin()) if keep.any() else float(means.idxmin()),
        "n_soft": int(soft_a.size),
    }


def _score(stats: dict) -> tuple:
    """Lower is better. Reject deep-tail violators with large penalty."""
    med = stats["soft_median"]
    deep = stats["soft_deepest"]
    mean_c = stats["mean_curve_min"]
    penalty = 0.0
    if deep < DEEPEST_FLOOR:
        penalty += 50.0 + abs(deep - DEEPEST_FLOOR)
    if mean_c < MEAN_CURVE_FLOOR:
        penalty += 20.0 + abs(mean_c - MEAN_CURVE_FLOOR)
    return (penalty, abs(med - LIT_TARGET), -med)


def main() -> int:
    tag = os.environ.get("DDC_ON5D_TAG", "hybrid_mm_lever2_on5_distill")
    ckpt_root = Path(os.environ.get("DDC_ON5D_CKPT_DIR", ROOT / "artifacts/lj_scales/ckpts"))
    run_dir = Path(os.environ["DDC_ON5D_RUN_DIR"]) if os.environ.get("DDC_ON5D_RUN_DIR") else _newest_run(ckpt_root, tag)
    sidecar = run_dir / "hybrid_mm.json"
    if not sidecar.is_file():
        sidecar = Path(os.environ.get("DDC_ON5D_SIDECAR", DEFAULT_SIDECAR_FALLBACK))
    data = Path(os.environ.get("DDC_ON5D_DATA", DEFAULT_DATA))
    out = Path(os.environ.get("DDC_ON5D_EVAL_OUT", DEFAULT_OUT))
    out.mkdir(parents=True, exist_ok=True)

    epochs_s = os.environ.get("DDC_ON5D_SWEEP_EPOCHS", ",".join(str(e) for e in EPOCHS_DEFAULT))
    epochs = [int(x) for x in epochs_s.split(",") if x.strip()]

    import jax

    print(f"run_dir={run_dir}")
    print(f"sidecar={sidecar}")
    print(f"epochs={epochs}")
    print("JAX devices:", jax.devices())

    rows = []
    for ep in epochs:
        ep_dir = run_dir / f"epoch-{ep}"
        if not ep_dir.is_dir():
            print(f"[skip] missing {ep_dir}")
            continue
        tag_ep = f"distill_ep{ep}"
        summary = run_scan(
            checkpoint=ep_dir,
            sidecar=sidecar,
            data=data,
            out_dir=out,
            tag=tag_ep,
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
        stats = _soft_stats(out / f"{tag_ep}_components.csv")
        row = {"epoch": ep, "checkpoint": str(ep_dir), **stats, "run_scan": summary}
        row["deploy_ready"] = bool(
            LIT_LO - 0.5 <= stats["soft_median"] <= LIT_HI + 0.5
            and stats["soft_deepest"] >= DEEPEST_FLOOR
            and stats["mean_curve_min"] >= MEAN_CURVE_FLOOR
        )
        rows.append(row)
        print(
            f"[ep {ep}] soft_median={stats['soft_median']:.2f} "
            f"deepest={stats['soft_deepest']:.2f} "
            f"mean_curve={stats['mean_curve_min']:.2f} "
            f"ready={row['deploy_ready']}",
            flush=True,
        )

    # Also score newest portable params JSON if present.
    portables = sorted(ckpt_root.glob(f"params_{tag}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if portables:
        ck = portables[0]
        tag_ep = "distill_portable_best"
        summary = run_scan(
            checkpoint=ck,
            sidecar=sidecar,
            data=data,
            out_dir=out,
            tag=tag_ep,
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
        stats = _soft_stats(out / f"{tag_ep}_components.csv")
        rows.append(
            {
                "epoch": "portable_best",
                "checkpoint": str(ck),
                **stats,
                "run_scan": summary,
                "deploy_ready": bool(
                    LIT_LO - 0.5 <= stats["soft_median"] <= LIT_HI + 0.5
                    and stats["soft_deepest"] >= DEEPEST_FLOOR
                    and stats["mean_curve_min"] >= MEAN_CURVE_FLOOR
                ),
            }
        )

    if not rows:
        raise SystemExit("no epochs evaluated")

    ranked = sorted(rows, key=_score)
    best = ranked[0]
    report = {
        "run_dir": str(run_dir),
        "sidecar": str(sidecar),
        "literature_kcal": [LIT_LO, LIT_HI],
        "gates": {
            "deepest_floor": DEEPEST_FLOOR,
            "mean_curve_floor": MEAN_CURVE_FLOOR,
            "lit_target": LIT_TARGET,
        },
        "epochs": rows,
        "best": {
            "epoch": best["epoch"],
            "checkpoint": best["checkpoint"],
            "soft_median": best["soft_median"],
            "soft_deepest": best["soft_deepest"],
            "mean_curve_min": best["mean_curve_min"],
            "deploy_ready": best["deploy_ready"],
        },
        "verdict": (
            "deploy_ready" if best["deploy_ready"] else "not_deploy_ready — keep epoch222 + soft handoff"
        ),
    }
    (out / "epoch_sweep.json").write_text(json.dumps(report, indent=2) + "\n")
    # slim table
    slim = [
        {
            k: r[k]
            for k in (
                "epoch",
                "soft_median",
                "soft_mean",
                "soft_deepest",
                "soft_p10",
                "mean_curve_min",
                "deploy_ready",
                "checkpoint",
            )
        }
        for r in rows
    ]
    pd.DataFrame(slim).to_csv(out / "epoch_sweep.csv", index=False)
    print(json.dumps(report["best"], indent=2))
    print("verdict:", report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
