#!/usr/bin/env python3
"""Ablate DCM–DCM overbinding: ES-off (lever 1) + earlier MM handoff (lever 2).

Reuses the epoch222 hybrid checkpoint. Writes CSVs + comparison plots + summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

EV_TO_KCAL = 23.0605


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


def _load_model(checkpoint: Path, pad: int, *, es_off: bool):
    from mmml.cli.misc.physnet_evaluate import _load_physnet_checkpoint
    from mmml.utils.model_checkpoint import build_physnet_from_config
    from mmml.models.physnetjax.physnetjax.models.model import PhysNet as StandardEF
    import json as _json

    path, params, model = _load_physnet_checkpoint(checkpoint, pad, use_ema=True)
    if not es_off:
        return path, params, model

    # Rebuild identical architecture with electrostatics disabled (same weights).
    loaded = _json.loads(Path(path).read_text())
    config = dict(loaded["config"])
    config["include_electrostatics"] = False
    model2 = build_physnet_from_config(config, model_cls=StandardEF, max_padded_atoms=pad)
    model2.max_padded_atoms = pad
    if "zbl" in config:
        model2.zbl = bool(config["zbl"])
    return path, params, model2


def run_scan(
    *,
    checkpoint: Path,
    sidecar: Path,
    data: Path,
    out_dir: Path,
    tag: str,
    es_off: bool,
    mm_switch_on: float,
    ml_switch_width: float,
    mm_switch_width: float,
    n_directions: int,
    n_orientations: int,
    n_r: int,
    r_min: float,
    r_max: float,
    batch_size: int,
) -> dict:
    import jax
    import jax.numpy as jnp
    from mmml.models.hybrid_energy import HYBRID_MM_BATCH_KEYS, hybrid_forward
    from mmml.models.physnetjax.physnetjax.data.batches import prepare_batches_jit

    out_dir.mkdir(parents=True, exist_ok=True)
    n_mono = 5
    pad = 2 * n_mono

    raw = dict(np.load(data, allow_pickle=True))
    Z1 = np.asarray(raw["Z"][0])[:n_mono]
    R1 = np.asarray(raw["R"][0])[:n_mono]
    R1 = R1 - R1.mean(axis=0)
    t1 = np.asarray(raw["cgenff_type_idx"][0])[:n_mono]
    q1 = np.asarray(raw["cgenff_charge"][0])[:n_mono]

    side = json.loads(sidecar.read_text())
    sig_scale = jnp.asarray(side["mm_lj_sigma_scale"], dtype=jnp.float32)
    eps_scale = jnp.asarray(side["mm_lj_epsilon_scale"], dtype=jnp.float32)
    master_sig = jnp.asarray(raw["cgenff_master_sigmas"])
    master_eps = jnp.asarray(raw["cgenff_master_epsilons"])

    dirs = fibonacci_sphere(n_directions)
    quats = super_fibonacci(n_orientations)
    rs = np.linspace(r_min, r_max, n_r)
    n_rays = len(dirs) * len(quats)
    n_tot = n_rays * len(rs)
    print(
        f"[{tag}] es_off={es_off} on={mm_switch_on} ml_w={ml_switch_width} "
        f"mm_w={mm_switch_width}: {n_rays} rays × {len(rs)} r = {n_tot}",
        flush=True,
    )

    R_all = np.zeros((n_tot, pad, 3), dtype=np.float64)
    Z_all = np.zeros((n_tot, pad), dtype=np.int32)
    T_all = np.full((n_tot, pad), -1, dtype=np.int32)
    Q_all = np.zeros((n_tot, pad), dtype=np.float64)
    M_all = np.full((n_tot, pad), -1, dtype=np.int32)
    ray_of = np.zeros(n_tot, dtype=np.int32)
    ir_of = np.zeros(n_tot, dtype=np.int32)

    n = 0
    for di, dvec in enumerate(dirs):
        for qi, q in enumerate(quats):
            Rb0 = R1 @ quat_to_matrix(q).T
            for ri, r in enumerate(rs):
                R_all[n, :n_mono] = R1 - 0.5 * r * dvec
                R_all[n, n_mono:pad] = Rb0 + 0.5 * r * dvec
                Z_all[n, :n_mono] = Z1
                Z_all[n, n_mono:pad] = Z1
                T_all[n, :n_mono] = t1
                T_all[n, n_mono:pad] = t1
                Q_all[n, :n_mono] = q1
                Q_all[n, n_mono:pad] = q1
                M_all[n, :n_mono] = 0
                M_all[n, n_mono:pad] = 1
                ray_of[n] = di * len(quats) + qi
                ir_of[n] = ri
                n += 1

    n_pad = -(-n_tot // batch_size) * batch_size
    if n_pad > n_tot:
        extra = n_pad - n_tot
        R_all = np.concatenate([R_all, np.repeat(R_all[:1], extra, 0)])
        Z_all = np.concatenate([Z_all, np.repeat(Z_all[:1], extra, 0)])
        T_all = np.concatenate([T_all, np.repeat(T_all[:1], extra, 0)])
        Q_all = np.concatenate([Q_all, np.repeat(Q_all[:1], extra, 0)])
        M_all = np.concatenate([M_all, np.repeat(M_all[:1], extra, 0)])

    _, params, model = _load_model(checkpoint, pad, es_off=es_off)

    d = {
        "R": jnp.asarray(R_all),
        "Z": jnp.asarray(Z_all),
        "F": jnp.zeros_like(jnp.asarray(R_all)),
        "E": jnp.zeros((n_pad, 1)),
        "N": jnp.full((n_pad,), pad),
        "D": jnp.zeros((n_pad, 3)),
        "cgenff_type_idx": jnp.asarray(T_all),
        "cgenff_charge": jnp.asarray(Q_all),
        "mol_id": jnp.asarray(M_all),
        "id": jnp.arange(n_pad),
    }
    keys = [
        "R",
        "Z",
        "F",
        "E",
        "N",
        "D",
        "dst_idx",
        "src_idx",
        "batch_segments",
        "id",
    ] + list(HYBRID_MM_BATCH_KEYS)
    batches = prepare_batches_jit(
        jax.random.PRNGKey(0),
        d,
        batch_size,
        num_atoms=pad,
        data_keys=keys,
        include_id=True,
    )

    fwd = jax.jit(
        lambda b: hybrid_forward(
            model.apply,
            params,
            b,
            batch_size,
            master_sig,
            master_eps,
            mm_switch_on=mm_switch_on,
            mm_switch_width=mm_switch_width,
            ml_switch_width=ml_switch_width,
            learn_mm_lj_scales=True,
            mm_lj_sigma_scale=sig_scale,
            mm_lj_epsilon_scale=eps_scale,
            lr_solver="mic",
            include_lj=True,
        )
    )

    E = np.full(n_pad, np.nan)
    E_MM = np.full(n_pad, np.nan)
    S = np.full(n_pad, np.nan)
    for bi, b in enumerate(batches):
        out = fwd(b)
        e = np.asarray(out["energy"]).reshape(batch_size)
        emm = np.asarray(out["e_mm"]).reshape(batch_size)
        s = np.asarray(out["ml_scale"]).reshape(batch_size)
        ids = np.asarray(b["id"])
        E[ids] = e
        E_MM[ids] = emm
        S[ids] = s
        if bi % 20 == 0:
            print(f"  [{tag}] batch {bi}/{len(batches)}", flush=True)

    from mmml.analysis.dimer_scans import (
        DEFAULT_ORIENT_MIN_CONTACT_A,
        intermolecular_min_distance,
    )

    min_contact = DEFAULT_ORIENT_MIN_CONTACT_A
    rows = []
    well_e = []
    soft_e = []
    for ray in range(n_rays):
        sel = np.where(ray_of == ray)[0]
        order = np.argsort(ir_of[sel])
        sel = sel[order]
        e = E[sel]
        emm = E_MM[sel]
        s = S[sel]
        if np.isnan(e).any():
            continue
        e_int = (e - e[-1]) * EV_TO_KCAL
        emm_int = (emm - emm[-1]) * EV_TO_KCAL
        eml_int = e_int - emm_int
        di = int(ray // len(quats))
        qi = int(ray % len(quats))
        dvec = dirs[di]
        quat = quats[qi]
        Rb0 = R1 @ quat_to_matrix(quat).T
        dmins = []
        for r in rs:
            Ra = R1 - 0.5 * r * dvec
            Rb = Rb0 + 0.5 * r * dvec
            dmins.append(intermolecular_min_distance(Ra, Rb))
        dmins = np.asarray(dmins)
        contact_ok = dmins >= min_contact
        if contact_ok.any():
            imin = int(np.flatnonzero(contact_ok)[np.argmin(e_int[contact_ok])])
            well_e.append(float(e_int[imin]))
            soft_ok = contact_ok & (rs >= 3.4)
            if soft_ok.any():
                soft_e.append(float(e_int[soft_ok].min()))
        for ri, r in enumerate(rs):
            rows.append(
                dict(
                    tag=tag,
                    ray=ray,
                    r_A=float(r),
                    E_int_kcal=float(e_int[ri]),
                    E_MM_kcal=float(emm_int[ri]),
                    E_ML_kcal=float(eml_int[ri]),
                    ml_scale=float(s[ri]),
                    dmin_A=float(dmins[ri]),
                    contact_ok=bool(contact_ok[ri]),
                )
            )

    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{tag}_components.csv"
    df.to_csv(csv_path, index=False)

    ok = df[df["contact_ok"]]
    g = ok.groupby("r_A")[["E_int_kcal", "E_MM_kcal", "E_ML_kcal", "ml_scale"]].mean()
    r_mean_well = float(g["E_int_kcal"].idxmin()) if len(g) else float("nan")
    mean_well = float(g["E_int_kcal"].min()) if len(g) else float("nan")
    summary = {
        "tag": tag,
        "es_off": es_off,
        "mm_switch_on": mm_switch_on,
        "ml_switch_width": ml_switch_width,
        "mm_switch_width": mm_switch_width,
        "min_contact_A": min_contact,
        "n_rays": n_rays,
        "n_r": int(n_r),
        "csv": str(csv_path),
        "mean_of_ray_minima_kcal": float(np.mean(well_e)) if well_e else None,
        "mean_soft_well_kcal": float(np.mean(soft_e)) if soft_e else None,
        "median_soft_well_kcal": float(np.median(soft_e)) if soft_e else None,
        "deepest_soft_kcal": float(np.min(soft_e)) if soft_e else None,
        "mean_curve_min_kcal": mean_well,
        "r_at_mean_curve_min_A": r_mean_well,
        "ml_full_below_A": mm_switch_on - ml_switch_width,
        "frac_points_contact_ok": float(df["contact_ok"].mean()) if len(df) else 0.0,
    }
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{tag}] mean_curve_min={mean_well:.2f} @ {r_mean_well:.2f} Å; "
          f"mean_soft={summary['mean_soft_well_kcal']:.2f}; "
          f"mean_ray_min={summary['mean_of_ray_minima_kcal']:.2f}", flush=True)
    return summary


def plot_compare(summaries: list[dict], out_dir: Path, baseline_csv: Path | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    # Left: E_int mean curves
    ax = axes[0]
    def _mean_curve(path: Path) -> pd.Series:
        d = pd.read_csv(path)
        if "contact_ok" in d.columns:
            d = d[d["contact_ok"]]
        elif "dmin_A" in d.columns:
            from mmml.analysis.dimer_scans import DEFAULT_ORIENT_MIN_CONTACT_A

            d = d[d["dmin_A"] >= DEFAULT_ORIENT_MIN_CONTACT_A]
        return d.groupby("r_A")["E_int_kcal"].mean()

    if baseline_csv and baseline_csv.is_file():
        bg = _mean_curve(baseline_csv)
        ax.plot(bg.index, bg.values, "k-", lw=2.0, label="baseline (ES on, on=8)")
    for s in summaries:
        g = _mean_curve(Path(s["csv"]))
        ax.plot(g.index, g.values, lw=1.6, label=s["tag"])
    ax.axhline(0, color="0.5", lw=0.6)
    ax.axhspan(-5, -3, color="0.85", alpha=0.5, label="lit. DCM dimer ~−3…−5")
    ax.set_xlim(2.5, 10)
    ax.set_ylim(-20, 15)
    ax.set_xlabel("COM–COM r (Å)")
    ax.set_ylabel(r"$E_\mathrm{int}$ (kcal/mol)")
    ax.set_title("Mean interaction profile")
    ax.legend(fontsize=7, loc="lower right")

    # Right: zoom + bars
    ax = axes[1]
    tags = []
    vals = []
    if baseline_csv and baseline_csv.is_file():
        bdf = pd.read_csv(baseline_csv)
        soft = bdf[bdf.r_A >= 3.4]
        idx = soft.groupby("ray")["E_int_kcal"].idxmin()
        tags.append("baseline")
        vals.append(float(soft.loc[idx, "E_int_kcal"].mean()))
    for s in summaries:
        tags.append(s["tag"])
        vals.append(s["mean_soft_well_kcal"])
    colors = ["0.3"] + [plt.cm.tab10(i % 10) for i in range(len(summaries))]
    ax.bar(range(len(tags)), vals, color=colors[: len(tags)])
    ax.axhline(-4, color="0.4", ls="--", lw=1, label="lit. ~−4")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean soft well (kcal/mol)")
    ax.set_title("Soft well depth (r ≥ 3.4 Å)")
    ax.legend(fontsize=8)

    fig.savefig(out_dir / "overbind_ablation_compare.png", dpi=160)
    fig.savefig(out_dir / "overbind_ablation_compare.pdf")
    plt.close(fig)

    # Component panel for handoff variants
    handoff = [s for s in summaries if s["tag"].startswith("handoff_")]
    if handoff:
        fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
        for s in handoff:
            df = pd.read_csv(s["csv"])
            g = df.groupby("r_A")[["E_ML_kcal", "E_MM_kcal"]].mean()
            ax.plot(g.index, g["E_ML_kcal"], lw=1.5, label=f"{s['tag']} ML")
            ax.plot(g.index, g["E_MM_kcal"], lw=1.2, ls="--", label=f"{s['tag']} MM")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlim(2.5, 10)
        ax.set_ylim(-40, 20)
        ax.set_xlabel("COM–COM r (Å)")
        ax.set_ylabel("kcal/mol")
        ax.set_title("Earlier handoff: mean ML vs MM")
        ax.legend(fontsize=7, ncol=2)
        fig.savefig(out_dir / "overbind_handoff_components.png", dpi=160)
        plt.close(fig)


def main() -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/lj_scales/ckpts/params_hybrid_mm_fixed_lj_scales_epoch222.json"),
    )
    p.add_argument(
        "--sidecar",
        type=Path,
        default=Path(
            "artifacts/lj_scales/ckpts/"
            "hybrid_mm_fixed_lj_scales-4d68132d-c686-4ded-9887-efc16d5b2638/hybrid_mm.json"
        ),
    )
    p.add_argument("--data", type=Path, default=Path("artifacts/lj_scales/dataset_cgenff.npz"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/lj_scales/dense_dt_campaign/overbind_ablation"),
    )
    p.add_argument("--n-directions", type=int, default=8)
    p.add_argument("--n-orientations", type=int, default=8)
    p.add_argument("--n-r", type=int, default=36)
    p.add_argument("--r-min", type=float, default=2.5)
    p.add_argument("--r-max", type=float, default=12.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("artifacts/lj_scales/dense_dt_campaign/dimer_scans/orient_components.csv"),
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    common = dict(
        checkpoint=args.checkpoint,
        sidecar=args.sidecar,
        data=args.data,
        out_dir=args.out,
        n_directions=args.n_directions,
        n_orientations=args.n_orientations,
        n_r=args.n_r,
        r_min=args.r_min,
        r_max=args.r_max,
        batch_size=args.batch_size,
        mm_switch_width=5.0,
    )

    runs = [
        # Lever 1: ES-off, same handoff as training/MD
        dict(tag="es_off_on8", es_off=True, mm_switch_on=8.0, ml_switch_width=1.5),
        # Lever 2: earlier MM handoff (ES on)
        dict(tag="handoff_on6_w1p5", es_off=False, mm_switch_on=6.0, ml_switch_width=1.5),
        dict(tag="handoff_on5_w1p5", es_off=False, mm_switch_on=5.0, ml_switch_width=1.5),
        dict(tag="handoff_on4p5_w1", es_off=False, mm_switch_on=4.5, ml_switch_width=1.0),
        # Combined: early handoff + ES off
        dict(tag="es_off_handoff_on5", es_off=True, mm_switch_on=5.0, ml_switch_width=1.5),
    ]

    summaries = []
    for run in runs:
        summaries.append(run_scan(**common, **run))

    plot_compare(summaries, args.out, args.baseline_csv if args.baseline_csv.is_file() else None)

    report = {
        "baseline_csv": str(args.baseline_csv) if args.baseline_csv.is_file() else None,
        "literature_dcm_dimer_kcal": [-5.0, -3.0],
        "runs": summaries,
        "verdict": {
            "lever1_es_off": (
                "If es_off soft well ≈ baseline, PhysNet Coulomb is not the driver; "
                "overbinding is neural local ML energy — needs retrain/constraint, not charge off."
            ),
            "lever2_early_handoff": (
                "If earlier mm_switch_on shallows the soft well toward −3…−5, contact pairs "
                "must leave pure-ML sooner (deploy-time lever; retrain with matching switches)."
            ),
        },
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
