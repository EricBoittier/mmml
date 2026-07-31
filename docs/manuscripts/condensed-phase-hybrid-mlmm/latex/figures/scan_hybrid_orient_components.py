#!/usr/bin/env python3
"""Multi-orientation hybrid dimer scans with energy components and mean(|F|).

Reads a portable PhysNet JSON + hybrid_mm.json LJ-scale sidecar and an NPZ
with CGenFF arrays (DCM dimers for the Menshutkin/LJ-scale training set).
Writes CSV + manuscript PNG/PDF panels.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def main() -> int:
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default="/mmhome/boittier/home/mmml/artifacts/nh3_ch3cl/ckpts/"
        "params_hybrid_mm_fixed_lj_scales_2026-07-31_13-39-37.json",
    )
    p.add_argument(
        "--sidecar",
        default="/mmhome/boittier/home/mmml/artifacts/nh3_ch3cl/ckpts/"
        "hybrid_mm_fixed_lj_scales-f7be8ce9-6b0c-4eae-bcc1-c50def501d13/hybrid_mm.json",
    )
    p.add_argument(
        "--data",
        default="/mmhome/boittier/home/mmml/artifacts/nh3_ch3cl/dataset_cgenff.npz",
    )
    p.add_argument("--n-mono", type=int, default=5, help="atoms per monomer")
    p.add_argument("--n-directions", type=int, default=6)
    p.add_argument("--n-orientations", type=int, default=8)
    p.add_argument("--r-min", type=float, default=2.8)
    p.add_argument("--r-max", type=float, default=12.0)
    p.add_argument("--n-r", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--mm-switch-on", type=float, default=8.0)
    p.add_argument("--ml-switch-width", type=float, default=1.5)
    p.add_argument("--mm-switch-width", type=float, default=5.0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "hybrid_orient_scan",
    )
    args = p.parse_args()

    import jax
    import jax.numpy as jnp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mmml.cli.misc.physnet_evaluate import _load_physnet_checkpoint
    from mmml.models.hybrid_energy import HYBRID_MM_BATCH_KEYS, hybrid_forward
    from mmml.models.physnetjax.physnetjax.data.batches import prepare_batches_jit
    from mmml.models.short_range_wall import inter_monomer_wall_energy

    raw = dict(np.load(args.data, allow_pickle=True))
    n_mono = int(args.n_mono)
    pad = 2 * n_mono
    Z1 = np.asarray(raw["Z"][0])[:n_mono]
    R1 = np.asarray(raw["R"][0])[:n_mono]
    R1 = R1 - R1.mean(axis=0)
    t1 = np.asarray(raw["cgenff_type_idx"][0])[:n_mono]
    q1 = np.asarray(raw["cgenff_charge"][0])[:n_mono]

    side = json.loads(Path(args.sidecar).read_text())
    sig_scale = jnp.asarray(side["mm_lj_sigma_scale"], dtype=jnp.float32)
    eps_scale = jnp.asarray(side["mm_lj_epsilon_scale"], dtype=jnp.float32)
    master_sig = jnp.asarray(raw["cgenff_master_sigmas"])
    master_eps = jnp.asarray(raw["cgenff_master_epsilons"])

    dirs = fibonacci_sphere(args.n_directions)
    quats = super_fibonacci(args.n_orientations)
    rs = np.linspace(args.r_min, args.r_max, args.n_r)
    n_rays = len(dirs) * len(quats)
    n_tot = n_rays * len(rs)
    print(
        f"DCM-like monomer Z={Z1.tolist()}: {len(dirs)} dirs × {len(quats)} oris "
        f"= {n_rays} rays × {len(rs)} r = {n_tot} evals"
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

    n_pad = -(-n_tot // args.batch_size) * args.batch_size
    if n_pad > n_tot:
        extra = n_pad - n_tot
        R_all = np.concatenate([R_all, np.repeat(R_all[:1], extra, 0)])
        Z_all = np.concatenate([Z_all, np.repeat(Z_all[:1], extra, 0)])
        T_all = np.concatenate([T_all, np.repeat(T_all[:1], extra, 0)])
        Q_all = np.concatenate([Q_all, np.repeat(Q_all[:1], extra, 0)])
        M_all = np.concatenate([M_all, np.repeat(M_all[:1], extra, 0)])

    _, params, model = _load_physnet_checkpoint(
        Path(args.checkpoint), pad, use_ema=True
    )

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
        args.batch_size,
        num_atoms=pad,
        data_keys=keys,
        include_id=True,
    )
    print(f"{len(batches)} batches of {args.batch_size}")

    fwd = jax.jit(
        lambda b: hybrid_forward(
            model.apply,
            params,
            b,
            args.batch_size,
            master_sig,
            master_eps,
            mm_switch_on=args.mm_switch_on,
            mm_switch_width=args.mm_switch_width,
            ml_switch_width=args.ml_switch_width,
            learn_mm_lj_scales=True,
            mm_lj_sigma_scale=sig_scale,
            mm_lj_epsilon_scale=eps_scale,
            lr_solver="mic",
            include_lj=True,
        )
    )
    wall_fn = jax.jit(jax.vmap(inter_monomer_wall_energy))

    E = np.full(n_pad, np.nan)
    E_MM = np.full(n_pad, np.nan)
    S = np.full(n_pad, np.nan)
    W = np.full(n_pad, np.nan)
    F_all = np.full((n_pad, pad, 3), np.nan)

    for bi, b in enumerate(batches):
        out = fwd(b)
        e = np.asarray(out["energy"]).reshape(args.batch_size)
        emm = np.asarray(out["e_mm"]).reshape(args.batch_size)
        s = np.asarray(out["ml_scale"]).reshape(args.batch_size)
        f = np.asarray(out["forces"]).reshape(args.batch_size, pad, 3)
        w = np.asarray(
            wall_fn(b["R"].reshape(args.batch_size, pad, 3), b["mol_id"])
        )
        ids = np.asarray(b["id"])
        E[ids] = e
        E_MM[ids] = emm
        S[ids] = s
        W[ids] = w
        F_all[ids] = f
        if bi % 10 == 0:
            print(f"  batch {bi}/{len(batches)}", flush=True)

    # Per-ray tables (interaction = relative to largest r)
    rows = []
    curves = []  # list of dicts with arrays per ray
    for ray in range(n_rays):
        sel = np.where(ray_of == ray)[0]
        order = np.argsort(ir_of[sel])
        sel = sel[order]
        e = E[sel]
        emm = E_MM[sel]
        s = S[sel]
        w = W[sel]
        F = F_all[sel]
        if np.isnan(e).any() or np.isnan(F).any():
            continue
        e_int = e - e[-1]
        emm_int = emm - emm[-1]
        eml_int = e_int - emm_int
        # Raw |F| keeps intramolecular residuals; use ΔF vs asymptote + COM force.
        dF = F - F[-1]
        dF_norm = np.linalg.norm(dF, axis=-1)
        fmean_d = dF_norm.mean(axis=1) * EV_TO_KCAL
        fmax_d = dF_norm.max(axis=1) * EV_TO_KCAL
        fcom_a = F[:, :n_mono, :].sum(axis=1)
        fcom_b = F[:, n_mono:pad, :].sum(axis=1)
        fcom_d = (
            0.5
            * (
                np.linalg.norm(fcom_a - fcom_a[-1], axis=-1)
                + np.linalg.norm(fcom_b - fcom_b[-1], axis=-1)
            )
            * EV_TO_KCAL
        )
        fnorm = np.linalg.norm(F, axis=-1)
        fmean_raw = fnorm.mean(axis=1)
        fmax_raw = fnorm.max(axis=1)
        di = int(ray // len(quats))
        qi = int(ray % len(quats))
        imin = int(np.argmin(e_int))
        curves.append(
            dict(
                ray=ray,
                direction=di,
                orientation=qi,
                r=rs.copy(),
                e_int_kcal=e_int * EV_TO_KCAL,
                e_mm_kcal=emm_int * EV_TO_KCAL,
                e_ml_kcal=eml_int * EV_TO_KCAL,
                ml_scale=s.copy(),
                wall_eV=w.copy(),
                fmean_d=fmean_d.copy(),
                fmax_d=fmax_d.copy(),
                fcom_d=fcom_d.copy(),
                e_min_kcal=float(e_int[imin] * EV_TO_KCAL),
                r_at_min=float(rs[imin]),
            )
        )
        for ri, r in enumerate(rs):
            rows.append(
                {
                    "ray": ray,
                    "direction": di,
                    "orientation": qi,
                    "r_A": float(r),
                    "E_int_kcal": float(e_int[ri] * EV_TO_KCAL),
                    "E_MM_kcal": float(emm_int[ri] * EV_TO_KCAL),
                    "E_ML_kcal": float(eml_int[ri] * EV_TO_KCAL),
                    "ml_scale": float(s[ri]),
                    "wall_eV": float(w[ri]),
                    "mean_abs_F_eV_A": float(fmean_raw[ri]),
                    "max_abs_F_eV_A": float(fmax_raw[ri]),
                    "mean_abs_dF_kcal_A": float(fmean_d[ri]),
                    "max_abs_dF_kcal_A": float(fmax_d[ri]),
                    "dFcom_kcal_A": float(fcom_d[ri]),
                }
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(__file__).resolve().parent

    # CSV
    import csv

    with (out_dir / "orient_components.csv").open("w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "sidecar": str(args.sidecar),
        "data": str(args.data),
        "n_rays": n_rays,
        "n_r": len(rs),
        "r_min": args.r_min,
        "r_max": args.r_max,
        "mm_switch_on": args.mm_switch_on,
        "ml_switch_width": args.ml_switch_width,
        "mm_switch_width": args.mm_switch_width,
        "deepest_kcal": float(min(c["e_min_kcal"] for c in curves)),
        "mean_well_kcal": float(np.mean([c["e_min_kcal"] for c in curves])),
        "Z_monomer": Z1.tolist(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (fig_dir / "hybrid_orient_scan_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Stack for mean ± std
    Emat = np.stack([c["e_int_kcal"] for c in curves], axis=0)
    EMMmat = np.stack([c["e_mm_kcal"] for c in curves], axis=0)
    EMLmat = np.stack([c["e_ml_kcal"] for c in curves], axis=0)
    Fdmat = np.stack([c["fmean_d"] for c in curves], axis=0)
    Fmaxdmat = np.stack([c["fmax_d"] for c in curves], axis=0)
    Fcommat = np.stack([c["fcom_d"] for c in curves], axis=0)
    Smat = np.stack([c["ml_scale"] for c in curves], axis=0)
    soft_idx = int(np.argmin(np.abs(rs - (args.mm_switch_on - args.ml_switch_width))))
    summary["median_mean_abs_dF_at_handoff"] = float(np.median(Fdmat[:, soft_idx]))
    summary["median_peak_dFcom_soft_kcal_A"] = float(
        np.median(Fcommat[:, rs >= 3.5].max(axis=1))
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (fig_dir / "hybrid_orient_scan_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Soft-region wells (ignore contact spikes when ranking)
    soft_mask = rs >= 3.5
    for c in curves:
        e_soft = c["e_int_kcal"][soft_mask]
        r_soft = c["r"][soft_mask]
        i = int(np.argmin(e_soft))
        c["e_min_soft_kcal"] = float(e_soft[i])
        c["r_at_min_soft"] = float(r_soft[i])
    ranked = sorted(curves, key=lambda c: c["e_min_soft_kcal"])
    showcase = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
    summary["deepest_soft_kcal"] = float(ranked[0]["e_min_soft_kcal"])
    summary["mean_soft_well_kcal"] = float(
        np.mean([c["e_min_soft_kcal"] for c in curves])
    )
    summary["r_at_deepest_soft"] = float(ranked[0]["r_at_min_soft"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (fig_dir / "hybrid_orient_scan_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    def _save(fig, name: str) -> None:
        for dest in [out_dir / name, fig_dir / name]:
            fig.savefig(dest, bbox_inches="tight")
            fig.savefig(dest.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    handoff_lo = args.mm_switch_on - args.ml_switch_width
    handoff_hi = args.mm_switch_on

    # --- Fig 1: all orientations E_int(r), well-focused y-lim ---
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=160)
    for c in curves:
        ax.plot(c["r"], c["e_int_kcal"], color="#1f4e5f", alpha=0.18, lw=0.8)
    mu, sd = Emat.mean(0), Emat.std(0)
    ax.plot(rs, mu, color="#b85c38", lw=2.0, label="mean over orientations")
    ax.fill_between(rs, mu - sd, mu + sd, color="#b85c38", alpha=0.2, label="±1σ")
    ax.axvline(handoff_lo, color="k", ls=":", lw=0.8)
    ax.axvline(handoff_hi, color="k", ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.set_xlabel(r"$r_{\mathrm{COM}}$ / Å")
    ax.set_ylabel(r"$E_{\mathrm{int}}$ / kcal mol$^{-1}$")
    ax.set_title(
        f"Hybrid dimer scans ({n_rays} orientations)\n"
        f"DCM-like · learnable LJ scales"
    )
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(args.r_min, args.r_max)
    ax.set_ylim(-12.0, 40.0)
    _save(fig, "hybrid_orient_Eint.png")

    # --- Fig 1b: percentile envelope (robust to contact outliers) ---
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=160)
    p10, p50, p90 = np.percentile(Emat, [10, 50, 90], axis=0)
    ax.fill_between(rs, p10, p90, color="#1f4e5f", alpha=0.18, label="10–90%")
    ax.plot(rs, p50, color="#1f4e5f", lw=2.0, label="median")
    ax.plot(rs, mu, color="#b85c38", lw=1.6, label="mean")
    ax.axvline(handoff_lo, color="k", ls=":", lw=0.8)
    ax.axvline(handoff_hi, color="k", ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.set_xlabel(r"$r_{\mathrm{COM}}$ / Å")
    ax.set_ylabel(r"$E_{\mathrm{int}}$ / kcal mol$^{-1}$")
    ax.set_title("Orientation envelope (median / mean)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(3.2, args.r_max)
    ax.set_ylim(-10.0, 20.0)
    _save(fig, "hybrid_orient_Eint_zoom.png")

    # --- Fig 2: components for showcase rays ---
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3), dpi=160, sharey=False)
    for ax, c in zip(axes, showcase):
        ax.plot(c["r"], c["e_int_kcal"], color="#1f4e5f", lw=1.8, label=r"$E_{\mathrm{int}}$")
        ax.plot(c["r"], c["e_ml_kcal"], color="#2a9d8f", lw=1.4, label=r"$E_{\mathrm{ML}}$")
        ax.plot(c["r"], c["e_mm_kcal"], color="#b85c38", lw=1.4, label=r"$E_{\mathrm{MM}}$")
        ax2 = ax.twinx()
        ax2.plot(c["r"], c["ml_scale"], color="#c4a35a", ls="--", lw=1.0, alpha=0.9)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel(r"$s_{\mathrm{ML}}$", color="#c4a35a", fontsize=8)
        ax.axvline(handoff_lo, color="k", ls=":", lw=0.7)
        ax.axvline(handoff_hi, color="k", ls="--", lw=0.7)
        ax.axhline(0, color="k", lw=0.4, alpha=0.35)
        ax.set_xlabel(r"$r$ / Å")
        ax.set_xlim(3.0, args.r_max)
        ax.set_ylim(-12.0, 25.0)
        ax.set_title(
            f"ray {c['ray']} (dir {c['direction']}, ori {c['orientation']})\n"
            f"well {c['e_min_soft_kcal']:.2f} kcal/mol @ {c['r_at_min_soft']:.2f} Å",
            fontsize=8,
        )
    axes[0].set_ylabel(r"energy / kcal mol$^{-1}$")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Hybrid energy components along selected orientations", fontsize=10)
    fig.tight_layout()
    _save(fig, "hybrid_orient_components.png")

    # --- Fig 3: mean components across orientations ---
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=160)
    ax.plot(rs, Emat.mean(0), color="#1f4e5f", lw=2.0, label=r"$\langle E_{\mathrm{int}}\rangle$")
    ax.plot(rs, EMLmat.mean(0), color="#2a9d8f", lw=1.6, label=r"$\langle E_{\mathrm{ML}}\rangle$")
    ax.plot(rs, EMMmat.mean(0), color="#b85c38", lw=1.6, label=r"$\langle E_{\mathrm{MM}}\rangle$")
    ax.fill_between(
        rs,
        Emat.mean(0) - Emat.std(0),
        Emat.mean(0) + Emat.std(0),
        color="#1f4e5f",
        alpha=0.15,
    )
    ax2 = ax.twinx()
    ax2.plot(
        rs, Smat.mean(0), color="#c4a35a", ls="--", lw=1.2, label=r"$\langle s_{\mathrm{ML}}\rangle$"
    )
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel(r"$s_{\mathrm{ML}}$", color="#c4a35a")
    ax.axvline(handoff_lo, color="k", ls=":", lw=0.8)
    ax.axvline(handoff_hi, color="k", ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.4, alpha=0.35)
    ax.set_xlabel(r"$r_{\mathrm{COM}}$ / Å")
    ax.set_ylabel(r"energy / kcal mol$^{-1}$")
    ax.set_title("Mean hybrid components over orientations")
    ax.set_xlim(3.2, args.r_max)
    ax.set_ylim(-10.0, 25.0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "hybrid_orient_components_mean.png")

    # --- Fig 4: |F - F_infty| ---
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=160)
    for c in curves:
        ax.plot(c["r"], c["fmean_d"], color="#1f4e5f", alpha=0.15, lw=0.7)
    mu, sd = Fdmat.mean(0), Fdmat.std(0)
    ax.plot(rs, mu, color="#b85c38", lw=2.0, label=r"mean $\langle|F-F_\infty|\rangle$")
    ax.fill_between(rs, mu - sd, mu + sd, color="#b85c38", alpha=0.2)
    ax.axvline(handoff_lo, color="k", ls=":", lw=0.8)
    ax.axvline(handoff_hi, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"$r_{\mathrm{COM}}$ / Å")
    ax.set_ylabel(r"mean $|F-F_\infty|$ / kcal mol$^{-1}$ Å$^{-1}$")
    ax.set_title(r"Force magnitude relative to separated limit")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(args.r_min, args.r_max)
    ax.set_ylim(0.0, 80.0)
    fig.tight_layout()
    _save(fig, "hybrid_orient_meanF.png")

    # --- Fig 4b: ΔF + COM interaction force ---
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=160)
    ax = axes[0]
    p10, p50, p90 = np.percentile(Fdmat, [10, 50, 90], axis=0)
    ax.fill_between(rs, p10, p90, color="#1f4e5f", alpha=0.2, label="10–90%")
    ax.plot(rs, p50, color="#1f4e5f", lw=2.0, label=r"median mean $|F-F_\infty|$")
    ax.plot(
        rs,
        np.median(Fmaxdmat, 0),
        color="#b85c38",
        lw=1.5,
        label=r"median max $|F-F_\infty|$",
    )
    ax.axvline(handoff_lo, color="k", ls=":", lw=0.8)
    ax.axvline(handoff_hi, color="k", ls="--", lw=0.8)
    ax.set_xlim(3.2, args.r_max)
    ax.set_ylim(0.0, 40.0)
    ax.set_xlabel(r"$r_{\mathrm{COM}}$ / Å")
    ax.set_ylabel(r"$|F-F_\infty|$ / kcal mol$^{-1}$ Å$^{-1}$")
    ax.set_title("Atomic forces vs separated asymptote")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    p10, p50, p90 = np.percentile(Fcommat, [10, 50, 90], axis=0)
    for c in curves:
        ax.plot(c["r"], c["fcom_d"], color="#1f4e5f", alpha=0.12, lw=0.6)
    ax.fill_between(rs, p10, p90, color="#1f4e5f", alpha=0.2, label="10–90%")
    ax.plot(rs, p50, color="#b85c38", lw=2.0, label=r"median $|\Delta F_{\mathrm{COM}}|$")
    ax.axvline(handoff_lo, color="k", ls=":", lw=0.8)
    ax.axvline(handoff_hi, color="k", ls="--", lw=0.8)
    ax.set_xlim(3.2, args.r_max)
    ax.set_ylim(0.0, 80.0)
    ax.set_xlabel(r"$r_{\mathrm{COM}}$ / Å")
    ax.set_ylabel(r"$|\Delta F_{\mathrm{COM}}|$ / kcal mol$^{-1}$ Å$^{-1}$")
    ax.set_title("Net monomer COM force (interaction)")
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Force metrics across orientations", fontsize=10)
    fig.tight_layout()
    _save(fig, "hybrid_orient_meanF_zoom.png")

    # copy CSV into figures/
    import shutil

    shutil.copy2(out_dir / "orient_components.csv", fig_dir / "hybrid_orient_components.csv")

    print(json.dumps(summary, indent=2))
    print(f"wrote panels under {out_dir} and {fig_dir}")
    return 0


if __name__ == "__main__":
    # Ensure mmml repo is importable when launched from manuscript tree.
    repo = Path("/mmhome/boittier/home/mmml")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    raise SystemExit(main())
