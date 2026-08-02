#!/usr/bin/env python3
"""Plot NPT instantaneous pressure parts from a jaxmd-unified trajectory.npz.

Expects keys written by ``mmml.md.drivers.jaxmd.JaxmdDriver``:
``pressures_bar``, optionally ``pressures_kin_bar`` / ``pressures_vir_bar``,
``target_pressure_bar``, ``volumes_A3``.

Example::

    uv run python scripts/plot_npt_pressure.py \\
      --traj artifacts/npt_argon_water/runs/ar1_90k_mmonly_unit/trajectory.npz \\
      --out docs/images/npt_argon_water/ar1_90k_mmonly_pressure.png \\
      --title "AR1 90 K MM-only unit scales"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmml.utils.plotting.styles import apply_plot_style, comparison_colors, legend_outside


def _times_ps(z, n: int) -> np.ndarray:
    for key in ("times_ps", "time_ps"):
        if key in z.files:
            t = np.asarray(z[key], dtype=float).reshape(-1)
            if t.size == n:
                return t
    # Fallback: uniform index (unknown dt)
    return np.arange(n, dtype=float)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traj", type=Path, required=True, help="trajectory.npz")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="")
    p.add_argument("--target-atm", type=float, default=None, help="optional target overlay in atm")
    p.add_argument("--dt-fs", type=float, default=None, help="if times missing, build t from dt")
    p.add_argument("--record-every", type=int, default=100)
    args = p.parse_args()

    apply_plot_style("icml")
    colors = comparison_colors("icml", n=4)
    z = np.load(args.traj, allow_pickle=True)
    if "pressures_bar" not in z.files:
        raise SystemExit(f"{args.traj} has no pressures_bar; need jaxmd-unified NPT")

    p_tot = np.asarray(z["pressures_bar"], dtype=float)
    n = len(p_tot)
    if "times_ps" in z.files or "time_ps" in z.files:
        t = _times_ps(z, n)
    elif args.dt_fs is not None:
        t = np.arange(n) * (args.record_every * args.dt_fs * 1e-3)
    else:
        t = np.arange(n, dtype=float)

    p_kin = np.asarray(z["pressures_kin_bar"], dtype=float) if "pressures_kin_bar" in z.files else None
    p_vir = np.asarray(z["pressures_vir_bar"], dtype=float) if "pressures_vir_bar" in z.files else None
    target = float(z["target_pressure_bar"]) if "target_pressure_bar" in z.files else None
    if args.target_atm is not None:
        target = float(args.target_atm) * 1.01325  # atm → bar

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.2]})
    ax = axes[0]
    ax.plot(t, p_tot, color=colors[0], lw=1.4, label=r"$P$")
    if p_kin is not None:
        ax.plot(t, p_kin, color=colors[1], lw=1.1, alpha=0.9, label=r"$P_\mathrm{kin}$")
    if p_vir is not None:
        ax.plot(t, p_vir, color=colors[2], lw=1.1, alpha=0.9, label=r"$P_\mathrm{vir}$")
    if target is not None:
        ax.axhline(target, color="0.25", ls="--", lw=1.0, label=rf"$P_\mathrm{{target}}={target:.3g}$ bar")
    ax.set_ylabel("pressure (bar)")
    if args.title:
        ax.set_title(args.title, fontsize=10)
    legend_outside(ax)

    axv = axes[1]
    if "volumes_A3" in z.files:
        V = np.asarray(z["volumes_A3"], dtype=float)
        L = np.cbrt(np.maximum(V, 0.0))
        axv.plot(t, L, color=colors[3], lw=1.3, label=r"$L=V^{1/3}$")
        axv.set_ylabel(r"box side (Å)")
        legend_outside(axv)
    else:
        axv.text(0.5, 0.5, "no volumes_A3", ha="center", va="center", transform=axv.transAxes)
    axv.set_xlabel("time (ps)")

    # Summary strip
    finite = np.isfinite(p_tot)
    summary = (
        f"n={int(finite.sum())}  "
        f"P_mean={float(np.nanmean(p_tot)):.3g} bar  "
        f"P_std={float(np.nanstd(p_tot)):.3g} bar"
    )
    if target is not None and finite.any():
        summary += f"  ⟨P⟩−P_t={float(np.nanmean(p_tot) - target):+.3g} bar"
    fig.text(0.02, 0.01, summary, fontsize=8, color="0.25")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
