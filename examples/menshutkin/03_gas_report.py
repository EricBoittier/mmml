#!/usr/bin/env python3
"""Turn a Menshutkin umbrella run into a PMF profile, diagnostics, and figures.

Reports the quantities a reader of Turan, Brickel & Meuwly (J. Phys. Chem. B
126, 1951 (2022)) will look for -- barrier height, transition-state position,
reaction free energy -- plus the sampling diagnostics that decide whether those
numbers mean anything: neighbouring-window histogram overlap, and whether the
fragments stayed associated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EV_TO_KCAL = 23.060547830619027
# NH3 + MeCl gas-phase barrier, Turan et al. Table 1 (MS-ARMD / MP2).
TURAN_GAS_BARRIER_KCAL = 35.8


def _load(run_dir: Path) -> tuple[dict, dict]:
    summary_path = run_dir / "umbrella_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"no umbrella_summary.json in {run_dir}")
    summary = json.loads(summary_path.read_text())
    mbar = summary.get("mbar")
    if not mbar:
        raise SystemExit(
            f"{summary_path} has no 'mbar' block -- run `mmml umbrella-mbar "
            f"--run-dir {run_dir}` first"
        )
    if "error" in mbar:
        raise SystemExit(f"MBAR did not complete: {mbar['error']}")
    return summary, mbar


def _cv_trajectories(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cv_traj, positions, Z)`` from the snapshot NPZ."""
    snap = np.load(run_dir / "umbrella_snapshots.npz", allow_pickle=True)
    return (
        np.asarray(snap["cv_traj"], dtype=np.float64),
        np.asarray(snap["positions"], dtype=np.float64),
        np.asarray(snap["Z"], dtype=np.int32),
    )


def overlap_fraction(a: np.ndarray, b: np.ndarray, bins: np.ndarray) -> float:
    """Histogram overlap coefficient of two window CV samples, in [0, 1].

    Below ~0.03 the windows barely share configurations and MBAR is
    extrapolating between them rather than reweighting.
    """
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    width = np.diff(bins)
    return float(np.sum(np.minimum(ha, hb) * width))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-barrier",
        type=float,
        default=TURAN_GAS_BARRIER_KCAL,
        help="Literature barrier to compare against (kcal/mol)",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    summary, mbar = _load(run_dir)

    xi = np.asarray(mbar["xi0"], dtype=np.float64)
    pmf = np.asarray(mbar["pmf_rel_kcal_mol"], dtype=np.float64)
    d_pmf = np.asarray(mbar["d_pmf_rel_kcal_mol"], dtype=np.float64)
    cv_traj, positions, z = _cv_trajectories(run_dir)
    cv = cv_traj[..., 0]  # (K, n_frames)

    # Reference the profile to the reactant basin (most negative xi), which is
    # what a barrier is measured from; MBAR references to the global minimum.
    pmf = pmf - pmf[0]

    ts = int(np.argmax(pmf))
    barrier = float(pmf[ts])
    barrier_err = float(d_pmf[ts])
    xi_ts = float(xi[ts])
    # Product basin = lowest point beyond the barrier.
    if ts < len(pmf) - 1:
        prod = ts + 1 + int(np.argmin(pmf[ts + 1 :]))
        dg_rxn = float(pmf[prod])
        xi_prod = float(xi[prod])
    else:
        prod, dg_rxn, xi_prod = -1, float("nan"), float("nan")

    print(f"run: {run_dir}")
    print(f"CV:  {summary.get('cv_label', ['?'])[0]}")
    print(
        f"windows={len(xi)}  frames/window={cv.shape[1]}  "
        f"T={mbar['temperature_K']:.0f} K"
    )
    print()
    print("  xi0     PMF(kcal/mol)    +/-     <xi>     sd(xi)   r(C-Cl)+r(C-N)")
    r_sum = np.linalg.norm(positions[:, :, 0] - positions[:, :, 2], axis=-1) + (
        np.linalg.norm(positions[:, :, 2] - positions[:, :, 1], axis=-1)
    )
    for w in range(len(xi)):
        print(
            f"  {xi[w]:+.2f}   {pmf[w]:10.3f}   {d_pmf[w]:6.3f}   "
            f"{cv[w].mean():+.3f}   {cv[w].std():.3f}    {r_sum[w].mean():.2f}"
        )

    print()
    print(f"barrier          {barrier:7.2f} +/- {barrier_err:.2f} kcal/mol at xi = {xi_ts:+.2f} A")
    if prod >= 0:
        print(f"reaction free E  {dg_rxn:7.2f} kcal/mol at xi = {xi_prod:+.2f} A")
    print(
        f"reference        {args.reference_barrier:7.2f} kcal/mol "
        f"(Turan et al. 2022, gas phase)   deviation "
        f"{barrier - args.reference_barrier:+.2f}"
    )

    # --- diagnostics --------------------------------------------------------
    print()
    print("diagnostics")
    lo, hi = float(cv.min()), float(cv.max())
    bins = np.linspace(lo, hi, 200)
    overlaps = np.array(
        [overlap_fraction(cv[w], cv[w + 1], bins) for w in range(len(xi) - 1)]
    )
    worst = int(np.argmin(overlaps))
    print(
        f"  neighbour histogram overlap: min={overlaps.min():.3f} "
        f"(windows {worst}-{worst + 1}, xi {xi[worst]:+.2f}/{xi[worst + 1]:+.2f})  "
        f"median={np.median(overlaps):.3f}"
    )
    if overlaps.min() < 0.03:
        print("    WARNING: poor overlap -- add windows or soften k there")

    drift = np.abs(cv.mean(axis=1) - xi)
    print(
        f"  |<xi> - xi0|: max={drift.max():.3f} A at window {int(np.argmax(drift))}"
    )
    if drift.max() > 0.15:
        print("    WARNING: windows are pulled off-center; the bias is fighting the PES")

    print(
        f"  r(C-Cl)+r(C-N): min={r_sum.mean(axis=1).min():.2f} "
        f"max={r_sum.mean(axis=1).max():.2f} A"
    )
    if r_sum.mean(axis=1).max() > 8.0:
        print("    WARNING: a window looks dissociated, not reacting")

    n_eff = np.asarray(mbar["N_k_effective"], dtype=float)
    print(f"  MBAR effective samples: min={n_eff.min():.0f} median={np.median(n_eff):.0f}")
    if n_eff.min() < 20:
        print("    WARNING: too few uncorrelated samples in at least one window")

    if summary.get("rex_acceptance") is not None:
        print(f"  replica exchange acceptance: {summary['rex_acceptance']:.3f}")

    profile = {
        "xi_A": xi.tolist(),
        "pmf_kcal_mol": pmf.tolist(),
        "d_pmf_kcal_mol": d_pmf.tolist(),
        "barrier_kcal_mol": barrier,
        "barrier_err_kcal_mol": barrier_err,
        "xi_ts_A": xi_ts,
        "dg_reaction_kcal_mol": dg_rxn,
        "xi_product_A": xi_prod,
        "reference_barrier_kcal_mol": args.reference_barrier,
        "min_neighbour_overlap": float(overlaps.min()),
        "max_center_drift_A": float(drift.max()),
        "cv_mean": cv.mean(axis=1).tolist(),
        "cv_std": cv.std(axis=1).tolist(),
    }
    profile_path = run_dir / "pmf_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    print(f"\nWrote {profile_path}")

    if args.no_plot:
        return 0

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figures")
        return 0

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), height_ratios=[2, 1], sharex=True)

    ax = axes[0]
    ax.errorbar(xi, pmf, yerr=d_pmf, marker="o", ms=4, lw=1.5, capsize=2, color="#1f4e79")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.axvline(xi_ts, color="#c00", ls="--", lw=1, label=f"TS  $\\xi$ = {xi_ts:+.2f} Å")
    ax.axhline(
        args.reference_barrier,
        color="#888",
        ls=":",
        lw=1.2,
        label=f"Turan 2022  {args.reference_barrier:.1f} kcal/mol",
    )
    ax.annotate(
        f"$\\Delta G^{{\\ddagger}}$ = {barrier:.1f} ± {barrier_err:.1f} kcal/mol",
        xy=(xi_ts, barrier),
        xytext=(0.05, 0.85),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "0.4"},
    )
    ax.set_ylabel("PMF (kcal/mol)")
    ax.set_title("NH$_3$ + CH$_3$Cl, gas phase — PhysNet/JAX-MD umbrella + MBAR")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    for w in range(len(xi)):
        ax.hist(cv[w], bins=60, histtype="step", lw=0.8, density=True)
    ax.set_xlabel(r"$\xi = r(\mathrm{C-Cl}) - r(\mathrm{C-N})$  (Å)")
    ax.set_ylabel("window density")
    ax.set_title(
        f"window overlap (min {overlaps.min():.2f}, median {np.median(overlaps):.2f})",
        fontsize=9,
    )

    fig.tight_layout()
    fig_path = run_dir / "pmf_gas.png"
    fig.savefig(fig_path, dpi=200)
    print(f"Wrote {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
