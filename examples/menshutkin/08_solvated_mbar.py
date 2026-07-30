#!/usr/bin/env python3
"""MBAR on the solvated umbrella windows, from xi(t) alone.

Every window shares the same unbiased Hamiltonian and differs only in the
harmonic bias, so the reduced potentials enter MBAR only through differences
``u_l - u_k = beta*(W_l - W_k)``, which depend on the collective variable and
nothing else. The common configurational energy is a per-sample constant and
cancels out of MBAR's weights, so there is no need to re-evaluate the full
ML/MM energy of every frame of a 1400-atom system -- xi(t) is sufficient and
exact.

Reports the barrier, the contact-ion-pair and solvent-separated-ion-pair minima,
and the desolvation bump between them, plus the sampling diagnostics that decide
whether any of it is meaningful.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EV_TO_KCAL = 23.060547830619027
K_B_EV = 8.617333262145e-5


def overlap(a, b, bins):
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    return float(np.sum(np.minimum(ha, hb) * np.diff(bins)))


def find_features(xi, pmf):
    """Locate barrier, CIP minimum, desolvation bump and SSIP minimum."""
    out = {}
    ts = int(np.argmax(pmf[xi < 1.0])) if np.any(xi < 1.0) else 0
    out["barrier"] = (float(xi[ts]), float(pmf[ts]))

    # Contact ion pair: lowest point just past the barrier.
    post = np.where(xi > xi[ts])[0]
    if post.size:
        cip = post[int(np.argmin(pmf[post]))]
        out["cip"] = (float(xi[cip]), float(pmf[cip]))
        # Desolvation bump: highest point after the CIP.
        tail = np.where(xi > xi[cip])[0]
        if tail.size > 2:
            bump = tail[int(np.argmax(pmf[tail]))]
            out["bump"] = (float(xi[bump]), float(pmf[bump]))
            beyond = np.where(xi > xi[bump])[0]
            if beyond.size:
                ssip = beyond[int(np.argmin(pmf[beyond]))]
                out["ssip"] = (float(xi[ssip]), float(pmf[ssip]))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--discard-frac", type=float, default=0.0,
                   help="Extra leading fraction of each window to drop")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    data = json.loads((args.run_dir / "umbrella_windows.json").read_text())
    windows = data["windows"]
    keys = sorted(windows, key=lambda k: int(k))
    xi0 = np.array([windows[k]["xi0"] for k in keys])
    traces = [np.asarray(windows[k]["xi"]) for k in keys]
    if args.discard_frac > 0:
        traces = [t[int(len(t) * args.discard_frac):] for t in traces]
    k_ev = float(data["k_ev_A2"])
    T = float(data["temperature_K"])
    beta = 1.0 / (K_B_EV * T)

    n_below = sum(int(windows[k].get("below_training_floor", 0)) for k in keys)
    if n_below:
        print(f"WARNING: {n_below} frames fell below the training-set energy floor; "
              "those windows sampled off the fitted surface\n")

    try:
        from pymbar import MBAR, timeseries
    except ImportError:
        raise SystemExit("pymbar required: uv sync --extra mbar")

    # Subsample each window to uncorrelated frames.
    sub = []
    g_k = []
    for t in traces:
        if t.size < 4:
            sub.append(t)
            g_k.append(1.0)
            continue
        g = max(1.0, float(timeseries.statistical_inefficiency(t)))
        idx = np.asarray(timeseries.subsample_correlated_data(t, g=g), dtype=int)
        sub.append(t[idx] if idx.size else t[-1:])
        g_k.append(g)

    K = len(sub)
    N_k = np.array([s.size for s in sub], dtype=np.int64)
    N_max = int(N_k.max())
    u_kln = np.zeros((K, K, N_max))
    for k, s in enumerate(sub):
        for l in range(K):
            # Only the bias differs between states; the shared configurational
            # energy is a per-sample constant and drops out of MBAR.
            u_kln[k, l, : s.size] = beta * 0.5 * k_ev * (s - xi0[l]) ** 2

    mbar = MBAR(u_kln, N_k)
    fe = mbar.compute_free_energy_differences(compute_uncertainty=True)
    f = np.asarray(fe["Delta_f"])[0]
    df = np.asarray(fe["dDelta_f"])[0]
    pmf = (f - f[0]) * K_B_EV * T * EV_TO_KCAL
    dpmf = df * K_B_EV * T * EV_TO_KCAL

    print(f"solvent   {data['solvent']}   {data['n_atoms']} atoms   T = {T:.0f} K")
    print(f"windows   {K}   k = {k_ev} eV/A^2 ({k_ev * EV_TO_KCAL:.0f} kcal/mol/A^2)")
    print()
    print("   xi0     PMF(kcal/mol)    +/-      <xi>    sd     N_eff")
    for i, k in enumerate(keys):
        print(f"  {xi0[i]:+5.2f}   {pmf[i]:10.3f}   {dpmf[i]:6.3f}   "
              f"{sub[i].mean():+6.3f}  {sub[i].std():.3f}  {N_k[i]:5d}")

    feats = find_features(xi0, pmf)
    print()
    for name, label in [("barrier", "SN2 barrier"), ("cip", "contact ion pair"),
                        ("bump", "desolvation bump"), ("ssip", "solvent-separated")]:
        if name in feats:
            x, y = feats[name]
            print(f"  {label:22s} xi = {x:+5.2f} A   {y:8.2f} kcal/mol")
    if "cip" in feats and "bump" in feats:
        print(f"  {'CIP -> bump height':22s} "
              f"{feats['bump'][1] - feats['cip'][1]:8.2f} kcal/mol")

    lo, hi = min(t.min() for t in traces), max(t.max() for t in traces)
    bins = np.linspace(lo, hi, 400)
    ov = np.array([overlap(traces[i], traces[i + 1], bins) for i in range(K - 1)])
    print()
    print(f"  neighbour overlap  min={ov.min():.3f} (windows "
          f"{int(np.argmin(ov))}-{int(np.argmin(ov)) + 1})  median={np.median(ov):.3f}")
    if ov.min() < 0.03:
        print("    WARNING: poor overlap; add windows there")
    print(f"  N_eff              min={N_k.min()}  median={int(np.median(N_k))}")
    if N_k.min() < 20:
        print("    WARNING: too few uncorrelated samples for trustworthy errors")

    result = {
        "solvent": data["solvent"],
        "xi_A": xi0.tolist(),
        "pmf_kcal_mol": pmf.tolist(),
        "d_pmf_kcal_mol": dpmf.tolist(),
        "features": feats,
        "min_overlap": float(ov.min()),
        "N_eff": N_k.tolist(),
        "g_k": g_k,
    }
    (args.run_dir / "pmf_solvated.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nWrote {args.run_dir / 'pmf_solvated.json'}")

    if args.no_plot:
        return 0
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(xi0, pmf, yerr=dpmf, marker="o", ms=3.5, lw=1.5, capsize=2,
                color="#1f4e79")
    ax.axhline(0, color="0.8", lw=0.8)
    for name, colour, label in [("barrier", "#c00", "TS"), ("cip", "#0a0", "CIP"),
                                ("bump", "#e80", "bump"), ("ssip", "#08c", "SSIP")]:
        if name in feats:
            x, y = feats[name]
            ax.plot([x], [y], "o", ms=8, mfc="none", mec=colour)
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 6),
                        color=colour, fontsize=9)
    ax.set_xlabel(r"$\xi = r(\mathrm{C-Cl}) - r(\mathrm{C-N})$  (Å)")
    ax.set_ylabel("PMF (kcal/mol)")
    ax.set_title(f"NH$_3$ + CH$_3$Cl in {data['solvent']} — ML/MM umbrella + MBAR")
    fig.tight_layout()
    fig.savefig(args.run_dir / "pmf_solvated.png", dpi=200)
    print(f"Wrote {args.run_dir / 'pmf_solvated.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
