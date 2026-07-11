#!/usr/bin/env python
"""Plot SMD work profiles and the Hummer-Szabo PMF from cg_jaxmd's smd_jarzynski.npz.

Usage:
    python examples/plot_smd_jarzynski.py [path/to/smd_jarzynski.npz] [-o out.png]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cumulative_jarzynski(work, beta):
    """Running Jarzynski free-energy estimate along lambda: -kT ln<exp(-beta W)>.

    work: (n_pulls, n_blocks) cumulative external work per block. Returns (n_blocks,).
    Uses a per-block min shift for numerical stability.
    """
    wmin = work.min(axis=0, keepdims=True)
    return -(1.0 / beta) * (np.log(np.mean(np.exp(-beta * (work - wmin)), axis=0))) + wmin[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", nargs="?", default="smd_jarzynski.npz",
                    help="Path to smd_jarzynski.npz written by cg_jaxmd.py")
    ap.add_argument("-o", "--out", default=None, help="Output image path (default: <npz stem>.png)")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    d = np.load(npz_path)
    z = d["z_profile"]              # (n_pulls, n_blocks)
    W = d["work_profile"]           # (n_pulls, n_blocks)
    lam = d["lambda_profile"]       # (n_blocks,)
    centers = d["pmf_centers"]      # (bins,)
    pmf = d["pmf_ev"]               # (bins,)
    kB = 8.617333262145e-5          # eV/K
    T = float(d["temperature_k"])
    beta = 1.0 / (kB * T)
    n_pulls = W.shape[0]

    jarz = cumulative_jarzynski(W, beta)   # (n_blocks,)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    # (A) Work vs lambda: individual pulls + mean + Jarzynski estimate.
    axA = axes[0]
    for i in range(n_pulls):
        axA.plot(lam, W[i], color="0.75", lw=0.8, alpha=0.7,
                 label="individual pulls" if i == 0 else None)
    axA.plot(lam, W.mean(axis=0), color="tab:blue", lw=2.0, label=r"$\langle W \rangle$ (mean work)")
    axA.plot(lam, jarz, color="tab:red", lw=2.0,
             label=r"$-k_BT\,\ln\langle e^{-\beta W}\rangle$ (Jarzynski)")
    axA.set_xlabel(r"restraint center $\lambda$  (Å)")
    axA.set_ylabel("cumulative work  (eV)")
    axA.set_title(f"Work profiles  ({n_pulls} pulls, T={T:.0f} K)")
    axA.legend(fontsize=8, loc="upper left")

    # (B) CV tracking: end-to-end distance vs lambda (ideal = diagonal).
    axB = axes[1]
    for i in range(n_pulls):
        axB.plot(lam, z[i], color="0.8", lw=0.7, alpha=0.7)
    axB.plot(lam, z.mean(axis=0), color="tab:green", lw=2.0, label=r"$\langle d \rangle$")
    axB.plot(lam, lam, color="k", ls="--", lw=1.0, label=r"$d=\lambda$ (perfect tracking)")
    axB.set_xlabel(r"restraint center $\lambda$  (Å)")
    axB.set_ylabel("end-to-end distance  (Å)")
    axB.set_title("CV tracking (lag = dissipation)")
    axB.legend(fontsize=8, loc="upper left")

    # (C) Hummer-Szabo PMF.
    axC = axes[2]
    finite = np.isfinite(pmf)
    axC.plot(centers[finite], pmf[finite], color="tab:purple", lw=2.0, marker="o", ms=3)
    axC.set_xlabel("end-to-end distance  (Å)")
    axC.set_ylabel("PMF  $G$  (eV)")
    axC.set_title("Hummer-Szabo PMF")
    axC.grid(True, alpha=0.3)

    out = Path(args.out) if args.out else npz_path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
