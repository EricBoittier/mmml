#!/usr/bin/env python3
"""Plot 1D dihedral umbrella PMF from ``umbrella_summary.json`` (MBAR block)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--title", default="")
    p.add_argument("--xlabel", default="ξ (deg)")
    args = p.parse_args()

    summary = json.loads((args.run_dir / "umbrella_summary.json").read_text())
    mbar = summary.get("mbar") or {}
    if "pmf_rel_kcal_mol" not in mbar:
        raise SystemExit(f"No mbar.pmf_rel_kcal_mol in {args.run_dir}")

    xi = np.asarray(mbar["xi0"], dtype=float)
    pmf = np.asarray(mbar["pmf_rel_kcal_mol"], dtype=float)
    dpmf = np.asarray(mbar.get("d_pmf_rel_kcal_mol") or np.zeros_like(pmf), dtype=float)
    mask = np.isfinite(pmf)
    xi, pmf, dpmf = xi[mask], pmf[mask], dpmf[mask]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.errorbar(xi, pmf, yerr=dpmf, fmt="o-", color="#1f4e79", capsize=3, lw=1.5)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel("ΔF (kcal/mol)")
    ax.set_title(args.title or f"Umbrella PMF — {args.run_dir.name}")
    ax.set_xlim(-180, 180)
    ax.axhline(0.0, color="0.7", lw=0.8)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
