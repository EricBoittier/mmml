#!/usr/bin/env python3
"""Per-atom charge trajectories, conservation, and pooled distributions vs distance."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
ELEMENT_SYMBOLS = {1: "H", 6: "C", 8: "O", 17: "Cl"}
ELEMENT_COLORS = {1: "tab:gray", 6: "tab:blue", 8: "tab:red", 17: "tab:green"}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    for epoch_label, fname in [("epoch-10", "charges_vs_distance_ep10.csv"), ("epoch-13", "charges_vs_distance_ep13.csv")]:
        rows = load(BASE / "charges_scan" / fname)
        pairs = sorted({r["pair"] for r in rows})

        # --- 1. per-atom charge trajectories + sum_charges, one figure per pair ---
        for pair in pairs:
            sub = [r for r in rows if r["pair"] == pair]
            by_atom = defaultdict(list)  # (fragment, atom_index, Z) -> [(d, q)]
            sums = {}
            for r in sub:
                key = (r["fragment"], int(r["atom_index"]), int(r["element_Z"]))
                by_atom[key].append((f(r["distance"]), f(r["charge"])))
                sums[f(r["distance"])] = f(r["sum_charges_total"])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            for (frag, idx, z), pts in sorted(by_atom.items()):
                pts.sort()
                xs, ys = zip(*pts)
                ls = "-" if frag == "A" else "--"
                ax1.plot(
                    xs, ys, linestyle=ls, marker="o", markersize=2.5,
                    color=ELEMENT_COLORS.get(z, "black"),
                    label=f"{ELEMENT_SYMBOLS.get(z, z)}[{frag}]",
                    alpha=0.85,
                )
            ax1.axhline(0, color="#ccc", lw=0.8)
            ax1.set_xlabel("Center distance / Å")
            ax1.set_ylabel("Per-atom predicted charge / e")
            ax1.set_title(f"{pair} ({epoch_label}): per-atom charges")
            # de-dup legend
            handles, labels = ax1.get_legend_handles_labels()
            seen = {}
            for h, lab in zip(handles, labels):
                seen.setdefault(lab, h)
            ax1.legend(seen.values(), seen.keys(), fontsize=7, loc="best", ncol=2)
            ax1.grid(alpha=0.25)

            xs_sum = sorted(sums)
            ys_sum = [sums[x] for x in xs_sum]
            ax2.plot(xs_sum, ys_sum, "o-", color="black")
            ax2.axhline(0, color="tab:red", lw=1.2, linestyle="--", label="target (neutral)")
            ax2.set_xlabel("Center distance / Å")
            ax2.set_ylabel("Sum of predicted charges (whole dimer) / e")
            ax2.set_title(f"{pair} ({epoch_label}): total charge conservation")
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.25)

            fig.tight_layout()
            out = BASE / f"charges_{pair.replace('+', '_')}_{epoch_label}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}")

        # --- 2. pooled charge-magnitude distribution across all pairs/points ---
        fig, ax = plt.subplots(figsize=(8, 5))
        by_element = defaultdict(list)
        for r in rows:
            by_element[int(r["element_Z"])].append(f(r["charge"]))
        for z, vals in sorted(by_element.items()):
            ax.hist(vals, bins=40, alpha=0.55, label=f"{ELEMENT_SYMBOLS.get(z, z)} (n={len(vals)})", color=ELEMENT_COLORS.get(z, "black"))
        ax.set_xlabel("Predicted per-atom charge / e")
        ax.set_ylabel("Count (pooled over all pairs & scan points)")
        ax.set_title(f"Charge distribution by element ({epoch_label}, TIP3/MEOH/DCM/ACE scans)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = BASE / f"charge_distribution_{epoch_label}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
