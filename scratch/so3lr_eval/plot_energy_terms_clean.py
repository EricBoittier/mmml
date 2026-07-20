#!/usr/bin/env python3
"""Plot the CGenFF-free (training-consistent) energy-term decomposition vs distance."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
HARTREE_TO_KCAL = 627.5094740631


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def series(rows, pair, key):
    sub = [r for r in rows if r["pair"] == pair]
    sub.sort(key=lambda r: f(r["distance"]))
    xs = [f(r["distance"]) for r in sub]
    ys = [f(r[key]) for r in sub]
    return [(x, y) for x, y in zip(xs, ys) if y == y]


def qm_series(rows, molecule_a, molecule_b, backend):
    sub = [r for r in rows if r["molecule_a"] == molecule_a and r["molecule_b"] == molecule_b and r["backend"] == backend]
    sub.sort(key=lambda r: f(r["distance_angstrom"]))
    out = []
    for r in sub:
        ed, ea, eb = f(r["comp_Edimer_hartree"]), f(r["comp_EfragA_hartree"]), f(r["comp_EfragB_hartree"])
        if ed != ed or ea != ea or eb != eb:
            continue
        out.append((f(r["distance_angstrom"]), (ed - ea - eb) * HARTREE_TO_KCAL))
    return out


def main():
    ep10 = load(BASE / "energy_terms_clean_ep10.csv")
    ep13 = load(BASE / "energy_terms_clean_ep13.csv")
    ref = load(BASE / "full_comparison_prior" / "scan_results_with_references.csv")

    pairs = ["TIP3+TIP3", "MEOH+MEOH", "TIP3+MEOH", "DCM+DCM", "ACE+ACE"]
    terms = [
        ("Eint_total_kcal_mol", "total", "black", "-", 2.2),
        ("Eint_neural_kcal_mol", "neural (short-range)", "tab:blue", "-", 1.4),
        ("Eint_electrostatics_kcal_mol", "electrostatics", "tab:red", "-", 1.4),
        ("Eint_zbl_kcal_mol", "ZBL repulsion", "tab:green", "-", 1.4),
    ]
    qm_backends = [
        ("ccsd_def2svpd_gpu4pyscf_cp", "CCSD/def2-SVPD", "grey"),
        ("mp2_def2svp_gpu4pyscf_cp", "MP2/def2-SVP", "darkkhaki"),
        ("pbe0_def2svp_gpu4pyscf_d3bj_cp", "PBE0-D3BJ/def2-SVP", "olive"),
    ]

    for pair in pairs:
        a, b = pair.split("+")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        for ax, (rows, label) in zip(axes, [(ep10, "epoch-10"), (ep13, "epoch-13")]):
            ax.axhline(0, color="#bbb", lw=0.8)
            for key, term_label, color, ls, lw in terms:
                pts = series(rows, pair, key)
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, label=term_label, color=color, linestyle=ls, linewidth=lw, marker="o", markersize=3)
            for qm_backend, qm_label, qm_color in qm_backends:
                pts = qm_series(ref, a, b, qm_backend)
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, label=qm_label, color=qm_color, linestyle=":", linewidth=1.6)
            ax.set_title(f"{pair}: {label} (CGenFF-free)")
            ax.set_xlabel("Center distance / Å")
            if ax is axes[0]:
                ax.set_ylabel("Interaction energy / kcal mol$^{-1}$")
            ax.grid(alpha=0.25)
            ax.set_ylim(-45, 40)
        handles, labels = axes[1].get_legend_handles_labels()
        seen = {}
        for h, lab in zip(handles, labels):
            seen.setdefault(lab, h)
        fig.legend(seen.values(), seen.keys(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        fig.suptitle(f"Training-consistent energy-term decomposition: {pair}")
        fig.tight_layout(rect=(0, 0, 0.86, 1))
        out_path = BASE / f"energy_terms_clean_{a}_{b}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
