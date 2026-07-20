#!/usr/bin/env python3
"""Plot per-term energy decomposition vs distance for epoch-10/13, with ab initio refs where available."""

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


def series(rows, molecule_a, molecule_b, backend, key="comp_Eint_kcal_mol", offset=0.0):
    sub = [
        r
        for r in rows
        if r["molecule_a"] == molecule_a and r["molecule_b"] == molecule_b and r["backend"] == backend
        and f(r.get("offset_angstrom", 0.0)) == offset
    ]
    sub.sort(key=lambda r: f(r["distance_angstrom"]))
    xs = [f(r["distance_angstrom"]) for r in sub]
    ys = [f(r[key]) for r in sub]
    pairs = [(x, y) for x, y in zip(xs, ys) if y == y]
    return pairs


def qm_series(rows, molecule_a, molecule_b, backend):
    sub = [
        r
        for r in rows
        if r["molecule_a"] == molecule_a and r["molecule_b"] == molecule_b and r["backend"] == backend
    ]
    sub.sort(key=lambda r: f(r["distance_angstrom"]))
    out = []
    for r in sub:
        ed, ea, eb = f(r["comp_Edimer_hartree"]), f(r["comp_EfragA_hartree"]), f(r["comp_EfragB_hartree"])
        if ed != ed or ea != ea or eb != eb:
            continue
        out.append((f(r["distance_angstrom"]), (ed - ea - eb) * HARTREE_TO_KCAL))
    return out


def main():
    ep10 = load(BASE / "dimer_scans_ep10" / "scan_results.csv")
    ep13 = load(BASE / "dimer_scans_ep13" / "scan_results.csv")
    ref = load(BASE / "full_comparison_prior" / "scan_results_with_references.csv")

    pairs = [("TIP3", "TIP3"), ("MEOH", "MEOH"), ("TIP3", "MEOH"), ("DCM", "DCM"), ("ACE", "ACE")]
    terms = [
        ("comp_Eint_kcal_mol", "total (hybrid)", "black", "-", 2.2),
        ("comp_Eint_neural_energy_kcal_mol", "neural (short-range)", "tab:blue", "-", 1.4),
        ("comp_Eint_electrostatics_energy_kcal_mol", "electrostatics", "tab:red", "-", 1.4),
        ("comp_Eint_zbl_repulsion_energy_kcal_mol", "ZBL repulsion", "tab:green", "-", 1.4),
        ("comp_Eint_cgenff_vdw_energy_kcal_mol", "CGenFF vdW", "tab:orange", "--", 1.0),
        ("comp_Eint_mbd_energy_kcal_mol", "MBD", "tab:purple", "--", 1.0),
    ]
    qm_backends = [
        ("ccsd_def2svpd_gpu4pyscf_cp", "CCSD/def2-SVPD", "grey"),
        ("ccsd_def2svp_gpu4pyscf_cp", "CCSD/def2-SVP", "silver"),
        ("mp2_def2svp_gpu4pyscf_cp", "MP2/def2-SVP", "darkkhaki"),
        ("pbe0_def2svp_gpu4pyscf_d3bj_cp", "PBE0-D3BJ/def2-SVP", "olive"),
    ]

    for a, b in pairs:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
        for ax, (epoch_rows, epoch_label, backend_suffix) in zip(
            axes, [(ep10, "epoch-10", "muon3_ep10"), (ep13, "epoch-13", "muon3_ep13")]
        ):
            ax.axhline(0, color="#bbb", lw=0.8)
            for key, label, color, ls, lw in terms:
                backend = f"spookynet_hybrid_{backend_suffix}" if key != "comp_Eint_kcal_mol" else f"spookynet_hybrid_{backend_suffix}"
                pts = series(epoch_rows, a, b, backend, key=key)
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, label=label, color=color, linestyle=ls, linewidth=lw, marker="o", markersize=3)
            for qm_backend, qm_label, qm_color in qm_backends:
                pts = qm_series(ref, a, b, qm_backend)
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, label=qm_label, color=qm_color, linestyle=":", linewidth=1.6)
            ax.set_title(f"{a}+{b}: {epoch_label}")
            ax.set_xlabel("Center distance / Å")
            if ax is axes[0]:
                ax.set_ylabel("Interaction energy / kcal mol$^{-1}$")
            ax.grid(alpha=0.25)
            ax.set_ylim(-30, 40)
        # Shared, deduplicated legend on the right
        handles, labels = axes[1].get_legend_handles_labels()
        seen = {}
        for h, lab in zip(handles, labels):
            seen.setdefault(lab, h)
        fig.legend(seen.values(), seen.keys(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        fig.suptitle(f"Energy-term decomposition: {a}+{b}")
        fig.tight_layout(rect=(0, 0, 0.86, 1))
        out_path = BASE / f"energy_terms_{a}_{b}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
