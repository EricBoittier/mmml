#!/usr/bin/env python3
"""Drop energy/force outliers and rebuild a stratified train/valid/test split.

Dimer-scan datasets built with a small ``--min-contact`` put a lot of weight
on the handful of structures at the closest approach -- e.g. the PBE0-D4 set
has DCM,DCM forces reaching >200 eV/A at the contact floor, ~50x a typical
value, which can dominate a force loss disproportionately (see
gfn2_nms_hybrid.yaml's own warning about this). This drops the top
``--pct`` percent of structures by referenced energy AND by max per-atom
force magnitude (union: dropped if either is extreme), refits the per-element
references on the retained pool only, then splits 80/10/10 *within each
``res_name`` stratum* so every split keeps the same monomer/dimer-type mix
instead of one random global permutation skewing that mix.

    python scripts/filter_and_stratify_dataset.py \\
        --data pbe0_nms15_train.npz pbe0_nms15_valid.npz pbe0_nms15_test.npz \\
        --pct 5 --out pbe0_nms15_clean --plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EV_TO_KCAL = 23.0605


def _fit_atom_refs(E: np.ndarray, Z: np.ndarray, N: np.ndarray):
    elems = np.unique(np.concatenate([Z[i][: N[i]] for i in range(len(Z))]))
    elems = elems[elems > 0]
    C = np.zeros((len(E), len(elems)))
    for i in range(len(E)):
        z = Z[i][: N[i]]
        for j, e in enumerate(elems):
            C[i, j] = (z == e).sum()
    coef, *_ = np.linalg.lstsq(C, E, rcond=None)
    refs = np.zeros(int(elems.max()) + 1)
    refs[elems] = coef
    return refs, C @ coef, elems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True,
                   help="npz file(s) to pool together before filtering/re-splitting "
                   "(e.g. all three existing train/valid/test splits)")
    p.add_argument("--pct", type=float, default=5.0,
                   help="drop this top percent by referenced-E and by max |F| (union)")
    p.add_argument("--out", default="dataset_clean")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()

    KEEP_KEYS = ["R", "Z", "N", "E_total", "F", "D", "mol_id", "cgenff_type_idx",
                 "cgenff_charge", "res_name"]
    srcs = [dict(np.load(f, allow_pickle=True)) for f in args.data]
    pooled = {k: np.concatenate([s[k] for s in srcs], axis=0) for k in KEEP_KEYS}
    units = srcs[0]["_mmml_units"]
    sig = np.asarray(srcs[0]["cgenff_master_sigmas"])
    eps = np.asarray(srcs[0]["cgenff_master_epsilons"])
    n_tot = len(pooled["R"])
    print(f"pooled {n_tot} structures from {len(args.data)} file(s)")

    # --- referenced E and per-structure max |F|, on the FULL pool for outlier
    # detection (refit again below on the retained subset for training) -------
    E_total = pooled["E_total"].ravel()
    Z, N = pooled["Z"], pooled["N"]
    refs0, fitted0, elems0 = _fit_atom_refs(E_total, Z, N)
    E_ref0 = E_total - fitted0

    fmax = np.zeros(n_tot)
    for i in range(n_tot):
        n = int(N[i])
        fmax[i] = np.linalg.norm(pooled["F"][i, :n], axis=-1).max()

    e_thresh = np.percentile(E_ref0, 100 - args.pct)
    f_thresh = np.percentile(fmax, 100 - args.pct)
    is_e_outlier = E_ref0 > e_thresh
    is_f_outlier = fmax > f_thresh
    outlier = is_e_outlier | is_f_outlier
    keep = ~outlier
    print(f"E_ref outlier threshold (top {args.pct:g}%): {e_thresh:.2f} eV "
          f"({is_e_outlier.sum()} structures)")
    print(f"|F|max outlier threshold (top {args.pct:g}%): {f_thresh:.2f} eV/A "
          f"({is_f_outlier.sum()} structures)")
    print(f"dropped {outlier.sum()}/{n_tot} ({100 * outlier.mean():.1f}%, "
          f"union of both criteria) -- kept {keep.sum()}")

    res_all = np.array([str(x) for x in pooled["res_name"]])
    print("\ncomposition before / after filtering:")
    for cat in sorted(set(res_all)):
        m = res_all == cat
        print(f"  {cat:12s} before={m.sum():6d}  after={((m) & keep).sum():6d}  "
              f"dropped={((m) & outlier).sum():6d} "
              f"({100 * ((m) & outlier).sum() / max(m.sum(), 1):.1f}%)")

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mmml.utils.plotting.styles import apply_plot_style

        apply_plot_style("icml")
        out_dir = Path(f"{args.out}_plots")
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=140)
        axes[0].hist(E_ref0, bins=100, color="#94a3b8", alpha=0.9, label="all")
        axes[0].hist(E_ref0[keep], bins=100, color="#2563eb", alpha=0.9, label="kept")
        axes[0].axvline(e_thresh, color="#dc2626", ls="--", lw=1, label=f"top {args.pct:g}%")
        axes[0].set_xlabel("referenced E (eV)")
        axes[0].set_ylabel("structures")
        axes[0].legend()
        axes[0].set_title("Energy outlier cut")

        axes[1].hist(fmax, bins=100, color="#94a3b8", alpha=0.9, label="all")
        axes[1].hist(fmax[keep], bins=100, color="#059669", alpha=0.9, label="kept")
        axes[1].axvline(f_thresh, color="#dc2626", ls="--", lw=1, label=f"top {args.pct:g}%")
        axes[1].set_xlabel("max |F| per structure (eV/A)")
        axes[1].legend()
        axes[1].set_title("Force outlier cut")
        fig.tight_layout()
        fig.savefig(out_dir / "outlier_cuts.png", bbox_inches="tight")
        plt.close(fig)
        print(f"\n-> {out_dir}/outlier_cuts.png")

    # --- refit atom refs on the retained pool only, for training ------------
    idx_keep = np.where(keep)[0]
    E_kept_total = E_total[idx_keep]
    refs, fitted, elems = _fit_atom_refs(E_kept_total, Z[idx_keep], N[idx_keep])
    E_kept_ref = E_kept_total - fitted
    print(f"\nrefit per-element refs on retained pool: "
          + ", ".join(f"Z={int(z)}:{refs[int(z)]:.3f}" for z in elems))

    # --- stratified split: 80/10/10 WITHIN each res_name category -----------
    rng = np.random.default_rng(args.seed)
    res_kept = res_all[idx_keep]
    split_of = np.empty(len(idx_keep), dtype="<U5")
    for cat in sorted(set(res_kept)):
        cat_idx = np.where(res_kept == cat)[0]
        perm = rng.permutation(cat_idx)
        n_tr = int(round(args.train_frac * len(perm)))
        n_va = int(round(args.valid_frac * len(perm)))
        split_of[perm[:n_tr]] = "train"
        split_of[perm[n_tr:n_tr + n_va]] = "valid"
        split_of[perm[n_tr + n_va:]] = "test"

    print("\nstratified split composition:")
    for cat in sorted(set(res_kept)):
        row = [f"{cat:12s}"]
        for tag in ("train", "valid", "test"):
            row.append(f"{tag}={((res_kept == cat) & (split_of == tag)).sum():5d}")
        print("  " + "  ".join(row))

    common = dict(
        atom_ref_energies=refs,
        cgenff_master_sigmas=sig,
        cgenff_master_epsilons=eps,
        _mmml_units=units,
    )
    for tag in ("train", "valid", "test"):
        sel = np.where(split_of == tag)[0]  # index into the kept/idx_keep arrays
        gi = idx_keep[sel]  # index into the original pooled arrays
        f = Path(f"{args.out}_{tag}.npz")
        np.savez(
            f,
            R=pooled["R"][gi], Z=pooled["Z"][gi], N=pooled["N"][gi],
            E=E_kept_ref[sel].reshape(-1, 1), E_total=E_total[gi].reshape(-1, 1),
            F=pooled["F"][gi], D=pooled["D"][gi],
            mol_id=pooled["mol_id"][gi], cgenff_type_idx=pooled["cgenff_type_idx"][gi],
            cgenff_charge=pooled["cgenff_charge"][gi], res_name=res_all[gi],
            **common,
        )
        print(f"-> {f}  ({len(gi)} structures)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
