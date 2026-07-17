#!/usr/bin/env python3
"""Validate the PBE0-D4/def2-TZVP collection against its GFN2-xTB source.

Both datasets were built from the same 22322 dimer-scan geometries
(``gfn2_nms15_{train,valid,test}.npz`` -> ``make_orca_array.py`` ->
``collect_orca_array.py`` -> ``pbe0_nms15_{train,valid,test}.npz``), but each
went through its own independent ``rng(0).permutation`` over a *different*
pre-shuffle ordering, so row j of a PBE0 split file is NOT row j of the
same-named GFN2 split file. This script reconstructs the correspondence from
first principles (replays collect_orca_array.py's permutation over the GFN2
train+valid+test concatenation), verifies it against the raw geometries
(coordinates must match exactly), and only then compares energies/forces.

Also validates the per-element ("per-atom") reference energies fitted at each
level of theory: reports the fit and plots the residual (E_raw minus the
composition-only linear fit) that atom_ref_energies leaves behind.

    python scripts/validate_pbe0_vs_gfn2_dataset.py \\
        --gfn2-dir acodcm --pbe0-dir acodcm --out validate_pbe0_gfn2
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
ELEMENT_SYMBOLS = {1: "H", 6: "C", 8: "O", 17: "Cl"}


def _load_splits(directory: Path, prefix: str) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for tag in ("train", "valid", "test"):
        f = directory / f"{prefix}_{tag}.npz"
        if not f.is_file():
            raise FileNotFoundError(f"missing {f}")
        out[tag] = dict(np.load(f, allow_pickle=True))
    return out


def _concat(splits: dict[str, dict[str, np.ndarray]], keys: list[str]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([splits[tag][k] for tag in ("train", "valid", "test")], axis=0)
            for k in keys}


def _reconstruct_pbe0_to_gfn2_index(n_gfn2: int, pbe0_counts: tuple[int, int, int]) -> np.ndarray:
    """Replay collect_orca_array.py's ``rng(0).permutation(n_gfn2)`` split.

    Returns an array ``orig_idx`` of length ``n_gfn2`` such that PBE0 split
    row ``j`` (in train/valid/test concatenation order) came from GFN2
    concatenation row ``orig_idx[j]``.
    """
    n_tr, n_va, n_te = pbe0_counts
    assert n_tr + n_va + n_te == n_gfn2, (n_tr, n_va, n_te, n_gfn2)
    perm = np.random.default_rng(0).permutation(n_gfn2)
    return perm  # perm[:n_tr] -> train rows in order, etc.


def _fit_atom_refs(E: np.ndarray, Z: np.ndarray, N: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elems = np.unique(np.concatenate([Z[i][: N[i]] for i in range(len(Z))]))
    elems = elems[elems > 0]
    C = np.zeros((len(E), len(elems)))
    for i in range(len(E)):
        z = Z[i][: N[i]]
        for j, e in enumerate(elems):
            C[i, j] = (z == e).sum()
    coef, *_ = np.linalg.lstsq(C, E, rcond=None)
    return coef, C @ coef, elems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gfn2-dir", type=Path, required=True)
    p.add_argument("--pbe0-dir", type=Path, required=True)
    p.add_argument("--gfn2-prefix", default="gfn2_nms15")
    p.add_argument("--pbe0-prefix", default="pbe0_nms15")
    p.add_argument("--out", type=Path, default=Path("validate_pbe0_gfn2"))
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mmml.utils.plotting.styles import apply_plot_style

    apply_plot_style("icml")
    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading GFN2 and PBE0 splits...")
    gfn2_splits = _load_splits(args.gfn2_dir, args.gfn2_prefix)
    pbe0_splits = _load_splits(args.pbe0_dir, args.pbe0_prefix)

    gfn2_counts = tuple(len(gfn2_splits[t]["R"]) for t in ("train", "valid", "test"))
    pbe0_counts = tuple(len(pbe0_splits[t]["R"]) for t in ("train", "valid", "test"))
    print(f"  GFN2 counts: {gfn2_counts}  (sum {sum(gfn2_counts)})")
    print(f"  PBE0 counts: {pbe0_counts}  (sum {sum(pbe0_counts)})")

    gfn2 = _concat(gfn2_splits, ["R", "Z", "N", "E", "E_total", "F", "D", "res_name",
                                  "mol_id"])
    n_gfn2 = len(gfn2["R"])

    # --- reconstruct the PBE0 <-> GFN2 correspondence ----------------------
    orig_idx = _reconstruct_pbe0_to_gfn2_index(n_gfn2, pbe0_counts)
    n_tr, n_va, n_te = pbe0_counts
    sel = {"train": orig_idx[:n_tr], "valid": orig_idx[n_tr:n_tr + n_va],
           "test": orig_idx[n_tr + n_va:]}

    # Sanity check: coordinates must match exactly at the mapped indices.
    max_err = 0.0
    for tag in ("train", "valid", "test"):
        r_pbe0 = pbe0_splits[tag]["R"]
        r_gfn2 = gfn2["R"][sel[tag]]
        err = float(np.abs(r_pbe0 - r_gfn2).max())
        max_err = max(max_err, err)
    print(f"  geometry correspondence check: max |R_pbe0 - R_gfn2[matched]| = {max_err:.3e} A")
    if max_err > 1e-8:
        print("  WARNING: correspondence does not line up -- energy/force comparisons "
              "below are NOT reliable. (Check that collect_orca_array.py's split logic "
              "hasn't changed.)", file=sys.stderr)
    else:
        print("  OK -- PBE0 rows map exactly onto their GFN2 source geometries.")

    pbe0 = _concat(pbe0_splits, ["R", "Z", "N", "E", "E_total", "F", "D"])
    gfn2_matched = {k: gfn2[k][orig_idx] for k in ("R", "Z", "N", "E", "E_total", "F", "D",
                                                     "res_name")}

    # =========================================================================
    # 1. Per-atom (per-element) reference energies
    # =========================================================================
    print("\n=== per-element reference energies ===")
    coef_gfn2, fit_gfn2, elems_gfn2 = _fit_atom_refs(gfn2["E_total"].ravel(), gfn2["Z"], gfn2["N"])
    coef_pbe0, fit_pbe0, elems_pbe0 = _fit_atom_refs(pbe0["E_total"].ravel(), pbe0["Z"], pbe0["N"])
    resid_gfn2 = gfn2["E_total"].ravel() - fit_gfn2
    resid_pbe0 = pbe0["E_total"].ravel() - fit_pbe0

    print(f"{'element':<10}{'GFN2 ref (eV)':>16}{'PBE0 ref (eV)':>16}")
    for z in sorted(set(elems_gfn2) | set(elems_pbe0)):
        sym = ELEMENT_SYMBOLS.get(int(z), f"Z={int(z)}")
        rg = coef_gfn2[list(elems_gfn2).index(z)] if z in elems_gfn2 else float("nan")
        rp = coef_pbe0[list(elems_pbe0).index(z)] if z in elems_pbe0 else float("nan")
        print(f"{sym:<10}{rg:>16.3f}{rp:>16.3f}")
    print(f"\nfit residual (composition-only linear model, should NOT be ~0 -- these "
          f"are dimer interaction + conformational energies, not atomization energies):")
    print(f"  GFN2: RMS {np.sqrt(np.mean(resid_gfn2**2)):.3f} eV, "
          f"range [{resid_gfn2.min():.2f}, {resid_gfn2.max():.2f}] eV")
    print(f"  PBE0: RMS {np.sqrt(np.mean(resid_pbe0**2)):.3f} eV, "
          f"range [{resid_pbe0.min():.2f}, {resid_pbe0.max():.2f}] eV")

    # stored atom_ref_energies (indexed by Z, sparse array) vs the refit here
    for name, splits, coef, elems in (("GFN2", gfn2_splits, coef_gfn2, elems_gfn2),
                                       ("PBE0", pbe0_splits, coef_pbe0, elems_pbe0)):
        stored = np.asarray(splits["train"]["atom_ref_energies"])
        mism = []
        for j, z in enumerate(elems):
            if int(z) < len(stored):
                d = abs(stored[int(z)] - coef[j])
                if d > 1e-6:
                    mism.append((int(z), stored[int(z)], coef[j], d))
        if mism:
            print(f"  {name}: stored atom_ref_energies MISMATCH vs refit -- "
                  f"{mism} (dataset's stored refs may be stale for this exact "
                  f"train/valid/test recombination)")
        else:
            print(f"  {name}: stored atom_ref_energies match a from-scratch refit on the "
                  f"full (train+valid+test) pool.")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=140)
    for ax, resid, name, color in ((axes[0], resid_gfn2, "GFN2-xTB", "#2563eb"),
                                    (axes[1], resid_pbe0, "PBE0-D4/def2-TZVP", "#dc2626")):
        ax.hist(resid, bins=80, color=color, alpha=0.85, edgecolor="white")
        ax.set_xlabel("E_raw - composition fit (eV)")
        ax.set_ylabel("Structures")
        ax.set_title(name)
    fig.suptitle("Per-element reference fit residual (interaction + conformational energy)")
    fig.tight_layout()
    fig.savefig(args.out / "atom_ref_fit_residual.png", bbox_inches="tight")
    plt.close(fig)

    # GFN2 is valence-only semi-empirical (refs near 0 eV); PBE0 is all-electron
    # (refs include core-electron energy, thousands of eV) -- a shared axis
    # would bury GFN2 entirely, so plot on separate axes.
    syms = [ELEMENT_SYMBOLS.get(int(z), str(int(z))) for z in elems_gfn2]
    xpos = np.arange(len(syms))
    pbe0_vals = [coef_pbe0[list(elems_pbe0).index(z)] if z in elems_pbe0 else 0.0
                 for z in elems_gfn2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), dpi=140)
    ax1.bar(xpos, coef_gfn2, color="#2563eb")
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(syms)
    ax1.set_ylabel("Fitted per-element reference (eV)")
    ax1.set_title("GFN2-xTB (valence-only)")
    ax2.bar(xpos, pbe0_vals, color="#dc2626")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(syms)
    ax2.set_title("PBE0-D4 (all-electron)")
    fig.suptitle("Per-atom reference energies by level of theory (note differing y-scales)")
    fig.tight_layout()
    fig.savefig(args.out / "atom_ref_by_element.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 2. Energy distributions per split
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=140, sharey=False)
    for ax, tag in zip(axes, ("train", "valid", "test")):
        ax.hist(gfn2_splits[tag]["E"].ravel(), bins=60, alpha=0.6, label="GFN2-xTB",
                color="#2563eb", density=True)
        ax.hist(pbe0_splits[tag]["E"].ravel(), bins=60, alpha=0.6, label="PBE0-D4",
                color="#dc2626", density=True)
        ax.set_title(tag)
        ax.set_xlabel("atom-referenced E (eV)")
    axes[0].set_ylabel("density")
    axes[0].legend()
    fig.suptitle("Referenced energy distribution per split")
    fig.tight_layout()
    fig.savefig(args.out / "energy_distribution_per_split.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 3. GFN2 vs PBE0 parity on matched geometries (referenced E, per-dataset ref)
    # =========================================================================
    e_gfn2 = gfn2_matched["E"].ravel() if "E" in gfn2_matched else None
    # gfn2_matched built from ["R","Z","N","E","E_total","F","D","res_name"]; has E already
    e_gfn2 = gfn2_matched["E"].ravel()
    e_pbe0 = pbe0["E"].ravel()
    mae = float(np.mean(np.abs(e_gfn2 - e_pbe0))) * EV_TO_KCAL
    rmse = float(np.sqrt(np.mean((e_gfn2 - e_pbe0) ** 2))) * EV_TO_KCAL
    corr = float(np.corrcoef(e_gfn2, e_pbe0)[0, 1])
    print(f"\n=== GFN2 vs PBE0 referenced-energy parity (matched geometries, own per-dataset refs) ===")
    print(f"  n={len(e_gfn2)}  Pearson r={corr:.4f}  MAE={mae:.2f} kcal/mol  RMSE={rmse:.2f} kcal/mol")
    print("  NOTE: each level's E is referenced against ITS OWN atom_ref_energies fit, so "
          "an offset/slope != 1 here reflects real differences between GFN2-xTB and "
          "PBE0-D4/def2-TZVP energetics, not a bookkeeping artifact.")

    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    ax.scatter(e_gfn2, e_pbe0, s=4, alpha=0.15, color="#2563eb", rasterized=True)
    lo, hi = min(e_gfn2.min(), e_pbe0.min()), max(e_gfn2.max(), e_pbe0.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="y=x")
    ax.set_xlabel("GFN2-xTB E (eV, referenced)")
    ax.set_ylabel("PBE0-D4/def2-TZVP E (eV, referenced)")
    ax.set_title(f"r={corr:.3f}  MAE={mae:.2f} kcal/mol")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "energy_parity_gfn2_vs_pbe0.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 4. Force comparison on matched geometries
    # =========================================================================
    n_at_gfn2 = gfn2_matched["N"]
    n_at_pbe0 = pbe0["N"]
    assert np.array_equal(n_at_gfn2, n_at_pbe0), "matched N mismatch -- correspondence broken"

    f_gfn2_list, f_pbe0_list = [], []
    for i in range(len(n_at_gfn2)):
        n = int(n_at_gfn2[i])
        f_gfn2_list.append(gfn2_matched["F"][i, :n])
        f_pbe0_list.append(pbe0["F"][i, :n])
    f_gfn2 = np.concatenate(f_gfn2_list, axis=0)
    f_pbe0 = np.concatenate(f_pbe0_list, axis=0)
    fmag_gfn2 = np.linalg.norm(f_gfn2, axis=-1)
    fmag_pbe0 = np.linalg.norm(f_pbe0, axis=-1)
    f_component_corr = float(np.corrcoef(f_gfn2.ravel(), f_pbe0.ravel())[0, 1])
    print(f"\n=== force comparison (matched geometries) ===")
    print(f"  per-component Pearson r={f_component_corr:.4f}")
    print(f"  |F| GFN2: mean {fmag_gfn2.mean():.3f}  PBE0: mean {fmag_pbe0.mean():.3f} eV/A")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=140)
    axes[0].hist(fmag_gfn2, bins=80, alpha=0.6, label="GFN2-xTB", color="#2563eb", density=True)
    axes[0].hist(fmag_pbe0, bins=80, alpha=0.6, label="PBE0-D4", color="#dc2626", density=True)
    axes[0].set_xlabel("|F| (eV/A)")
    axes[0].set_ylabel("density")
    axes[0].legend()
    axes[0].set_title("Force magnitude distribution")

    rng = np.random.default_rng(1)
    take = rng.choice(len(f_gfn2), size=min(20000, len(f_gfn2)), replace=False)
    axes[1].scatter(f_gfn2[take].ravel(), f_pbe0[take].ravel(),
                     s=2, alpha=0.1, color="#059669", rasterized=True)
    lo, hi = -8, 8
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    axes[1].set_xlim(lo, hi)
    axes[1].set_ylim(lo, hi)
    axes[1].set_xlabel("GFN2-xTB F component (eV/A)")
    axes[1].set_ylabel("PBE0-D4 F component (eV/A)")
    axes[1].set_title(f"r={f_component_corr:.3f}")
    fig.tight_layout()
    fig.savefig(args.out / "force_comparison.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\n-> plots written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
