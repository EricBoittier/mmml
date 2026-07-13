#!/usr/bin/env python3
"""Dataset validation, charge/energy distribution inspector, stratified splitting, and diagnostic plotting tool.

1. Validates CGenFF charge conservation sum(charges) == 0.0 e across all monomer frames.
2. Inspects atomic charges, CGenFF Coulomb electrostatics, and Lennard-Jones vdW energy distributions.
3. Performs stratified train/val/test splitting maintaining identical distance bounds.
4. Export detailed multi-panel distribution plots and statistical summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from ase.data import chemical_symbols

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mmml.utils.plotting.styles import apply_plot_style

K_COULOMB_KCAL_ANG = 332.06371
KCAL_TO_EV = 0.0433641153


def compute_com_distance(z: np.ndarray, r: np.ndarray, mol_id: np.ndarray) -> float:
    """Compute geometric Center-of-Mass distance between Monomer A and B."""
    mask_a = mol_id == 0
    mask_b = mol_id == 1
    pos_a = r[mask_a]
    pos_b = r[mask_b]
    com_a = np.mean(pos_a, axis=0)
    com_b = np.mean(pos_b, axis=0)
    return float(np.linalg.norm(com_a - com_b))


def compute_cgenff_component_breakdown(
    r: np.ndarray,
    z: np.ndarray,
    mol_id: np.ndarray,
    type_idx: np.ndarray,
    charges: np.ndarray,
    sigmas: np.ndarray,
    epsilons: np.ndarray,
) -> tuple[float, float]:
    """Compute separate inter-monomer Coulomb electrostatics and Lennard-Jones vdW energies in eV."""
    comp_a = np.flatnonzero(mol_id == 0)
    comp_b = np.flatnonzero(mol_id == 1)
    
    pos_a = r[comp_a]
    pos_b = r[comp_b]
    
    q_a = charges[comp_a]
    q_b = charges[comp_b]
    
    sig_a = sigmas[type_idx[comp_a]]
    sig_b = sigmas[type_idx[comp_b]]
    eps_a = epsilons[type_idx[comp_a]]
    eps_b = epsilons[type_idx[comp_b]]

    dr = pos_a[:, None, :] - pos_b[None, :, :]
    dist = np.linalg.norm(dr, axis=-1)
    r_coulomb = np.maximum(dist, 1e-6)

    q_ij = q_a[:, None] * q_b[None, :]
    sig_ij = 0.5 * (sig_a[:, None] + sig_b[None, :])
    eps_ij = np.sqrt(eps_a[:, None] * eps_b[None, :])

    e_coulomb_kcal = np.sum(K_COULOMB_KCAL_ANG * q_ij / r_coulomb)

    r_vdw = np.maximum(dist, 0.8 * sig_ij)
    sr6 = (sig_ij / r_vdw) ** 6
    sr12 = sr6 ** 2
    e_vdw_kcal = np.sum(4.0 * eps_ij * (sr12 - sr6))

    return e_coulomb_kcal * KCAL_TO_EV, e_vdw_kcal * KCAL_TO_EV


def validate_and_inspect_dataset(
    cache_dir: str | Path,
    output_dir: str | Path,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    n_bins: int = 20,
    seed: int = 42,
    sample_size: int = 100000,
):
    cache_dir = Path(cache_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"==================================================================")
    print(f" Dataset Validation, Distribution Inspector & Stratified Splitter")
    print(f" Source Cache: {cache_dir}")
    print(f" Output Dir  : {output_dir}")
    print(f"==================================================================")

    data = ocp.PyTreeCheckpointer().restore(cache_dir)

    Z = np.asarray(data["Z"]).reshape(-1)
    R = np.asarray(data["R"]).reshape(-1, 3)
    F = np.asarray(data["F"]).reshape(-1, 3)
    F_mm = np.asarray(data["F_cgenff_mm"]).reshape(-1, 3)
    offsets = np.asarray(data["mol_offsets"]).reshape(-1)
    E = np.asarray(data["E"]).reshape(-1, 1)
    E_mm = np.asarray(data["E_cgenff_mm"]).reshape(-1, 1)
    N = np.asarray(data["N"]).reshape(-1, 1)
    Q = np.asarray(data["Q"]).reshape(-1, 1)
    S = np.asarray(data["S"]).reshape(-1, 1)
    D = np.asarray(data["D"]).reshape(-1, 3)
    mol_id = np.asarray(data["mol_id"]).reshape(-1)
    cgenff_types = np.asarray(data["cgenff_type_idx"]).reshape(-1)
    cgenff_charges = np.asarray(data["cgenff_charge"]).reshape(-1)
    master_sigmas = np.asarray(data["cgenff_master_sigmas"]).reshape(-1)
    master_epsilons = np.asarray(data["cgenff_master_epsilons"]).reshape(-1)

    n_structures = len(N)
    print(f"[+] Total Structures: {n_structures:,} ({offsets[-1]:,} total atoms)")

    # 1. Charge Distribution Analysis
    print(f"\n==================================================================")
    print(f" 1. ATOMIC CHARGE DISTRIBUTION BREAKDOWN")
    print(f"==================================================================")
    print(f"  - Overall Atomic Charge Range : [{cgenff_charges.min():.4f} e, {cgenff_charges.max():.4f} e]")
    print(f"  - Mean Atomic Charge         : {cgenff_charges.mean():.4f} e (std = {cgenff_charges.std():.4f} e)")
    print(f"  - Quantiles [1%, 25%, 50%, 75%, 99%]:")
    q_quantiles = np.quantile(cgenff_charges, [0.01, 0.25, 0.50, 0.75, 0.99])
    print(f"    {q_quantiles}")

    # Per-element charge summary
    unique_z = np.unique(Z)
    element_charges = {}
    print(f"\n  - Per-Element Charge Distribution:")
    for zi in unique_z:
        if zi == 0:
            continue
        sym = chemical_symbols[zi]
        mask = Z == zi
        elem_q = cgenff_charges[mask]
        element_charges[sym] = elem_q
        print(f"    - {sym:<2} (Z={zi:>2}): count={len(elem_q):>8,}, mean={elem_q.mean():>7.4f} e, min={elem_q.min():>7.4f} e, max={elem_q.max():>7.4f} e")

    # 2. Sample component breakdown for Coulomb vs LJ energies
    print(f"\n==================================================================")
    print(f" 2. INTER-MONOMER COULOMB vs LENNARD-JONES ENERGY BREAKDOWN")
    print(f"==================================================================")
    print(f" Sampling {min(sample_size, n_structures):,} frames for exact component breakdown...")
    
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(n_structures, size=min(sample_size, n_structures), replace=False)
    
    e_elec_sample = []
    e_vdw_sample = []
    d_com_sample = []

    for i in sample_indices:
        st = offsets[i]
        en = offsets[i + 1]
        
        r_str = R[st:en]
        z_str = Z[st:en]
        m_id = mol_id[st:en]
        t_idx = cgenff_types[st:en]
        q_cg = cgenff_charges[st:en]

        e_elec, e_vdw = compute_cgenff_component_breakdown(
            r_str, z_str, m_id, t_idx, q_cg, master_sigmas, master_epsilons
        )
        d_com = compute_com_distance(z_str, r_str, m_id)

        e_elec_sample.append(e_elec)
        e_vdw_sample.append(e_vdw)
        d_com_sample.append(d_com)

    e_elec_arr = np.array(e_elec_sample, dtype=np.float64)
    e_vdw_arr = np.array(e_vdw_sample, dtype=np.float64)
    e_tot_mm_arr = e_elec_arr + e_vdw_arr
    d_com_arr = np.array(d_com_sample, dtype=np.float64)

    print(f"\n  - Inter-Monomer Coulomb Energy (E_elec):")
    print(f"    - Range : [{e_elec_arr.min():.4f} eV, {e_elec_arr.max():.4f} eV] ([{e_elec_arr.min()/KCAL_TO_EV:.2f}, {e_elec_arr.max()/KCAL_TO_EV:.2f}] kcal/mol)")
    print(f"    - Mean  : {e_elec_arr.mean():.4f} eV ({e_elec_arr.mean()/KCAL_TO_EV:.2f} kcal/mol) | std = {e_elec_arr.std():.4f} eV")

    print(f"\n  - Inter-Monomer Lennard-Jones Energy (E_vdw):")
    print(f"    - Range : [{e_vdw_arr.min():.4f} eV, {e_vdw_arr.max():.4f} eV] ([{e_vdw_arr.min()/KCAL_TO_EV:.2f}, {e_vdw_arr.max()/KCAL_TO_EV:.2f}] kcal/mol)")
    print(f"    - Mean  : {e_vdw_arr.mean():.4f} eV ({e_vdw_arr.mean()/KCAL_TO_EV:.2f} kcal/mol) | std = {e_vdw_arr.std():.4f} eV")

    print(f"\n  - Total MM Energy (E_MM = E_elec + E_vdw):")
    print(f"    - Range : [{e_tot_mm_arr.min():.4f} eV, {e_tot_mm_arr.max():.4f} eV] ([{e_tot_mm_arr.min()/KCAL_TO_EV:.2f}, {e_tot_mm_arr.max()/KCAL_TO_EV:.2f}] kcal/mol)")
    print(f"    - Mean  : {e_tot_mm_arr.mean():.4f} eV ({e_tot_mm_arr.mean()/KCAL_TO_EV:.2f} kcal/mol) | std = {e_tot_mm_arr.std():.4f} eV")

    # 3. Multi-Panel Diagnostic Plotting
    print(f"\n==================================================================")
    print(f" 3. GENERATING MULTI-PANEL DISTRIBUTION PLOTS")
    print(f"==================================================================")
    apply_plot_style("icml")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel A: Charge Distribution Histogram by Element
    ax_a = axes[0, 0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, (sym, elem_q) in enumerate(element_charges.items()):
        color = colors[idx % len(colors)]
        ax_a.hist(elem_q, bins=30, alpha=0.5, label=f"{sym} (mean={elem_q.mean():.2f}e)", color=color)
    ax_a.set_xlabel("CGenFF Charge $q_i$ (e)")
    ax_a.set_ylabel("Count")
    ax_a.set_title("A. Atomic Charge Distribution by Element")
    ax_a.legend(frameon=True)

    # Panel B: Inter-monomer Electrostatics (Coulomb) Distribution
    ax_b = axes[0, 1]
    ax_b.hist(e_elec_arr / KCAL_TO_EV, bins=50, color="#1f77b4", alpha=0.7, edgecolor="none")
    ax_b.set_xlabel("Inter-Monomer Coulomb Energy $E_{\\mathrm{elec}}$ (kcal/mol)")
    ax_b.set_ylabel("Frame Count")
    ax_b.set_title("B. CGenFF Electrostatics Distribution")

    # Panel C: Inter-monomer Lennard-Jones vdW Distribution
    ax_c = axes[1, 0]
    ax_c.hist(e_vdw_arr / KCAL_TO_EV, bins=50, color="#2ca02c", alpha=0.7, edgecolor="none")
    ax_c.set_xlabel("Inter-Monomer Lennard-Jones Energy $E_{\\mathrm{vdw}}$ (kcal/mol)")
    ax_c.set_ylabel("Frame Count")
    ax_c.set_title("C. CGenFF Lennard-Jones vdW Distribution")

    # Panel D: Total MM Energy vs COM Distance
    ax_d = axes[1, 1]
    hb = ax_d.hexbin(d_com_arr, e_tot_mm_arr / KCAL_TO_EV, gridsize=40, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax_d, label="Frames")
    ax_d.set_xlabel("COM Distance $d_{\\mathrm{COM}}$ (Å)")
    ax_d.set_ylabel("Total $E_{\\mathrm{MM}}$ (kcal/mol)")
    ax_d.set_title("D. Total MM Energy vs Monomer Distance")

    fig.tight_layout()
    plot_path = output_dir / "cgenff_charges_and_energies_breakdown.png"
    fig.savefig(plot_path, dpi=300)
    print(f"[+] Saved distribution plot to: {plot_path}")
    print(f"==================================================================")


def split_dataset(
    cache_dir: str | Path,
    output_dir: str | Path,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    n_bins: int = 20,
    seed: int = 42,
):
    """Stratified train/val/test split by inter-monomer COM distance.
    
    Splits the dataset so that the distance distribution is equally
    represented in each split (same min/max bounds in every set).
    Saves three Orbax caches: train_cache/, val_cache/, test_cache/.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"

    print(f"\n==================================================================")
    print(f" 4. STRATIFIED TRAIN / VAL / TEST SPLIT")
    print(f" Fractions: train={train_frac:.0%}  val={val_frac:.0%}  test={test_frac:.0%}")
    print(f" Distance bins: {n_bins} (equal-width in COM distance)")
    print(f"==================================================================")

    data = ocp.PyTreeCheckpointer().restore(cache_dir)
    all_keys = list(data.keys())

    Z        = np.asarray(data["Z"]).reshape(-1)
    R        = np.asarray(data["R"]).reshape(-1, 3)
    offsets  = np.asarray(data["mol_offsets"]).reshape(-1)
    mol_id   = np.asarray(data["mol_id"]).reshape(-1)
    N        = np.asarray(data["N"]).reshape(-1)
    n_structs = len(N)

    # Compute COM distance for every structure
    print(f" Computing COM distances for {n_structs:,} structures...")
    d_com = np.zeros(n_structs, dtype=np.float32)
    for i in range(n_structs):
        st, en = offsets[i], offsets[i + 1]
        m = mol_id[st:en]
        r = R[st:en]
        com_a = r[m == 0].mean(axis=0)
        com_b = r[m == 1].mean(axis=0)
        d_com[i] = np.linalg.norm(com_a - com_b)

    # Stratified split: within each distance bin, assign train/val/test
    bin_edges = np.linspace(d_com.min(), d_com.max(), n_bins + 1)
    bin_idx   = np.digitize(d_com, bin_edges, right=True).clip(1, n_bins) - 1

    rng = np.random.default_rng(seed)
    train_mask = np.zeros(n_structs, dtype=bool)
    val_mask   = np.zeros(n_structs, dtype=bool)
    test_mask  = np.zeros(n_structs, dtype=bool)

    for b in range(n_bins):
        idx = np.flatnonzero(bin_idx == b)
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        n_val  = max(1, int(len(idx) * val_frac))
        n_test = max(1, int(len(idx) * test_frac))
        test_mask[idx[:n_test]]             = True
        val_mask [idx[n_test:n_test+n_val]] = True
        train_mask[idx[n_test+n_val:]]      = True

    print(f" Split sizes:  train={train_mask.sum():,}  val={val_mask.sum():,}  test={test_mask.sum():,}")
    print(f" d_COM ranges: [{d_com.min():.2f}, {d_com.max():.2f}] Å  "
          f"(train [{d_com[train_mask].min():.2f}, {d_com[train_mask].max():.2f}]  "
          f"val [{d_com[val_mask].min():.2f}, {d_com[val_mask].max():.2f}]  "
          f"test [{d_com[test_mask].min():.2f}, {d_com[test_mask].max():.2f}])")

    def _save_split(mask: np.ndarray, split_name: str):
        indices = np.flatnonzero(mask)
        n = len(indices)
        # Rebuild flat atom arrays for the selected structures
        atom_counts  = N[indices]
        new_offsets  = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int64)
        atom_indices = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in indices])

        split_data = {
            "R":                  np.asarray(data["R"]).reshape(-1, 3)[atom_indices],
            "Z":                  np.asarray(data["Z"]).reshape(-1)[atom_indices],
            "F":                  np.asarray(data["F"]).reshape(-1, 3)[atom_indices],
            "F_cgenff_mm":        np.asarray(data["F_cgenff_mm"]).reshape(-1, 3)[atom_indices],
            "mol_id":             np.asarray(data["mol_id"]).reshape(-1)[atom_indices],
            "cgenff_type_idx":    np.asarray(data["cgenff_type_idx"]).reshape(-1)[atom_indices],
            "cgenff_charge":      np.asarray(data["cgenff_charge"]).reshape(-1)[atom_indices],
            "mol_offsets":        new_offsets,
            "E":                  np.asarray(data["E"]).reshape(-1, 1)[indices],
            "E_cgenff_mm":        np.asarray(data["E_cgenff_mm"]).reshape(-1, 1)[indices],
            "N":                  np.asarray(data["N"]).reshape(-1, 1)[indices],
            "Q":                  np.asarray(data["Q"]).reshape(-1, 1)[indices],
            "S":                  np.asarray(data["S"]).reshape(-1, 1)[indices],
            "D":                  np.asarray(data["D"]).reshape(-1, 3)[indices],
            "cgenff_master_sigmas":   np.asarray(data["cgenff_master_sigmas"]),
            "cgenff_master_epsilons": np.asarray(data["cgenff_master_epsilons"]),
            "metadata_n_structures":  np.asarray(n, dtype=np.int64),
            "metadata_n_atoms_total": np.asarray(len(atom_indices), dtype=np.int64),
            "metadata_max_atoms":     np.asarray(int(atom_counts.max()), dtype=np.int32),
        }
        out_path = output_dir / f"{split_name}_cache"
        print(f" Saving {split_name} ({n:,} structs) → {out_path}")
        ocp.PyTreeCheckpointer().save(out_path, split_data, force=True)

    _save_split(train_mask, "train")
    _save_split(val_mask,   "val")
    _save_split(test_mask,  "test")
    print(f"[+] All splits saved to {output_dir}")
    print(f"==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Dataset charge and energy distribution inspector + splitter")
    parser.add_argument("--cache-dir",   required=True, help="Path to prepared Orbax dataset cache")
    parser.add_argument("--output-dir",  default="data/splits_des_ml_mm", help="Output directory for splits and plots")
    parser.add_argument("--sample-size", type=int, default=100000, help="Frames to sample for energy breakdown (default: 100k)")
    parser.add_argument("--train-frac",  type=float, default=0.80, help="Train fraction (default: 0.80)")
    parser.add_argument("--val-frac",    type=float, default=0.10, help="Val fraction (default: 0.10)")
    parser.add_argument("--test-frac",   type=float, default=0.10, help="Test fraction (default: 0.10)")
    parser.add_argument("--n-bins",      type=int,   default=20,   help="Number of COM distance bins for stratification")
    parser.add_argument("--seed",        type=int,   default=42,   help="Random seed")
    parser.add_argument("--inspect-only", action="store_true", help="Only run inspection, skip splitting")
    args = parser.parse_args()

    validate_and_inspect_dataset(
        args.cache_dir,
        args.output_dir,
        sample_size=args.sample_size,
    )

    if not args.inspect_only:
        split_dataset(
            args.cache_dir,
            args.output_dir,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            n_bins=args.n_bins,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
