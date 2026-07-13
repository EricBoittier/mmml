#!/usr/bin/env python3
"""Diagnostic tool to analyze SpookyNet model checkpoints.

Decomposes interaction energies into component terms (neural features, explicit
electrostatics, ZBL repulsion, MBD dispersion) and inspects atomic charge predictions
across radial distances and lateral offsets to diagnose pathological energy surfaces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ase import Atoms
from mmml.analysis.dimer_molecules import MOLECULES, make_oriented_scan_geometries
from mmml.models.spookynet_calc import SpookyNetCalculator, EV_TO_KCAL_MOL


def analyze_checkpoint(
    checkpoint_path: str | Path,
    pair_name: str = "TIP3+TIP3",
    offsets: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0),
    d_min: float = 2.0,
    d_max: float = 10.0,
    n_points: int = 41,
    output_dir: str | Path = "./spookynet_diagnostics",
):
    checkpoint_path = Path(checkpoint_path).expanduser()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"==================================================================")
    print(f" SpookyNet Checkpoint Diagnostic: {checkpoint_path.name}")
    print(f" Pair: {pair_name} | Grid: d={d_min}..{d_max} Å ({n_points} pts), offsets={offsets}")
    print(f"==================================================================")

    # Initialize calculator
    calc = SpookyNetCalculator(checkpoint=checkpoint_path)

    # Check MBD status
    mbd_status = "ACTIVE" if calc.mbd_calc is not None else "NOT LOADED / RESIDUAL ONLY"
    print(f"\n[+] MBD Correction Status: {mbd_status}")
    if calc.mbd_calc is not None:
        mbd_path = getattr(calc.mbd_calc, 'checkpoint', 'Loaded')
        print(f"    MBD Checkpoint: {mbd_path}")
        print(f"    MBD Weight: {calc.mbd_weight}")
    else:
        print("    [!] WARNING: Evaluating SpookyNet residual neural network alone.")

    mon_a_name, mon_b_name = pair_name.split("+")
    mol_a = MOLECULES[mon_a_name]
    mol_b = MOLECULES[mon_b_name]

    # Pre-evaluate isolated monomers
    calc.calculate(mol_a)
    mon_a_spooky = calc.results["spooky_energy"]
    mon_a_mbd = calc.results.get("mbd_energy", 0.0)
    mon_a_total = calc.results["energy"]

    calc.calculate(mol_b)
    mon_b_spooky = calc.results["spooky_energy"]
    mon_b_mbd = calc.results.get("mbd_energy", 0.0)
    mon_b_total = calc.results["energy"]

    distances = np.linspace(d_min, d_max, n_points)
    rows = []

    for offset in offsets:
        geoms = make_oriented_scan_geometries(mol_a, mol_b, distances=distances, offset=offset)

        for d, dimer in zip(distances, geoms):
            # Calculate dimer using internal model to inspect raw outputs
            n_real = len(dimer)
            pad = calc.max_atoms - n_real
            z = np.asarray(dimer.get_atomic_numbers(), dtype=np.int32)
            pos = np.asarray(dimer.get_positions(), dtype=np.float32)

            if pad > 0:
                far = 1.0e4 + 100.0 * np.arange(pad, dtype=np.float32)
                pad_pos = np.stack([far, np.zeros(pad, dtype=np.float32), np.zeros(pad, dtype=np.float32)], axis=1)
                z = np.concatenate([z, np.zeros(pad, dtype=np.int32)])
                pos = np.concatenate([pos, pad_pos], axis=0)

            import e3x
            import jax.numpy as jnp

            dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(calc.max_atoms)
            atom_mask = (z > 0).astype(np.float32)
            valid_pairs = (atom_mask[dst_idx] > 0) & (atom_mask[src_idx] > 0)
            batch_mask = valid_pairs.astype(np.float32)

            output = calc._apply(
                jnp.asarray(z),
                jnp.asarray(pos),
                jnp.asarray(dst_idx),
                jnp.asarray(src_idx),
                jnp.asarray(atom_mask),
                jnp.asarray(batch_mask),
                calc.charge,
                calc.spin_multiplicity,
            )

            # Extract breakdown from model outputs
            total_spooky = float(np.asarray(output["energy"]).squeeze()) * EV_TO_KCAL_MOL
            charges = np.asarray(output["charges"]).squeeze()[:n_real] if "charges" in output else np.zeros(n_real)
            
            # Electrostatics component (if available)
            if "electrostatics" in output and output["electrostatics"] is not None:
                e_elec = float(np.asarray(output["electrostatics"]).squeeze()) * EV_TO_KCAL_MOL
            else:
                e_elec = 0.0

            # Repulsion component (if available)
            if "repulsion" in output and output["repulsion"] is not None:
                e_rep = float(np.asarray(output["repulsion"]).squeeze()) * EV_TO_KCAL_MOL
            else:
                e_rep = 0.0

            # MBD contribution
            if calc.mbd_calc is not None:
                mbd_out = calc.mbd_calc.predict_mbd(dimer)
                e_mbd = calc.mbd_weight * mbd_out["energy_ev"] * EV_TO_KCAL_MOL
            else:
                e_mbd = 0.0

            total_energy = total_spooky + e_mbd
            e_neural = total_spooky - e_elec - e_rep

            # Monomer interaction decomposition
            delta_total = total_energy - (mon_a_total + mon_b_total) * EV_TO_KCAL_MOL
            delta_spooky = total_spooky - (mon_a_spooky + mon_b_spooky) * EV_TO_KCAL_MOL
            delta_mbd = e_mbd - (mon_a_mbd + mon_b_mbd) * EV_TO_KCAL_MOL

            n_a = len(mol_a)
            charges_a = charges[:n_a]
            charges_b = charges[n_a:]

            rows.append({
                "offset": offset,
                "distance": d,
                "delta_E_total": delta_total,
                "delta_E_spooky": delta_spooky,
                "delta_E_mbd": delta_mbd,
                "E_elec": e_elec,
                "E_rep": e_rep,
                "E_neural": e_neural,
                "q_max_abs": np.max(np.abs(charges)),
                "q_rms": np.sqrt(np.mean(charges**2)),
                "q_net_monA": np.sum(charges_a),
                "q_net_monB": np.sum(charges_b),
            })

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"spookynet_diagnostic_{checkpoint_path.name}_{pair_name.replace('+', '_')}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Tabular diagnostic saved to: {csv_path}")

    # Report Canary point: d ≈ 5.9 Å at offset = 2.0 Å
    canary_sub = df[(np.isclose(df["offset"], 2.0)) & (np.isclose(df["distance"], 5.9, atol=0.25))]
    if not canary_sub.empty:
        c_row = canary_sub.iloc[0]
        print(f"\n==================================================================")
        print(f" CANARY GEOMETRY SUMMARY (d ≈ 5.9 Å, offset = 2.0 Å)")
        print(f"==================================================================")
        print(f"  Interaction Energy (ΔE_total): {c_row['delta_E_total']:.3f} kcal/mol")
        print(f"  Spooky Component (ΔE_spooky) : {c_row['delta_E_spooky']:.3f} kcal/mol")
        print(f"  Explicit Electrostatics (E_elec): {c_row['E_elec']:.3f} kcal/mol")
        print(f"  ZBL Repulsion (E_rep)           : {c_row['E_rep']:.3f} kcal/mol")
        print(f"  Neural Feature Component        : {c_row['E_neural']:.3f} kcal/mol")
        print(f"  MBD Energy (E_mbd)              : {c_row['delta_E_mbd']:.3f} kcal/mol")
        print(f"  Max Atomic Charge (|q_max|)     : {c_row['q_max_abs']:.4f} e")
        print(f"  Monomer Charges (A / B)         : {c_row['q_net_monA']:+.4f} / {c_row['q_net_monB']:+.4f} e")
        print(f"==================================================================")

    # Plot Diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Interaction Energy Curves across offsets
    for off in offsets:
        sub = df[df["offset"] == off]
        axes[0].plot(sub["distance"], sub["delta_E_total"], label=f"offset={off:.1f} Å")
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.6)
    axes[0].set_ylim(-100, 50)
    axes[0].set_xlabel("Distance d (Å)")
    axes[0].set_ylabel("Interaction Energy ΔE (kcal/mol)")
    axes[0].set_title(f"Interaction Energy Curves ({pair_name})")
    axes[0].legend()

    # Panel 2: Component Breakdown at Canary Offset (2.0 Å)
    sub_canary = df[df["offset"] == 2.0]
    axes[1].plot(sub_canary["distance"], sub_canary["delta_E_total"], "k-", linewidth=2, label="ΔE Total")
    axes[1].plot(sub_canary["distance"], sub_canary["E_elec"], "r--", label="Explicit Electrostatics")
    axes[1].plot(sub_canary["distance"], sub_canary["E_neural"], "g--", label="Neural Network")
    if calc.mbd_calc is not None:
        axes[1].plot(sub_canary["distance"], sub_canary["delta_E_mbd"], "b--", label="MBD Dispersion")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.6)
    axes[1].set_xlabel("Distance d (Å)")
    axes[1].set_ylabel("Energy (kcal/mol)")
    axes[1].set_title(f"Component Breakdown at offset=2.0 Å")
    axes[1].legend()

    # Panel 3: Predicted Charges
    axes[2].plot(sub_canary["distance"], sub_canary["q_max_abs"], "m-", label="Max |q_i|")
    axes[2].plot(sub_canary["distance"], sub_canary["q_rms"], "c-", label="RMS q_i")
    axes[2].set_xlabel("Distance d (Å)")
    axes[2].set_ylabel("Charge Magnitude (e)")
    axes[2].set_title("Predicted Atomic Charge Magnitudes")
    axes[2].legend()

    fig.tight_layout()
    plot_path = output_dir / f"spookynet_diagnostic_{checkpoint_path.name}_{pair_name.replace('+', '_')}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Diagnostic plot saved to: {plot_path}\n")


def main():
    parser = argparse.ArgumentParser(description="SpookyNet Model Diagnostic Suite")
    parser.add_argument("--checkpoint", required=True, help="Path to SpookyNet checkpoint directory or JSON file")
    parser.add_argument("--pair", default="TIP3+TIP3", help="Molecular dimer pair (e.g. TIP3+TIP3, DCM+DCM)")
    parser.add_argument("--offsets", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0], help="Lateral offsets in Å")
    parser.add_argument("--output-dir", default="./spookynet_diagnostics", help="Output directory for plots & CSVs")
    args = parser.parse_args()

    analyze_checkpoint(
        checkpoint_path=args.checkpoint,
        pair_name=args.pair,
        offsets=tuple(args.offsets),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
