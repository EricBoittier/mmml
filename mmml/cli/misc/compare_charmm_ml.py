#!/usr/bin/env python3
"""
Compare dipoles and ESPs from CHARMM PSF point charges vs ML model predictions.

Loads a trained joint PhysNet-DCMNet model, validation data, and CHARMM charges,
then computes dipole and ESP for each validation sample and compares against
QM reference.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import jax

# Add mmml to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mmml.utils.electrostatics import (
    compute_dipole_from_point_charges,
    compute_esp_from_point_charges,
)
from mmml.cli.misc.train_joint import (
    load_combined_data,
    precompute_edge_lists,
    prepare_batch_data,
    eval_step,
    JointPhysNetDCMNet,
    JointPhysNetNonEquivariant,
    LossTerm,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def _check_pycharmm():
    """Lazy check for PyCHARMM availability (avoids importing CHARMM setup at module load)."""
    try:
        from mmml.interfaces.pycharmmInterface.setupBox import setup_box_generic
        from pycharmm import psf
        return True, setup_box_generic, psf
    except Exception:
        return False, None, None


def load_checkpoint_and_model(checkpoint_path: Path) -> Tuple[Any, Any]:
    """Load model params and rebuild model from checkpoint directory."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.is_file():
        ckpt_dir = checkpoint_path.parent
        params_path = checkpoint_path
    else:
        ckpt_dir = checkpoint_path
        params_path = ckpt_dir / "best_params.pkl"

    if not params_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {params_path}")

    import pickle
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    config_path = ckpt_dir / "model_config.pkl"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {config_path}. "
            "Ensure the checkpoint directory contains model_config.pkl."
        )
    with open(config_path, "rb") as f:
        model_config = pickle.load(f)

    physnet_config = dict(model_config["physnet_config"])
    physnet_config.setdefault("natoms", 64)
    mix_coulomb_energy = model_config.get("mix_coulomb_energy", False)

    if "dcmnet_config" in model_config:
        model = JointPhysNetDCMNet(
            physnet_config=physnet_config,
            dcmnet_config=model_config["dcmnet_config"],
            mix_coulomb_energy=mix_coulomb_energy,
        )
    elif "noneq_config" in model_config:
        model = JointPhysNetNonEquivariant(
            physnet_config=physnet_config,
            noneq_config=model_config["noneq_config"],
            mix_coulomb_energy=mix_coulomb_energy,
        )
    else:
        raise ValueError("Model config must contain dcmnet_config or noneq_config")

    return params, model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CHARMM vs ML dipoles and ESPs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to train_joint checkpoint (dir with best_params.pkl or path to best_params.pkl)")
    parser.add_argument("--valid-efd", type=Path, required=True,
                        help="Validation energies/forces/dipoles NPZ file")
    parser.add_argument("--valid-esp", type=Path, required=True,
                        help="Validation ESP grids NPZ file")
    parser.add_argument("--pdb", type=Path, required=True,
                        help="PDB path for CHARMM setup (same molecule as validation data)")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of validation samples to evaluate")
    parser.add_argument("--out-dir", type=Path, default=Path("charmm_ml_comparison"),
                        help="Output directory for plots")
    parser.add_argument("--cutoff", type=float, default=10.0,
                        help="Cutoff for prepare_batch_data")
    parser.add_argument("--subtract-atom-energies", action="store_true",
                        help="Subtract reference atomic energies (match training)")
    args = parser.parse_args()

    has_pycharmm, setup_box_generic, psf_module = _check_pycharmm()
    if not has_pycharmm:
        print("ERROR: PyCHARMM is required. Install pycharmm and ensure CHARMM is available.")
        return 1

    if not HAS_MATPLOTLIB:
        print("WARNING: Matplotlib not available. Plots will be skipped.")

    print("=" * 70)
    print("CHARMM vs ML Dipole/ESP Comparison")
    print("=" * 70)

    # Load validation data
    print("\nLoading validation data...")
    valid_data = load_combined_data(
        args.valid_efd,
        args.valid_esp,
        subtract_atom_energies=args.subtract_atom_energies,
        verbose=True,
    )
    valid_data = precompute_edge_lists(valid_data, cutoff=args.cutoff, verbose=True)
    natoms = valid_data["R"].shape[1]
    n_samples_total = len(valid_data["E"])
    n_total = min(args.n_samples, n_samples_total)
    print(f"  Samples: {n_total} (of {n_samples_total})")
    print(f"  Natoms (padded): {natoms}")

    # Load model and params
    print("\nLoading checkpoint...")
    params, model = load_checkpoint_and_model(args.checkpoint)
    if hasattr(model, "dcmnet_config"):
        n_dcm = model.dcmnet_config["n_dcm"]
    else:
        n_dcm = model.noneq_config["n_dcm"]

    # Default loss terms for eval_step
    dipole_terms = (LossTerm(source="physnet", weight=1.0), LossTerm(source="dcmnet", weight=1.0))
    esp_terms = (LossTerm(source="physnet", weight=1.0), LossTerm(source="dcmnet", weight=1.0))

    # CHARMM setup
    print("\nSetting up CHARMM and loading PSF charges...")
    pdb_path = Path(args.pdb).resolve()
    if not pdb_path.exists():
        print(f"ERROR: PDB not found: {pdb_path}")
        return 1

    # setup_box_generic writes to pdb/ and psf/ relative to cwd
    for d in ("pdb", "psf"):
        Path(d).mkdir(exist_ok=True)

    setup_box_generic(
        str(pdb_path),
        side_length=25,
        skip_energy_show=True,
    )
    charges_charmm = np.array(psf_module.get_charges())
    n_charmm = len(charges_charmm)
    if n_charmm < natoms:
        # Truncate to first molecule if box has multiple
        max_n = int(np.max(valid_data["N"]))
        if n_charmm < max_n:
            print(f"ERROR: CHARMM has {n_charmm} atoms but validation data has up to {max_n} atoms.")
            return 1
    charges_charmm = charges_charmm[:natoms].copy()

    print(f"  CHARMM charges: {len(charges_charmm)} atoms")
    print(f"  Charge sum: {charges_charmm.sum():.4f} e")

    # Collect predictions
    dipoles_physnet = []
    dipoles_dcmnet = []
    dipoles_charmm = []
    dipoles_true = []
    esp_physnet_list = []
    esp_dcmnet_list = []
    esp_charmm_list = []
    esp_true_list = []
    esp_grid_list = []

    print(f"\nEvaluating {n_total} samples...")
    for i in range(n_total):
        batch = prepare_batch_data(valid_data, np.array([i]), cutoff=args.cutoff)
        _, losses, output = eval_step(
            params=params,
            batch=batch,
            model_apply=model.apply,
            energy_w=1.0,
            forces_w=1.0,
            mono_w=1.0,
            charge_reg_w=1.0,
            batch_size=1,
            n_dcm=n_dcm,
            dipole_terms=dipole_terms,
            esp_terms=esp_terms,
            esp_min_distance=0.0,
            esp_max_value=1e10,
        )
        jax.block_until_ready(output)

        n_atoms = int(batch["N"][0])
        positions = np.array(batch["R"]).reshape(-1, 3)[:n_atoms]
        Z = np.array(batch["Z"]).flatten()[:n_atoms]
        vdw = np.array(batch["vdw_surface"][0])

        q_charmm = charges_charmm[:n_atoms]
        dipole_charmm = compute_dipole_from_point_charges(q_charmm, positions, Z)
        esp_charmm = compute_esp_from_point_charges(q_charmm, positions, vdw, atom_mask=None)

        dipoles_physnet.append(np.array(output["dipoles"][0]))
        dipoles_dcmnet.append(np.array(output["dipoles_dcmnet"][0]))
        dipoles_charmm.append(dipole_charmm)
        dipoles_true.append(np.array(batch["D"][0]))

        esp_physnet_list.append(np.array(output["esp_physnet"][0]))
        esp_dcmnet_list.append(np.array(output["esp_dcmnet"][0]))
        esp_charmm_list.append(esp_charmm)
        esp_true_list.append(np.array(batch["esp"][0]))
        esp_grid_list.append(vdw)

    # Convert to arrays
    dipoles_physnet = np.array(dipoles_physnet)
    dipoles_dcmnet = np.array(dipoles_dcmnet)
    dipoles_charmm = np.array(dipoles_charmm)
    dipoles_true = np.array(dipoles_true)

    esp_physnet_flat = np.concatenate([e.reshape(-1) for e in esp_physnet_list])
    esp_dcmnet_flat = np.concatenate([e.reshape(-1) for e in esp_dcmnet_list])
    esp_charmm_flat = np.concatenate([e.reshape(-1) for e in esp_charmm_list])
    esp_true_flat = np.concatenate([e.reshape(-1) for e in esp_true_list])

    # Mask NaN from invalid grid points
    valid_mask = ~np.isnan(esp_charmm_flat) & ~np.isnan(esp_true_flat)
    esp_physnet_valid = esp_physnet_flat[valid_mask]
    esp_dcmnet_valid = esp_dcmnet_flat[valid_mask]
    esp_charmm_valid = esp_charmm_flat[valid_mask]
    esp_true_valid = esp_true_flat[valid_mask]

    # Print metrics
    print("\n" + "=" * 70)
    print("METRICS")
    print("=" * 70)

    mae_dipole_physnet = np.abs(dipoles_physnet - dipoles_true).mean()
    mae_dipole_dcmnet = np.abs(dipoles_dcmnet - dipoles_true).mean()
    mae_dipole_charmm = np.abs(dipoles_charmm - dipoles_true).mean()
    print(f"\nDipole MAE (e·Å):")
    print(f"  PhysNet:  {mae_dipole_physnet:.4f}")
    print(f"  DCMNet:   {mae_dipole_dcmnet:.4f}")
    print(f"  CHARMM:   {mae_dipole_charmm:.4f}")

    def rmse(x, y):
        return np.sqrt(np.mean((x - y) ** 2))

    def r2(x, y):
        ss_res = np.sum((y - x) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0

    rmse_esp_physnet = rmse(esp_physnet_valid, esp_true_valid)
    rmse_esp_dcmnet = rmse(esp_dcmnet_valid, esp_true_valid)
    rmse_esp_charmm = rmse(esp_charmm_valid, esp_true_valid)
    r2_esp_physnet = r2(esp_physnet_valid, esp_true_valid)
    r2_esp_dcmnet = r2(esp_dcmnet_valid, esp_true_valid)
    r2_esp_charmm = r2(esp_charmm_valid, esp_true_valid)

    print(f"\nESP RMSE (Hartree/e):")
    print(f"  PhysNet:  {rmse_esp_physnet:.6f}")
    print(f"  DCMNet:   {rmse_esp_dcmnet:.6f}")
    print(f"  CHARMM:   {rmse_esp_charmm:.6f}")
    print(f"\nESP R²:")
    print(f"  PhysNet:  {r2_esp_physnet:.4f}")
    print(f"  DCMNet:   {r2_esp_dcmnet:.4f}")
    print(f"  CHARMM:   {r2_esp_charmm:.4f}")

    # Plots
    if HAS_MATPLOTLIB:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving plots to {args.out_dir}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Dipole scatter: True vs PhysNet, DCMNet, CHARMM
        ax = axes[0, 0]
        ax.scatter(dipoles_true.flatten(), dipoles_physnet.flatten(), alpha=0.5, s=20, label="PhysNet", color="green")
        ax.scatter(dipoles_true.flatten(), dipoles_dcmnet.flatten(), alpha=0.5, s=20, label="DCMNet", color="purple")
        ax.scatter(dipoles_true.flatten(), dipoles_charmm.flatten(), alpha=0.5, s=20, label="CHARMM", color="orange")
        lims = [
            min(dipoles_true.min(), dipoles_physnet.min(), dipoles_dcmnet.min(), dipoles_charmm.min()),
            max(dipoles_true.max(), dipoles_physnet.max(), dipoles_dcmnet.max(), dipoles_charmm.max()),
        ]
        ax.plot(lims, lims, "r--", alpha=0.5, label="Perfect")
        ax.set_xlabel("True Dipole (e·Å)")
        ax.set_ylabel("Predicted Dipole (e·Å)")
        ax.set_title("Dipole: True vs PhysNet / DCMNet / CHARMM")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ESP scatter: True vs PhysNet, DCMNet, CHARMM
        ax = axes[0, 1]
        ax.scatter(esp_true_valid, esp_physnet_valid, alpha=0.3, s=5, label="PhysNet", color="green")
        ax.scatter(esp_true_valid, esp_dcmnet_valid, alpha=0.3, s=5, label="DCMNet", color="purple")
        ax.scatter(esp_true_valid, esp_charmm_valid, alpha=0.3, s=5, label="CHARMM", color="orange")
        lims = [
            min(esp_true_valid.min(), esp_physnet_valid.min(), esp_dcmnet_valid.min(), esp_charmm_valid.min()),
            max(esp_true_valid.max(), esp_physnet_valid.max(), esp_dcmnet_valid.max(), esp_charmm_valid.max()),
        ]
        ax.plot(lims, lims, "r--", alpha=0.5, label="Perfect")
        ax.set_xlabel("True ESP (Hartree/e)")
        ax.set_ylabel("Predicted ESP (Hartree/e)")
        ax.set_title("ESP: True vs PhysNet / DCMNet / CHARMM")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Dipole error histogram
        ax = axes[1, 0]
        err_phys = (dipoles_physnet - dipoles_true).flatten()
        err_dcm = (dipoles_dcmnet - dipoles_true).flatten()
        err_charmm = (dipoles_charmm - dipoles_true).flatten()
        ax.hist(err_phys, bins=30, alpha=0.5, label=f"PhysNet (MAE={mae_dipole_physnet:.3f})", color="green")
        ax.hist(err_dcm, bins=30, alpha=0.5, label=f"DCMNet (MAE={mae_dipole_dcmnet:.3f})", color="purple")
        ax.hist(err_charmm, bins=30, alpha=0.5, label=f"CHARMM (MAE={mae_dipole_charmm:.3f})", color="orange")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Dipole Error (e·Å)")
        ax.set_ylabel("Count")
        ax.set_title("Dipole Error Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ESP error histogram
        ax = axes[1, 1]
        err_phys_esp = esp_physnet_valid - esp_true_valid
        err_dcm_esp = esp_dcmnet_valid - esp_true_valid
        err_charmm_esp = esp_charmm_valid - esp_true_valid
        ax.hist(err_phys_esp, bins=50, alpha=0.5, label=f"PhysNet (RMSE={rmse_esp_physnet:.4f})", color="green")
        ax.hist(err_dcm_esp, bins=50, alpha=0.5, label=f"DCMNet (RMSE={rmse_esp_dcmnet:.4f})", color="purple")
        ax.hist(err_charmm_esp, bins=50, alpha=0.5, label=f"CHARMM (RMSE={rmse_esp_charmm:.4f})", color="orange")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("ESP Error (Hartree/e)")
        ax.set_ylabel("Count")
        ax.set_title("ESP Error Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.out_dir / "charmm_ml_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {args.out_dir / 'charmm_ml_comparison.png'}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
