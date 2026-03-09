#!/usr/bin/env python3
"""
Debug ESP calculations - check values and units.
"""

import sys
from pathlib import Path
import numpy as np
import pickle

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from trainer import JointPhysNetDCMNet, prepare_batch_data, load_combined_data, precompute_edge_lists

def debug_esp(checkpoint_path, data_efd, data_esp):
    """Debug ESP calculation and check values."""
    
    print("="*70)
    print("ESP Debug Analysis")
    print("="*70)
    
    # Load checkpoint
    print(f"\n1. Loading checkpoint...")
    with open(checkpoint_path / 'best_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    with open(checkpoint_path / 'model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    print("✅ Loaded")
    
    # Create model
    model = JointPhysNetDCMNet(
        physnet_config=config['physnet_config'],
        dcmnet_config=config['dcmnet_config']
    )
    
    # Load one sample
    print(f"\n2. Loading data...")
    data = load_combined_data(data_efd, data_esp, verbose=False)
    data = precompute_edge_lists(data, cutoff=10.0, verbose=False)
    
    batch = prepare_batch_data(data, np.array([0]), cutoff=10.0)
    
    print(f"   Sample 0:")
    print(f"   N atoms: {int(batch['N'][0])}")
    print(f"   ESP target shape: {batch['esp'][0].shape}")
    print(f"   ESP target range: [{batch['esp'][0].min():.6f}, {batch['esp'][0].max():.6f}] Ha/e")
    print(f"   ESP target mean: {batch['esp'][0].mean():.6f} Ha/e")
    print(f"   VDW surface shape: {batch['vdw_surface'][0].shape}")
    
    # Run model
    print(f"\n3. Running model...")
    output = model.apply(
        params,
        atomic_numbers=batch["Z"],
        positions=batch["R"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=1,
        batch_mask=batch["batch_mask"],
        atom_mask=batch["atom_mask"],
    )
    
    print(f"✅ Model ran")
    
    # Check charges
    print(f"\n4. Checking charges...")
    charges = output['charges_as_mono']
    natoms_model = len(charges)
    n_real = int(batch['N'][0])
    
    print(f"   Charges shape: {charges.shape}")
    print(f"   Charges (real atoms): {np.array(charges[:n_real])}")
    print(f"   Total charge: {np.sum(charges[:n_real]):.6f} e")
    
    # Check distributed multipoles
    print(f"\n5. Checking DCMNet multipoles...")
    mono_dist = output['mono_dist'][:n_real]  # (n_atoms, n_dcm)
    dipo_dist = output['dipo_dist'][:n_real]  # (n_atoms, n_dcm, 3)
    
    print(f"   Distributed monopoles shape: {mono_dist.shape}")
    print(f"   Per-atom monopole sums:")
    for i in range(n_real):
        print(f"     Atom {i}: {np.sum(mono_dist[i]):.6f} e (should ≈ PhysNet charge)")
    
    print(f"\n   Distributed dipole positions shape: {dipo_dist.shape}")
    print(f"   Atomic positions (Å):")
    positions = batch['R'][:n_real]
    for i in range(n_real):
        print(f"     Atom {i}: {np.array(positions[i])}")
    
    # Compute ESP from both methods
    print(f"\n6. Computing ESP from distributed multipoles...")
    from mmml.dcmnet.dcmnet.electrostatics import calc_esp
    
    mono_flat = mono_dist.reshape(-1)
    dipo_flat = jnp.moveaxis(dipo_dist, -1, -2).reshape(-1, 3)
    
    esp_pred_dcmnet = calc_esp(dipo_flat, mono_flat, batch["vdw_surface"][0])
    
    print(f"   ESP prediction range: [{esp_pred_dcmnet.min():.6f}, {esp_pred_dcmnet.max():.6f}] Ha/e")
    print(f"   ESP prediction mean: {esp_pred_dcmnet.mean():.6f} Ha/e")
    
    # Compute ESP from PhysNet point charges
    print(f"\n7. Computing ESP from PhysNet point charges...")
    charges_real = charges[:n_real]
    positions_real = batch['R'][:n_real]
    vdw_grid = batch["vdw_surface"][0]
    
    # Manual calculation
    distances = jnp.linalg.norm(vdw_grid[:, None, :] - positions_real[None, :, :], axis=2)
    esp_pred_physnet = jnp.sum(charges_real[None, :] / (distances + 1e-10), axis=1)
    
    print(f"   ESP prediction range: [{esp_pred_physnet.min():.6f}, {esp_pred_physnet.max():.6f}] Ha/e")
    print(f"   ESP prediction mean: {esp_pred_physnet.mean():.6f} Ha/e")
    
    # Compare
    print(f"\n8. Comparison with target:")
    esp_target = batch['esp'][0]
    
    rmse_dcmnet = np.sqrt(np.mean((esp_pred_dcmnet - esp_target)**2))
    rmse_physnet = np.sqrt(np.mean((esp_pred_physnet - esp_target)**2))
    
    print(f"   Target range: [{esp_target.min():.6f}, {esp_target.max():.6f}] Ha/e")
    print(f"   Target mean: {esp_target.mean():.6f} Ha/e")
    print(f"   ")
    print(f"   DCMNet RMSE: {rmse_dcmnet:.6f} Ha/e ({rmse_dcmnet * 627.509:.2f} kcal/mol/e)")
    print(f"   PhysNet RMSE: {rmse_physnet:.6f} Ha/e ({rmse_physnet * 627.509:.2f} kcal/mol/e)")
    
    # Check if ESP calculation has wrong units
    print(f"\n9. Unit check on calc_esp:")
    print(f"   Testing with q=1e at distance r=1Å:")
    test_charge_pos = jnp.array([[0.0, 0.0, 0.0]])
    test_charge_val = jnp.array([1.0])  # 1 elementary charge
    test_grid = jnp.array([[1.0, 0.0, 0.0]])  # 1 Angstrom away
    
    esp_test = calc_esp(test_charge_pos, test_charge_val, test_grid)
    esp_test_val = float(esp_test[0]) if esp_test.ndim > 0 else float(esp_test)
    print(f"   calc_esp result: {esp_test_val:.6f}")
    print(f"   Expected (atomic units): 1/(1.88973) = 0.529177 Ha/e")
    print(f"   Match: {'✅' if abs(esp_test_val - 0.529177) < 1e-5 else '❌'}")
    
    # Expected in SI: V = (1/4πε₀) × (e/r)
    # In eV: V = 14.4 eV·Å / r = 14.4 eV at r=1Å
    # In Hartree: 14.4 eV / 27.211 = 0.529 Ha
    print(f"   As sanity check: 14.4 eV·Å / 1Å = 14.4 eV = {14.4/27.211:.6f} Ha ✓")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-efd', type=Path, required=True)
    parser.add_argument('--data-esp', type=Path, required=True)
    
    args = parser.parse_args()
    
    debug_esp(args.checkpoint, args.data_efd, args.data_esp)

