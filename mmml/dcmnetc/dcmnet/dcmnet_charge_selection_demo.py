#!/usr/bin/env python3
"""
Corrected DCMNET MCTS Example - Charge-Level Selection

This script demonstrates the corrected approach where we select individual
charges from different DCMNET models, not entire models.
"""

import numpy as np
import jax
import jax.numpy as jnp
import sys
import os

# Add the dcmnet directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dcmnet_mcts import optimize_dcmnet_combination, DCMNETSelectionEnv

def create_example_data():
    """Create example data for demonstration."""
    
    # Example molecular data (CH4)
    molecular_data = {
        'atomic_numbers': jnp.array([6, 1, 1, 1, 1]),  # C, H, H, H, H
        'positions': jnp.array([[0.0, 0.0, 0.0],      # C at origin
                               [1.0, 1.0, 1.0],        # H1
                               [-1.0, -1.0, 1.0],      # H2
                               [-1.0, 1.0, -1.0],      # H3
                               [1.0, -1.0, -1.0]]),   # H4
        'dst_idx': jnp.array([0, 0, 0, 0]),            # All H atoms connected to C
        'src_idx': jnp.array([1, 2, 3, 4])             # H atom indices
    }
    
    # Example ESP target and VdW surface
    esp_target = jnp.random.normal(0, 1, (100,))
    vdw_surface = jnp.random.normal(0, 2, (100, 3))
    
    # Example model charges - each model predicts different numbers of charges per atom
    n_atoms = len(molecular_data['atomic_numbers'])
    model_charges = {
        0: jnp.array([[0.5], [0.1], [0.1], [0.1], [0.1]]),  # DCM1: 1 charge per atom
        1: jnp.array([[0.3, 0.2], [0.05, 0.05], [0.05, 0.05], [0.05, 0.05], [0.05, 0.05]]),  # DCM2: 2 charges per atom
        2: jnp.array([[0.2, 0.15, 0.15], [0.03, 0.03, 0.04], [0.03, 0.03, 0.04], [0.03, 0.03, 0.04], [0.03, 0.03, 0.04]]),  # DCM3: 3 charges per atom
        3: jnp.array([[0.15, 0.1, 0.1, 0.15], [0.025, 0.025, 0.025, 0.025], [0.025, 0.025, 0.025, 0.025], [0.025, 0.025, 0.025, 0.025], [0.025, 0.025, 0.025, 0.025]]),  # DCM4: 4 charges per atom
    }
    
    return molecular_data, esp_target, vdw_surface, model_charges

def demonstrate_charge_selection():
    """Demonstrate the charge-level selection approach."""
    
    print("=== DCMNET Charge-Level Selection Demo ===\n")
    
    # Create example data
    molecular_data, esp_target, vdw_surface, model_charges = create_example_data()
    
    print("1. Molecular System:")
    print(f"   - {len(molecular_data['atomic_numbers'])} atoms")
    print(f"   - Atomic numbers: {molecular_data['atomic_numbers']}")
    print(f"   - Positions shape: {molecular_data['positions'].shape}")
    
    print("\n2. Available Models and Charges:")
    total_charges = 0
    for model_id, charges in model_charges.items():
        n_charges = charges.shape[1]
        total_charges += n_charges
        print(f"   - DCM{model_id+1}: {n_charges} charges per atom")
        print(f"     Example charges for atom 0: {charges[0].tolist()}")
    
    print(f"\n   Total charges per atom across all models: {total_charges}")
    print(f"   Total possible charge combinations: {total_charges ** len(molecular_data['atomic_numbers'])}")
    
    print("\n3. Creating DCMNET Selection Environment...")
    env = DCMNETSelectionEnv(molecular_data, esp_target, vdw_surface, model_charges)
    
    print(f"   - Environment state shape: {env.selected_charges.shape}")
    print(f"   - Charge mapping: {env.charge_mapping}")
    print(f"   - Legal actions: {len(env.legal_actions())} possible (atom_idx, charge_idx) pairs")
    
    print("\n4. Running MCTS Optimization...")
    best_selection, best_loss = optimize_dcmnet_combination(
        molecular_data=molecular_data,
        esp_target=esp_target,
        vdw_surface=vdw_surface,
        model_charges=model_charges,
        n_simulations=100,  # Small number for demo
        temperature=1.0
    )
    
    print(f"\n5. Results:")
    print(f"   - Best ESP loss: {best_loss:.6f}")
    print(f"   - Selection matrix shape: {best_selection.shape}")
    
    print(f"\n6. Selected Charges per Atom:")
    for atom_idx in range(len(molecular_data['atomic_numbers'])):
        selected_charges = np.where(best_selection[atom_idx])[0]
        print(f"   Atom {atom_idx} (atomic number {molecular_data['atomic_numbers'][atom_idx]}):")
        print(f"     Selected charge indices: {selected_charges.tolist()}")
        
        # Show which models these charges come from
        for charge_idx in selected_charges:
            model_id, charge_within_model = env.charge_mapping[charge_idx]
            charge_value = model_charges[model_id][atom_idx, charge_within_model]
            print(f"       Charge {charge_idx} -> DCM{model_id+1}, charge {charge_within_model} = {charge_value:.3f}")
    
    print(f"\n7. Summary:")
    total_selected = np.sum(best_selection)
    print(f"   - Total charges selected: {total_selected}")
    print(f"   - Average charges per atom: {total_selected / len(molecular_data['atomic_numbers']):.1f}")
    
    # Show which models are being used
    used_models = set()
    for atom_idx in range(len(molecular_data['atomic_numbers'])):
        for charge_idx in np.where(best_selection[atom_idx])[0]:
            model_id, _ = env.charge_mapping[charge_idx]
            used_models.add(model_id)
    
    print(f"   - Models being used: DCM{[m+1 for m in sorted(used_models)]}")

def compare_with_model_level_selection():
    """Compare charge-level selection with model-level selection."""
    
    print("\n=== Comparison: Charge-Level vs Model-Level Selection ===\n")
    
    molecular_data, esp_target, vdw_surface, model_charges = create_example_data()
    
    print("Charge-Level Selection (Current Approach):")
    print("- Select individual charges from different models")
    print("- Can mix charges from DCM1, DCM2, DCM3, DCM4 for each atom")
    print("- More granular control over charge selection")
    print("- Larger search space but potentially better results")
    
    print("\nModel-Level Selection (Previous Approach):")
    print("- Select entire models (all charges from DCM1, or all from DCM2, etc.)")
    print("- Simpler but less flexible")
    print("- Smaller search space but may miss optimal combinations")
    
    print("\nExample:")
    print("Charge-level: Atom 0 could use charge 0 from DCM1 + charge 1 from DCM3")
    print("Model-level: Atom 0 would use ALL charges from DCM1 OR ALL from DCM3")
    
    print("\nThe charge-level approach allows for more sophisticated combinations!")

def main():
    """Main demonstration function."""
    
    try:
        demonstrate_charge_selection()
        compare_with_model_level_selection()
        
        print("\n=== Key Points ===")
        print("✓ Fixed conceptual mistake: now selecting individual charges, not entire models")
        print("✓ Feature representation: (n_atoms * total_charges_per_atom) + step + esp_loss")
        print("✓ Actions: (atom_idx, charge_idx) pairs for toggling charge selection")
        print("✓ State: Binary matrix of shape (n_atoms, total_charges_per_atom)")
        print("✓ Goal: Find optimal combination of individual charges to minimize ESP loss")
        
    except Exception as e:
        print(f"Error running demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
