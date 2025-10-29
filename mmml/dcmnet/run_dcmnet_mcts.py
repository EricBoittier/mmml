#!/usr/bin/env python3
"""
Quick integration script for DCMNET MCTS optimization.

This script shows how to integrate the MCTS-based model selection
into your existing DCMNET workflow.
"""

import numpy as np
import jax
import jax.numpy as jnp
import sys
import os

# Add the dcmnet directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dcmnet_mcts import optimize_dcmnet_combination, DCMNETSelectionEnv
from dcmnet_ensemble_example import DCMNETEnsembleOptimizer

def quick_optimization_example():
    """
    Quick example of how to use MCTS for DCMNET model selection.
    Replace the example data with your actual molecular data.
    """
    
    print("=== DCMNET MCTS Optimization Example ===\n")
    
    # Example molecular data (REPLACE WITH YOUR ACTUAL DATA)
    print("1. Loading molecular data...")
    molecular_data = {
        'atomic_numbers': jnp.array([6, 1, 1, 1, 1]),  # CH4 example
        'positions': jnp.array([[0.0, 0.0, 0.0],
                               [1.0, 1.0, 1.0],
                               [-1.0, -1.0, 1.0],
                               [-1.0, 1.0, -1.0],
                               [1.0, -1.0, -1.0]]),
        'dst_idx': jnp.array([0, 0, 0, 0]),
        'src_idx': jnp.array([1, 2, 3, 4])
    }
    
    # Example ESP target and VdW surface (REPLACE WITH YOUR ACTUAL DATA)
    esp_target = jnp.random.normal(0, 1, (100,))
    vdw_surface = jnp.random.normal(0, 2, (100, 3))
    
    print("2. Running MCTS optimization...")
    best_selection, best_loss = optimize_dcmnet_combination(
        molecular_data=molecular_data,
        esp_target=esp_target,
        vdw_surface=vdw_surface,
        n_simulations=200,  # Adjust based on your computational budget
        temperature=1.0
    )
    
    print("3. Results:")
    print(f"   Best model selection: {best_selection}")
    print(f"   Best ESP loss: {best_loss:.6f}")
    
    # Show which models were selected
    selected_models = [f"DCM{i+1}" for i in range(7) if best_selection[i]]
    print(f"   Selected models: {selected_models}")
    
    return best_selection, best_loss

def advanced_optimization_example():
    """
    Advanced example using the full DCMNETEnsembleOptimizer class.
    """
    
    print("\n=== Advanced DCMNET Ensemble Optimization ===\n")
    
    # Load your data (replace with actual data loading)
    molecular_data = {
        'atomic_numbers': jnp.array([6, 1, 1, 1, 1]),
        'positions': jnp.array([[0.0, 0.0, 0.0],
                               [1.0, 1.0, 1.0],
                               [-1.0, -1.0, 1.0],
                               [-1.0, 1.0, -1.0],
                               [1.0, -1.0, -1.0]]),
        'dst_idx': jnp.array([0, 0, 0, 0]),
        'src_idx': jnp.array([1, 2, 3, 4])
    }
    
    esp_target = jnp.random.normal(0, 1, (100,))
    vdw_surface = jnp.random.normal(0, 2, (100, 3))
    
    # Initialize optimizer
    optimizer = DCMNETEnsembleOptimizer(
        molecular_data=molecular_data,
        esp_target=esp_target,
        vdw_surface=vdw_surface
    )
    
    # Run optimization with multiple restarts
    print("Running MCTS optimization with multiple restarts...")
    best_selection, best_loss = optimizer.optimize_with_mcts(
        n_simulations=300,
        temperature=1.0,
        n_restarts=3
    )
    
    # Print detailed results
    optimizer.print_results()
    
    return optimizer

def integration_with_your_workflow():
    """
    Example of how to integrate this into your existing DCMNET workflow.
    """
    
    print("\n=== Integration with Your Workflow ===\n")
    
    # Step 1: Load your molecular data
    print("Step 1: Load your molecular data")
    print("   - atomic_numbers: atomic numbers for each atom")
    print("   - positions: 3D coordinates of atoms")
    print("   - dst_idx, src_idx: connectivity information")
    print("   - esp_target: target ESP values")
    print("   - vdw_surface: VdW surface points")
    
    # Step 2: Run MCTS optimization
    print("\nStep 2: Run MCTS optimization")
    print("   - This will find the best combination of DCM1-DCM7 models")
    print("   - Adjust n_simulations based on your computational budget")
    print("   - Use multiple restarts for better results")
    
    # Step 3: Use the results
    print("\nStep 3: Use the optimized combination")
    print("   - The best_selection array tells you which models to use")
    print("   - Combine predictions from selected models")
    print("   - Use this combination for your ESP calculations")
    
    # Example code snippet
    print("\nExample code snippet:")
    print("""
    # Your data loading code here
    molecular_data = load_your_molecular_data()
    esp_target = load_your_esp_target()
    vdw_surface = load_your_vdw_surface()
    
    # Run MCTS optimization
    optimizer = DCMNETEnsembleOptimizer(molecular_data, esp_target, vdw_surface)
    best_selection, best_loss = optimizer.optimize_with_mcts(
        n_simulations=1000,  # Adjust based on your needs
        n_restarts=5
    )
    
    # Use the results
    selected_models = optimizer.get_best_models()
    print(f"Use these models: {selected_models}")
    """)

def main():
    """Main function demonstrating the MCTS optimization."""
    
    try:
        # Quick example
        quick_optimization_example()
        
        # Advanced example
        optimizer = advanced_optimization_example()
        
        # Integration guide
        integration_with_your_workflow()
        
        print("\n=== Summary ===")
        print("The MCTS optimization has been successfully adapted for DCMNET!")
        print("Key features:")
        print("- Finds optimal combinations of DCM1-DCM7 models")
        print("- Minimizes ESP loss through intelligent search")
        print("- Handles missing models gracefully")
        print("- Provides detailed optimization history")
        print("- Easy to integrate with existing workflows")
        
    except Exception as e:
        print(f"Error running example: {e}")
        print("Make sure you have all the required dependencies installed.")
        print("You may need to adjust the import paths based on your setup.")

if __name__ == "__main__":
    main()
