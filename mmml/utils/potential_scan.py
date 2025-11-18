"""
Potential Scan Utilities

This module provides functions for scanning potential energy surfaces,
particularly for dimer systems by varying center-of-mass distances.
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict, List, Optional
import ase
from ase import Atoms
import matplotlib.pyplot as plt
from pathlib import Path


def compute_com(positions: np.ndarray, masses: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute center of mass for a set of positions.
    
    Args:
        positions: Atomic positions (n_atoms, 3)
        masses: Atomic masses (n_atoms,). If None, uses equal masses.
    
    Returns:
        Center of mass position (3,)
    """
    if masses is None:
        return np.mean(positions, axis=0)
    else:
        total_mass = np.sum(masses)
        return np.sum(positions * masses[:, None], axis=0) / total_mass


def generate_dimer_com_scan_trajectory(
    atoms: Atoms,
    n_monomers: int = 2,
    atoms_per_monomer: int = 10,
    distances: Optional[np.ndarray] = None,
    direction: Optional[np.ndarray] = None,
    center_at_origin: bool = True,
) -> List[Atoms]:
    """
    Generate a trajectory by scanning center-of-mass distances between dimers.
    
    This function takes a dimer system and generates a series of configurations
    by translating one dimer relative to another along the COM-COM vector.
    
    Args:
        atoms: ASE Atoms object containing the dimer system
        n_monomers: Number of monomers (should be 2 for dimers)
        atoms_per_monomer: Number of atoms per monomer
        distances: Array of COM distances to scan (in Angstroms).
                   If None, generates a default range.
        direction: Direction vector for translation (3,). If None, uses COM-COM vector.
        center_at_origin: If True, centers the first monomer at origin
    
    Returns:
        List of ASE Atoms objects, one for each distance
    """
    if n_monomers != 2:
        raise ValueError(f"Currently only supports n_monomers=2, got {n_monomers}")
    
    positions = atoms.get_positions()
    n_atoms = len(positions)
    
    if n_atoms != n_monomers * atoms_per_monomer:
        raise ValueError(
            f"Expected {n_monomers * atoms_per_monomer} atoms, got {n_atoms}"
        )
    
    # Split into monomers
    monomer1_positions = positions[:atoms_per_monomer]
    monomer2_positions = positions[atoms_per_monomer:]
    
    # Get atomic numbers and masses
    atomic_numbers = atoms.get_atomic_numbers()
    monomer1_Z = atomic_numbers[:atoms_per_monomer]
    monomer2_Z = atomic_numbers[atoms_per_monomer:]
    
    # Get masses
    masses = atoms.get_masses()
    monomer1_masses = masses[:atoms_per_monomer]
    monomer2_masses = masses[atoms_per_monomer:]
    
    # Compute initial COM positions
    com1_initial = compute_com(monomer1_positions, monomer1_masses)
    com2_initial = compute_com(monomer2_positions, monomer2_masses)
    
    # Compute COM-COM vector
    com_vector = com2_initial - com1_initial
    initial_distance = np.linalg.norm(com_vector)
    
    # Normalize direction
    if direction is None:
        if initial_distance > 1e-6:
            direction = com_vector / initial_distance
        else:
            # Default to x-axis if monomers are at same position
            direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)
    
    # Generate distance array if not provided
    if distances is None:
        # Default: scan from 2.0 to 15.0 Angstroms in 0.5 A steps
        distances = np.arange(2.0, 15.5, 0.5)
    
    # Center first monomer at origin if requested
    if center_at_origin:
        monomer1_positions = monomer1_positions - com1_initial
        com1_initial = np.zeros(3)
    
    trajectory = []
    
    for dist in distances:
        # Compute new COM position for monomer 2
        com2_new = com1_initial + direction * dist
        
        # Translate monomer 2 to new position
        com2_current = compute_com(monomer2_positions, monomer2_masses)
        translation = com2_new - com2_current
        monomer2_positions_new = monomer2_positions + translation
        
        # Combine into full system
        full_positions = np.vstack([monomer1_positions, monomer2_positions_new])
        full_Z = np.concatenate([monomer1_Z, monomer2_Z])
        
        # Create new Atoms object
        new_atoms = Atoms(full_Z, full_positions)
        new_atoms.set_masses(atoms.get_masses())  # Copy masses
        
        trajectory.append(new_atoms)
    
    return trajectory


def scan_potential_with_calculator(
    atoms: Atoms,
    calculator,
    n_monomers: int = 2,
    atoms_per_monomer: int = 10,
    distances: Optional[np.ndarray] = None,
    direction: Optional[np.ndarray] = None,
    extract_outputs: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Scan potential energy surface by varying dimer COM distances.
    
    This function generates a trajectory and computes energies/forces
    using the provided calculator, extracting detailed outputs if available.
    
    Args:
        atoms: Initial ASE Atoms object (dimer system)
        calculator: ASE calculator (should be hybrid MM/ML calculator)
        n_monomers: Number of monomers (should be 2)
        atoms_per_monomer: Number of atoms per monomer
        distances: Array of COM distances to scan. If None, uses default range.
        direction: Direction vector for translation. If None, uses COM-COM vector.
        extract_outputs: If True, tries to extract detailed outputs (ML/MM breakdown)
    
    Returns:
        Dictionary with:
            - "distances": Array of COM distances (Angstroms)
            - "energies": Total energies (eV)
            - "forces_norm": Norm of forces (eV/Angstrom)
            - "ml_energies": ML energies (if available)
            - "mm_energies": MM energies (if available)
            - "internal_energies": Internal ML energies (if available)
            - "ml_2b_energies": ML dimer energies (if available)
    """
    # Generate trajectory
    trajectory = generate_dimer_com_scan_trajectory(
        atoms, n_monomers, atoms_per_monomer, distances, direction
    )
    
    if distances is None:
        distances = np.arange(2.0, 15.5, 0.5)
    
    n_configs = len(trajectory)
    
    # Initialize result arrays
    results = {
        "distances": distances,
        "energies": np.zeros(n_configs),
        "forces_norm": np.zeros(n_configs),
    }
    
    # Optional outputs (if calculator provides them)
    if extract_outputs:
        results["ml_energies"] = np.zeros(n_configs)
        results["mm_energies"] = np.zeros(n_configs)
        results["internal_energies"] = np.zeros(n_configs)
        results["ml_2b_energies"] = np.zeros(n_configs)
    
    # Compute energies and forces for each configuration
    for i, config_atoms in enumerate(trajectory):
        config_atoms.calc = calculator
        
        try:
            # Compute energy
            energy = config_atoms.get_potential_energy()
            results["energies"][i] = energy
            
            # Compute forces
            forces = config_atoms.get_forces()
            results["forces_norm"][i] = np.linalg.norm(forces)
            
            # Try to extract detailed outputs
            if extract_outputs and hasattr(calculator, 'results'):
                calc_results = calculator.results
                
                # Try to get model_output if available
                if "model_output" in calc_results:
                    model_out = calc_results["model_output"]
                    if hasattr(model_out, 'ml_2b_E'):
                        results["ml_2b_energies"][i] = float(model_out.ml_2b_E)
                    if hasattr(model_out, 'internal_E'):
                        results["internal_energies"][i] = float(model_out.internal_E)
                    if hasattr(model_out, 'mm_E'):
                        results["mm_energies"][i] = float(model_out.mm_E)
                    # ML energy = total - MM (or from model_output if available)
                    if hasattr(model_out, 'energy'):
                        total_e = float(model_out.energy)
                        if hasattr(model_out, 'mm_E'):
                            results["ml_energies"][i] = total_e - float(model_out.mm_E)
                        else:
                            results["ml_energies"][i] = total_e
                
        except Exception as e:
            print(f"Warning: Failed to compute energy for distance {distances[i]:.2f} A: {e}")
            results["energies"][i] = np.nan
            results["forces_norm"][i] = np.nan
    
    return results


def plot_potential_scan(
    results: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    show_components: bool = True,
    show_forces: bool = True,
):
    """
    Plot potential energy scan results.
    
    Args:
        results: Dictionary from scan_potential_with_calculator
        save_path: Path to save figure. If None, shows interactively.
        show_components: If True, shows ML/MM breakdown
        show_forces: If True, shows force norm on secondary axis
    """
    distances = results["distances"]
    energies = results["energies"]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot total energy
    ax1.plot(distances, energies, 'k-', linewidth=2, label='Total Energy')
    
    # Plot components if available
    if show_components:
        if "ml_energies" in results and not np.all(np.isnan(results["ml_energies"])):
            ax1.plot(distances, results["ml_energies"], 'b--', linewidth=1.5, label='ML Energy')
        
        if "mm_energies" in results and not np.all(np.isnan(results["mm_energies"])):
            ax1.plot(distances, results["mm_energies"], 'r--', linewidth=1.5, label='MM Energy')
        
        if "internal_energies" in results and not np.all(np.isnan(results["internal_energies"])):
            ax1.plot(distances, results["internal_energies"], 'g--', linewidth=1.5, label='ML Internal')
        
        if "ml_2b_energies" in results and not np.all(np.isnan(results["ml_2b_energies"])):
            ax1.plot(distances, results["ml_2b_energies"], 'm--', linewidth=1.5, label='ML Dimer')
    
    ax1.set_xlabel('COM Distance (Å)', fontsize=12)
    ax1.set_ylabel('Energy (eV)', fontsize=12, color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Plot forces on secondary axis if requested
    if show_forces and "forces_norm" in results:
        ax2 = ax1.twinx()
        ax2.plot(distances, results["forces_norm"], 'orange', linewidth=1.5, linestyle=':', label='Force Norm')
        ax2.set_ylabel('Force Norm (eV/Å)', fontsize=12, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.legend(loc='upper right')
    
    plt.title('Potential Energy Scan: Dimer COM Distance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

