from ase.io import read
import numpy as np


def save_traj_to_npz(traj_file, output_file, FORCES=True, ENERGY=True):
    """Efficiently saves ASE trajectory data to compressed NPZ format."""
    positions_list = []
    atomic_numbers = None  # Assume all structures have the same atomic numbers
    cell_list = []
    forces_list = []
    energies_list = []
    N_list = []
    for i, atoms in enumerate(read(traj_file, index=':')):  # Load one by one
        positions_list.append(atoms.get_positions())
        cell_list.append(atoms.get_cell())
        forces_list.append(atoms.get_forces() if FORCES else None)
        energies_list.append(atoms.get_potential_energy() if ENERGY else None)
        N_list.append(len(atoms))
        if atomic_numbers is None:
            atomic_numbers = atoms.get_atomic_numbers()  # Same for all frames

    # Convert lists to NumPy arrays
    positions_array = np.array(positions_list)
    cell_array = np.array(cell_list)
    forces_array = np.array(forces_list) 
    energies_array = np.array(energies_list) 
    N_array = np.array(N_list)
    # Save in compressed format
    np.savez_compressed(output_file, 
                        R=positions_array,
                        Z=atomic_numbers,
                        cell=cell_array,
                        F=forces_array,
                        E=energies_array,
                        N=N_array)









