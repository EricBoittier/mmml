"""JAX-MD hybrid ML/MM energy function construction and neighbor list utilities."""

import jax
import jax.numpy as jnp
import numpy as np

from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    nonbonded_energy_and_forces,
    _build_pair_indices,
)
from mmml.data.units import KCAL_MOL_TO_EV


def make_monomer_energy_fn(model, params, jax_z, monomer_indices, displacement_fn):
    """Factory creating an intramolecular energy function that evaluates grouped monomers.
    
    Batches evaluations using jax.vmap and unfolds monomer coordinates under periodic boundary 
    conditions (PBC) using the displacement function to prevent bond stretching artifacts.
    """
    from collections import defaultdict
    # Group monomer indices by their size (atom count)
    by_size = defaultdict(list)
    for idx in monomer_indices:
        by_size[len(idx)].append(idx)
        
    # Pre-build size-specific parameters
    size_params = {}
    for size, indices_list in by_size.items():
        stacked_indices = jnp.stack(indices_list)
        n_atoms = size
        dst_idx, src_idx = np.where(~np.eye(n_atoms, dtype=bool))
        dst_idx = jnp.array(dst_idx, dtype=jnp.int32)
        src_idx = jnp.array(src_idx, dtype=jnp.int32)
        group_z = jax_z[stacked_indices]
        
        vmapped_displacement = jax.vmap(
            jax.vmap(displacement_fn, in_axes=(0, None)),
            in_axes=(0, 0)
        )
        
        size_params[size] = {
            "stacked_idx": stacked_indices,
            "dst_idx": dst_idx,
            "src_idx": src_idx,
            "gz": group_z,
            "v_disp": vmapped_displacement,
        }

    def eval_energy_and_forces(r, compute_forces=False):
        tot_energy = 0.0
        tot_forces = jnp.zeros_like(r)
        
        for size, p in size_params.items():
            stacked_idx = p["stacked_idx"]
            dst_idx = p["dst_idx"]
            src_idx = p["src_idx"]
            gz = p["gz"]
            v_disp = p["v_disp"]
            
            group_pos = r[stacked_idx]
            ref_pos = group_pos[:, 0, :]
            displacements = v_disp(group_pos, ref_pos)
            unfolded_pos = ref_pos[:, None, :] + displacements
            com = jnp.mean(unfolded_pos, axis=1, keepdims=True)
            centered_pos = unfolded_pos - com
            
            vmapped_apply = jax.vmap(
                lambda pos, atomic_nums, d_idx=dst_idx, s_idx=src_idx: model.apply(
                    params,
                    atomic_numbers=atomic_nums,
                    positions=pos,
                    dst_idx=d_idx,
                    src_idx=s_idx,
                    compute_forces=compute_forces,
                ),
                in_axes=(0, 0)
            )
            
            outputs = vmapped_apply(centered_pos, gz)
            tot_energy += jnp.sum(outputs["energy"])
            
            if compute_forces:
                group_forces = outputs["forces"]
                tot_forces = tot_forces.at[stacked_idx].add(group_forces)
                
        return tot_energy, tot_forces

    @jax.custom_jvp
    def total_monomer_energy(r):
        energy, _ = eval_energy_and_forces(r, compute_forces=False)
        return energy
        
    @total_monomer_energy.defjvp
    def total_monomer_energy_jvp(primals, tangents):
        r, = primals
        r_dot, = tangents
        r_primal = jax.lax.stop_gradient(r)
        energy, forces = eval_energy_and_forces(r_primal, compute_forces=True)
        return energy, jnp.sum(-forces * r_dot)
        
    return total_monomer_energy



def make_peptide_water_ml_energy_fn(model, params, jax_z, peptide_idx, water_indices, displacement_fn):
    """Factory creating an energy function where peptide-water interactions are computed via ML.
    
    Evaluates peptide-water dimers in parallel using jax.vmap and subtracts redundant peptide
    energies to yield the sum of monomer energies plus peptide-water intermolecular energies.
    """
    # Construct stacked dimer indices of shape (N_waters, 45) containing the peptide and each water
    dimer_indices = [np.concatenate([peptide_idx, idx]) for idx in water_indices]
    stacked_dimer_indices = jnp.stack(dimer_indices)
    
    # 1. Setup dimer evaluation (45 atoms)
    n_atoms_dimer = 45
    dst_idx_dimer, src_idx_dimer = np.where(~np.eye(n_atoms_dimer, dtype=bool))
    dst_idx_dimer = jnp.array(dst_idx_dimer, dtype=jnp.int32)
    src_idx_dimer = jnp.array(src_idx_dimer, dtype=jnp.int32)
    
    dimer_z = jax_z[stacked_dimer_indices]
    
    # Vectorized displacement to calculate relative coordinates of dimer under PBC relative to first atom
    vmapped_displacement_dimer = jax.vmap(
        jax.vmap(displacement_fn, in_axes=(0, None)),
        in_axes=(0, 0)
    )
    
    # 2. Setup peptide evaluation (42 atoms)
    n_atoms_pep = len(peptide_idx)
    dst_idx_pep, src_idx_pep = np.where(~np.eye(n_atoms_pep, dtype=bool))
    dst_idx_pep = jnp.array(dst_idx_pep, dtype=jnp.int32)
    src_idx_pep = jnp.array(src_idx_pep, dtype=jnp.int32)
    
    pep_z = jax_z[peptide_idx]
    
    vmapped_displacement_pep = jax.vmap(displacement_fn, in_axes=(0, None))
    
    n_waters = len(water_indices)
    
    def eval_energy_and_forces(r, compute_forces=False):
        # 1. Dimer evaluation
        group_pos = r[stacked_dimer_indices]
        ref_pos = group_pos[:, 0, :]
        displacements = vmapped_displacement_dimer(group_pos, ref_pos)
        unfolded_dimer_pos = ref_pos[:, None, :] + displacements
        
        # Center dimer coordinates to its COM
        dimer_com = jnp.mean(unfolded_dimer_pos, axis=1, keepdims=True)
        centered_dimer = unfolded_dimer_pos - dimer_com
        
        vmapped_apply_dimer = jax.vmap(
            lambda pos, atomic_nums: model.apply(
                params,
                atomic_numbers=atomic_nums,
                positions=pos,
                dst_idx=dst_idx_dimer,
                src_idx=src_idx_dimer,
                compute_forces=compute_forces,
            ),
            in_axes=(0, 0)
        )
        
        outputs_dimer = vmapped_apply_dimer(centered_dimer, dimer_z)
        e_dimer_sum = jnp.sum(outputs_dimer["energy"])
        
        # 2. Peptide evaluation
        pep_pos = r[peptide_idx]
        ref_pos_pep = pep_pos[0]
        unfolded_pep = ref_pos_pep + vmapped_displacement_pep(pep_pos, ref_pos_pep)
        
        # Center peptide coordinates to its COM
        pep_com = jnp.mean(unfolded_pep, axis=0, keepdims=True)
        centered_pep = unfolded_pep - pep_com
        
        outputs_pep = model.apply(
            params,
            atomic_numbers=pep_z,
            positions=centered_pep,
            dst_idx=dst_idx_pep,
            src_idx=src_idx_pep,
            compute_forces=compute_forces,
        )
        e_pep = jnp.sum(outputs_pep["energy"])
        
        # Subtract (N_waters - 1) * E_peptide to get:
        # E_peptide_ML + sum(E_water_i_ML) + sum(E_inter_peptide_water_ML)
        tot_energy = e_dimer_sum - (n_waters - 1) * e_pep
        
        if compute_forces:
            # Scatter dimer forces back to global coordinates
            tot_forces = jnp.zeros_like(r)
            tot_forces = tot_forces.at[stacked_dimer_indices].add(outputs_dimer["forces"])
            
            # Subtract scaled peptide forces scattered back to peptide_idx
            pep_forces_scaled = (n_waters - 1) * outputs_pep["forces"]
            tot_forces = tot_forces.at[peptide_idx].add(-pep_forces_scaled)
            
            return tot_energy, tot_forces
            
        return tot_energy, None

    @jax.custom_jvp
    def dimer_energy_fn(r):
        energy, _ = eval_energy_and_forces(r, compute_forces=False)
        return energy
        
    @dimer_energy_fn.defjvp
    def dimer_energy_fn_jvp(primals, tangents):
        r, = primals
        r_dot, = tangents
        r_primal = jax.lax.stop_gradient(r)
        energy, forces = eval_energy_and_forces(r_primal, compute_forces=True)
        return energy, jnp.sum(-forces * r_dot)
        
    return dimer_energy_fn


def get_intermolecular_pairs(positions, cell_matrix, excluded, cutoff, mol_id, peptide_water_ml=False):
    """Filters out intramolecular pairs (where molecule IDs match) on the host.
    
    Optionally also filters out peptide-water pairs if they are treated with ML.
    """
    pair_i_raw, pair_j_raw = _build_pair_indices(positions, cell_matrix, excluded, cutoff)
    # Exclude pairs where both atoms belong to the same molecule
    inter = mol_id[pair_i_raw] != mol_id[pair_j_raw]
    
    if peptide_water_ml:
        # Also exclude peptide-water pairs: one is 0 (peptide) and the other is > 0 (water)
        is_pep_i = mol_id[pair_i_raw] == 0
        is_pep_j = mol_id[pair_j_raw] == 0
        pep_wat = (is_pep_i & ~is_pep_j) | (is_pep_j & ~is_pep_i)
        inter = inter & ~pep_wat
        
    return pair_i_raw[inter], pair_j_raw[inter]
