"""JAX-MD hybrid ML/MM energy function construction and neighbor list utilities."""

import jax
import jax.numpy as jnp
import numpy as np

from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    nonbonded_energy_and_forces,
    _build_pair_indices,
)
from mmml.data.units import KCAL_MOL_TO_EV


def make_monomer_energy_fn(model, params, jax_z, monomer_indices, displacement_fn, charges=None, spins=None):
    """Factory creating an intramolecular energy function that evaluates grouped monomers.
    
    Batches evaluations using jax.vmap and unfolds monomer coordinates under periodic boundary 
    conditions (PBC) using the displacement function to prevent bond stretching artifacts.
    """
    from collections import defaultdict
    # Group monomer indices by their size (atom count)
    by_size = defaultdict(list)
    for idx in monomer_indices:
        by_size[len(idx)].append(idx)
    import inspect
    def _model_accepts_kwarg(model, name):
        fn = getattr(model, "__call__", None)
        if fn is None:
            return False
        try:
            return name in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False
    
    supports_charges = _model_accepts_kwarg(model, "charges")
    supports_spins = _model_accepts_kwarg(model, "spins")
    is_spooky = supports_charges and supports_spins
    # old check: is_spooky = (
    #     hasattr(model, "charges")
    #     and hasattr(model, "total_charge")
    # ) or "spooky" in str(type(model)).lower()

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
        
        # Determine charge and spin for this size
        if isinstance(charges, dict):
            size_q = float(charges.get(size, 0.0))
        else:
            size_q = 0.0 if size == 3 else float(charges or 0.0)

        if isinstance(spins, dict):
            size_s = float(spins.get(size, 1.0))
        else:
            size_s = 1.0 if size == 3 else float(spins or 1.0)
        
        size_params[size] = {
            "stacked_idx": stacked_indices,
            "dst_idx": dst_idx,
            "src_idx": src_idx,
            "gz": group_z,
            "v_disp": vmapped_displacement,
            "q": size_q,
            "s": size_s,
        }

    def total_monomer_energy(r):
        tot_energy = 0.0
        
        for size, p in size_params.items():
            stacked_idx = p["stacked_idx"]
            dst_idx = p["dst_idx"]
            src_idx = p["src_idx"]
            gz = p["gz"]
            v_disp = p["v_disp"]
            size_q = p["q"]
            size_s = p["s"]
            
            group_pos = r[stacked_idx]
            ref_pos = group_pos[:, 0, :]
            displacements = v_disp(group_pos, ref_pos)
            unfolded_pos = ref_pos[:, None, :] + displacements
            com = jnp.mean(unfolded_pos, axis=1, keepdims=True)
            centered_pos = unfolded_pos - com
            
            n_atoms = size
            if is_spooky:
                vmapped_apply = jax.vmap(
                    lambda pos, atomic_nums, d_idx=dst_idx, s_idx=src_idx: model.apply(
                        params,
                        atomic_numbers=atomic_nums,
                        positions=pos,
                        charges=jnp.full((n_atoms, 1), size_q, dtype=jnp.float32),
                        spins=jnp.full((n_atoms, 1), size_s, dtype=jnp.float32),
                        dst_idx=d_idx,
                        src_idx=s_idx,
                        compute_forces=False,
                    ),
                    in_axes=(0, 0)
                )
            else:
                vmapped_apply = jax.vmap(
                    lambda pos, atomic_nums, d_idx=dst_idx, s_idx=src_idx: model.apply(
                        params,
                        atomic_numbers=atomic_nums,
                        positions=pos,
                        dst_idx=d_idx,
                        src_idx=s_idx,
                        compute_forces=False,
                    ),
                    in_axes=(0, 0)
                )
            
            outputs = vmapped_apply(centered_pos, gz)
            tot_energy += jnp.sum(outputs["energy"])

        return tot_energy
        
    return total_monomer_energy



def make_peptide_water_ml_energy_fn(model, params, jax_z, peptide_idx, water_indices, displacement_fn, charges=None, spins=None):
    """Factory creating an energy function where peptide-water interactions are computed via ML.
    
    Evaluates peptide-water dimers in parallel using jax.vmap and subtracts redundant peptide
    energies to yield the sum of monomer energies plus peptide-water intermolecular energies.
    """
    peptide_idx = jnp.asarray(peptide_idx, dtype=jnp.int32)
    water_indices = [jnp.asarray(idx, dtype=jnp.int32) for idx in water_indices]
    n_waters = len(water_indices)
    n_atoms_pep = int(peptide_idx.shape[0])
    n_atoms_water = int(water_indices[0].shape[0]) if n_waters else 0
    n_atoms_dimer = n_atoms_pep + n_atoms_water

    # 1. Setup peptide-water dimer evaluation.
    if n_waters:
        dimer_indices = [jnp.concatenate([peptide_idx, idx]) for idx in water_indices]
        stacked_dimer_indices = jnp.stack(dimer_indices)
        dimer_z = jax_z[stacked_dimer_indices]
    else:
        stacked_dimer_indices = jnp.zeros((0, n_atoms_dimer), dtype=jnp.int32)
        dimer_z = jnp.zeros((0, n_atoms_dimer), dtype=jnp.int32)

    dst_idx_dimer, src_idx_dimer = np.where(~np.eye(n_atoms_dimer, dtype=bool))
    dst_idx_dimer = jnp.array(dst_idx_dimer, dtype=jnp.int32)
    src_idx_dimer = jnp.array(src_idx_dimer, dtype=jnp.int32)
    
    # Vectorized displacement to calculate relative coordinates of dimer under PBC relative to first atom
    vmapped_displacement_dimer = jax.vmap(
        jax.vmap(displacement_fn, in_axes=(0, None)),
        in_axes=(0, 0)
    )
    
    # 2. Setup peptide evaluation.
    dst_idx_pep, src_idx_pep = np.where(~np.eye(n_atoms_pep, dtype=bool))
    dst_idx_pep = jnp.array(dst_idx_pep, dtype=jnp.int32)
    src_idx_pep = jnp.array(src_idx_pep, dtype=jnp.int32)
    
    pep_z = jax_z[peptide_idx]
    
    vmapped_displacement_pep = jax.vmap(displacement_fn, in_axes=(0, None))
    
    import inspect

    def _model_accepts_kwarg(model, name):
        fn = getattr(model, "__call__", None)
        if fn is None:
            return False
        try:
            return name in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False
            
    supports_charges = _model_accepts_kwarg(model, "charges")
    supports_spins = _model_accepts_kwarg(model, "spins")
    is_spooky = supports_charges and supports_spins
    # is_spooky = (
    #     hasattr(model, "charges")
    #     and hasattr(model, "total_charge")
    # ) or "spooky" in str(type(model)).lower()

    if isinstance(charges, dict):
        dimer_q = float(charges.get(n_atoms_dimer, charges.get(n_atoms_pep, 0.0)))
        pep_q = float(charges.get(n_atoms_pep, 0.0))
    else:
        dimer_q = float(charges or 0.0)
        pep_q = float(charges or 0.0)

    if isinstance(spins, dict):
        dimer_s = float(spins.get(n_atoms_dimer, spins.get(n_atoms_pep, 1.0)))
        pep_s = float(spins.get(n_atoms_pep, 1.0))
    else:
        dimer_s = float(spins or 1.0)
        pep_s = float(spins or 1.0)

    if is_spooky:
        vmapped_apply_dimer = jax.vmap(
            lambda pos, atomic_nums: model.apply(
                params,
                atomic_numbers=atomic_nums,
                positions=pos,
                charges=jnp.full((n_atoms_dimer, 1), dimer_q, dtype=jnp.float32),
                spins=jnp.full((n_atoms_dimer, 1), dimer_s, dtype=jnp.float32),
                dst_idx=dst_idx_dimer,
                src_idx=src_idx_dimer,
                compute_forces=False,
            ),
            in_axes=(0, 0)
        )

        def apply_peptide(pos):
            return model.apply(
                params,
                atomic_numbers=pep_z,
                positions=pos,
                charges=jnp.full((n_atoms_pep, 1), pep_q, dtype=jnp.float32),
                spins=jnp.full((n_atoms_pep, 1), pep_s, dtype=jnp.float32),
                dst_idx=dst_idx_pep,
                src_idx=src_idx_pep,
                compute_forces=False,
            )
    else:
        vmapped_apply_dimer = jax.vmap(
            lambda pos, atomic_nums: model.apply(
                params,
                atomic_numbers=atomic_nums,
                positions=pos,
                dst_idx=dst_idx_dimer,
                src_idx=src_idx_dimer,
                compute_forces=False,
            ),
            in_axes=(0, 0)
        )

        def apply_peptide(pos):
            return model.apply(
                params,
                atomic_numbers=pep_z,
                positions=pos,
                dst_idx=dst_idx_pep,
                src_idx=src_idx_pep,
                compute_forces=False,
            )
    
    def dimer_energy_fn(r):
        # Always evaluate the peptide monomer. With no waters this function
        # reduces to the peptide ML energy, which keeps gas-phase setups valid.
        pep_pos = r[peptide_idx]
        ref_pos_pep = pep_pos[0]
        unfolded_pep = ref_pos_pep + vmapped_displacement_pep(pep_pos, ref_pos_pep)
        pep_com = jnp.mean(unfolded_pep, axis=0, keepdims=True)
        centered_pep = unfolded_pep - pep_com
        outputs_pep = apply_peptide(centered_pep)
        e_pep = jnp.sum(outputs_pep["energy"])

        if n_waters == 0:
            return e_pep

        # 1. Dimer evaluation
        group_pos = r[stacked_dimer_indices]
        ref_pos = group_pos[:, 0, :]
        displacements = vmapped_displacement_dimer(group_pos, ref_pos)
        unfolded_dimer_pos = ref_pos[:, None, :] + displacements
        
        # Center dimer coordinates to its COM
        dimer_com = jnp.mean(unfolded_dimer_pos, axis=1, keepdims=True)
        centered_dimer = unfolded_dimer_pos - dimer_com
        
        outputs_dimer = vmapped_apply_dimer(centered_dimer, dimer_z)
        e_dimer_sum = jnp.sum(outputs_dimer["energy"])

        # Subtract (N_waters - 1) * E_peptide to get:
        # E_peptide_ML + sum(E_water_i_ML) + sum(E_inter_peptide_water_ML)
        return e_dimer_sum - (n_waters - 1) * e_pep
        
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
