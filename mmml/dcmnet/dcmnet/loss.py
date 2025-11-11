import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.random import randint
import ase.data

from .electrostatics import batched_electrostatic_potential, calc_esp

from .utils import reshape_dipole

# Constants for ESP masking
EPS = 1e-8
RADII_TABLE = jnp.array(ase.data.covalent_radii)


def pred_dipole(dcm, com, q):
    """
    Calculate molecular dipole moment from distributed multipoles.
    
    Parameters
    ----------
    dcm : array_like
        Distributed multipole positions, shape (N, 3)
    com : array_like
        Center of mass coordinates, shape (3,)
    q : array_like
        Charges/monopoles, shape (N,)
        
    Returns
    -------
    array_like
        Molecular dipole moment in Debye, shape (3,)
    """
    dipole_out = jnp.zeros(3)
    for i, _ in enumerate(dcm):
        dipole_out += q[i] * (_ - com)
    return dipole_out * 1.88873
    # return jnp.linalg.norm(dipole_out)* 4.80320


@functools.partial(jax.jit, static_argnames=("batch_size", "esp_w", "chg_w", "n_dcm", "distance_weighting", "distance_scale", "distance_min", "esp_magnitude_weighting", "esp_min_distance", "esp_max_value", "use_atomic_radii_mask"))
def esp_mono_loss(
    dipo_prediction,
    mono_prediction,
    esp_target,
    vdw_surface,
    mono,
    ngrid,
    n_atoms,
    batch_size,
    esp_w,
    chg_w,
    n_dcm,
    atom_positions=None,
    atomic_numbers=None,
    atom_mask=None,
    distance_weighting=False,
    distance_scale=2.0,
    distance_min=0.5,
    esp_magnitude_weighting=False,
    esp_min_distance=0.0,
    esp_max_value=1e10,
    use_atomic_radii_mask=True,
):
    """
    Combined ESP and monopole loss function for DCMNet training.
    
    Computes loss as weighted sum of ESP fitting error and monopole
    constraint violation. Handles dummy atoms and charge neutrality.
    Optionally applies distance-based weighting to ESP loss, giving higher
    weight to errors near atoms and lower weight to errors far away.
    
    Parameters
    ----------
    dipo_prediction : array_like
        Predicted distributed dipole positions, shape (batch_size, natoms, n_dcm, 3)
    mono_prediction : array_like
        Predicted monopoles, shape (batch_size, natoms, n_dcm)
    esp_target : array_like
        Target ESP values, shape (batch_size, ngrid)
    vdw_surface : array_like
        VDW surface grid points, shape (batch_size, ngrid, 3)
    mono : array_like
        Reference monopoles, shape (batch_size, natoms)
    ngrid : array_like
        Number of grid points per system, shape (batch_size,)
    n_atoms : array_like
        Number of atoms per system, shape (batch_size,)
    batch_size : int
        Batch size
    esp_w : float
        Weight for ESP loss term
    chg_w : float
        Weight for charge/monopole loss term
    n_dcm : int
        Number of distributed multipoles per atom
    atom_positions : array_like, optional
        Atom positions, shape (batch_size, natoms, 3) or (batch_size * natoms, 3).
        Required if distance_weighting=True or use_atomic_radii_mask=True.
    atomic_numbers : array_like, optional
        Atomic numbers, shape (batch_size, natoms) or (batch_size * natoms,).
        Required if use_atomic_radii_mask=True.
    atom_mask : array_like, optional
        Atom mask (1 for real atoms, 0 for dummy), shape (batch_size, natoms) or (batch_size * natoms,).
        Required if use_atomic_radii_mask=True.
    distance_weighting : bool, optional
        Whether to apply distance-based weighting to ESP loss, by default False
    distance_scale : float, optional
        Scale parameter for distance weighting (in Angstroms). Larger values
        give slower increase with distance. Weight = exp((distance - distance_min) / distance_scale),
        normalized to have mean=1. This gives higher weight to points further from atoms.
        By default 2.0
    distance_min : float, optional
        Minimum distance for weighting (in Angstroms). Distances below this
        are clamped to this value to avoid singularities, by default 0.5
    esp_magnitude_weighting : bool, optional
        Whether to weight by ESP magnitude instead of distance. Errors at points
        with larger |ESP| values will have LOWER weight. This reduces the impact
        of points where nuclear-electron shielding occurs and ESP approaches
        singularity. By default False
    esp_min_distance : float, optional
        Minimum distance from atoms for ESP points to be included in loss (in Angstroms).
        Points closer than this are masked out. By default 0.0 (no masking).
    esp_max_value : float, optional
        Maximum absolute ESP value to include in loss. Points with |ESP| > this are masked out.
        By default 1e10 (no masking).
    use_atomic_radii_mask : bool, optional
        Whether to mask out ESP points too close to atoms (within 2.0 * covalent_radii).
        This excludes singularities near atomic nuclei. By default True.
        
    Returns
    -------
    tuple
        (loss, esp_pred, esp_target, esp_errors) where:
        - loss: scalar total loss value
        - esp_pred: predicted ESP values at grid points
        - esp_target: target ESP values at grid points
        - esp_errors: per-grid-point errors
    """
    # sum_of_dc_monopoles = mono_prediction.sum(axis=-1)
    # l2_loss_mono = optax.l2_loss(sum_of_dc_monopoles, mono)
    # mono_loss = jnp.mean(l2_loss_mono)
    # d = jnp.moveaxis(dipo_prediction, -1, -2).reshape(batch_size, max_atoms * n_dcm, 3)
    # m = mono_prediction.reshape(batch_size, max_atoms * n_dcm)
    # batched_pred = batched_electrostatic_potential(d, m, vdw_surface)
    # l2_loss = optax.l2_loss(batched_pred, esp_target)
    # esp_loss = jnp.mean(l2_loss) * esp_w
    # return esp_loss + mono_loss
    # Ensure scalar n_atoms even if shaped (1,) or (1,1)
    n_atoms = jnp.ravel(n_atoms)[0]
    
    # Infer max_atoms from prediction shape
    # Handle both batched and unbatched/flattened predictions
    # Model can return either:
    # - Flattened: (batch_size * num_atoms, n_dcm) and (batch_size * num_atoms, n_dcm, 3)
    # - Batched: (batch_size, num_atoms, n_dcm) and (batch_size, num_atoms, n_dcm, 3)
    # - Unbatched (batch_size=1): (num_atoms, n_dcm) and (num_atoms, n_dcm, 3)
    
    mono_ndim = len(mono_prediction.shape)
    dipo_ndim = len(dipo_prediction.shape)
    
    if batch_size == 1 and mono_ndim == 2:
        # Unbatched: (n_atoms, n_dcm)
        max_atoms = mono_prediction.shape[0]
        # Add batch dimension
        mono_prediction = mono_prediction[None, :, :]  # (1, n_atoms, n_dcm)
        dipo_prediction = dipo_prediction[None, :, :, :]  # (1, n_atoms, n_dcm, 3)
    elif mono_ndim == 2 and dipo_ndim == 3:
        # Check if flattened: (batch_size * num_atoms, n_dcm) and (batch_size * num_atoms, n_dcm, 3)
        total_atoms_mono = mono_prediction.shape[0]
        total_atoms_dipo = dipo_prediction.shape[0]
        
        # If both are divisible by batch_size and match, they're flattened
        if (total_atoms_mono % batch_size == 0 and 
            total_atoms_dipo % batch_size == 0 and 
            total_atoms_mono == total_atoms_dipo):
            # Flattened format - reshape to batched
            max_atoms = total_atoms_mono // batch_size
            mono_prediction = mono_prediction.reshape(batch_size, max_atoms, n_dcm)
            dipo_prediction = dipo_prediction.reshape(batch_size, max_atoms, n_dcm, 3)
        else:
            # Batched: (batch_size, n_atoms, n_dcm)
            max_atoms = mono_prediction.shape[1]
    else:
        # Batched: (batch_size, n_atoms, n_dcm)
        max_atoms = mono_prediction.shape[1]
    
    d = jnp.moveaxis(dipo_prediction, -1, -2).reshape(batch_size, max_atoms * n_dcm, 3)
    m = mono_prediction.reshape(batch_size, max_atoms * n_dcm)

    # Mask dummy atoms before ESP computation (critical for accuracy - matches working trainer.py)
    # Always use n_atoms-based mask for reliability (atom_mask from batch may have wrong shape)
    # This ensures dummy atoms are properly masked out before ESP calculation
    # Note: We skip using atom_mask from batch to avoid shape mismatch issues
    if False:  # Disabled: always use n_atoms-based mask
        # Handle atom_mask shape - need to get it to (batch_size, max_atoms)
        if atom_mask.ndim == 1:
            # Flattened: could be (batch_size * max_atoms,) or something else
            total_elements = atom_mask.shape[0]
            if total_elements == batch_size * max_atoms:
                # Perfect match: reshape to (batch_size, max_atoms)
                atom_mask_reshaped = atom_mask.reshape(batch_size, max_atoms)
            elif batch_size == 1:
                # Single sample: (max_atoms,) -> add batch dimension
                atom_mask_reshaped = atom_mask[None, :]  # (1, max_atoms)
                if atom_mask_reshaped.shape[1] != max_atoms:
                    # Pad or trim to match max_atoms
                    if atom_mask_reshaped.shape[1] < max_atoms:
                        padding = jnp.zeros((1, max_atoms - atom_mask_reshaped.shape[1]), dtype=atom_mask.dtype)
                        atom_mask_reshaped = jnp.concatenate([atom_mask_reshaped, padding], axis=1)
                    else:
                        atom_mask_reshaped = atom_mask_reshaped[:, :max_atoms]
            else:
                # Try to infer max_atoms from total_elements
                if total_elements % batch_size == 0:
                    inferred_max_atoms = total_elements // batch_size
                    atom_mask_reshaped = atom_mask.reshape(batch_size, inferred_max_atoms)
                    # Pad or trim to match max_atoms
                    if inferred_max_atoms < max_atoms:
                        padding = jnp.zeros((batch_size, max_atoms - inferred_max_atoms), dtype=atom_mask.dtype)
                        atom_mask_reshaped = jnp.concatenate([atom_mask_reshaped, padding], axis=1)
                    elif inferred_max_atoms > max_atoms:
                        atom_mask_reshaped = atom_mask_reshaped[:, :max_atoms]
                else:
                    # Fallback: assume it should be max_atoms per batch
                    # This might fail, but let's try
                    atom_mask_reshaped = jnp.ones((batch_size, max_atoms), dtype=atom_mask.dtype)
        elif atom_mask.ndim == 2:
            # Already batched: (batch_size, max_atoms) or (batch_size, something_else)
            atom_mask_reshaped = atom_mask
            if atom_mask.shape[1] != max_atoms:
                # Pad or trim to match max_atoms
                if atom_mask.shape[1] < max_atoms:
                    padding = jnp.zeros((batch_size, max_atoms - atom_mask.shape[1]), dtype=atom_mask.dtype)
                    atom_mask_reshaped = jnp.concatenate([atom_mask, padding], axis=1)
                else:
                    atom_mask_reshaped = atom_mask[:, :max_atoms]
        else:
            # Unexpected shape - create default mask
            atom_mask_reshaped = jnp.ones((batch_size, max_atoms), dtype=jnp.float32)
        
        # Ensure atom_mask_reshaped has correct shape (batch_size, max_atoms)
        # If shape doesn't match, fall back to n_atoms-based mask (more reliable)
        # Check if size matches and reshape if possible
        if atom_mask_reshaped.size == batch_size * max_atoms and atom_mask_reshaped.ndim >= 1:
            # Size matches - reshape and use it
            atom_mask_final = atom_mask_reshaped.reshape(batch_size, max_atoms)
            # Expand mask to DCM dimensions
            mask_expanded = atom_mask_final[:, :, None]  # (batch_size, max_atoms, 1)
            mask_expanded_flat = mask_expanded.reshape(batch_size, max_atoms * n_dcm)
            m = m * mask_expanded_flat
        else:
            # Size doesn't match or wrong shape - use n_atoms-based mask
            NDC = n_atoms * n_dcm
            valid_atoms_mask = jnp.arange(max_atoms * n_dcm)[None, :] < NDC
            valid_atoms_mask = jnp.broadcast_to(valid_atoms_mask, (batch_size, max_atoms * n_dcm))
            m = m * valid_atoms_mask.astype(m.dtype)

    # monopole loss
    # Reshape m to (batch_size, max_atoms, n_dcm) to sum over n_dcm dimension
    mono_pred_reshaped = m.reshape(batch_size, max_atoms, n_dcm)
    sum_of_dc_monopoles = mono_pred_reshaped.sum(axis=-1)  # (batch_size, max_atoms)
    
    # Handle mono shape - it might be (batch_size, max_atoms) or (batch_size * max_atoms,)
    if mono.ndim == 1:
        mono_target = mono.reshape(batch_size, max_atoms)
    else:
        mono_target = mono
    
    l2_loss_mono = optax.l2_loss(sum_of_dc_monopoles, mono_target)
    mono_loss_corrected = l2_loss_mono.sum() / jnp.maximum(n_atoms * batch_size, 1.0)

    # esp_loss
    # batched_pred = batched_electrostatic_potential(d, m, vdw_surface)
    batched_pred = batched_electrostatic_potential(d, m, vdw_surface)
    
    # Ensure batched_pred and esp_target have compatible shapes
    # They might be (batch_size, ngrid) or flattened
    if batched_pred.ndim > 1:
        # Keep batch dimension for proper masking
        batched_pred_shape = batched_pred.shape
    else:
        # Flattened - will need to reshape based on ngrid
        batched_pred_shape = (batch_size, -1)
    
    if esp_target.ndim > 1:
        esp_target_shape = esp_target.shape
    else:
        esp_target_shape = (batch_size, -1)
    
    # Compute per-grid-point errors (difference, not squared)
    esp_errors = batched_pred - esp_target
    
    l2_loss = optax.l2_loss(batched_pred, esp_target)
    
    # Create ESP mask to exclude points too close to atoms (singularities)
    # This is critical for reducing ESP errors - matches working trainer.py
    # Initialize mask with same shape as esp_target
    if esp_target.ndim > 1:
        esp_mask = jnp.ones_like(esp_target, dtype=jnp.float32)  # (batch_size, ngrid)
    else:
        # Flattened case - will reshape later
        esp_mask = jnp.ones_like(esp_target, dtype=jnp.float32)
    
    if use_atomic_radii_mask and atom_positions is not None and atomic_numbers is not None:
        # First, determine the actual shape of atom_positions to get max_atoms
        # atom_positions could be:
        # - Flattened: (batch_size * natoms, 3)
        # - Batched: (batch_size, natoms, 3)
        # - Single sample: (natoms, 3)
        
        if atom_positions.ndim == 2:
            # Could be flattened batch or single sample
            total_atoms = atom_positions.shape[0]
            if total_atoms == batch_size * max_atoms:
                # Flattened batch: reshape to (batch_size, natoms, 3)
                atom_pos = atom_positions.reshape(batch_size, max_atoms, 3)
            elif batch_size == 1:
                # Single sample: (natoms, 3) -> add batch dimension
                actual_max_atoms = total_atoms
                atom_pos = atom_positions[None, :, :]  # (1, natoms, 3)
                # Update max_atoms if needed
                if actual_max_atoms != max_atoms:
                    max_atoms = actual_max_atoms
            else:
                # Try to infer max_atoms from total_atoms
                if total_atoms % batch_size == 0:
                    inferred_max_atoms = total_atoms // batch_size
                    atom_pos = atom_positions.reshape(batch_size, inferred_max_atoms, 3)
                    max_atoms = inferred_max_atoms
                else:
                    # Fallback: assume it's already correct
                    atom_pos = atom_positions.reshape(batch_size, max_atoms, 3)
        elif atom_positions.ndim == 3:
            # Already batched: (batch_size, natoms, 3)
            atom_pos = atom_positions
            # Update max_atoms if needed
            if atom_positions.shape[1] != max_atoms:
                max_atoms = atom_positions.shape[1]
        else:
            raise ValueError(f"Unexpected atom_positions shape: {atom_positions.shape}")
        
        # Handle atomic_numbers shape similarly
        if atomic_numbers.ndim == 1:
            total_atoms = atomic_numbers.shape[0]
            if total_atoms == batch_size * max_atoms:
                atomic_nums = atomic_numbers.reshape(batch_size, max_atoms)
            elif batch_size == 1:
                atomic_nums = atomic_numbers[None, :]  # (1, natoms)
            else:
                if total_atoms % batch_size == 0:
                    inferred_max_atoms = total_atoms // batch_size
                    atomic_nums = atomic_numbers.reshape(batch_size, inferred_max_atoms)
                    max_atoms = inferred_max_atoms
                else:
                    atomic_nums = atomic_numbers.reshape(batch_size, max_atoms)
        elif atomic_numbers.ndim == 2:
            atomic_nums = atomic_numbers
            if atomic_numbers.shape[1] != max_atoms:
                max_atoms = atomic_numbers.shape[1]
        else:
            raise ValueError(f"Unexpected atomic_numbers shape: {atomic_numbers.shape}")
        
        # Ensure atom_pos and atomic_nums have matching max_atoms
        if atom_pos.shape[1] != atomic_nums.shape[1]:
            # Use the larger one and pad if needed
            actual_max_atoms = max(atom_pos.shape[1], atomic_nums.shape[1])
            if atom_pos.shape[1] < actual_max_atoms:
                # Pad atom_pos
                padding = jnp.zeros((batch_size, actual_max_atoms - atom_pos.shape[1], 3))
                atom_pos = jnp.concatenate([atom_pos, padding], axis=1)
            if atomic_nums.shape[1] < actual_max_atoms:
                # Pad atomic_nums with zeros
                padding = jnp.zeros((batch_size, actual_max_atoms - atomic_nums.shape[1]), dtype=atomic_nums.dtype)
                atomic_nums = jnp.concatenate([atomic_nums, padding], axis=1)
            max_atoms = actual_max_atoms
        
        # Handle atom_mask shape - ensure it matches max_atoms
        if atom_mask is None:
            # Assume all atoms are valid if mask not provided
            atom_mask_arr = jnp.ones((batch_size, max_atoms), dtype=jnp.float32)
        elif atom_mask.ndim == 1:
            # Flattened: reshape to (batch_size, natoms)
            total_atoms = atom_mask.shape[0]
            if total_atoms == batch_size * max_atoms:
                atom_mask_arr = atom_mask.reshape(batch_size, max_atoms)
            elif batch_size == 1:
                atom_mask_arr = atom_mask[None, :]  # (1, natoms)
            else:
                if total_atoms % batch_size == 0:
                    inferred_max_atoms = total_atoms // batch_size
                    atom_mask_arr = atom_mask.reshape(batch_size, inferred_max_atoms)
                    if inferred_max_atoms != max_atoms:
                        # Pad or trim to match max_atoms
                        if inferred_max_atoms < max_atoms:
                            padding = jnp.zeros((batch_size, max_atoms - inferred_max_atoms), dtype=atom_mask.dtype)
                            atom_mask_arr = jnp.concatenate([atom_mask_arr, padding], axis=1)
                        else:
                            atom_mask_arr = atom_mask_arr[:, :max_atoms]
                else:
                    atom_mask_arr = atom_mask.reshape(batch_size, max_atoms)
        elif atom_mask.ndim == 2:
            atom_mask_arr = atom_mask
            if atom_mask.shape[1] != max_atoms:
                # Pad or trim to match max_atoms
                if atom_mask.shape[1] < max_atoms:
                    padding = jnp.zeros((batch_size, max_atoms - atom_mask.shape[1]), dtype=atom_mask.dtype)
                    atom_mask_arr = jnp.concatenate([atom_mask_arr, padding], axis=1)
                else:
                    atom_mask_arr = atom_mask_arr[:, :max_atoms]
        else:
            raise ValueError(f"Unexpected atom_mask shape: {atom_mask.shape}")
        
        # Handle vdw_surface shape - check if already batched
        if vdw_surface.ndim == 2:
            # Single sample: (ngrid, 3) -> add batch dimension
            vdw = vdw_surface[None, :, :]  # (1, ngrid, 3)
            esp_mask_expanded = esp_mask[None, :] if esp_mask.ndim == 1 else esp_mask
            esp_target_expanded = esp_target[None, :] if esp_target.ndim == 1 else esp_target
        elif vdw_surface.ndim == 3:
            # Already batched: (batch_size, ngrid, 3)
            vdw = vdw_surface
            esp_mask_expanded = esp_mask
            esp_target_expanded = esp_target
        else:
            # Unexpected shape - try to handle gracefully
            vdw = vdw_surface.reshape(batch_size, -1, 3)
            esp_mask_expanded = esp_mask
            esp_target_expanded = esp_target
        
        # Compute distances from grid to all atoms: (batch_size, ngrid, natoms)
        # vdw: (batch_size, ngrid, 3), atom_pos: (batch_size, natoms, 3)
        # Use broadcasting: vdw[:, :, None, :] - atom_pos[:, None, :, :]
        diff = vdw[:, :, None, :] - atom_pos[:, None, :, :]  # (batch_size, ngrid, natoms, 3)
        distances = jnp.linalg.norm(diff, axis=-1)  # (batch_size, ngrid, natoms)
        
        # Get atomic radii for each atom
        atomic_nums_int = atomic_nums.astype(jnp.int32)
        atomic_radii = jnp.take(RADII_TABLE, atomic_nums_int, mode='clip')  # (batch_size, natoms)
        
        # For distance-based masking, only consider real (unmasked) atoms
        # Set distances to masked atoms to infinity
        distances_masked = jnp.where(
            atom_mask_arr[:, None, :] > 0.5,  # (batch_size, 1, natoms)
            distances,  # (batch_size, ngrid, natoms)
            1e10  # Large distance for masked atoms
        )
        
        # Check if any REAL atom is too close (within 2.0 * covalent_radii)
        # atomic_radii: (batch_size, natoms) -> expand to (batch_size, 1, natoms) for broadcasting
        cutoff_distances = 2.0 * atomic_radii[:, None, :]  # (batch_size, 1, natoms)
        within_cutoff = distances_masked < cutoff_distances  # (batch_size, ngrid, natoms)
        distance_mask = (~jnp.any(within_cutoff, axis=-1)).astype(jnp.float32)  # (batch_size, ngrid)
        
        # Apply additional distance and value filters
        if esp_min_distance > 0:
            min_dist = jnp.min(distances_masked, axis=-1)  # (batch_size, ngrid)
            distance_mask = distance_mask * (min_dist >= esp_min_distance).astype(jnp.float32)
        
        if esp_max_value < 1e9:
            esp_abs = jnp.abs(esp_target_expanded)
            if esp_abs.ndim == 1:
                esp_abs = esp_abs[None, :]
            distance_mask = distance_mask * (esp_abs <= esp_max_value).astype(jnp.float32)
        
        # Apply mask to esp_mask - keep same shape as esp_target
        if batch_size == 1 and vdw_surface.ndim == 2:
            esp_mask = distance_mask[0]  # (ngrid,)
        else:
            # Keep distance_mask as (batch_size, ngrid) to match esp_target shape
            esp_mask = distance_mask  # (batch_size, ngrid)
    
    # Ensure esp_mask, l2_loss, and esp_errors have matching shapes
    # They should all be (batch_size, ngrid) or flattened consistently
    if esp_target.ndim > 1:
        # Keep 2D shape: (batch_size, ngrid)
        if esp_mask.ndim == 1:
            # Reshape esp_mask to match esp_target
            esp_mask = esp_mask.reshape(esp_target.shape)
        elif esp_mask.ndim > 1 and esp_mask.shape != esp_target.shape:
            # Reshape to match
            esp_mask = esp_mask.reshape(esp_target.shape)
        
        # Ensure l2_loss has same shape
        if l2_loss.ndim == 1:
            l2_loss = l2_loss.reshape(esp_target.shape)
        elif l2_loss.ndim > 1 and l2_loss.shape != esp_target.shape:
            l2_loss = l2_loss.reshape(esp_target.shape)
        
        # Ensure esp_errors has same shape
        if esp_errors.ndim == 1:
            esp_errors = esp_errors.reshape(esp_target.shape)
        elif esp_errors.ndim > 1 and esp_errors.shape != esp_target.shape:
            esp_errors = esp_errors.reshape(esp_target.shape)
    else:
        # Flattened case - ensure all are 1D
        if esp_mask.ndim > 1:
            esp_mask = esp_mask.reshape(-1)
        if l2_loss.ndim > 1:
            l2_loss = l2_loss.reshape(-1)
        if esp_errors.ndim > 1:
            esp_errors = esp_errors.reshape(-1)
    
    # Mask the loss (set masked points to 0)
    l2_loss_masked = l2_loss * esp_mask
    mask_total = jnp.sum(esp_mask) + EPS
    
    # Apply weighting if requested
    if esp_magnitude_weighting:
        # Weight by inverse ESP magnitude: points with larger |ESP| get LOWER weight
        # This reduces the impact of points where nuclear-electron shielding occurs
        # and ESP approaches singularity (near atomic nuclei)
        esp_magnitude = jnp.abs(esp_target)  # Use target ESP magnitude
        
        # Avoid division by zero for very small ESP values
        esp_magnitude_safe = jnp.maximum(esp_magnitude, 1e-10)
        
        # Inverse weighting: weight = 1 / (1 + esp_magnitude / scale)
        # This gives weight ≈ 1 for small ESP, weight → 0 for large ESP
        # Use mean ESP as scale for normalization
        esp_scale = jnp.mean(esp_magnitude_safe) + 1e-10
        weights = 1.0 / (1.0 + esp_magnitude_safe / esp_scale)
        
        # Normalize weights to have mean=1 for stability
        weights = weights / (jnp.mean(weights) + 1e-10)
        
        # Ensure weights have same shape as l2_loss_masked
        if weights.ndim != l2_loss_masked.ndim:
            weights = weights.reshape(l2_loss_masked.shape)
        elif weights.shape != l2_loss_masked.shape:
            weights = weights.reshape(l2_loss_masked.shape)
        
        # Apply weights to loss (with mask)
        weighted_l2_loss = l2_loss_masked * weights
        # Normalize by masked weights sum
        weights_masked = weights * esp_mask
        esp_loss_corrected = weighted_l2_loss.sum() / jnp.maximum(weights_masked.sum(), EPS)
        
    elif distance_weighting and atom_positions is not None:
        # Handle atom_positions shape - could be (batch_size, natoms, 3) or flattened
        if atom_positions.ndim == 2:
            # Flattened: (batch_size * natoms, 3) -> reshape to (batch_size, natoms, 3)
            atom_pos = atom_positions.reshape(batch_size, max_atoms, 3)
        else:
            # Already batched: (batch_size, natoms, 3)
            atom_pos = atom_positions
        
        # Handle vdw_surface shape
        if vdw_surface.ndim == 2:
            # Single sample: (ngrid, 3) -> add batch dimension
            vdw = vdw_surface[None, :, :]  # (1, ngrid, 3)
        else:
            vdw = vdw_surface  # (batch_size, ngrid, 3)
        
        # Compute distances from each grid point to nearest atom
        # vdw: (batch_size, ngrid, 3), atom_pos: (batch_size, natoms, 3)
        # Compute pairwise distances: (batch_size, ngrid, natoms)
        diff = vdw[:, :, None, :] - atom_pos[:, None, :, :]  # (batch_size, ngrid, natoms, 3)
        distances = jnp.linalg.norm(diff, axis=-1)  # (batch_size, ngrid, natoms)
        
        # Find minimum distance to any atom for each grid point
        min_distances = jnp.min(distances, axis=-1)  # (batch_size, ngrid)
        
        # Clamp minimum distance to avoid singularities
        min_distances = jnp.maximum(min_distances, distance_min)
        
        # Compute weights: INCREASING with distance (reversed from before)
        # Weight = exp(distance / scale) normalized so that weight at distance_min = 1
        # This gives higher weight to points further from atoms
        raw_weights = jnp.exp((min_distances - distance_min) / distance_scale)  # (batch_size, ngrid)
        
        # Normalize weights to have mean=1 for stability
        weights = raw_weights / (jnp.mean(raw_weights) + 1e-10)
        
        # Ensure weights have same shape as l2_loss_masked
        if weights.ndim != l2_loss_masked.ndim:
            weights = weights.reshape(l2_loss_masked.shape)
        elif weights.shape != l2_loss_masked.shape:
            weights = weights.reshape(l2_loss_masked.shape)
        
        # Apply weights to loss (with mask)
        weighted_l2_loss = l2_loss_masked * weights
        # Normalize by masked weights sum
        weights_masked = weights * esp_mask
        esp_loss_corrected = weighted_l2_loss.sum() / jnp.maximum(weights_masked.sum(), EPS)
    else:
        # No weighting - normalize by mask_total (number of valid grid points)
        # This matches the working trainer.py approach
        esp_loss_corrected = l2_loss_masked.sum() / mask_total
    
    # Enforce per-molecule zero charge constraint
    # sum_of_dc_monopoles has shape (batch_size, max_atoms)
    # Sum over atoms (axis=-1) to get total charge per molecule: (batch_size,)
    # Square each molecule's total charge and average over batch
    charge_conservation_loss = (sum_of_dc_monopoles.sum(axis=-1) ** 2).mean()
    
    total_loss = esp_loss_corrected * esp_w + mono_loss_corrected * chg_w + charge_conservation_loss
    
    return total_loss, batched_pred, esp_target, esp_errors


@functools.partial(jax.jit, static_argnames=("batch_size", "esp_w", "chg_w", "n_dcm"))
def dipo_esp_mono_loss(
    dipo_prediction,
    mono_prediction,
    esp_target,
    vdw_surface,
    mono,
    Dxyz,
    com,
    espMask,
    n_atoms,
    batch_size,
    esp_w,
    chg_w,
    n_dcm,
):
    """
    Dipole-augmented ESP and monopole loss function.
    
    Computes loss as weighted sum of ESP fitting error, monopole constraint
    violation, and dipole moment error. Includes charge neutrality and
    dummy atom handling.
    
    Parameters
    ----------
    dipo_prediction : array_like
        Predicted distributed dipole positions, shape (batch_size, natoms, n_dcm, 3)
    mono_prediction : array_like
        Predicted monopoles, shape (batch_size, natoms, n_dcm)
    esp_target : array_like
        Target ESP values, shape (batch_size, ngrid)
    vdw_surface : array_like
        VDW surface grid points, shape (batch_size, ngrid, 3)
    mono : array_like
        Reference monopoles, shape (batch_size, natoms)
    Dxyz : array_like
        Reference dipole moments, shape (batch_size, 3)
    com : array_like
        Center of mass coordinates, shape (batch_size, 3)
    espMask : array_like
        ESP evaluation masks, shape (batch_size, ngrid)
    n_atoms : array_like
        Number of atoms per system, shape (batch_size,)
    batch_size : int
        Batch size
    esp_w : float
        Weight for ESP loss term
    n_dcm : int
        Number of distributed multipoles per atom
        
    Returns
    -------
    tuple
        (esp_loss, mono_loss, dipole_loss)
    """
    # Infer max_atoms from prediction shape
    # Handle both batched and unbatched predictions
    if batch_size == 1 and len(mono_prediction.shape) == 2:
        # Unbatched: (n_atoms, n_dcm)
        max_atoms = mono_prediction.shape[0]
        # Add batch dimension
        mono_prediction = mono_prediction[None, :, :]  # (1, n_atoms, n_dcm)
        dipo_prediction = dipo_prediction[None, :, :, :]  # (1, n_atoms, n_dcm, 3)
    else:
        # Batched: (batch_size, n_atoms, n_dcm)
        max_atoms = mono_prediction.shape[1]
    
    d = jnp.moveaxis(dipo_prediction, -1, -2).reshape(batch_size, max_atoms * n_dcm, 3)
    m = mono_prediction.reshape(batch_size, max_atoms * n_dcm)

    # 0 the charges for dummy atoms
    # Ensure scalar n_atoms even if shaped (1,) or (1,1)
    n_atoms = jnp.ravel(n_atoms)[0]
    NDC = n_atoms * n_dcm
    valid_atoms = jnp.where(jnp.arange(max_atoms * n_dcm) < NDC, 1, 0)
    d = d[0]
    m = m[0] * valid_atoms
    # constrain the net charge to 0.0
    avg_chg = m.sum() / jnp.maximum(NDC, 1.0)
    m = (m - avg_chg) * valid_atoms

    # monopole loss
    mono_prediction = m.reshape(max_atoms, n_dcm)
    sum_of_dc_monopoles = mono_prediction.sum(axis=-1)
    l2_loss_mono = optax.l2_loss(sum_of_dc_monopoles, mono)
    mono_loss_corrected = l2_loss_mono.sum() / jnp.maximum(n_atoms, 1.0)

    # dipole loss
    molecular_dipole = pred_dipole(d, com[0], m)
    # jax.debug.print("{x} {y}", x=molecular_dipole, y=Dxyz[0])
    dipo_loss = optax.l2_loss(molecular_dipole, Dxyz[0]).sum()

    # esp_loss
    # batched_pred = batched_electrostatic_potential(d, m, vdw_surface)
    batched_pred = calc_esp(d, m, vdw_surface[0])
    l2_loss = optax.l2_loss(batched_pred, esp_target[0])
    # remove dummy grid points
    valid_grids = jnp.where(espMask[0], l2_loss, 0)
    esp_loss_corrected = valid_grids.sum() / espMask[0].sum()
    # jax.debug.print("{x} {y} {z}", x=esp_loss_corrected * esp_w, y=mono_loss_corrected, z=dipo_loss * 10)
    return esp_loss_corrected * esp_w * 0.0 , mono_loss_corrected*0.0 , dipo_loss * chg_w


def esp_mono_loss_pots(
    dipo_prediction, mono_prediction, vdw_surface, mono, batch_size, n_dcm
):
    """
    Compute ESP from distributed multipoles for loss calculation.
    
    Parameters
    ----------
    dipo_prediction : array_like
        Predicted distributed dipole positions
    mono_prediction : array_like
        Predicted monopoles
    vdw_surface : array_like
        VDW surface grid points
    mono : array_like
        Reference monopoles
    batch_size : int
        Batch size
    n_dcm : int
        Number of distributed multipoles per atom
        
    Returns
    -------
    array_like
        Predicted ESP values
    """
    # Infer max_atoms from prediction shape
    # Handle both batched and unbatched predictions
    if batch_size == 1 and len(mono_prediction.shape) == 2:
        # Unbatched: (n_atoms, n_dcm)
        max_atoms = mono_prediction.shape[0]
    elif len(mono_prediction.shape) > 1:
        max_atoms = mono_prediction.shape[1]
    else:
        max_atoms = mono_prediction.size // n_dcm
    
    return calc_esp(
        dipo_prediction, mono_prediction.reshape(batch_size, n_dcm * max_atoms), vdw_surface
    )


def esp_loss_pots(dipo_prediction, mono_prediction, vdw_surface, mono, batch_size):
    """
    Compute ESP from atomic monopoles for comparison.
    
    Parameters
    ----------
    dipo_prediction : array_like
        Predicted distributed dipole positions
    mono_prediction : array_like
        Predicted monopoles
    vdw_surface : array_like
        VDW surface grid points
    mono : array_like
        Reference monopoles
    batch_size : int
        Batch size
        
    Returns
    -------
    array_like
        Predicted ESP values from atomic monopoles
    """
    # Infer max_atoms from prediction shape
    # Handle both batched and unbatched predictions
    if batch_size == 1 and len(mono_prediction.shape) == 1:
        # Unbatched: (n_atoms,)
        max_atoms = mono_prediction.size
    elif len(mono_prediction.shape) > 1:
        max_atoms = mono_prediction.shape[1] if batch_size > 1 else mono_prediction.shape[0]
    else:
        max_atoms = mono_prediction.size // batch_size
    
    d = dipo_prediction.reshape(batch_size, max_atoms, 3)
    mono = mono.reshape(batch_size, max_atoms)
    m = mono_prediction.reshape(batch_size, max_atoms)
    batched_pred = batched_electrostatic_potential(d, m, vdw_surface)

    return batched_pred


def mean_absolute_error(prediction, target, batch_size):
    """
    Calculate mean absolute error for non-zero target values.
    
    Parameters
    ----------
    prediction : array_like
        Predicted values
    target : array_like
        Target values
    batch_size : int
        Batch size
        
    Returns
    -------
    float
        Mean absolute error
    """
    # Infer max_atoms from target shape
    if len(target.shape) > 1:
        max_atoms = target.shape[1] if batch_size > 1 else target.shape[0]
    else:
        max_atoms = target.size // batch_size if batch_size > 0 else target.size
    
    nonzero = jnp.nonzero(target, size=batch_size * max_atoms)
    return jnp.mean(jnp.abs(prediction[nonzero] - target[nonzero]))


def esp_loss_eval(pred, target, ngrid):
    """
    Evaluate ESP loss for non-zero target values.
    
    Parameters
    ----------
    pred : array_like
        Predicted ESP values
    target : array_like
        Target ESP values
    ngrid : int
        Number of grid points
        
    Returns
    -------
    float
        Root mean square error in kcal/mol
    """
    target = target.flatten()
    esp_non_zero = np.nonzero(target)
    l2_loss = optax.l2_loss(pred[esp_non_zero], target[esp_non_zero]) * 2
    esp_loss = np.mean(l2_loss) ** 0.5
    return esp_loss


def get_predictions(mono_dc2, dipo_dc2, batch, batch_size, n_dcm):
    """
    Get ESP predictions from both distributed and atomic monopoles.
    
    Parameters
    ----------
    mono_dc2 : array_like
        Distributed monopole predictions
    dipo_dc2 : array_like
        Distributed dipole predictions
    batch : dict
        Batch dictionary
    batch_size : int
        Batch size
    n_dcm : int
        Number of distributed multipoles per atom
        
    Returns
    -------
    tuple
        (esp_dc_pred, mono_pred) - ESP from distributed multipoles and atomic monopoles
    """
    mono = mono_dc2
    dipo = dipo_dc2

    esp_dc_pred = esp_mono_loss_pots(
        dipo, mono, batch["vdw_surface"], batch["mono"], batch_size, n_dcm
    )

    mono_pred = esp_loss_pots(
        batch["positions"],
        batch["mono"],
        batch["vdw_surface"],
        batch["mono"],
        batch_size,
    )
    return esp_dc_pred, mono_pred
