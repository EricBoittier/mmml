#!/usr/bin/env python3
"""
Joint PhysNet-DCMNet Training Script for CO2 Data

This script trains PhysNet and DCMNet simultaneously with end-to-end gradient flow.
PhysNet predicts atomic charges (supervised by molecular dipole), which are fed as
monopoles into DCMNet to predict distributed multipoles and ESP on VDW surfaces.

Usage:
    python trainer.py --train-efd energies_forces_dipoles_train.npz \
                      --train-esp grids_esp_train.npz \
                      --valid-efd energies_forces_dipoles_valid.npz \
                      --valid-esp grids_esp_valid.npz
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pickle
from typing import Dict, Tuple, Any
import time

# Add mmml to path
repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import functools

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import PhysNet components
from mmml.physnetjax.physnetjax.models.model import EF
from mmml.physnetjax.physnetjax.directories import BASE_CKPT_DIR

# Import DCMNet components
from mmml.dcmnet.dcmnet.modules import MessagePassingModel
from mmml.dcmnet.dcmnet.electrostatics import calc_esp

# Import data utilities
from mmml.data import load_npz, DataConfig


class JointPhysNetDCMNet(nn.Module):
    """
    Joint PhysNet-DCMNet model for end-to-end charge and ESP prediction.
    
    Architecture:
    1. PhysNet predicts atomic charges (supervised by molecular dipole)
    2. Charges are fed as monopoles into DCMNet
    3. DCMNet predicts distributed multipoles for ESP fitting
    4. Full gradient flow from ESP loss back to PhysNet parameters
    
    Optional: Mix PhysNet energy with DCMNet Coulomb energy via learnable λ
    
    Attributes
    ----------
    physnet_config : dict
        Configuration for PhysNet EF model
    dcmnet_config : dict
        Configuration for DCMNet MessagePassingModel
    mix_coulomb_energy : bool
        If True, mix PhysNet energy with DCMNet Coulomb energy using learnable λ
    """
    physnet_config: Dict[str, Any]
    dcmnet_config: Dict[str, Any]
    mix_coulomb_energy: bool = False
    
    def setup(self):
        """Initialize both PhysNet and DCMNet models."""
        # PhysNet must have charges=True to predict atomic charges
        self.physnet = EF(**self.physnet_config)
        self.dcmnet = MessagePassingModel(**self.dcmnet_config)
        
        # Learnable mixing parameter for Coulomb energy
        if self.mix_coulomb_energy:
            self.coulomb_lambda = self.param(
                "coulomb_lambda",
                lambda rng, shape: jnp.ones(shape) * 0.1,  # Initialize to 0.1
                (1,)
            )
    
    @nn.compact
    def __call__(
        self,
        atomic_numbers: jnp.ndarray,
        positions: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
        batch_mask: jnp.ndarray,
        atom_mask: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass through both models.
        
        Parameters
        ----------
        atomic_numbers : jnp.ndarray
            Atomic numbers, shape (batch_size * natoms,)
        positions : jnp.ndarray
            Atomic positions, shape (batch_size * natoms, 3)
        dst_idx : jnp.ndarray
            Destination indices for edges
        src_idx : jnp.ndarray
            Source indices for edges
        batch_segments : jnp.ndarray
            Batch segment indices
        batch_size : int
            Batch size
        batch_mask : jnp.ndarray
            Batch mask
        atom_mask : jnp.ndarray
            Atom mask
            
        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary containing:
            - energy, forces, dipoles, charges (from PhysNet)
            - mono_dist, dipo_dist (from DCMNet)
            - charges_as_mono (charges formatted for monopole loss)
        """
        # 1. PhysNet forward pass: predict E, F, D, and atomic charges
        physnet_output = self.physnet(
            atomic_numbers=atomic_numbers,
            positions=positions,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            batch_size=batch_size,
            batch_mask=batch_mask,
            atom_mask=atom_mask,
        )
        
        # 2. Extract and reshape charges for DCMNet
        # PhysNet charges shape: (batch*natoms, 1, 1, 1)
        charges = physnet_output["charges"]
        charges_squeezed = jnp.squeeze(charges)  # (batch*natoms,)
        
        # Compute sum of charges per molecule (respecting atom_mask)
        charges_masked = charges_squeezed * atom_mask
        # Sum per batch segment
        sum_charges = jax.ops.segment_sum(
            charges_masked,
            segment_ids=batch_segments,
            num_segments=batch_size
        )
        
        # 3. DCMNet forward pass: predict distributed multipoles
        # DCMNet uses charges as the monopole constraint
        mono_dist, dipo_dist = self.dcmnet(
            atomic_numbers=atomic_numbers,
            positions=positions,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            batch_size=batch_size,
        )
        
        # 4. Optionally compute and mix Coulomb energy from DCMNet charges
        energy_reshaped = physnet_output["energy"].reshape(batch_size)
        
        if self.mix_coulomb_energy:
            # Compute Coulomb energy from DCMNet distributed charges
            # E_coulomb = (1/2) Σᵢⱼ qᵢqⱼ/rᵢⱼ (pairwise interactions)
            natoms = mono_dist.shape[0] // batch_size
            
            # Get all charge positions and values
            charges_flat = mono_dist.reshape(-1)  # (batch*natoms*n_dcm,)
            positions_flat = jnp.moveaxis(dipo_dist, -1, -2).reshape(-1, 3)  # (batch*natoms*n_dcm, 3)
            
            # For batch_size==1, compute Coulomb energy
            if batch_size == 1:
                # Pairwise distances between all distributed charges
                diff = positions_flat[:, None, :] - positions_flat[None, :, :]  # (N, N, 3)
                distances = jnp.linalg.norm(diff, axis=-1)  # (N, N)
                # Avoid self-interaction
                distances = jnp.where(distances < 1e-6, 1e6, distances)
                # Coulomb energy in atomic units: E = (1/2) Σᵢⱼ qᵢqⱼ/(rᵢⱼ * 1.88973)
                pairwise_energy = charges_flat[:, None] * charges_flat[None, :] / (distances * 1.88973)
                coulomb_energy = 0.5 * jnp.sum(pairwise_energy)
                
                # Mix energies: E_total = E_physnet + λ * E_coulomb
                lambda_val = self.coulomb_lambda[0]
                energy_mixed = energy_reshaped + lambda_val * coulomb_energy
                energy_reshaped = energy_mixed
                
                # Store for monitoring
                coulomb_energy_out = coulomb_energy
                lambda_out = lambda_val
            else:
                # For batched, would need to compute per-molecule
                coulomb_energy_out = jnp.array(0.0)
                lambda_out = self.coulomb_lambda[0]
        else:
            coulomb_energy_out = jnp.array(0.0)
            lambda_out = jnp.array(0.0)
        
        # Reshape PhysNet outputs to proper shapes
        # PhysNet forces shape is (batch_size*natoms, 1, 1, 3) -> reshape to (batch_size*natoms, 3)
        # PhysNet dipoles shape is (batch_size, 1, 1, 3) -> reshape to (batch_size, 3)
        forces_reshaped = physnet_output["forces"].reshape(-1, 3)
        dipoles_reshaped = physnet_output["dipoles"].reshape(batch_size, 3)
        
        return {
            # PhysNet outputs (reshaped to proper dimensions)
            "energy": energy_reshaped,  # (batch_size,)
            "forces": forces_reshaped,  # (batch_size*natoms, 3)
            "dipoles": dipoles_reshaped,  # (batch_size, 3)
            "charges": charges,
            "sum_charges": physnet_output.get("sum_charges", sum_charges),  # (batch_size,)
            # DCMNet outputs
            "mono_dist": mono_dist,  # (batch*natoms, n_dcm)
            "dipo_dist": dipo_dist,  # (batch*natoms, n_dcm, 3)
            # For monopole constraint loss
            "charges_as_mono": charges_squeezed,  # (batch*natoms,)
            # Energy mixing (if enabled)
            "coulomb_energy": coulomb_energy_out,
            "coulomb_lambda": lambda_out,
        }


def load_combined_data(efd_file: Path, esp_file: Path, verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Load and combine EFD and ESP data from separate NPZ files.
    
    Parameters
    ----------
    efd_file : Path
        Path to energies_forces_dipoles NPZ file
    esp_file : Path
        Path to grids_esp NPZ file
    verbose : bool
        Whether to print loading information
        
    Returns
    -------
    Dict[str, np.ndarray]
        Combined data dictionary with all fields
    """
    if verbose:
        print(f"  Loading EFD: {efd_file}")
    efd_data = np.load(efd_file)
    
    if verbose:
        print(f"  Loading ESP: {esp_file}")
    esp_data = np.load(esp_file)
    
    # Combine data - ESP file should have R, Z, N as well
    combined = {
        # Molecular properties
        'R': esp_data['R'],
        'Z': esp_data['Z'],
        'N': esp_data['N'],
        'E': efd_data['E'],
        'F': efd_data['F'],
        'Dxyz': efd_data.get('Dxyz', efd_data.get('D')),
        # ESP properties
        'esp': esp_data['esp'],
        'vdw_surface': esp_data['vdw_surface'],
    }
    
    if verbose:
        print(f"  Combined data shapes:")
        for key, val in combined.items():
            print(f"    {key}: {val.shape}")
        print(f"  Data padding: {combined['R'].shape[1]} atoms")
        print(f"  Max actual atoms: {int(np.max(combined['N']))}")
    
    return combined


def resize_data_padding(
    data: Dict[str, np.ndarray],
    target_natoms: int,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Resize padded arrays to a different number of atoms.
    
    Useful when data is padded to N atoms but you want to use M < N atoms
    to save memory/computation (as long as M >= max(N) in the data).
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dataset dictionary
    target_natoms : int
        Target padding size
    verbose : bool
        Print information
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dataset with resized arrays
    """
    current_natoms = data['R'].shape[1]
    
    if current_natoms == target_natoms:
        return data
    
    # Check if safe
    max_atoms = int(np.max(data['N']))
    if max_atoms > target_natoms:
        raise ValueError(
            f"Cannot resize to {target_natoms} atoms: data contains molecules "
            f"with up to {max_atoms} atoms. Use --natoms >= {max_atoms}"
        )
    
    if verbose:
        print(f"\n📏 Resizing arrays from {current_natoms} to {target_natoms} atoms...")
    
    n_samples = len(data['N'])
    resized = data.copy()
    
    # Resize (n_samples, natoms, 3) arrays
    for key in ['R', 'F']:
        if key in data:
            old = data[key]
            if target_natoms < current_natoms:
                # Truncate
                resized[key] = old[:, :target_natoms, :]
            else:
                # Expand with zeros
                new = np.zeros((n_samples, target_natoms, 3), dtype=old.dtype)
                new[:, :current_natoms, :] = old
                resized[key] = new
            if verbose:
                print(f"  {key}: {old.shape} → {resized[key].shape}")
    
    # Resize (n_samples, natoms) arrays
    for key in ['Z']:
        if key in data:
            old = data[key]
            if target_natoms < current_natoms:
                # Truncate
                resized[key] = old[:, :target_natoms]
            else:
                # Expand with zeros
                new = np.zeros((n_samples, target_natoms), dtype=old.dtype)
                new[:, :current_natoms] = old
                resized[key] = new
            if verbose:
                print(f"  {key}: {old.shape} → {resized[key].shape}")
    
    if verbose:
        print(f"✅ Arrays resized to {target_natoms} atoms")
    
    return resized


def precompute_edge_lists(
    data: Dict[str, np.ndarray],
    cutoff: float,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Pre-compute edge lists for all samples in the dataset.
    
    This is much faster than computing edge lists on-the-fly for each batch.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dataset dictionary
    cutoff : float
        Cutoff distance for edge list construction
    verbose : bool
        Print progress
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dataset with added edge list arrays
    """
    if verbose:
        print(f"  Pre-computing edge lists (cutoff={cutoff:.1f} Å)...")
    
    R = data['R']  # (n_samples, natoms, 3)
    N = data['N']  # (n_samples,)
    n_samples = len(N)
    natoms = R.shape[1]
    
    # Pre-compute edge lists for each sample
    all_dst_idx = []
    all_src_idx = []
    all_edge_counts = []
    
    for sample_idx in range(n_samples):
        n_atoms = int(N[sample_idx])
        pos = R[sample_idx, :n_atoms]
        
        dst_list = []
        src_list = []
        
        # Compute pairwise distances
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    dist = np.linalg.norm(pos[i] - pos[j])
                    if dist < cutoff:
                        dst_list.append(i)
                        src_list.append(j)
        
        all_dst_idx.append(np.array(dst_list, dtype=np.int32))
        all_src_idx.append(np.array(src_list, dtype=np.int32))
        all_edge_counts.append(len(dst_list))
        
        if verbose and (sample_idx + 1) % 1000 == 0:
            print(f"    Processed {sample_idx + 1}/{n_samples} samples...")
    
    # Store as object arrays (variable length per sample)
    data_with_edges = data.copy()
    data_with_edges['dst_idx'] = np.array(all_dst_idx, dtype=object)
    data_with_edges['src_idx'] = np.array(all_src_idx, dtype=object)
    data_with_edges['edge_counts'] = np.array(all_edge_counts, dtype=np.int32)
    
    if verbose:
        avg_edges = np.mean(all_edge_counts)
        print(f"  ✅ Pre-computed edge lists: avg {avg_edges:.1f} edges/sample")
    
    return data_with_edges


def prepare_batch_data(
    data: Dict[str, np.ndarray],
    indices: np.ndarray,
    cutoff: float = 6.0,
    verbose: bool = False,
) -> Dict[str, jnp.ndarray]:
    """
    Prepare a batch of data using pre-computed edge lists.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Full dataset with pre-computed edge lists
    indices : np.ndarray
        Indices for this batch
    cutoff : float
        Cutoff distance (not used, kept for compatibility)
    verbose : bool
        Print debugging information
        
    Returns
    -------
    Dict[str, jnp.ndarray]
        Batch with edge lists and masks
    """
    batch_size = len(indices)
    
    # Extract batch data
    R = data['R'][indices]  # (batch_size, natoms, 3)
    Z = data['Z'][indices]  # (batch_size, natoms)
    N = data['N'][indices]  # (batch_size,)
    E = data['E'][indices]  # (batch_size,)
    F = data['F'][indices]  # (batch_size, natoms, 3)
    D = data['Dxyz'][indices]  # (batch_size, 3)
    esp = data['esp'][indices]  # (batch_size, ngrid)
    vdw_surface = data['vdw_surface'][indices]  # (batch_size, ngrid, 3)
    
    # Create masks
    natoms = R.shape[1]
    atom_mask = np.zeros((batch_size, natoms), dtype=np.float32)
    for i, n in enumerate(N):
        atom_mask[i, :n] = 1.0
    
    # batch_mask needs to be per-edge for PhysNet, but create it per-batch for now
    # It will be expanded to match edges in the edge list
    batch_mask = np.ones(batch_size, dtype=np.float32)
    
    # Flatten batch dimension
    R_flat = R.reshape(-1, 3)
    Z_flat = Z.reshape(-1)
    
    # Create batch segments
    batch_segments = np.repeat(np.arange(batch_size), natoms)
    
    # Extract pre-computed edge lists and offset for batching
    dst_idx_list = []
    src_idx_list = []
    
    for batch_idx, sample_idx in enumerate(indices):
        offset = batch_idx * natoms
        dst = data['dst_idx'][sample_idx] + offset
        src = data['src_idx'][sample_idx] + offset
        dst_idx_list.append(dst)
        src_idx_list.append(src)
    
    # Concatenate all edge lists
    if dst_idx_list:
        dst_idx = np.concatenate(dst_idx_list).astype(np.int32)
        src_idx = np.concatenate(src_idx_list).astype(np.int32)
        # Create edge mask (1 for valid edges)
        n_edges = len(dst_idx)
        edge_batch_mask = np.ones(n_edges, dtype=np.float32)
    else:
        dst_idx = np.array([], dtype=np.int32)
        src_idx = np.array([], dtype=np.int32)
        edge_batch_mask = np.array([], dtype=np.float32)
    
    return {
        'R': jnp.array(R_flat),
        'Z': jnp.array(Z_flat),
        'N': jnp.array(N),
        'E': jnp.array(E),
        'F': jnp.array(F.reshape(-1, 3)),
        'D': jnp.array(D),
        'esp': jnp.array(esp),
        'vdw_surface': jnp.array(vdw_surface),
        'atom_mask': jnp.array(atom_mask.reshape(-1)),
        'batch_mask': jnp.array(edge_batch_mask),  # Per-edge mask
        'batch_segments': jnp.array(batch_segments),
        'dst_idx': jnp.array(dst_idx),
        'src_idx': jnp.array(src_idx),
    }


@functools.partial(jax.jit, static_argnames=('batch_size', 'n_dcm', 'dipole_source'))
def compute_loss(
    output: Dict[str, jnp.ndarray],
    batch: Dict[str, jnp.ndarray],
    energy_w: float,
    forces_w: float,
    dipole_w: float,
    esp_w: float,
    mono_w: float,
    batch_size: int,
    n_dcm: int,
    dipole_source: str = 'physnet',
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute joint loss for PhysNet and DCMNet.
    
    Parameters
    ----------
    output : Dict[str, jnp.ndarray]
        Model outputs
    batch : Dict[str, jnp.ndarray]
        Batch data
    energy_w, forces_w, dipole_w, esp_w, mono_w : float
        Loss weights
    batch_size : int
        Batch size
    n_dcm : int
        Number of distributed multipoles per atom
        
    Returns
    -------
    Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]
        Total loss and individual loss components
    """
    # PhysNet losses
    loss_energy = optax.l2_loss(output["energy"], batch["E"]).mean()
    
    # Forces loss (only for real atoms)
    forces_masked = output["forces"] * batch["atom_mask"][:, None]
    forces_target_masked = batch["F"] * batch["atom_mask"][:, None]
    loss_forces = optax.l2_loss(forces_masked, forces_target_masked).mean()
    
    # Dipole loss - choose source
    if dipole_source == 'physnet':
        # Use PhysNet's dipole (from charges × positions)
        dipole_pred = output["dipoles"]
    elif dipole_source == 'dcmnet':
        # Compute dipole from DCMNet distributed multipoles
        # Dipole = Σ (q_i * r_i) for distributed charges
        natoms = output["mono_dist"].shape[0] // batch_size
        mono_reshaped = output["mono_dist"].reshape(batch_size, natoms, n_dcm)
        dipo_reshaped = output["dipo_dist"].reshape(batch_size, natoms, n_dcm, 3)
        
        # For each molecule, compute dipole from distributed charges
        # mono_reshaped: (batch, natoms, n_dcm) - charges
        # dipo_reshaped: (batch, natoms, n_dcm, 3) - positions
        # Dipole = Σ_atoms Σ_dcm (charge * position)
        dipole_pred = jnp.sum(mono_reshaped[..., None] * dipo_reshaped, axis=(1, 2))  # (batch, 3)
    else:
        raise ValueError(f"Unknown dipole_source: {dipole_source}")
    
    loss_dipole = optax.l2_loss(dipole_pred, batch["D"]).mean()
    
    # DCMNet losses
    # ESP loss - compute ESP from distributed multipoles
    # Need to reshape for calc_esp
    natoms = output["mono_dist"].shape[0] // batch_size
    
    # Reshape distributed multipoles
    mono_reshaped = output["mono_dist"].reshape(batch_size, natoms, n_dcm)
    dipo_reshaped = output["dipo_dist"].reshape(batch_size, natoms, n_dcm, 3)
    
    # For batch_size=1, calc_esp expects single molecule
    if batch_size == 1:
        mono_for_esp = mono_reshaped[0]  # (natoms, n_dcm)
        dipo_for_esp = dipo_reshaped[0]  # (natoms, n_dcm, 3)
        vdw_for_esp = batch["vdw_surface"][0]  # (ngrid, 3)
        esp_target = batch["esp"][0]  # (ngrid,)
        
        # Flatten distributed charges: (natoms*n_dcm,)
        mono_flat = mono_for_esp.reshape(-1)
        # Reshape dipoles: (natoms*n_dcm, 3)
        dipo_flat = jnp.moveaxis(dipo_for_esp, -1, -2).reshape(-1, 3)
        
        # Compute ESP
        esp_pred = calc_esp(dipo_flat, mono_flat, vdw_for_esp)
        loss_esp = optax.l2_loss(esp_pred, esp_target).mean()
    else:
        # For batched, need to loop (or use vmap)
        def single_esp_loss(mono_mol, dipo_mol, vdw_mol, esp_mol):
            mono_flat = mono_mol.reshape(-1)
            dipo_flat = jnp.moveaxis(dipo_mol, -1, -2).reshape(-1, 3)
            esp_pred = calc_esp(dipo_flat, mono_flat, vdw_mol)
            return optax.l2_loss(esp_pred, esp_mol).mean()
        
        esp_losses = jax.vmap(single_esp_loss)(
            mono_reshaped, dipo_reshaped, batch["vdw_surface"], batch["esp"]
        )
        loss_esp = esp_losses.mean()
    
    # PhysNet total charge constraint: sum of charges should equal total_charge (0.0)
    total_charge_pred = output.get("sum_charges", jnp.sum(output["charges_as_mono"] * batch["atom_mask"]))
    total_charge_target = jnp.zeros_like(total_charge_pred)  # Should be 0.0 for neutral molecules
    loss_total_charge = optax.l2_loss(total_charge_pred, total_charge_target).mean()
    
    # Monopole constraint: sum of distributed charges should equal PhysNet charge per atom
    # mono_dist shape: (batch*natoms, n_dcm)
    mono_sum = output["mono_dist"].sum(axis=-1)  # (batch*natoms,)
    charges_target = output["charges_as_mono"]  # (batch*natoms,)
    
    # Only compute loss for real atoms
    mono_sum_masked = mono_sum * batch["atom_mask"]
    charges_masked = charges_target * batch["atom_mask"]
    loss_mono = optax.l2_loss(mono_sum_masked, charges_masked).mean()
    
    # Total loss
    total_loss = (
        energy_w * loss_energy +
        forces_w * loss_forces +
        dipole_w * loss_dipole +
        esp_w * loss_esp +
        mono_w * loss_mono +
        mono_w * loss_total_charge  # Use same weight as monopole constraint
    )
    
    losses = {
        "total": total_loss,
        "energy": loss_energy,
        "forces": loss_forces,
        "dipole": loss_dipole,
        "esp": loss_esp,
        "monopole": loss_mono,
        "total_charge": loss_total_charge,
    }
    
    return total_loss, losses


@functools.partial(jax.jit, static_argnames=('model_apply', 'optimizer_update', 'batch_size', 'n_dcm', 'clip_norm', 'dipole_source'))
def train_step(
    params: Any,
    opt_state: Any,
    batch: Dict[str, jnp.ndarray],
    model_apply: Any,
    optimizer_update: Any,
    energy_w: float,
    forces_w: float,
    dipole_w: float,
    esp_w: float,
    mono_w: float,
    batch_size: int,
    n_dcm: int,
    clip_norm: float = None,
    dipole_source: str = 'physnet',
) -> Tuple[Any, Any, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Single training step with gradient computation and optional clipping.
    
    Returns
    -------
    Tuple
        (updated_params, updated_opt_state, total_loss, loss_dict)
    """
    def loss_fn(params):
        output = model_apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
            batch_mask=batch["batch_mask"],
            atom_mask=batch["atom_mask"],
        )
        total_loss, losses = compute_loss(
            output, batch, energy_w, forces_w, dipole_w, esp_w, mono_w, batch_size, n_dcm, dipole_source
        )
        return total_loss, (output, losses)
    
    (loss, (output, losses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Clip gradients if requested
    if clip_norm is not None:
        # Compute global norm
        grad_norm = optax.global_norm(grads)
        # Scale gradients if norm exceeds clip_norm
        grads = jax.tree_util.tree_map(
            lambda g: g * jnp.minimum(clip_norm / grad_norm, 1.0),
            grads
        )
    
    # Update parameters
    updates, opt_state = optimizer_update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss, losses


@functools.partial(jax.jit, static_argnames=('model_apply', 'batch_size', 'n_dcm', 'dipole_source'))
def eval_step(
    params: Any,
    batch: Dict[str, jnp.ndarray],
    model_apply: Any,
    energy_w: float,
    forces_w: float,
    dipole_w: float,
    esp_w: float,
    mono_w: float,
    batch_size: int,
    n_dcm: int,
    dipole_source: str = 'physnet',
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    Evaluation step without gradient computation.
    
    Returns
    -------
    Tuple
        (total_loss, loss_dict, output_dict)
    """
    output = model_apply(
        params,
        atomic_numbers=batch["Z"],
        positions=batch["R"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=batch_size,
        batch_mask=batch["batch_mask"],
        atom_mask=batch["atom_mask"],
    )
    
    total_loss, losses = compute_loss(
        output, batch, energy_w, forces_w, dipole_w, esp_w, mono_w, batch_size, n_dcm, dipole_source
    )
    
    # Compute MAE metrics
    mae_energy = jnp.abs(output["energy"] - batch["E"]).mean()
    mae_forces = jnp.abs(output["forces"] * batch["atom_mask"][:, None] - 
                         batch["F"] * batch["atom_mask"][:, None]).mean()
    
    # Compute MAE for BOTH dipole sources (regardless of which is used for loss)
    # PhysNet dipole
    mae_dipole_physnet = jnp.abs(output["dipoles"] - batch["D"]).mean()
    
    # DCMNet dipole
    natoms = output["mono_dist"].shape[0] // batch_size
    mono_reshaped = output["mono_dist"].reshape(batch_size, natoms, n_dcm)
    dipo_reshaped = output["dipo_dist"].reshape(batch_size, natoms, n_dcm, 3)
    dipole_dcmnet = jnp.sum(mono_reshaped[..., None] * dipo_reshaped, axis=(1, 2))
    mae_dipole_dcmnet = jnp.abs(dipole_dcmnet - batch["D"]).mean()
    
    # Compute ESP RMSE for BOTH methods (only for batch_size == 1)
    # For larger batches, this would need batched computation
    # DCMNet ESP (from distributed multipoles)
    mono_for_esp = mono_reshaped[0]  # (natoms, n_dcm)
    dipo_for_esp = dipo_reshaped[0]  # (natoms, n_dcm, 3)
    vdw_for_esp = batch["vdw_surface"][0]  # (ngrid, 3)
    esp_target = batch["esp"][0]  # (ngrid,)
    
    # DCMNet ESP
    mono_flat = mono_for_esp.reshape(-1)
    dipo_flat = jnp.moveaxis(dipo_for_esp, -1, -2).reshape(-1, 3)
    from mmml.dcmnet.dcmnet.electrostatics import calc_esp
    esp_pred_dcmnet = calc_esp(dipo_flat, mono_flat, vdw_for_esp)
    
    # PhysNet ESP (from atomic charges on atom centers)
    # Get charges and positions for first molecule in batch
    charges_physnet = output["charges_as_mono"]  # (batch*natoms,)
    positions_physnet = batch["R"]  # (batch*natoms, 3)
    atom_mask_batch = batch["atom_mask"]  # (batch*natoms,)
    
    # For batch_size == 1: use first natoms atoms
    # Apply mask to handle padding (padded atoms have charge 0)
    charges_masked = charges_physnet[:natoms] * atom_mask_batch[:natoms]
    positions_masked = positions_physnet[:natoms]  # (natoms, 3)
    
    # Compute ESP from point charges: V(r) = Σ q_i / |r - r_i|
    # Vectorized computation over grid points
    # grid: (ngrid, 3), positions: (natoms, 3)
    # distances: (ngrid, natoms)
    distances = jnp.linalg.norm(vdw_for_esp[:, None, :] - positions_masked[None, :, :], axis=2)
    # ESP at each grid point: sum over atoms (padded atoms have charge=0 so don't contribute)
    esp_pred_physnet = jnp.sum(charges_masked[None, :] / (distances + 1e-10), axis=1)
    
    # Compute RMSE for both
    rmse_esp_dcmnet = jnp.sqrt(jnp.mean((esp_pred_dcmnet - esp_target)**2))
    rmse_esp_physnet = jnp.sqrt(jnp.mean((esp_pred_physnet - esp_target)**2))
    
    losses["mae_energy"] = mae_energy
    losses["mae_forces"] = mae_forces
    losses["mae_dipole_physnet"] = mae_dipole_physnet
    losses["mae_dipole_dcmnet"] = mae_dipole_dcmnet
    losses["rmse_esp_physnet"] = rmse_esp_physnet
    losses["rmse_esp_dcmnet"] = rmse_esp_dcmnet
    # Keep backward compatibility
    losses["mae_dipole"] = mae_dipole_physnet if dipole_source == 'physnet' else mae_dipole_dcmnet
    
    # Add Coulomb energy info
    losses["coulomb_energy"] = output.get("coulomb_energy", jnp.array(0.0))
    losses["coulomb_lambda"] = output.get("coulomb_lambda", jnp.array(0.0))
    
    return total_loss, losses, output


def plot_validation_results(
    params: Any,
    model: JointPhysNetDCMNet,
    valid_data: Dict[str, np.ndarray],
    cutoff: float,
    energy_w: float,
    forces_w: float,
    dipole_w: float,
    esp_w: float,
    mono_w: float,
    n_dcm: int,
    save_dir: Path,
    n_samples: int = 100,
    n_esp_examples: int = 2,
    epoch: int = None,
    dipole_source: str = 'physnet',
) -> None:
    """
    Create validation set plots: scatter plots and ESP examples.
    
    Parameters
    ----------
    params : Any
        Model parameters
    model : JointPhysNetDCMNet
        Joint model
    valid_data : Dict[str, np.ndarray]
        Validation dataset
    cutoff : float
        Cutoff distance
    energy_w, forces_w, dipole_w, esp_w, mono_w : float
        Loss weights
    n_dcm : int
        Number of distributed multipoles
    save_dir : Path
        Directory to save plots
    n_samples : int
        Number of samples to plot
    n_esp_examples : int
        Number of ESP examples to visualize
    """
    if not HAS_MATPLOTLIB:
        print("\n⚠️  Matplotlib not available, skipping plots")
        return
    
    epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
    print(f"\n{'#'*70}")
    print(f"# Creating Validation Plots{epoch_str}")
    print(f"{'#'*70}\n")
    
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Add epoch suffix to filenames if specified
    suffix = f"_epoch{epoch}" if epoch is not None else ""
    
    # Collect predictions for first n_samples
    n_total = min(n_samples, len(valid_data['E']))
    
    energies_pred = []
    energies_true = []
    forces_pred = []
    forces_true = []
    dipoles_physnet_pred = []
    dipoles_dcmnet_pred = []
    dipoles_true = []
    esp_pred_dcmnet_list = []
    esp_pred_physnet_list = []
    esp_true_list = []
    
    print(f"Evaluating {n_total} validation samples...")
    
    for i in range(n_total):
        batch = prepare_batch_data(valid_data, np.array([i]), cutoff=cutoff)
        
        _, losses, output = eval_step(
            params=params,
            batch=batch,
            model_apply=model.apply,
            energy_w=energy_w,
            forces_w=forces_w,
            dipole_w=dipole_w,
            esp_w=esp_w,
            mono_w=mono_w,
            batch_size=1,
            n_dcm=n_dcm,
            dipole_source=dipole_source,
        )
        
        energies_pred.append(float(output['energy']))
        energies_true.append(float(batch['E']))
        
        forces_pred.append(np.array(output['forces']))
        forces_true.append(np.array(batch['F']))
        
        # Store BOTH dipole predictions
        dipoles_physnet_pred.append(np.array(output['dipoles']))
        
        # Compute DCMNet dipole
        natoms = output["mono_dist"].shape[0]
        mono_reshaped = output["mono_dist"].reshape(1, natoms, n_dcm)
        dipo_reshaped = output["dipo_dist"].reshape(1, natoms, n_dcm, 3)
        dipole_dcmnet = np.sum(mono_reshaped[..., None] * dipo_reshaped, axis=(1, 2))
        dipoles_dcmnet_pred.append(np.array(dipole_dcmnet))
        
        dipoles_true.append(np.array(batch['D']))
        
        # ESP (only store first n_esp_examples) - compute from BOTH methods
        if i < n_esp_examples:
            natoms = output["mono_dist"].shape[0]
            mono_reshaped = output["mono_dist"].reshape(1, natoms, n_dcm)
            dipo_reshaped = output["dipo_dist"].reshape(1, natoms, n_dcm, 3)
            
            # DCMNet ESP
            mono_flat = mono_reshaped[0].reshape(-1)
            dipo_flat = jnp.moveaxis(dipo_reshaped[0], -1, -2).reshape(-1, 3)
            from mmml.dcmnet.dcmnet.electrostatics import calc_esp
            esp_pred_dcmnet = calc_esp(dipo_flat, mono_flat, batch["vdw_surface"][0])
            
            # PhysNet ESP (from atomic charges)
            charges_physnet = output["charges_as_mono"]
            positions_physnet = batch["R"].reshape(natoms, 3)
            atom_mask_plot = batch["atom_mask"].reshape(natoms)
            
            # Apply mask to handle padding
            charges_masked = charges_physnet * atom_mask_plot
            
            # Vectorized ESP computation
            vdw_grid = batch["vdw_surface"][0]
            distances = jnp.linalg.norm(vdw_grid[:, None, :] - positions_physnet[None, :, :], axis=2)
            esp_pred_physnet = jnp.sum(charges_masked[None, :] / (distances + 1e-10), axis=1)
            
            esp_pred_dcmnet_list.append(np.array(esp_pred_dcmnet))
            esp_pred_physnet_list.append(np.array(esp_pred_physnet))
            esp_true_list.append(np.array(batch['esp'][0]))
    
    # Convert to arrays
    energies_pred = np.array(energies_pred)
    energies_true = np.array(energies_true)
    
    forces_pred = np.concatenate([f.reshape(-1) for f in forces_pred])
    forces_true = np.concatenate([f.reshape(-1) for f in forces_true])
    # Remove padding zeros
    forces_mask = forces_true != 0
    forces_pred = forces_pred[forces_mask]
    forces_true = forces_true[forces_mask]
    
    dipoles_physnet_pred = np.concatenate([d.reshape(-1) for d in dipoles_physnet_pred])
    dipoles_dcmnet_pred = np.concatenate([d.reshape(-1) for d in dipoles_dcmnet_pred])
    dipoles_true = np.concatenate([d.reshape(-1) for d in dipoles_true])
    
    # Create figure with 6 subplots (2x3) to show both dipole sources
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Energy scatter
    ax = axes[0, 0]
    ax.scatter(energies_true, energies_pred, alpha=0.5, s=20)
    lims = [min(energies_true.min(), energies_pred.min()),
            max(energies_true.max(), energies_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.set_xlabel('True Energy (eV)')
    ax.set_ylabel('Predicted Energy (eV)')
    ax.set_title(f'Energy\nMAE: {np.abs(energies_true - energies_pred).mean():.3f} eV')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Forces scatter
    ax = axes[0, 1]
    ax.scatter(forces_true, forces_pred, alpha=0.3, s=10)
    lims = [min(forces_true.min(), forces_pred.min()),
            max(forces_true.max(), forces_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.set_xlabel('True Forces (eV/Å)')
    ax.set_ylabel('Predicted Forces (eV/Å)')
    ax.set_title(f'Forces\nMAE: {np.abs(forces_true - forces_pred).mean():.3f} eV/Å')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PhysNet Dipole scatter
    ax = axes[1, 0]
    ax.scatter(dipoles_true, dipoles_physnet_pred, alpha=0.5, s=20)
    lims = [min(dipoles_true.min(), dipoles_physnet_pred.min()),
            max(dipoles_true.max(), dipoles_physnet_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.set_xlabel('True Dipole (D)')
    ax.set_ylabel('Predicted Dipole (D)')
    marker = ' *' if dipole_source == 'physnet' else ''
    ax.set_title(f'Dipole - PhysNet{marker}\nMAE: {np.abs(dipoles_true - dipoles_physnet_pred).mean():.3f} D')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # DCMNet Dipole scatter
    ax = axes[1, 1]
    ax.scatter(dipoles_true, dipoles_dcmnet_pred, alpha=0.5, s=20)
    lims = [min(dipoles_true.min(), dipoles_dcmnet_pred.min()),
            max(dipoles_true.max(), dipoles_dcmnet_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.set_xlabel('True Dipole (D)')
    ax.set_ylabel('Predicted Dipole (D)')
    marker = ' *' if dipole_source == 'dcmnet' else ''
    ax.set_title(f'Dipole - DCMNet{marker}\nMAE: {np.abs(dipoles_true - dipoles_dcmnet_pred).mean():.3f} D')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ESP scatter - DCMNet
    ax = axes[1, 2]
    if esp_pred_dcmnet_list:
        esp_pred_dcmnet_all = np.concatenate([e.reshape(-1) for e in esp_pred_dcmnet_list])
        esp_true_all = np.concatenate([e.reshape(-1) for e in esp_true_list])
        ax.scatter(esp_true_all, esp_pred_dcmnet_all, alpha=0.3, s=10)
        lims = [min(esp_true_all.min(), esp_pred_dcmnet_all.min()),
                max(esp_true_all.max(), esp_pred_dcmnet_all.max())]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
        ax.set_xlabel('True ESP (Hartree/e)')
        ax.set_ylabel('Predicted ESP (Hartree/e)')
        rmse = np.sqrt(np.mean((esp_pred_dcmnet_all - esp_true_all)**2))
        # Compute R²
        ss_res = np.sum((esp_true_all - esp_pred_dcmnet_all)**2)
        ss_tot = np.sum((esp_true_all - np.mean(esp_true_all))**2)
        r2 = 1 - (ss_res / ss_tot)
        ax.set_title(f'ESP - DCMNet\nRMSE: {rmse:.6f} Ha/e, R²: {r2:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Add PhysNet ESP scatter in the empty slot
    ax = axes[0, 2]
    if esp_pred_physnet_list:
        esp_pred_physnet_all = np.concatenate([e.reshape(-1) for e in esp_pred_physnet_list])
        esp_true_all = np.concatenate([e.reshape(-1) for e in esp_true_list])
        ax.scatter(esp_true_all, esp_pred_physnet_all, alpha=0.3, s=10, color='orange')
        lims = [min(esp_true_all.min(), esp_pred_physnet_all.min()),
                max(esp_true_all.max(), esp_pred_physnet_all.max())]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
        ax.set_xlabel('True ESP (Hartree/e)')
        ax.set_ylabel('Predicted ESP (Hartree/e)')
        rmse = np.sqrt(np.mean((esp_pred_physnet_all - esp_true_all)**2))
        # Compute R²
        ss_res = np.sum((esp_true_all - esp_pred_physnet_all)**2)
        ss_tot = np.sum((esp_true_all - np.mean(esp_true_all))**2)
        r2 = 1 - (ss_res / ss_tot)
        ax.set_title(f'ESP - PhysNet Charges\nRMSE: {rmse:.6f} Ha/e, R²: {r2:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_path = save_dir / f'validation_scatter{suffix}.png'
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved scatter plots: {scatter_path}")
    
    # Create ESP example plots - compare PhysNet vs DCMNet
    for idx in range(min(n_esp_examples, len(esp_pred_dcmnet_list))):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        esp_true = esp_true_list[idx]
        esp_pred_dcmnet = esp_pred_dcmnet_list[idx]
        esp_pred_physnet = esp_pred_physnet_list[idx]
        esp_error_dcmnet = esp_pred_dcmnet - esp_true
        esp_error_physnet = esp_pred_physnet - esp_true
        
        # True ESP (shared)
        ax = axes[0, 0]
        sc = ax.scatter(range(len(esp_true)), esp_true, c=esp_true, 
                       cmap='RdBu_r', s=30, alpha=0.7)
        ax.set_xlabel('Grid Point Index')
        ax.set_ylabel('ESP (Hartree/e)')
        ax.set_title(f'True ESP (Sample {idx}){epoch_str}')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax)
        
        # DCMNet ESP
        ax = axes[0, 1]
        sc = ax.scatter(range(len(esp_pred_dcmnet)), esp_pred_dcmnet, c=esp_pred_dcmnet,
                       cmap='RdBu_r', s=30, alpha=0.7)
        ax.set_xlabel('Grid Point Index')
        ax.set_ylabel('ESP (Hartree/e)')
        ax.set_title(f'DCMNet ESP{epoch_str}')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax)
        
        # DCMNet Error
        ax = axes[0, 2]
        sc = ax.scatter(range(len(esp_error_dcmnet)), esp_error_dcmnet, c=esp_error_dcmnet,
                       cmap='RdBu_r', s=30, alpha=0.7)
        ax.set_xlabel('Grid Point Index')
        ax.set_ylabel('ESP Error (Hartree/e)')
        rmse = np.sqrt(np.mean(esp_error_dcmnet**2))
        # Compute R²
        ss_res = np.sum((esp_true - esp_pred_dcmnet)**2)
        ss_tot = np.sum((esp_true - np.mean(esp_true))**2)
        r2 = 1 - (ss_res / ss_tot)
        ax.set_title(f'DCMNet Error\nRMSE: {rmse:.6f}, R²: {r2:.4f}')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        plt.colorbar(sc, ax=ax)
        
        # Empty
        axes[1, 0].axis('off')
        
        # PhysNet ESP
        ax = axes[1, 1]
        sc = ax.scatter(range(len(esp_pred_physnet)), esp_pred_physnet, c=esp_pred_physnet,
                       cmap='RdBu_r', s=30, alpha=0.7, marker='s')
        ax.set_xlabel('Grid Point Index')
        ax.set_ylabel('ESP (Hartree/e)')
        ax.set_title(f'PhysNet ESP{epoch_str}')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax)
        
        # PhysNet Error
        ax = axes[1, 2]
        sc = ax.scatter(range(len(esp_error_physnet)), esp_error_physnet, c=esp_error_physnet,
                       cmap='RdBu_r', s=30, alpha=0.7, marker='s')
        ax.set_xlabel('Grid Point Index')
        ax.set_ylabel('ESP Error (Hartree/e)')
        rmse = np.sqrt(np.mean(esp_error_physnet**2))
        # Compute R²
        ss_res = np.sum((esp_true - esp_pred_physnet)**2)
        ss_tot = np.sum((esp_true - np.mean(esp_true))**2)
        r2 = 1 - (ss_res / ss_tot)
        ax.set_title(f'PhysNet Error\nRMSE: {rmse:.6f}, R²: {r2:.4f}')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        plt.colorbar(sc, ax=ax)
        
        plt.tight_layout()
        esp_path = save_dir / f'esp_example_{idx}{suffix}.png'
        plt.savefig(esp_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved ESP example {idx}: {esp_path}")
    
    print(f"\n✅ All plots saved to: {save_dir}")


def train_model(
    model: JointPhysNetDCMNet,
    train_data: Dict[str, np.ndarray],
    valid_data: Dict[str, np.ndarray],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    energy_w: float,
    forces_w: float,
    dipole_w: float,
    esp_w: float,
    mono_w: float,
    n_dcm: int,
    cutoff: float,
    seed: int,
    ckpt_dir: Path,
    name: str,
    print_freq: int = 1,
    grad_clip_norm: float = None,
    plot_freq: int = None,
    plot_samples: int = 100,
    plot_esp_examples: int = 2,
    dipole_source: str = 'physnet',
    restart_params: Any = None,
    start_epoch: int = 1,
) -> Any:
    """
    Main training loop.
    
    Returns
    -------
    Any
        Final model parameters
    """
    # Pre-compute edge lists for all data (huge speedup!)
    print("\nPre-computing edge lists...")
    train_data = precompute_edge_lists(train_data, cutoff=cutoff, verbose=True)
    valid_data = precompute_edge_lists(valid_data, cutoff=cutoff, verbose=True)
    
    # Initialize model
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    
    # Initialize or load parameters
    if restart_params is not None:
        print("\n🔄 Restarting from checkpoint...")
        params = restart_params
        print(f"✅ Loaded {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")
    else:
        # Create dummy batch for initialization
        dummy_batch = prepare_batch_data(train_data, np.array([0]), cutoff=cutoff)
        
        # Initialize parameters
        print("\nInitializing model parameters...")
        params = model.init(
            init_key,
            atomic_numbers=dummy_batch["Z"],
            positions=dummy_batch["R"],
            dst_idx=dummy_batch["dst_idx"],
            src_idx=dummy_batch["src_idx"],
            batch_segments=dummy_batch["batch_segments"],
            batch_size=1,
            batch_mask=dummy_batch["batch_mask"],
            atom_mask=dummy_batch["atom_mask"],
        )
        
        print(f"✅ Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")
    
    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Prepare training indices
    n_train = len(train_data['E'])
    n_valid = len(valid_data['E'])
    
    print(f"\nTraining samples: {n_train}")
    print(f"Validation samples: {n_valid}")
    print(f"Batch size: {batch_size}")
    
    # Training loop
    best_valid_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        
        # Shuffle training data
        key, shuffle_key = jax.random.split(key)
        train_indices = jax.random.permutation(shuffle_key, n_train)
        
        # Training phase
        train_losses = []
        n_batches = (n_train + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            batch_indices = train_indices[start_idx:end_idx]
            
            # Prepare batch
            batch = prepare_batch_data(
                train_data, 
                np.array(batch_indices), 
                cutoff=cutoff
            )
            
            # Training step
            params, opt_state, loss, losses = train_step(
                params=params,
                opt_state=opt_state,
                batch=batch,
                model_apply=model.apply,
                optimizer_update=optimizer.update,
                energy_w=energy_w,
                forces_w=forces_w,
                dipole_w=dipole_w,
                esp_w=esp_w,
                mono_w=mono_w,
                batch_size=len(batch_indices),
                n_dcm=n_dcm,
                clip_norm=grad_clip_norm,
                dipole_source=dipole_source,
            )
            
            train_losses.append({k: float(v) for k, v in losses.items()})
        
        # Average training losses
        train_loss_avg = {
            k: np.mean([loss[k] for loss in train_losses])
            for k in train_losses[0].keys()
        }
        
        # Validation phase
        valid_losses = []
        n_valid_batches = (n_valid + batch_size - 1) // batch_size
        
        # Collect predictions for statistics
        all_energy_pred = []
        all_energy_true = []
        all_forces_pred = []
        all_forces_true = []
        
        for batch_idx in range(n_valid_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_valid)
            batch_indices = np.arange(start_idx, end_idx)
            
            batch = prepare_batch_data(
                valid_data,
                batch_indices,
                cutoff=cutoff
            )
            
            _, losses, output = eval_step(
                params=params,
                batch=batch,
                model_apply=model.apply,
                energy_w=energy_w,
                forces_w=forces_w,
                dipole_w=dipole_w,
                esp_w=esp_w,
                mono_w=mono_w,
                batch_size=len(batch_indices),
                n_dcm=n_dcm,
                dipole_source=dipole_source,
            )
            
            valid_losses.append({k: float(v) for k, v in losses.items()})
            
            # Collect for statistics
            all_energy_pred.extend(np.array(output['energy']).flatten())
            all_energy_true.extend(np.array(batch['E']).flatten())
            all_forces_pred.extend(np.array(output['forces']).flatten())
            all_forces_true.extend(np.array(batch['F']).flatten())
        
        # Average validation losses
        valid_loss_avg = {
            k: np.mean([loss[k] for loss in valid_losses])
            for k in valid_losses[0].keys()
        }
        
        # Compute validation set statistics
        valid_stats = {
            'energy_mean': np.mean(all_energy_true),
            'energy_std': np.std(all_energy_true),
            'forces_mean': np.mean([f for f in all_forces_true if f != 0]),  # Exclude padding
            'forces_std': np.std([f for f in all_forces_true if f != 0]),
        }
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        if epoch % print_freq == 0:
            print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss_avg['total']:.6f}")
            print(f"    Energy: {train_loss_avg['energy']:.6f}")
            print(f"    Forces: {train_loss_avg['forces']:.6f}")
            print(f"    Dipole: {train_loss_avg['dipole']:.6f}")
            print(f"    ESP: {train_loss_avg['esp']:.6f}")
            print(f"    Monopole: {train_loss_avg['monopole']:.6f}")
            print(f"    Total Charge: {train_loss_avg['total_charge']:.6f}")
            print(f"  Valid Loss: {valid_loss_avg['total']:.6f}")
            
            # Print Coulomb mixing info if enabled
            if 'coulomb_energy' in valid_loss_avg and valid_loss_avg.get('coulomb_lambda', 0) != 0:
                print(f"  Coulomb Mixing:")
                print(f"    λ (learned): {valid_loss_avg['coulomb_lambda']:.6f}")
                print(f"    E_coulomb: {valid_loss_avg['coulomb_energy']:.6f} eV")
            if 'mae_energy' in valid_loss_avg:
                # Conversion factors
                eV_to_kcal = 23.0605        # 1 eV = 23.0605 kcal/mol
                Ha_to_kcal = 627.509        # 1 Hartree = 627.509 kcal/mol
                Debye_to_eA = 0.208194      # 1 Debye = 0.208194 e·Å
                
                mae_energy_ev = valid_loss_avg['mae_energy']
                mae_forces_ev = valid_loss_avg['mae_forces']
                mae_dipole_physnet_D = valid_loss_avg['mae_dipole_physnet']
                mae_dipole_dcmnet_D = valid_loss_avg['mae_dipole_dcmnet']
                rmse_esp_physnet_Ha = valid_loss_avg['rmse_esp_physnet']
                rmse_esp_dcmnet_Ha = valid_loss_avg['rmse_esp_dcmnet']
                
                # Get validation set statistics
                e_mean = valid_stats.get('energy_mean', 0)
                e_std = valid_stats.get('energy_std', 1)
                f_mean = valid_stats.get('forces_mean', 0)
                f_std = valid_stats.get('forces_std', 1)
                
                print(f"    MAE Energy: {mae_energy_ev:.6f} eV  ({mae_energy_ev * eV_to_kcal:.6f} kcal/mol) [μ={e_mean:.3f}, σ={e_std:.3f} eV]")
                print(f"    MAE Forces: {mae_forces_ev:.6f} eV/Å  ({mae_forces_ev * eV_to_kcal:.6f} kcal/mol/Å) [μ={f_mean:.3f}, σ={f_std:.3f} eV/Å]")
                print(f"    MAE Dipole (PhysNet): {mae_dipole_physnet_D:.6f} D  ({mae_dipole_physnet_D * Debye_to_eA:.6f} e·Å)")
                print(f"    MAE Dipole (DCMNet): {mae_dipole_dcmnet_D:.6f} D  ({mae_dipole_dcmnet_D * Debye_to_eA:.6f} e·Å)")
                print(f"    RMSE ESP (PhysNet): {rmse_esp_physnet_Ha:.6f} Ha/e  ({rmse_esp_physnet_Ha * Ha_to_kcal:.6f} (kcal/mol)/e)")
                print(f"    RMSE ESP (DCMNet): {rmse_esp_dcmnet_Ha:.6f} Ha/e  ({rmse_esp_dcmnet_Ha * Ha_to_kcal:.6f} (kcal/mol)/e)")
            
            # Print constraint violations if available
            if 'total_charge' in valid_loss_avg:
                print(f"    Total Charge Violation: {valid_loss_avg['total_charge']:.6f}")
        
        # Save best model
        if valid_loss_avg['total'] < best_valid_loss:
            best_valid_loss = valid_loss_avg['total']
            save_path = ckpt_dir / name
            save_path.mkdir(exist_ok=True, parents=True)
            
            with open(save_path / 'best_params.pkl', 'wb') as f:
                pickle.dump(params, f)
            
            # Also save model config for later use
            model_config = {
                'physnet_config': dict(model.physnet_config),
                'dcmnet_config': dict(model.dcmnet_config),
                'mix_coulomb_energy': model.mix_coulomb_energy,
            }
            with open(save_path / 'model_config.pkl', 'wb') as f:
                pickle.dump(model_config, f)
            
            print(f"  💾 Saved best model (valid_loss: {best_valid_loss:.6f})")
        
        # Create plots periodically if requested
        if plot_freq is not None and epoch % plot_freq == 0 and HAS_MATPLOTLIB:
            print(f"\n📊 Creating plots at epoch {epoch}...")
            plot_validation_results(
                params=params,
                model=model,
                valid_data=valid_data,
                cutoff=cutoff,
                energy_w=energy_w,
                forces_w=forces_w,
                dipole_w=dipole_w,
                esp_w=esp_w,
                mono_w=mono_w,
                n_dcm=n_dcm,
                save_dir=ckpt_dir / name / 'plots',
                n_samples=plot_samples,
                n_esp_examples=plot_esp_examples,
                epoch=epoch,
                dipole_source=dipole_source,
            )
    
    return params


def main():
    parser = argparse.ArgumentParser(
        description="Joint PhysNet-DCMNet training for CO2 data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train-efd', type=Path, required=True,
                       help='Training energies/forces/dipoles NPZ file')
    parser.add_argument('--train-esp', type=Path, required=True,
                       help='Training ESP grids NPZ file')
    parser.add_argument('--valid-efd', type=Path, required=True,
                       help='Validation energies/forces/dipoles NPZ file')
    parser.add_argument('--valid-esp', type=Path, required=True,
                       help='Validation ESP grids NPZ file')
    
    # PhysNet hyperparameters
    parser.add_argument('--physnet-features', type=int, default=64,
                       help='PhysNet: number of features')
    parser.add_argument('--physnet-iterations', type=int, default=2,
                       help='PhysNet: message passing iterations')
    parser.add_argument('--physnet-basis', type=int, default=64,
                       help='PhysNet: number of basis functions')
    parser.add_argument('--physnet-cutoff', type=float, default=6.0,
                       help='PhysNet: cutoff distance (Angstroms)')
    parser.add_argument('--physnet-n-res', type=int, default=3,
                       help='PhysNet: number of residual blocks')
    
    # DCMNet hyperparameters
    parser.add_argument('--dcmnet-features', type=int, default=32,
                       help='DCMNet: number of features')
    parser.add_argument('--dcmnet-iterations', type=int, default=2,
                       help='DCMNet: message passing iterations')
    parser.add_argument('--dcmnet-basis', type=int, default=32,
                       help='DCMNet: number of basis functions')
    parser.add_argument('--dcmnet-cutoff', type=float, default=10.0,
                       help='DCMNet: cutoff distance (Angstroms)')
    parser.add_argument('--n-dcm', type=int, default=3,
                       help='DCMNet: distributed multipoles per atom')
    parser.add_argument('--max-degree', type=int, default=1,
                       help='DCMNet: maximum spherical harmonic degree')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (start with 1 for debugging)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Loss weights
    parser.add_argument('--energy-weight', type=float, default=1.0,
                       help='Energy loss weight')
    parser.add_argument('--forces-weight', type=float, default=50.0,
                       help='Forces loss weight')
    parser.add_argument('--dipole-weight', type=float, default=25.0,
                       help='Dipole loss weight')
    parser.add_argument('--esp-weight', type=float, default=10000.0,
                       help='ESP loss weight')
    parser.add_argument('--mono-weight', type=float, default=100.0,
                       help='Monopole constraint loss weight (enforce distributed charges sum to atomic charges)')
    parser.add_argument('--dipole-source', type=str, default='physnet',
                       choices=['physnet', 'dcmnet'],
                       help='Source for dipole in loss: physnet (from charges) or dcmnet (from distributed multipoles)')
    parser.add_argument('--mix-coulomb-energy', action='store_true', default=False,
                       help='Mix PhysNet energy with DCMNet Coulomb energy via learnable lambda')
    
    # General options
    parser.add_argument('--natoms', type=int, default=None,
                       help='Maximum number of atoms (default: auto-detect from data)')
    parser.add_argument('--max-atomic-number', type=int, default=28,
                       help='Maximum atomic number')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0,
                       help='Gradient clipping norm (None to disable)')
    parser.add_argument('--name', type=str, default='co2_joint_physnet_dcmnet',
                       help='Experiment name')
    parser.add_argument('--ckpt-dir', type=Path, default=None,
                       help='Checkpoint directory')
    parser.add_argument('--restart', type=Path, default=None,
                       help='Restart from checkpoint (path to best_params.pkl or checkpoint directory)')
    parser.add_argument('--print-freq', type=int, default=1,
                       help='Print frequency (epochs)')
    parser.add_argument('--plot-results', action='store_true', default=False,
                       help='Create validation plots after training')
    parser.add_argument('--plot-freq', type=int, default=None,
                       help='Create validation plots every N epochs during training (None to disable)')
    parser.add_argument('--plot-samples', type=int, default=100,
                       help='Number of validation samples to plot')
    parser.add_argument('--plot-esp-examples', type=int, default=2,
                       help='Number of ESP examples to visualize')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Joint PhysNet-DCMNet Training - CO2 Data")
    print("="*70)
    
    # Validate input files
    for name, path in [
        ('Train EFD', args.train_efd),
        ('Train ESP', args.train_esp),
        ('Valid EFD', args.valid_efd),
        ('Valid ESP', args.valid_esp),
    ]:
        if not path.exists():
            print(f"❌ Error: {name} file not found: {path}")
            sys.exit(1)
    
    print(f"\n📁 Data Files:")
    print(f"  Train EFD:  {args.train_efd}")
    print(f"  Train ESP:  {args.train_esp}")
    print(f"  Valid EFD:  {args.valid_efd}")
    print(f"  Valid ESP:  {args.valid_esp}")
    
    # Setup checkpoint directory
    if args.ckpt_dir is None:
        ckpt_dir = BASE_CKPT_DIR
    else:
        ckpt_dir = args.ckpt_dir
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"  Checkpoints: {ckpt_dir / args.name}")
    
    # Load data
    print(f"\n{'#'*70}")
    print("# Loading Data")
    print(f"{'#'*70}\n")
    
    print("Loading training data...")
    train_data = load_combined_data(args.train_efd, args.train_esp, verbose=args.verbose)
    
    print("\nLoading validation data...")
    valid_data = load_combined_data(args.valid_efd, args.valid_esp, verbose=args.verbose)
    
    # Auto-detect or validate natoms
    data_natoms = train_data['R'].shape[1]
    max_actual_atoms = int(max(np.max(train_data['N']), np.max(valid_data['N'])))
    
    if args.natoms is None:
        # Auto-detect: use data padding
        args.natoms = data_natoms
        print(f"\n✅ Auto-detected natoms={args.natoms} from data padding")
    else:
        # User specified: validate
        if args.natoms < max_actual_atoms:
            print(f"\n❌ Error: --natoms {args.natoms} is too small!")
            print(f"   Data contains molecules with up to {max_actual_atoms} atoms")
            print(f"   Use --natoms >= {max_actual_atoms}")
            sys.exit(1)
        elif args.natoms < data_natoms:
            # Valid reduction: resize arrays
            print(f"\n📏 Resizing data from {data_natoms} to {args.natoms} atoms...")
            train_data = resize_data_padding(train_data, args.natoms, verbose=args.verbose)
            valid_data = resize_data_padding(valid_data, args.natoms, verbose=args.verbose)
        elif args.natoms > data_natoms:
            # Expansion: resize arrays
            print(f"\n📏 Expanding data from {data_natoms} to {args.natoms} atoms...")
            train_data = resize_data_padding(train_data, args.natoms, verbose=args.verbose)
            valid_data = resize_data_padding(valid_data, args.natoms, verbose=args.verbose)
    
    print(f"\n✅ Data loaded:")
    print(f"  Training samples: {len(train_data['E'])}")
    print(f"  Validation samples: {len(valid_data['E'])}")
    print(f"  Padded to: {args.natoms} atoms")
    print(f"  Max actual atoms: {max_actual_atoms}")
    
    # Build models
    print(f"\n{'#'*70}")
    print("# Building Joint Model")
    print(f"{'#'*70}\n")
    
    physnet_config = {
        'features': args.physnet_features,
        'max_degree': 0,  # PhysNet typically uses degree 0
        'num_iterations': args.physnet_iterations,
        'num_basis_functions': args.physnet_basis,
        'cutoff': args.physnet_cutoff,
        'max_atomic_number': args.max_atomic_number,
        'charges': True,  # MUST be True for charge prediction
        'natoms': args.natoms,
        'total_charge': 0.0,
        'n_res': args.physnet_n_res,
        'zbl': False,
        'use_energy_bias': True,
        'debug': False,
        'efa': False,
    }
    
    dcmnet_config = {
        'features': args.dcmnet_features,
        'max_degree': args.max_degree,
        'num_iterations': args.dcmnet_iterations,
        'num_basis_functions': args.dcmnet_basis,
        'cutoff': args.dcmnet_cutoff,
        'max_atomic_number': args.max_atomic_number,
        'n_dcm': args.n_dcm,
        'include_pseudotensors': False,
    }
    
    print("PhysNet configuration:")
    for k, v in physnet_config.items():
        print(f"  {k}: {v}")
    
    print("\nDCMNet configuration:")
    for k, v in dcmnet_config.items():
        print(f"  {k}: {v}")
    
    model = JointPhysNetDCMNet(
        physnet_config=physnet_config,
        dcmnet_config=dcmnet_config,
        mix_coulomb_energy=args.mix_coulomb_energy,
    )
    
    print(f"\n✅ Joint model created")
    
    # Training setup
    print(f"\n{'#'*70}")
    print("# Training Setup")
    print(f"{'#'*70}\n")
    
    print(f"Training hyperparameters:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Random seed: {args.seed}")
    
    print(f"\nLoss weights:")
    print(f"  Energy: {args.energy_weight} (mix Coulomb: {args.mix_coulomb_energy})")
    print(f"  Forces: {args.forces_weight}")
    print(f"  Dipole: {args.dipole_weight} (source: {args.dipole_source})")
    print(f"  ESP: {args.esp_weight}")
    print(f"  Monopole constraint: {args.mono_weight}")
    
    print(f"\nTraining stability:")
    print(f"  Gradient clipping: {args.grad_clip_norm if args.grad_clip_norm else 'disabled'}")
    
    # Handle restart
    restart_params = None
    start_epoch = 1
    
    if args.restart:
        # Determine restart path
        if args.restart.is_dir():
            restart_path = args.restart / 'best_params.pkl'
        else:
            restart_path = args.restart
        
        if not restart_path.exists():
            print(f"\n❌ Error: Restart checkpoint not found: {restart_path}")
            sys.exit(1)
        
        print(f"\n🔄 Restart checkpoint: {restart_path}")
        with open(restart_path, 'rb') as f:
            restart_params = pickle.load(f)
        print(f"✅ Loaded checkpoint with {sum(x.size for x in jax.tree_util.tree_leaves(restart_params)):,} parameters")
        
        # Try to determine start epoch from checkpoint directory name or metadata
        # For now, user should manually adjust --epochs if needed
        print(f"⚠️  Starting from epoch 1 (adjust --epochs if continuing a longer run)")
        print(f"   Example: if you ran 50 epochs before, use --epochs 100 to train 50 more")
    
    # Start training
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    try:
        final_params = train_model(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            energy_w=args.energy_weight,
            forces_w=args.forces_weight,
            dipole_w=args.dipole_weight,
            esp_w=args.esp_weight,
            mono_w=args.mono_weight,
            n_dcm=args.n_dcm,
            cutoff=max(args.physnet_cutoff, args.dcmnet_cutoff),
            seed=args.seed,
            ckpt_dir=ckpt_dir,
            name=args.name,
            print_freq=args.print_freq,
            grad_clip_norm=args.grad_clip_norm,
            plot_freq=args.plot_freq,
            plot_samples=args.plot_samples,
            plot_esp_examples=args.plot_esp_examples,
            dipole_source=args.dipole_source,
            restart_params=restart_params,
            start_epoch=start_epoch,
        )
        
        print(f"\n{'='*70}")
        print("✅ TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"\nFinal parameters saved to: {ckpt_dir / args.name}")
        
        # Create validation plots if requested
        if args.plot_results:
            if HAS_MATPLOTLIB:
                # Load best parameters
                best_params_path = ckpt_dir / args.name / 'best_params.pkl'
                if best_params_path.exists():
                    with open(best_params_path, 'rb') as f:
                        plot_params = pickle.load(f)
                else:
                    plot_params = final_params
                
                plot_validation_results(
                    params=plot_params,
                    model=model,
                    valid_data=valid_data,
                    cutoff=max(args.physnet_cutoff, args.dcmnet_cutoff),
                    energy_w=args.energy_weight,
                    forces_w=args.forces_weight,
                    dipole_w=args.dipole_weight,
                    esp_w=args.esp_weight,
                    mono_w=args.mono_weight,
                    n_dcm=args.n_dcm,
                    save_dir=ckpt_dir / args.name / 'plots',
                    n_samples=args.plot_samples,
                    n_esp_examples=args.plot_esp_examples,
                    dipole_source=args.dipole_source,
                )
            else:
                print("\n⚠️  Matplotlib not installed, cannot create plots")
                print("   Install with: pip install matplotlib")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted by user")
        print(f"Checkpoints saved to: {ckpt_dir / args.name}")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
