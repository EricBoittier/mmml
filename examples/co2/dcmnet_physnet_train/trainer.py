#!/usr/bin/env python3
"""
Joint PhysNet-DCMNet Training Script with Learnable Dipole/ESP Mixing

This script trains PhysNet and DCMNet simultaneously with end-to-end gradient flow.

KEY CAPABILITIES:
---------------
1. Flexible Loss Configuration
   - Train on PhysNet, DCMNet, or mixed dipole/ESP predictions
   - Configurable loss terms with weights and metrics (L2, MAE, RMSE)
   - Support for JSON/YAML loss configuration files
   - Example: Supervise both raw and mixed outputs simultaneously

2. Learnable Charge Orientation Mixing
   - Neural network that learns mixing weights (λ) from charge distributions
   - Combines PhysNet (atom-centered) and DCMNet (distributed) predictions
   - E(3)-equivariant using spherical harmonics for orientation encoding
   - Outputs: mixed_dipole = λ·DCMNet + (1-λ)·PhysNet

3. Exponential Moving Average (EMA)
   - Smooths validation weights for better generalization (decay=0.999)
   - All evaluation uses EMA-parameterized model

4. Comprehensive Validation & Plotting
   - Validation metrics work with any batch size
   - 3D ESP error visualization
   - Detailed charge distribution plots
   - Automated checkpoint management

ARCHITECTURE:
------------
PhysNet → Atomic Charges → DCMNet → Distributed Multipoles → ESP
         ↓                          ↓                         ↓
     Dipole (Phys)             Dipole (DCM)              ESP (Phys/DCM)
                                        ↓                      ↓
                                   [Charge Mixer] → Mixed Dipole/ESP

Usage:
    python trainer.py --train-efd energies_forces_dipoles_train.npz \
                      --train-esp grids_esp_train.npz \
                      --valid-efd energies_forces_dipoles_valid.npz \
                      --valid-esp grids_esp_valid.npz \
                      --dipole-loss-sources physnet mixed \
                      --esp-loss-sources dcmnet
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pickle
from typing import Dict, Tuple, Any, Sequence, Optional
import time
import json
from dataclasses import dataclass

# Add mmml to path
repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict, freeze, unfreeze
import optax
import functools

import e3x
import ase.data

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

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


EPS = 1e-8
RADII_TABLE = jnp.array(ase.data.covalent_radii)


# ============================================================================
# Optimizer Configuration and Recommendations
# ============================================================================

def get_recommended_optimizer_config(
    dataset_size: int,
    num_features: int,
    num_atoms: int,
    optimizer_name: str = 'adamw'
) -> Dict[str, Any]:
    """
    Get recommended optimizer hyperparameters based on dataset properties.
    
    Args:
        dataset_size: Number of training samples
        num_features: Total number of model features (PhysNet + DCMNet)
        num_atoms: Maximum number of atoms in molecules
        optimizer_name: One of 'adam', 'adamw', 'rmsprop', 'muon'
    
    Returns:
        Dictionary with recommended hyperparameters
    """
    # Heuristics based on dataset complexity
    is_small_dataset = dataset_size < 1000
    is_large_model = num_features > 256 or num_atoms > 50
    
    if optimizer_name.lower() == 'adam':
        base_lr = 0.001 if is_small_dataset else 0.0005
        return {
            'learning_rate': base_lr * (0.5 if is_large_model else 1.0),
            'b1': 0.9,
            'b2': 0.999,
            'eps': 1e-8,
            'weight_decay': 0.0,  # Adam without weight decay
        }
    
    elif optimizer_name.lower() == 'adamw':
        base_lr = 0.001 if is_small_dataset else 0.0005
        # Larger models benefit from more regularization
        wd = 1e-3 if is_large_model else 1e-4
        return {
            'learning_rate': base_lr * (0.5 if is_large_model else 1.0),
            'b1': 0.9,
            'b2': 0.999,
            'eps': 1e-8,
            'weight_decay': wd,
        }
    
    elif optimizer_name.lower() == 'rmsprop':
        base_lr = 0.001 if is_small_dataset else 0.0005
        return {
            'learning_rate': base_lr * (0.7 if is_large_model else 1.0),
            'decay': 0.9,
            'eps': 1e-8,
            'momentum': 0.0,
            'weight_decay': 1e-4 if is_large_model else 1e-5,
        }
    
    elif optimizer_name.lower() == 'muon':
        # Muon (Momentum Orthogonalized by Newton's method)
        # Typically works well with higher learning rates
        base_lr = 0.01 if is_small_dataset else 0.005
        return {
            'learning_rate': base_lr * (0.5 if is_large_model else 1.0),
            'momentum': 0.95,
            'weight_decay': 1e-3 if is_large_model else 1e-4,
            'nesterov': True,
        }
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_optimizer(
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    **kwargs
) -> optax.GradientTransformation:
    """
    Create an optimizer with the specified configuration.
    
    Args:
        optimizer_name: One of 'adam', 'adamw', 'rmsprop', 'muon'
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer-specific parameters
    
    Returns:
        Optax optimizer
    """
    name = optimizer_name.lower()
    
    if name == 'adam':
        b1 = kwargs.get('b1', 0.9)
        b2 = kwargs.get('b2', 0.999)
        eps = kwargs.get('eps', 1e-8)
        return optax.adam(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
    
    elif name == 'adamw':
        b1 = kwargs.get('b1', 0.9)
        b2 = kwargs.get('b2', 0.999)
        eps = kwargs.get('eps', 1e-8)
        return optax.adamw(
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay
        )
    
    elif name == 'rmsprop':
        decay = kwargs.get('decay', 0.9)
        eps = kwargs.get('eps', 1e-8)
        momentum = kwargs.get('momentum', 0.0)
        # RMSprop with weight decay
        optimizer = optax.rmsprop(
            learning_rate=learning_rate,
            decay=decay,
            eps=eps,
            momentum=momentum
        )
        if weight_decay > 0:
            optimizer = optax.chain(
                optimizer,
                optax.add_decayed_weights(weight_decay)
            )
        return optimizer
    
    elif name == 'muon':
        # Muon optimizer - high-momentum optimizer with Newton-like preconditioning
        # Implementation using SGD with heavy ball momentum + orthogonalization
        momentum = kwargs.get('momentum', 0.95)
        nesterov = kwargs.get('nesterov', True)
        
        # Basic Muon approximation using heavy ball + weight decay
        optimizer = optax.sgd(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov
        )
        if weight_decay > 0:
            optimizer = optax.chain(
                optimizer,
                optax.add_decayed_weights(weight_decay)
            )
        
        print("⚠️  Note: Using SGD with heavy ball momentum as Muon approximation.")
        print("    For true Muon optimizer, consider installing: pip install muon-optimizer")
        
        return optimizer
    
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Supported: adam, adamw, rmsprop, muon"
        )


OPTIMIZER_CONFIGS = {
    'adam': lambda ds_size, features, atoms: get_recommended_optimizer_config(
        ds_size, features, atoms, 'adam'
    ),
    'adamw': lambda ds_size, features, atoms: get_recommended_optimizer_config(
        ds_size, features, atoms, 'adamw'
    ),
    'rmsprop': lambda ds_size, features, atoms: get_recommended_optimizer_config(
        ds_size, features, atoms, 'rmsprop'
    ),
    'muon': lambda ds_size, features, atoms: get_recommended_optimizer_config(
        ds_size, features, atoms, 'muon'
    ),
}

LOSS_SOURCE_CHOICES = ("physnet", "dcmnet", "mixed")
LOSS_METRIC_CHOICES = ("l2", "mae", "rmse")


@dataclass(frozen=True)
class LossTerm:
    """Configuration for an individual loss component."""

    source: str
    weight: float
    metric: str = "l2"
    name: Optional[str] = None

    @property
    def key(self) -> str:
        return self.name or self.source


def _build_loss_term(entry: Dict[str, Any], *, default_metric: str = "l2") -> LossTerm:
    """Build a LossTerm from a dictionary entry."""
    source = entry.get("source")
    if source not in LOSS_SOURCE_CHOICES:
        raise ValueError(f"Invalid loss source '{source}'. Valid options: {LOSS_SOURCE_CHOICES}")
    metric = entry.get("metric", default_metric)
    if metric not in LOSS_METRIC_CHOICES:
        raise ValueError(f"Invalid loss metric '{metric}'. Valid options: {LOSS_METRIC_CHOICES}")
    weight = float(entry.get("weight", 1.0))
    name = entry.get("name")
    return LossTerm(source=source, weight=weight, metric=metric, name=name)


def load_loss_terms_config(path: Path) -> Tuple[Tuple[LossTerm, ...], Tuple[LossTerm, ...]]:
    """Load dipole and ESP loss term configurations from JSON or YAML file."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yml", ".yaml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to load YAML loss configs but is not installed.")
            data = yaml.safe_load(fh)
        else:
            data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError("Loss configuration file must define a mapping with 'dipole' and/or 'esp' keys")

    dipole_cfg = data.get("dipole", [])
    esp_cfg = data.get("esp", [])

    if dipole_cfg and not isinstance(dipole_cfg, (list, tuple)):
        raise ValueError("'dipole' section of loss config must be a list")
    if esp_cfg and not isinstance(esp_cfg, (list, tuple)):
        raise ValueError("'esp' section of loss config must be a list")

    dipole_terms = tuple(_build_loss_term(entry, default_metric="l2") for entry in dipole_cfg)
    esp_terms = tuple(_build_loss_term(entry, default_metric="l2") for entry in esp_cfg)

    return dipole_terms, esp_terms


def _orientation_feature(positions: jnp.ndarray, weights: jnp.ndarray, max_degree: int) -> jnp.ndarray:
    """Compute spherical harmonic orientation descriptors weighted by charges.
    
    Uses weight-based masking instead of boolean indexing for JIT compatibility.
    Positions/weights with near-zero weights are automatically down-weighted in the computation.
    """
    # Compute features using all positions but weighted by their weights
    # Near-zero weights will have negligible contribution
    w_abs = jnp.abs(weights)
    total = jnp.sum(w_abs) + EPS
    
    # Weighted center of mass
    center = jnp.sum(positions * w_abs[:, None], axis=0) / total
    rel = positions - center
    norms = jnp.linalg.norm(rel, axis=1)
    rel_unit = rel / jnp.maximum(norms[:, None], EPS)
    
    # Compute spherical harmonics for all positions
    sh = e3x.so3.spherical_harmonics(rel_unit, max_degree=max_degree)
    
    # Weight-averaged spherical harmonics
    weighted_sh = jnp.sum(sh * weights[:, None], axis=0) / total
    
    # Radial statistics (weighted)
    mean_radial = jnp.sum(norms * w_abs) / total
    var_radial = jnp.sum(((norms - mean_radial) ** 2) * w_abs) / total
    radial_stats = jnp.array([mean_radial, var_radial])
    
    feature = jnp.concatenate([weighted_sh, radial_stats])
    
    # Zero out features if total weight is negligible (all charges masked)
    has_valid = total > 1e-6
    return jnp.where(has_valid, feature, jnp.zeros_like(feature))


def _compute_esp_single(
    mono_mol: jnp.ndarray,
    dipo_mol: jnp.ndarray,
    vdw_mol: jnp.ndarray,
    esp_target: jnp.ndarray,
    atom_pos: jnp.ndarray,
    atom_mask_mol: jnp.ndarray,
    atomic_nums_mol: jnp.ndarray,
    phys_charges: jnp.ndarray,
    esp_min_distance: float,
    esp_max_value: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute ESP predictions using masked operations (no boolean indexing for JIT)."""
    
    # Use masking instead of boolean indexing - weight by atom_mask
    # mono_mol: (natoms, n_dcm), dipo_mol: (natoms, n_dcm, 3)
    # Expand mask to DCM dimensions
    mask_expanded = atom_mask_mol[:, None]  # (natoms, 1)
    mono_masked = mono_mol * mask_expanded  # (natoms, n_dcm)
    dipo_masked = dipo_mol * mask_expanded[..., None]  # (natoms, n_dcm, 3)
    
    # Flatten for ESP calculation
    mono_flat = mono_masked.reshape(-1)  # (natoms*n_dcm,)
    dipo_flat = dipo_masked.reshape(-1, 3)  # (natoms*n_dcm, 3)
    esp_pred_dcm = calc_esp(dipo_flat, mono_flat, vdw_mol)
    
    # Compute distances from grid to all atoms (including masked)
    distances = jnp.linalg.norm(vdw_mol[:, None, :] - atom_pos[None, :, :], axis=2)  # (ngrid, natoms)
    
    # Get atomic radii and apply mask
    atomic_nums_int = atomic_nums_mol.astype(jnp.int32)
    atomic_radii = jnp.take(RADII_TABLE, atomic_nums_int)  # (natoms,)
    
    # For distance-based masking, only consider real (unmasked) atoms
    # Set distances to masked atoms to infinity
    distances_masked = jnp.where(
        atom_mask_mol[None, :] > 0.5,
        distances,
        1e10  # Large distance for masked atoms
    )
    
    # Check if any REAL atom is too close
    within_cutoff = distances_masked < (2.0 * atomic_radii[None, :])
    distance_mask = (~jnp.any(within_cutoff, axis=1)).astype(jnp.float32)
    
    if esp_min_distance > 0:
        min_dist = jnp.min(distances_masked, axis=1)
        distance_mask = distance_mask * (min_dist >= esp_min_distance).astype(jnp.float32)
    
    if esp_max_value < 1e9:
        distance_mask = distance_mask * (jnp.abs(esp_target) <= esp_max_value).astype(jnp.float32)
    
    # PhysNet ESP: sum over atoms with masking
    # Use masked charges and distances
    charges_masked = phys_charges * atom_mask_mol  # (natoms,)
    esp_pred_phys = jnp.sum(charges_masked[None, :] / (distances + 1e-10), axis=1)
    
    return esp_pred_dcm, esp_pred_phys, distance_mask


def _compute_esp_batch(
    mono: jnp.ndarray,
    dipo: jnp.ndarray,
    vdw: jnp.ndarray,
    esp_target: jnp.ndarray,
    atom_pos: jnp.ndarray,
    atom_mask: jnp.ndarray,
    atomic_nums: jnp.ndarray,
    physnet_charges: jnp.ndarray,
    esp_min_distance: float,
    esp_max_value: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    compute_fn = lambda mono_mol, dipo_mol, vdw_mol, esp_mol, atom_pos_mol, atom_mask_mol, atomic_nums_mol, phys_charges_mol: _compute_esp_single(
        mono_mol,
        dipo_mol,
        vdw_mol,
        esp_mol,
        atom_pos_mol,
        atom_mask_mol,
        atomic_nums_mol,
        phys_charges_mol,
        esp_min_distance,
        esp_max_value,
    )
    return jax.vmap(compute_fn)(
        mono,
        dipo,
        vdw,
        esp_target,
        atom_pos,
        atom_mask,
        atomic_nums,
        physnet_charges,
    )


class NonEquivariantChargeModel(nn.Module):
    """
    Non-equivariant model that predicts distributed charges with explicit Cartesian displacements.
    
    Instead of using spherical harmonics (equivariant), this model directly predicts:
    - n charges per atom (scalars)
    - n displacement vectors per atom (3D Cartesian coordinates)
    
    This is simpler but breaks rotational equivariance since displacements are in Cartesian space.
    
    Attributes
    ----------
    features : int
        Hidden layer size for MLP
    n_dcm : int
        Number of distributed charges per atom
    max_atomic_number : int
        Maximum atomic number for embedding
    num_layers : int
        Number of MLP layers (default: 3)
    max_displacement : float
        Maximum displacement distance in Angstroms (default: 1.0)
    """
    features: int
    n_dcm: int
    max_atomic_number: int
    num_layers: int = 3
    max_displacement: float = 1.0
    
    @nn.compact
    def __call__(
        self,
        atomic_numbers: jnp.ndarray,
        positions: jnp.ndarray,
        charges_from_physnet: jnp.ndarray,
        atom_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict distributed charges and their positions.
        
        Parameters
        ----------
        atomic_numbers : jnp.ndarray
            Atomic numbers, shape (batch_size * natoms,)
        positions : jnp.ndarray
            Atomic positions, shape (batch_size * natoms, 3)
        charges_from_physnet : jnp.ndarray
            Atomic charges from PhysNet, shape (batch_size * natoms,)
        atom_mask : jnp.ndarray
            Atom mask, shape (batch_size * natoms,)
            
        Returns
        -------
        charges_dist : jnp.ndarray
            Distributed charges, shape (batch_size * natoms, n_dcm)
        positions_dist : jnp.ndarray
            Absolute positions of distributed charges, shape (batch_size * natoms, n_dcm, 3)
        """
        # Atomic number embedding
        atomic_embedding = nn.Embed(
            num_embeddings=self.max_atomic_number + 1,
            features=self.features,
            name="atomic_embedding"
        )(atomic_numbers)
        
        # Combine atomic embedding with charge information
        # charges_from_physnet: (batch*natoms,)
        x = jnp.concatenate([
            atomic_embedding,  # (batch*natoms, features)
            charges_from_physnet[:, None],  # (batch*natoms, 1)
        ], axis=-1)  # (batch*natoms, features+1)
        
        # MLP to process features
        for i in range(self.num_layers):
            x = nn.Dense(self.features, name=f"layer_{i}")(x)
            x = nn.silu(x)
        
        # Predict charges for each distributed charge point
        # Output: (batch*natoms, n_dcm)
        charges_dist = nn.Dense(self.n_dcm, name="charges_head")(x)
        
        # Predict displacement vectors for each distributed charge point
        # Output: (batch*natoms, n_dcm * 3)
        displacements_flat = nn.Dense(self.n_dcm * 3, name="displacements_head")(x)
        
        # Reshape to (batch*natoms, n_dcm, 3)
        displacements = displacements_flat.reshape(-1, self.n_dcm, 3)
        
        # Apply tanh to bound displacements, then scale by max_displacement
        # This ensures displacements stay within reasonable range
        displacements = jnp.tanh(displacements) * self.max_displacement
        
        # Compute absolute positions by adding displacements to atom positions
        # positions: (batch*natoms, 3) -> (batch*natoms, 1, 3)
        # displacements: (batch*natoms, n_dcm, 3)
        positions_dist = positions[:, None, :] + displacements  # (batch*natoms, n_dcm, 3)
        
        # Apply atom mask to outputs
        mask_expanded = atom_mask[:, None]  # (batch*natoms, 1)
        charges_dist = charges_dist * mask_expanded
        positions_dist = positions_dist * mask_expanded[:, :, None]
        
        return charges_dist, positions_dist


class ChargeOrientationMixer(nn.Module):
    """Learn scalar mixing factors from charge orientations."""

    hidden_dim: int = 64
    max_degree: int = 2
    num_layers: int = 2
    output_dim: int = 2

    @nn.compact
    def __call__(
        self,
        phys_positions: jnp.ndarray,
        phys_charges: jnp.ndarray,
        dcm_positions: jnp.ndarray,
        dcm_charges: jnp.ndarray,
        atom_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        def per_sample(pp, pc, dp, dc, mask):
            # Use mask weighting instead of boolean indexing for JIT compatibility
            # Pass masked arrays to _orientation_feature which handles masking internally
            phys_feat = _orientation_feature(pp, pc * mask, self.max_degree)
            
            # For DCM features, flatten and weight by mask
            # dp: (natoms, n_dcm, 3), dc: (natoms, n_dcm)
            dp_flat = dp.reshape(-1, 3)  # (natoms*n_dcm, 3)
            dc_flat = dc.reshape(-1)     # (natoms*n_dcm,)
            # Expand mask to match DCM shape
            mask_expanded = jnp.repeat(mask, dp.shape[1])  # (natoms*n_dcm,)
            dcm_feat = _orientation_feature(dp_flat, dc_flat * mask_expanded, self.max_degree)
            
            return jnp.concatenate([phys_feat, dcm_feat, dcm_feat - phys_feat])

        features = jax.vmap(per_sample)(phys_positions, phys_charges, dcm_positions, dcm_charges, atom_mask)

        x = nn.Dense(self.hidden_dim)(features)
        x = nn.silu(x)
        for _ in range(self.num_layers - 1):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.silu(x)
        x = nn.Dense(self.output_dim)(x)
        return nn.sigmoid(x)


class JointPhysNetNonEquivariant(nn.Module):
    """
    Joint PhysNet + Non-Equivariant charge model.
    
    Architecture:
    1. PhysNet predicts atomic charges (supervised by molecular dipole)
    2. Non-equivariant MLP predicts n charges and n Cartesian displacements per atom
    3. Distributed charges used for ESP fitting
    4. Full gradient flow from ESP loss back to PhysNet parameters
    
    This model is NOT rotationally equivariant (displacements are in Cartesian coords).
    
    Attributes
    ----------
    physnet_config : dict
        Configuration for PhysNet EF model
    noneq_config : dict
        Configuration for NonEquivariantChargeModel
    mix_coulomb_energy : bool
        If True, mix PhysNet energy with Coulomb energy from distributed charges
    """
    physnet_config: Dict[str, Any]
    noneq_config: Dict[str, Any]
    mix_coulomb_energy: bool = False
    mixer_config: Optional[Dict[str, Any]] = None
    
    def setup(self):
        """Initialize PhysNet and non-equivariant charge model."""
        self.physnet = EF(**self.physnet_config)
        self.noneq_model = NonEquivariantChargeModel(**self.noneq_config)
        
        if self.mix_coulomb_energy:
            self.coulomb_lambda = self.param(
                "coulomb_lambda",
                lambda rng, shape: jnp.ones(shape) * 0.1,
                (1,)
            )
        
        mixer_cfg = self.mixer_config or {}
        self.charge_mixer = ChargeOrientationMixer(**mixer_cfg)
    
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
        """Forward pass through both models."""
        
        # 1. PhysNet forward pass
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
        
        # 2. Extract charges
        charges = physnet_output["charges"]
        charges_squeezed = jnp.squeeze(charges)
        charges_masked = charges_squeezed * atom_mask
        sum_charges = jax.ops.segment_sum(
            charges_masked,
            segment_ids=batch_segments,
            num_segments=batch_size
        )
        
        # 3. Non-equivariant model predicts distributed charges and positions
        mono_dist, dipo_dist = self.noneq_model(
            atomic_numbers=atomic_numbers,
            positions=positions,
            charges_from_physnet=charges_squeezed,
            atom_mask=atom_mask,
        )
        # mono_dist: (batch*natoms, n_dcm) - charge values
        # dipo_dist: (batch*natoms, n_dcm, 3) - absolute positions
        
        # 4. Compute energies and mixing (same as DCMNet version)
        energy_reshaped = physnet_output["energy"].reshape(batch_size)
        natoms = charges_squeezed.shape[0] // batch_size
        n_dcm = mono_dist.shape[1]
        
        positions_batched = positions.reshape(batch_size, natoms, 3)
        charges_batched = charges_squeezed.reshape(batch_size, natoms)
        atom_mask_batched = atom_mask.reshape(batch_size, natoms)
        atomic_numbers_batched = atomic_numbers.reshape(batch_size, natoms)
        mono_batched = mono_dist.reshape(batch_size, natoms, n_dcm)
        dipo_batched = dipo_dist.reshape(batch_size, natoms, n_dcm, 3)
        
        if self.mix_coulomb_energy:
            charges_reshaped = mono_dist.reshape(batch_size, natoms * n_dcm)
            positions_reshaped = dipo_dist.reshape(batch_size, natoms * n_dcm, 3)
            
            def compute_coulomb_single(charges, positions):
                diff = positions[:, None, :] - positions[None, :, :]
                distances = jnp.linalg.norm(diff, axis=-1)
                distances = jnp.where(distances < 1e-6, 1e6, distances)
                pairwise_energy = charges[:, None] * charges[None, :] / (distances * 1.88973)
                coulomb_energy_hartree = 0.5 * jnp.sum(pairwise_energy)
                return coulomb_energy_hartree * 27.2114
            
            coulomb_energies = jax.vmap(compute_coulomb_single)(charges_reshaped, positions_reshaped)
            lambda_val = self.coulomb_lambda[0]
            energy_reshaped = energy_reshaped + lambda_val * coulomb_energies
            coulomb_energy_out = jnp.mean(coulomb_energies)
            lambda_out = lambda_val
        else:
            coulomb_energy_out = jnp.array(0.0)
            lambda_out = jnp.array(0.0)
        
        # Compute dipoles
        masses_lookup = jnp.array(ase.data.atomic_masses)
        masses = jnp.take(masses_lookup, atomic_numbers_batched.astype(jnp.int32)) * atom_mask_batched
        total_mass = jnp.sum(masses, axis=1) + EPS
        mass_weighted_pos = positions_batched * masses[..., None]
        com = jnp.sum(mass_weighted_pos, axis=1) / total_mass[:, None]
        
        mono_masked = mono_batched * atom_mask_batched[..., None]
        dipo_rel = dipo_batched - com[:, None, None, :]
        dipole_noneq = jnp.sum(mono_masked[..., None] * dipo_rel, axis=(1, 2))
        dipoles_physnet = physnet_output["dipoles"].reshape(batch_size, 3)
        
        lambda_outputs = self.charge_mixer(
            positions_batched,
            charges_batched * atom_mask_batched,
            dipo_batched,
            mono_masked,
            atom_mask_batched,
        )
        lambda_dipole = lambda_outputs[:, 0]
        lambda_esp = lambda_outputs[:, 1]
        dipoles_mixed = (
            lambda_dipole[:, None] * dipole_noneq
            + (1.0 - lambda_dipole[:, None]) * dipoles_physnet
        )
        
        forces_reshaped = physnet_output["forces"].reshape(-1, 3)
        dipoles_reshaped = physnet_output["dipoles"].reshape(batch_size, 3)
        
        return {
            "energy": energy_reshaped,
            "forces": forces_reshaped,
            "dipoles": dipoles_reshaped,
            "charges": charges,
            "sum_charges": physnet_output.get("sum_charges", sum_charges),
            "mono_dist": mono_dist,
            "dipo_dist": dipo_dist,
            "charges_as_mono": charges_squeezed,
            "coulomb_energy": coulomb_energy_out,
            "coulomb_lambda": lambda_out,
            "dipoles_dcmnet": dipole_noneq,  # Keep name for compatibility
            "dipoles_mixed": dipoles_mixed,
            "lambda_dipole": lambda_dipole,
            "lambda_esp": lambda_esp,
        }


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
    mixer_config: Optional[Dict[str, Any]] = None
    
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

        mixer_cfg = self.mixer_config or {}
        self.charge_mixer = ChargeOrientationMixer(**mixer_cfg)
    
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
        natoms = charges_squeezed.shape[0] // batch_size
        n_dcm = mono_dist.shape[1]

        positions_batched = positions.reshape(batch_size, natoms, 3)
        charges_batched = charges_squeezed.reshape(batch_size, natoms)
        atom_mask_batched = atom_mask.reshape(batch_size, natoms)
        atomic_numbers_batched = atomic_numbers.reshape(batch_size, natoms)
        mono_batched = mono_dist.reshape(batch_size, natoms, n_dcm)
        dipo_batched = dipo_dist.reshape(batch_size, natoms, n_dcm, 3)
        
        if self.mix_coulomb_energy:
            # Compute Coulomb energy from DCMNet distributed charges
            # Note: dipo_dist contains the POSITIONS of distributed charges, not dipole moments!
            # E_coulomb = (1/2) Σᵢⱼ qᵢqⱼ/rᵢⱼ (pairwise interactions)
            # Get all charge positions and values
            # mono_dist: (batch*natoms, n_dcm) - charge values
            # dipo_dist: (batch*natoms, n_dcm, 3) - charge positions in Angstrom
            charges_reshaped = mono_dist.reshape(batch_size, natoms * n_dcm)  # (batch, natoms*n_dcm)
            positions_reshaped = dipo_dist.reshape(batch_size, natoms * n_dcm, 3)  # (batch, natoms*n_dcm, 3)
            
            # Compute Coulomb energy per molecule using vmap
            def compute_coulomb_single(charges, positions):
                """Compute Coulomb energy for a single molecule."""
                # Pairwise distances between all distributed charges
                diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
                distances = jnp.linalg.norm(diff, axis=-1)  # (N, N)
                # Avoid self-interaction
                distances = jnp.where(distances < 1e-6, 1e6, distances)
                # Coulomb energy: E = (1/2) Σᵢⱼ qᵢqⱼ/rᵢⱼ
                # Convert distance from Angstrom to Bohr (rᵢⱼ * 1.88973)
                # Energy in Hartree, then convert to eV (* 27.2114)
                pairwise_energy = charges[:, None] * charges[None, :] / (distances * 1.88973)
                coulomb_energy_hartree = 0.5 * jnp.sum(pairwise_energy)
                return coulomb_energy_hartree * 27.2114  # Convert Ha to eV
            
            # Vectorize over batch dimension
            coulomb_energies = jax.vmap(compute_coulomb_single)(charges_reshaped, positions_reshaped)  # (batch_size,)
            
            # Mix energies: E_total = E_physnet + λ * E_coulomb
            lambda_val = self.coulomb_lambda[0]
            energy_reshaped = energy_reshaped + lambda_val * coulomb_energies
            
            # Store for monitoring (mean across batch)
            coulomb_energy_out = jnp.mean(coulomb_energies)
            lambda_out = lambda_val
        else:
            coulomb_energy_out = jnp.array(0.0)
            lambda_out = jnp.array(0.0)
        masses_lookup = jnp.array(ase.data.atomic_masses)
        masses = jnp.take(masses_lookup, atomic_numbers_batched.astype(jnp.int32)) * atom_mask_batched
        total_mass = jnp.sum(masses, axis=1) + EPS
        mass_weighted_pos = positions_batched * masses[..., None]
        com = jnp.sum(mass_weighted_pos, axis=1) / total_mass[:, None]

        mono_masked = mono_batched * atom_mask_batched[..., None]
        dipo_rel = dipo_batched - com[:, None, None, :]
        dipole_dcmnet = jnp.sum(mono_masked[..., None] * dipo_rel, axis=(1, 2))
        dipoles_physnet = physnet_output["dipoles"].reshape(batch_size, 3)

        lambda_outputs = self.charge_mixer(
            positions_batched,
            charges_batched * atom_mask_batched,
            dipo_batched,
            mono_masked,
            atom_mask_batched,
        )
        lambda_dipole = lambda_outputs[:, 0]
        lambda_esp = lambda_outputs[:, 1]
        dipoles_mixed = (
            lambda_dipole[:, None] * dipole_dcmnet
            + (1.0 - lambda_dipole[:, None]) * dipoles_physnet
        )
        
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
            "dipoles_dcmnet": dipole_dcmnet,
            "dipoles_mixed": dipoles_mixed,
            "lambda_dipole": lambda_dipole,
            "lambda_esp": lambda_esp,
        }


def load_combined_data(efd_file: Path, esp_file: Path, subtract_atom_energies: bool = True, verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Load and combine EFD and ESP data from separate NPZ files.
    
    Parameters
    ----------
    efd_file : Path
        Path to energies_forces_dipoles NPZ file
    esp_file : Path
        Path to grids_esp NPZ file
    subtract_atom_energies : bool
        Whether to subtract reference atomic energies from molecular energies (default: True)
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
    
    # Reference atomic energies (in eV) from PBE/def2-TZVP
    # These are approximate isolated atom energies
    ATOM_ENERGIES = {
        1: -13.587,      # H
        6: -1029.499,    # C
        7: -1484.274,    # N
        8: -2041.878,    # O
        9: -2713.473,    # F
        15: -8978.229,   # P
        16: -10831.086,  # S
        17: -12516.444,  # Cl
    }
    
    energies = efd_data['E'].copy()
    
    # Subtract atomic energies
    if subtract_atom_energies:
        Z = esp_data['Z']  # (n_samples, natoms)
        N = esp_data['N']  # (n_samples,)
        
        for i in range(len(energies)):
            n_atoms = int(N[i])
            atomic_nums = Z[i, :n_atoms]
            atom_energy_sum = sum(ATOM_ENERGIES.get(int(z), 0.0) for z in atomic_nums)
            energies[i] -= atom_energy_sum
        
        if verbose:
            print(f"  ✅ Subtracted atomic energies (now relative to isolated atoms)")
    
    # Align coordinate frames: center grid on molecular COM (VECTORIZED)
    # ESP grids and atom positions may be in different reference frames
    # Solution: shift vdw_surface so its COM matches atom COM for each molecule
    
    if verbose:
        print(f"  Aligning ESP grids to molecular reference frames...")
    
    # Vectorized computation of atom COMs
    # Create mask for real atoms
    n_samples = len(esp_data['N'])
    natoms_max = esp_data['R'].shape[1]
    atom_mask = np.arange(natoms_max)[None, :] < esp_data['N'][:, None]  # (n_samples, natoms)
    
    # Compute atom COM for each molecule (handling variable number of atoms)
    atom_positions_masked = esp_data['R'] * atom_mask[:, :, None]  # Zero out padding
    atom_com = atom_positions_masked.sum(axis=1) / esp_data['N'][:, None]  # (n_samples, 3)
    
    # Compute grid COM for each molecule
    grid_com = esp_data['vdw_surface'].mean(axis=1)  # (n_samples, 3)
    
    # Compute offset for each molecule
    offset = grid_com - atom_com  # (n_samples, 3)
    
    # Apply alignment (broadcast over grid points)
    vdw_surface_aligned = esp_data['vdw_surface'] - offset[:, None, :]  # (n_samples, ngrid, 3)
    
    if verbose:
        print(f"  ✅ Aligned ESP grids to molecular reference frames")
        # Show example alignment
        print(f"     Sample 0 - Atom COM: {atom_com[0]}")
        print(f"     Sample 0 - Grid COM before: {grid_com[0]}")
        print(f"     Sample 0 - Grid COM after: {vdw_surface_aligned[0].mean(axis=0)}")
        print(f"     Sample 0 - Offset corrected: {offset[0]} Å")
    
    # Combine data - ESP file should have R, Z, N as well
    # NOTE: vdw_surface is in Angstroms (do NOT convert from Bohr!)
    combined = {
        # Molecular properties
        'R': esp_data['R'],  # Angstroms
        'Z': esp_data['Z'],
        'N': esp_data['N'],
        'E': energies,  # eV
        'F': efd_data['F'],  # eV/Å
        'Dxyz': efd_data.get('Dxyz', efd_data.get('D')),  # e·Å
        # ESP properties
        'esp': esp_data['esp'],  # Hartree/e
        'vdw_surface': vdw_surface_aligned,  # Angstroms, aligned to molecular frame
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


@functools.partial(
    jax.jit,
    static_argnames=(
        "batch_size",
        "n_dcm",
        "dipole_terms",
        "esp_terms",
        "esp_min_distance",
        "esp_max_value",
    ),
)
def compute_loss(
    output: Dict[str, jnp.ndarray],
    batch: Dict[str, jnp.ndarray],
    energy_w: float,
    forces_w: float,
    mono_w: float,
    batch_size: int,
    n_dcm: int,
    dipole_terms: Sequence[LossTerm],
    esp_terms: Sequence[LossTerm],
    esp_min_distance: float = 0.0,
    esp_max_value: float = 1e10,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute weighted losses for energy, forces, dipoles, ESP, and charge constraints."""

    losses: Dict[str, jnp.ndarray] = {}

    loss_energy = optax.l2_loss(output["energy"], batch["E"]).mean()
    losses["energy"] = loss_energy

    forces_masked = output["forces"] * batch["atom_mask"][:, None]
    forces_target_masked = batch["F"] * batch["atom_mask"][:, None]
    loss_forces = optax.l2_loss(forces_masked, forces_target_masked).mean()
    losses["forces"] = loss_forces

    dipole_predictions = {
        "physnet": output["dipoles"],
        "dcmnet": output.get("dipoles_dcmnet"),
        "mixed": output.get("dipoles_mixed"),
    }

    total_dipole_loss = jnp.array(0.0, dtype=loss_energy.dtype)
    for term in dipole_terms:
        pred = dipole_predictions.get(term.source)
        if pred is None:
            raise ValueError(f"Unknown dipole source '{term.source}' in loss config")
        diff = pred - batch["D"]
        if term.metric == "l2":
            loss_term = optax.l2_loss(pred, batch["D"]).mean()
        elif term.metric == "mae":
            loss_term = jnp.abs(diff).mean()
        elif term.metric == "rmse":
            loss_term = jnp.sqrt(jnp.mean(diff ** 2))
        else:
            raise ValueError(f"Unsupported dipole metric '{term.metric}'")
        losses[f"dipole_{term.key}"] = loss_term
        total_dipole_loss = total_dipole_loss + term.weight * loss_term
    losses["dipole"] = total_dipole_loss

    natoms = output["mono_dist"].shape[0] // batch_size
    mono_reshaped = output["mono_dist"].reshape(batch_size, natoms, n_dcm)
    dipo_reshaped = output["dipo_dist"].reshape(batch_size, natoms, n_dcm, 3)
    atom_positions = batch["R"].reshape(batch_size, natoms, 3)
    atom_mask = batch["atom_mask"].reshape(batch_size, natoms)
    atomic_nums = batch["Z"].reshape(batch_size, natoms)
    physnet_charges = output["charges_as_mono"].reshape(batch_size, natoms)

    esp_pred_dcmnet, esp_pred_physnet, esp_mask = _compute_esp_batch(
        mono_reshaped,
        dipo_reshaped,
        batch["vdw_surface"],
        batch["esp"],
        atom_positions,
        atom_mask,
        atomic_nums,
        physnet_charges,
        esp_min_distance,
        esp_max_value,
    )

    esp_predictions = {
        "physnet": esp_pred_physnet,
        "dcmnet": esp_pred_dcmnet,
    }

    lambda_esp_values = output.get("lambda_esp")
    if lambda_esp_values is not None:
        esp_predictions["mixed"] = (
            lambda_esp_values[:, None] * esp_pred_dcmnet
            + (1.0 - lambda_esp_values[:, None]) * esp_pred_physnet
        )

    mask_total = jnp.sum(esp_mask) + EPS
    total_esp_loss = jnp.array(0.0, dtype=loss_energy.dtype)
    for term in esp_terms:
        pred = esp_predictions.get(term.source)
        if pred is None:
            raise ValueError(f"Unknown ESP source '{term.source}' in loss config")
        diff = (pred - batch["esp"]) * esp_mask
        if term.metric == "l2":
            loss_term = jnp.sum(diff ** 2) / mask_total
        elif term.metric == "mae":
            loss_term = jnp.sum(jnp.abs(diff)) / mask_total
        elif term.metric == "rmse":
            loss_term = jnp.sqrt(jnp.sum(diff ** 2) / mask_total)
        else:
            raise ValueError(f"Unsupported ESP metric '{term.metric}'")
        losses[f"esp_{term.key}"] = loss_term
        total_esp_loss = total_esp_loss + term.weight * loss_term
    losses["esp"] = total_esp_loss

    total_charge_pred = output.get("sum_charges", jnp.sum(output["charges_as_mono"] * batch["atom_mask"]))
    loss_total_charge = optax.l2_loss(total_charge_pred, jnp.zeros_like(total_charge_pred)).mean()
    losses["total_charge"] = loss_total_charge

    mono_sum = output["mono_dist"].sum(axis=-1)
    charges_target = output["charges_as_mono"]
    mono_sum_masked = mono_sum * batch["atom_mask"]
    charges_masked = charges_target * batch["atom_mask"]
    loss_mono = optax.l2_loss(mono_sum_masked, charges_masked).mean()
    losses["monopole"] = loss_mono

    total_loss = (
        energy_w * loss_energy
        + forces_w * loss_forces
        + total_dipole_loss
        + total_esp_loss
        + mono_w * loss_mono
        + mono_w * loss_total_charge
    )

    losses["coulomb_energy"] = output.get("coulomb_energy", jnp.array(0.0))
    losses["coulomb_lambda"] = output.get("coulomb_lambda", jnp.array(0.0))
    if output.get("lambda_dipole") is not None:
        losses["lambda_dipole_mean"] = jnp.mean(output["lambda_dipole"])
    if lambda_esp_values is not None:
        losses["lambda_esp_mean"] = jnp.mean(lambda_esp_values)

    losses["total"] = total_loss

    return total_loss, losses


@functools.partial(
    jax.jit,
    static_argnames=(
        "model_apply",
        "optimizer_update",
        "batch_size",
        "n_dcm",
        "clip_norm",
        "dipole_terms",
        "esp_terms",
        "esp_min_distance",
        "esp_max_value",
    ),
)
def train_step(
    params: Any,
    opt_state: Any,
    batch: Dict[str, jnp.ndarray],
    model_apply: Any,
    optimizer_update: Any,
    energy_w: float,
    forces_w: float,
    mono_w: float,
    batch_size: int,
    n_dcm: int,
    dipole_terms: Sequence[LossTerm],
    esp_terms: Sequence[LossTerm],
    clip_norm: float = None,
    esp_min_distance: float = 0.0,
    esp_max_value: float = 1e10,
) -> Tuple[Any, Any, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Single training step with gradient computation and optional clipping."""

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
            output,
            batch,
            energy_w,
            forces_w,
            mono_w,
            batch_size,
            n_dcm,
            dipole_terms,
            esp_terms,
            esp_min_distance,
            esp_max_value,
        )
        return total_loss, (output, losses)

    (loss, (output, losses)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    if clip_norm is not None:
        grad_norm = optax.global_norm(grads)
        grads = jax.tree_util.tree_map(
            lambda g: g * jnp.minimum(clip_norm / grad_norm, 1.0),
            grads,
        )

    updates, opt_state = optimizer_update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, losses


@functools.partial(
    jax.jit,
    static_argnames=(
        "model_apply",
        "batch_size",
        "n_dcm",
        "dipole_terms",
        "esp_terms",
        "esp_min_distance",
        "esp_max_value",
    ),
)
def eval_step(
    params: Any,
    batch: Dict[str, jnp.ndarray],
    model_apply: Any,
    energy_w: float,
    forces_w: float,
    mono_w: float,
    batch_size: int,
    n_dcm: int,
    dipole_terms: Sequence[LossTerm],
    esp_terms: Sequence[LossTerm],
    esp_min_distance: float = 0.0,
    esp_max_value: float = 1e10,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Evaluation step without gradients."""

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
        output,
        batch,
        energy_w,
        forces_w,
        mono_w,
        batch_size,
        n_dcm,
        dipole_terms,
        esp_terms,
        esp_min_distance,
        esp_max_value,
    )

    losses = dict(losses)

    mae_energy = jnp.abs(output["energy"] - batch["E"]).mean()
    mae_forces = jnp.abs(
        output["forces"] * batch["atom_mask"][:, None]
        - batch["F"] * batch["atom_mask"][:, None]
    ).mean()
    losses["mae_energy"] = mae_energy
    losses["mae_forces"] = mae_forces

    mae_dipole_physnet = jnp.abs(output["dipoles"] - batch["D"]).mean()
    losses["mae_dipole_physnet"] = mae_dipole_physnet

    if output.get("dipoles_dcmnet") is not None:
        mae_dipole_dcmnet = jnp.abs(output["dipoles_dcmnet"] - batch["D"]).mean()
        losses["mae_dipole_dcmnet"] = mae_dipole_dcmnet
    if output.get("dipoles_mixed") is not None:
        mae_dipole_mixed = jnp.abs(output["dipoles_mixed"] - batch["D"]).mean()
        losses["mae_dipole_mixed"] = mae_dipole_mixed

    natoms = output["mono_dist"].shape[0] // batch_size
    mono_reshaped = output["mono_dist"].reshape(batch_size, natoms, n_dcm)
    dipo_reshaped = output["dipo_dist"].reshape(batch_size, natoms, n_dcm, 3)
    atom_positions = batch["R"].reshape(batch_size, natoms, 3)
    atom_mask = batch["atom_mask"].reshape(batch_size, natoms)
    atomic_nums = batch["Z"].reshape(batch_size, natoms)
    physnet_charges = output["charges_as_mono"].reshape(batch_size, natoms)

    esp_pred_dcmnet, esp_pred_physnet, esp_mask = _compute_esp_batch(
        mono_reshaped,
        dipo_reshaped,
        batch["vdw_surface"],
        batch["esp"],
        atom_positions,
        atom_mask,
        atomic_nums,
        physnet_charges,
        esp_min_distance,
        esp_max_value,
    )

    if isinstance(output, FrozenDict):
        output_mutable = unfreeze(output)
    else:
        output_mutable = dict(output)

    output_mutable["esp_dcmnet"] = esp_pred_dcmnet
    output_mutable["esp_physnet"] = esp_pred_physnet
    output_mutable["esp_mask"] = esp_mask

    if batch_size == 1:
        mask = esp_mask[0]
        denom = jnp.sum(mask) + EPS
        esp_true = batch["esp"][0]
        rmse_dcmnet = jnp.sqrt(jnp.sum(((esp_pred_dcmnet[0] - esp_true) * mask) ** 2) / denom)
        rmse_physnet = jnp.sqrt(jnp.sum(((esp_pred_physnet[0] - esp_true) * mask) ** 2) / denom)
        losses["rmse_esp_dcmnet"] = rmse_dcmnet
        losses["rmse_esp_physnet"] = rmse_physnet
        if output.get("lambda_esp") is not None:
            esp_mixed = (
                output["lambda_esp"][0] * esp_pred_dcmnet[0]
                + (1.0 - output["lambda_esp"][0]) * esp_pred_physnet[0]
            )
            rmse_mixed = jnp.sqrt(jnp.sum(((esp_mixed - esp_true) * mask) ** 2) / denom)
            losses["rmse_esp_mixed"] = rmse_mixed
            output_mutable["esp_mixed"] = esp_mixed

    output = freeze(output_mutable)

    return total_loss, losses, output


def plot_validation_results(
    params: Any,
    model: JointPhysNetDCMNet,
    valid_data: Dict[str, np.ndarray],
    cutoff: float,
    energy_w: float,
    forces_w: float,
    mono_w: float,
    n_dcm: int,
    save_dir: Path,
    n_samples: int = 100,
    n_esp_examples: int = 2,
    dipole_terms: Sequence[LossTerm] = (),
    esp_terms: Sequence[LossTerm] = (),
    epoch: int = None,
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
    energy_w, forces_w, mono_w : float
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
    
    highlight_sources = {term.source for term in dipole_terms}

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
    esp_grid_positions_list = []
    
    print(f"Evaluating {n_total} validation samples...")
    
    for i in range(n_total):
        batch = prepare_batch_data(valid_data, np.array([i]), cutoff=cutoff)
        
        _, losses, output = eval_step(
            params=params,
            batch=batch,
            model_apply=model.apply,
            energy_w=energy_w,
            forces_w=forces_w,
            mono_w=mono_w,
            batch_size=1,
            n_dcm=n_dcm,
            dipole_terms=dipole_terms,
            esp_terms=esp_terms,
            esp_min_distance=0.0,  # No filtering for plotting
            esp_max_value=1e10,  # No magnitude filtering for plotting
        )
        
        # Extract scalar/first element from batch outputs
        energies_pred.append(float(output['energy'][0]))
        energies_true.append(float(batch['E'][0]))
        
        forces_pred.append(np.array(output['forces']))
        forces_true.append(np.array(batch['F']))
        
        # Store BOTH dipole predictions
        dipoles_physnet_pred.append(np.array(output['dipoles'][0]))
        
        # Compute DCMNet dipole (already batched, extract first)
        dipoles_dcmnet_pred.append(np.array(output['dipoles_dcmnet'][0]))
        
        dipoles_true.append(np.array(batch['D'][0]))
        
        # ESP (only store first n_esp_examples) - already computed in eval_step
        if i < n_esp_examples:
            esp_pred_dcmnet_list.append(np.array(output['esp_dcmnet']))
            esp_pred_physnet_list.append(np.array(output['esp_physnet']))
            esp_true_list.append(np.array(batch['esp'][0]))
            esp_grid_positions_list.append(np.array(batch['vdw_surface'][0]))
    
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
    
    # Create figure with scatter plots (row 1&3) and histograms (row 2&4)
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    
    # ========== ROW 0: SCATTER PLOTS ==========
    # Energy scatter
    ax = axes[0, 0]
    ax.scatter(energies_true, energies_pred, alpha=0.5, s=20)
    lims = [min(energies_true.min(), energies_pred.min()),
            max(energies_true.max(), energies_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.set_xlabel('True Energy (eV)')
    ax.set_ylabel('Predicted Energy (eV)')
    mae_energy = np.abs(energies_true - energies_pred).mean()
    ax.set_title(f'Energy\nMAE: {mae_energy:.3f} eV')
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
    mae_forces = np.abs(forces_true - forces_pred).mean()
    ax.set_title(f'Forces\nMAE: {mae_forces:.3f} eV/Å')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PhysNet Dipole scatter
    ax = axes[0, 2]
    ax.scatter(dipoles_true, dipoles_physnet_pred, alpha=0.5, s=20)
    lims = [min(dipoles_true.min(), dipoles_physnet_pred.min()),
            max(dipoles_true.max(), dipoles_physnet_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.set_xlabel('True Dipole (e·Å)')
    ax.set_ylabel('Predicted Dipole (e·Å)')
    marker = ' *' if 'physnet' in highlight_sources else ''
    mae_dipole_physnet = np.abs(dipoles_true - dipoles_physnet_pred).mean()
    ax.set_title(f'Dipole - PhysNet{marker}\nMAE: {mae_dipole_physnet:.3f} e·Å')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ========== ROW 1: ERROR HISTOGRAMS ==========
    # Energy error histogram
    ax = axes[1, 0]
    errors_energy = energies_pred - energies_true
    ax.hist(errors_energy, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Zero Error')
    ax.set_xlabel('Prediction Error (eV)')
    ax.set_ylabel('Count')
    ax.set_title(f'Energy Error Distribution\nStd: {errors_energy.std():.3f} eV')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Forces error histogram
    ax = axes[1, 1]
    errors_forces = forces_pred - forces_true
    ax.hist(errors_forces, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Zero Error')
    ax.set_xlabel('Prediction Error (eV/Å)')
    ax.set_ylabel('Count')
    ax.set_title(f'Forces Error Distribution\nStd: {errors_forces.std():.3f} eV/Å')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # PhysNet Dipole error histogram
    ax = axes[1, 2]
    errors_dipole_physnet = dipoles_physnet_pred - dipoles_true
    ax.hist(errors_dipole_physnet, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Zero Error')
    ax.set_xlabel('Prediction Error (e·Å)')
    ax.set_ylabel('Count')
    ax.set_title(f'Dipole (PhysNet) Error\nStd: {errors_dipole_physnet.std():.3f} e·Å')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # ========== ROW 2: MORE SCATTER PLOTS ==========
    # DCMNet Dipole scatter
    ax = axes[2, 0]
    ax.scatter(dipoles_true, dipoles_dcmnet_pred, alpha=0.5, s=20, color='orange')
    lims = [min(dipoles_true.min(), dipoles_dcmnet_pred.min()),
            max(dipoles_true.max(), dipoles_dcmnet_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.set_xlabel('True Dipole (e·Å)')
    ax.set_ylabel('Predicted Dipole (e·Å)')
    marker = ' *' if 'dcmnet' in highlight_sources else ''
    mae_dipole_dcmnet = np.abs(dipoles_true - dipoles_dcmnet_pred).mean()
    ax.set_title(f'Dipole - DCMNet{marker}\nMAE: {mae_dipole_dcmnet:.3f} e·Å')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ESP PhysNet scatter
    ax = axes[2, 1]
    if esp_pred_physnet_list:
        esp_pred_physnet_all = np.concatenate([e.reshape(-1) for e in esp_pred_physnet_list])
        esp_true_all = np.concatenate([e.reshape(-1) for e in esp_true_list])
        ax.scatter(esp_true_all, esp_pred_physnet_all, alpha=0.3, s=10, color='green')
        
        # Compute bounds at median ± 6 SD for better visualization
        true_median = np.median(esp_true_all)
        true_std = np.std(esp_true_all)
        pred_median = np.median(esp_pred_physnet_all)
        pred_std = np.std(esp_pred_physnet_all)
        
        xlim_min = true_median - 6 * true_std
        xlim_max = true_median + 6 * true_std
        ylim_min = pred_median - 6 * pred_std
        ylim_max = pred_median + 6 * pred_std
        
        # Plot perfect line across the bounds
        lims = [min(xlim_min, ylim_min), max(xlim_max, ylim_max)]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(ylim_min, ylim_max)
        
        ax.set_xlabel('True ESP (Hartree/e)')
        ax.set_ylabel('Predicted ESP (Hartree/e)')
        rmse_esp_physnet = np.sqrt(np.mean((esp_pred_physnet_all - esp_true_all)**2))
        # Compute R²
        ss_res = np.sum((esp_true_all - esp_pred_physnet_all)**2)
        ss_tot = np.sum((esp_true_all - np.mean(esp_true_all))**2)
        r2_esp_physnet = 1 - (ss_res / ss_tot)
        ax.set_title(f'ESP - PhysNet\nRMSE: {rmse_esp_physnet:.6f} Ha/e, R²: {r2_esp_physnet:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # ESP DCMNet scatter
    ax = axes[2, 2]
    if esp_pred_dcmnet_list:
        esp_pred_dcmnet_all = np.concatenate([e.reshape(-1) for e in esp_pred_dcmnet_list])
        esp_true_all = np.concatenate([e.reshape(-1) for e in esp_true_list])
        ax.scatter(esp_true_all, esp_pred_dcmnet_all, alpha=0.3, s=10, color='purple')
        
        # Compute bounds at median ± 6 SD for better visualization
        true_median = np.median(esp_true_all)
        true_std = np.std(esp_true_all)
        pred_median = np.median(esp_pred_dcmnet_all)
        pred_std = np.std(esp_pred_dcmnet_all)
        
        xlim_min = true_median - 6 * true_std
        xlim_max = true_median + 6 * true_std
        ylim_min = pred_median - 6 * pred_std
        ylim_max = pred_median + 6 * pred_std
        
        # Plot perfect line across the bounds
        lims = [min(xlim_min, ylim_min), max(xlim_max, ylim_max)]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(ylim_min, ylim_max)
        
        ax.set_xlabel('True ESP (Hartree/e)')
        ax.set_ylabel('Predicted ESP (Hartree/e)')
        rmse_esp_dcmnet = np.sqrt(np.mean((esp_pred_dcmnet_all - esp_true_all)**2))
        # Compute R²
        ss_res = np.sum((esp_true_all - esp_pred_dcmnet_all)**2)
        ss_tot = np.sum((esp_true_all - np.mean(esp_true_all))**2)
        r2_esp_dcmnet = 1 - (ss_res / ss_tot)
        ax.set_title(f'ESP - DCMNet\nRMSE: {rmse_esp_dcmnet:.6f} Ha/e, R²: {r2_esp_dcmnet:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # ========== ROW 3: MORE ERROR HISTOGRAMS ==========
    # DCMNet Dipole error histogram
    ax = axes[3, 0]
    errors_dipole_dcmnet = dipoles_dcmnet_pred - dipoles_true
    ax.hist(errors_dipole_dcmnet, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Zero Error')
    ax.set_xlabel('Prediction Error (e·Å)')
    ax.set_ylabel('Count')
    ax.set_title(f'Dipole (DCMNet) Error\nStd: {errors_dipole_dcmnet.std():.3f} e·Å')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # ESP PhysNet error histogram
    ax = axes[3, 1]
    if esp_pred_physnet_list:
        errors_esp_physnet = esp_pred_physnet_all - esp_true_all
        ax.hist(errors_esp_physnet, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Zero Error')
        ax.set_xlabel('Prediction Error (Hartree/e)')
        ax.set_ylabel('Count')
        ax.set_title(f'ESP (PhysNet) Error\nStd: {errors_esp_physnet.std():.6f} Ha/e')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # ESP DCMNet error histogram
    ax = axes[3, 2]
    if esp_pred_dcmnet_list:
        errors_esp_dcmnet = esp_pred_dcmnet_all - esp_true_all
        ax.hist(errors_esp_dcmnet, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Zero Error')
        ax.set_xlabel('Prediction Error (Hartree/e)')
        ax.set_ylabel('Count')
        ax.set_title(f'ESP (DCMNet) Error\nStd: {errors_esp_dcmnet.std():.6f} Ha/e')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plots_path = save_dir / f'validation_plots{suffix}.png'
    plt.savefig(plots_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved validation plots (scatter + histograms): {plots_path}")
    
    # ========== CENTERED SCATTER PLOTS (True - Mean vs Pred - Mean) ==========
    # These plots remove mean bias and show correlation patterns
    fig_centered, axes_centered = plt.subplots(2, 3, figsize=(18, 12))
    
    # Compute means
    energy_mean_true = energies_true.mean()
    energy_mean_pred = energies_pred.mean()
    forces_mean_true = forces_true.mean()
    forces_mean_pred = forces_pred.mean()
    dipoles_mean_true = dipoles_true.mean()
    dipoles_mean_physnet_pred = dipoles_physnet_pred.mean()
    dipoles_mean_dcmnet_pred = dipoles_dcmnet_pred.mean()
    
    # Energy centered
    ax = axes_centered[0, 0]
    energy_true_centered = energies_true - energy_mean_true
    energy_pred_centered = energies_pred - energy_mean_pred
    ax.scatter(energy_true_centered, energy_pred_centered, alpha=0.5, s=20)
    lims = [min(energy_true_centered.min(), energy_pred_centered.min()),
            max(energy_true_centered.max(), energy_pred_centered.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('True Energy - <True> (eV)')
    ax.set_ylabel('Predicted Energy - <Pred> (eV)')
    # Compute correlation coefficient
    corr_coef = np.corrcoef(energy_true_centered, energy_pred_centered)[0, 1]
    ax.set_title(f'Energy (Centered)\nR = {corr_coef:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Forces centered
    ax = axes_centered[0, 1]
    forces_true_centered = forces_true - forces_mean_true
    forces_pred_centered = forces_pred - forces_mean_pred
    ax.scatter(forces_true_centered, forces_pred_centered, alpha=0.3, s=10)
    lims = [min(forces_true_centered.min(), forces_pred_centered.min()),
            max(forces_true_centered.max(), forces_pred_centered.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('True Forces - <True> (eV/Å)')
    ax.set_ylabel('Predicted Forces - <Pred> (eV/Å)')
    corr_coef = np.corrcoef(forces_true_centered, forces_pred_centered)[0, 1]
    ax.set_title(f'Forces (Centered)\nR = {corr_coef:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PhysNet Dipole centered
    ax = axes_centered[0, 2]
    dipoles_true_centered = dipoles_true - dipoles_mean_true
    dipoles_physnet_pred_centered = dipoles_physnet_pred - dipoles_mean_physnet_pred
    ax.scatter(dipoles_true_centered, dipoles_physnet_pred_centered, alpha=0.5, s=20)
    lims = [min(dipoles_true_centered.min(), dipoles_physnet_pred_centered.min()),
            max(dipoles_true_centered.max(), dipoles_physnet_pred_centered.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('True Dipole - <True> (e·Å)')
    ax.set_ylabel('Predicted Dipole - <Pred> (e·Å)')
    corr_coef = np.corrcoef(dipoles_true_centered, dipoles_physnet_pred_centered)[0, 1]
    marker = ' *' if 'physnet' in highlight_sources else ''
    ax.set_title(f'Dipole - PhysNet{marker} (Centered)\nR = {corr_coef:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # DCMNet Dipole centered
    ax = axes_centered[1, 0]
    dipoles_dcmnet_pred_centered = dipoles_dcmnet_pred - dipoles_mean_dcmnet_pred
    ax.scatter(dipoles_true_centered, dipoles_dcmnet_pred_centered, alpha=0.5, s=20, color='orange')
    lims = [min(dipoles_true_centered.min(), dipoles_dcmnet_pred_centered.min()),
            max(dipoles_true_centered.max(), dipoles_dcmnet_pred_centered.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax.set_xlabel('True Dipole - <True> (e·Å)')
    ax.set_ylabel('Predicted Dipole - <Pred> (e·Å)')
    corr_coef = np.corrcoef(dipoles_true_centered, dipoles_dcmnet_pred_centered)[0, 1]
    marker = ' *' if 'dcmnet' in highlight_sources else ''
    ax.set_title(f'Dipole - DCMNet{marker} (Centered)\nR = {corr_coef:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ESP PhysNet centered
    ax = axes_centered[1, 1]
    if esp_pred_physnet_list:
        # Compute arrays for centered plots (may duplicate earlier computation, but simpler)
        esp_pred_physnet_all_centered = np.concatenate([e.reshape(-1) for e in esp_pred_physnet_list])
        esp_true_all_centered = np.concatenate([e.reshape(-1) for e in esp_true_list])
        esp_true_mean = esp_true_all_centered.mean()
        esp_pred_physnet_mean = esp_pred_physnet_all_centered.mean()
        esp_true_centered = esp_true_all_centered - esp_true_mean
        esp_pred_physnet_centered = esp_pred_physnet_all_centered - esp_pred_physnet_mean
        ax.scatter(esp_true_centered, esp_pred_physnet_centered, alpha=0.3, s=10, color='green')
        
        # Compute bounds at median ± 6 SD
        true_centered_median = np.median(esp_true_centered)
        true_centered_std = np.std(esp_true_centered)
        pred_centered_median = np.median(esp_pred_physnet_centered)
        pred_centered_std = np.std(esp_pred_physnet_centered)
        
        xlim_min = true_centered_median - 6 * true_centered_std
        xlim_max = true_centered_median + 6 * true_centered_std
        ylim_min = pred_centered_median - 6 * pred_centered_std
        ylim_max = pred_centered_median + 6 * pred_centered_std
        
        lims = [min(xlim_min, ylim_min), max(xlim_max, ylim_max)]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(ylim_min, ylim_max)
        ax.axhline(0, color='k', linestyle=':', alpha=0.3)
        ax.axvline(0, color='k', linestyle=':', alpha=0.3)
        ax.set_xlabel('True ESP - <True> (Hartree/e)')
        ax.set_ylabel('Predicted ESP - <Pred> (Hartree/e)')
        corr_coef = np.corrcoef(esp_true_centered, esp_pred_physnet_centered)[0, 1]
        ax.set_title(f'ESP - PhysNet (Centered)\nR = {corr_coef:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # ESP DCMNet centered
    ax = axes_centered[1, 2]
    if esp_pred_dcmnet_list:
        # Compute arrays for centered plots
        esp_pred_dcmnet_all_centered = np.concatenate([e.reshape(-1) for e in esp_pred_dcmnet_list])
        # Reuse esp_true_centered from PhysNet plot if available, otherwise compute
        if 'esp_true_centered' not in locals():
            esp_true_all_centered = np.concatenate([e.reshape(-1) for e in esp_true_list])
            esp_true_mean = esp_true_all_centered.mean()
            esp_true_centered = esp_true_all_centered - esp_true_mean
        esp_pred_dcmnet_mean = esp_pred_dcmnet_all_centered.mean()
        esp_pred_dcmnet_centered = esp_pred_dcmnet_all_centered - esp_pred_dcmnet_mean
        ax.scatter(esp_true_centered, esp_pred_dcmnet_centered, alpha=0.3, s=10, color='purple')
        
        # Compute bounds at median ± 6 SD
        true_centered_median = np.median(esp_true_centered)
        true_centered_std = np.std(esp_true_centered)
        pred_centered_median = np.median(esp_pred_dcmnet_centered)
        pred_centered_std = np.std(esp_pred_dcmnet_centered)
        
        xlim_min = true_centered_median - 6 * true_centered_std
        xlim_max = true_centered_median + 6 * true_centered_std
        ylim_min = pred_centered_median - 6 * pred_centered_std
        ylim_max = pred_centered_median + 6 * pred_centered_std
        
        lims = [min(xlim_min, ylim_min), max(xlim_max, ylim_max)]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(ylim_min, ylim_max)
        ax.axhline(0, color='k', linestyle=':', alpha=0.3)
        ax.axvline(0, color='k', linestyle=':', alpha=0.3)
        ax.set_xlabel('True ESP - <True> (Hartree/e)')
        ax.set_ylabel('Predicted ESP - <Pred> (Hartree/e)')
        corr_coef = np.corrcoef(esp_true_centered, esp_pred_dcmnet_centered)[0, 1]
        ax.set_title(f'ESP - DCMNet (Centered)\nR = {corr_coef:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Centered Scatter Plots (Correlation Analysis)', fontsize=14, weight='bold')
    plt.tight_layout()
    centered_path = save_dir / f'validation_centered{suffix}.png'
    plt.savefig(centered_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved centered scatter plots: {centered_path}")
    
    # ========== COMPREHENSIVE ESP ANALYSIS PLOTS ==========
    # Create one comprehensive analysis across ALL validation samples
    if esp_pred_physnet_list and esp_pred_dcmnet_list:
        fig_esp_analysis, axes_esp = plt.subplots(3, 4, figsize=(20, 15))
        
        # Collect ESP data (already computed earlier)
        # Reuse from scatter plots to avoid recomputation
        try:
            _ = esp_pred_physnet_all
        except NameError:
            esp_pred_physnet_all = np.concatenate([e.reshape(-1) for e in esp_pred_physnet_list])
        try:
            _ = esp_pred_dcmnet_all
        except NameError:
            esp_pred_dcmnet_all = np.concatenate([e.reshape(-1) for e in esp_pred_dcmnet_list])
        try:
            _ = esp_true_all
        except NameError:
            esp_true_all = np.concatenate([e.reshape(-1) for e in esp_true_list])
        
        # Row 0: Hexbin density plots
        ax = axes_esp[0, 0]
        hexbin = ax.hexbin(esp_true_all, esp_pred_physnet_all, gridsize=50, cmap='Greens', mincnt=1)
        ax.plot([-0.1, 0.9], [-0.1, 0.9], 'r--', alpha=0.5, label='Perfect')
        ax.set_xlabel('True ESP (Ha/e)')
        ax.set_ylabel('PhysNet Pred (Ha/e)')
        ax.set_title('PhysNet - Hexbin Density')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(hexbin, ax=ax, label='Count')
        
        ax = axes_esp[0, 1]
        hexbin = ax.hexbin(esp_true_all, esp_pred_dcmnet_all, gridsize=50, cmap='Purples', mincnt=1)
        ax.plot([-0.1, 0.9], [-0.1, 0.9], 'r--', alpha=0.5, label='Perfect')
        ax.set_xlabel('True ESP (Ha/e)')
        ax.set_ylabel('DCMNet Pred (Ha/e)')
        ax.set_title('DCMNet - Hexbin Density')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(hexbin, ax=ax, label='Count')
        
        # Row 0, Col 2-3: Residual plots (error vs true value)
        ax = axes_esp[0, 2]
        error_physnet = esp_pred_physnet_all - esp_true_all
        ax.scatter(esp_true_all, error_physnet, alpha=0.3, s=5, color='green')
        ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('True ESP (Ha/e)')
        ax.set_ylabel('Residual (Pred - True, Ha/e)')
        ax.set_title('PhysNet Residuals')
        ax.grid(True, alpha=0.3)
        
        ax = axes_esp[0, 3]
        error_dcmnet = esp_pred_dcmnet_all - esp_true_all
        ax.scatter(esp_true_all, error_dcmnet, alpha=0.3, s=5, color='purple')
        ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('True ESP (Ha/e)')
        ax.set_ylabel('Residual (Pred - True, Ha/e)')
        ax.set_title('DCMNet Residuals')
        ax.grid(True, alpha=0.3)
        
        # Row 1: 2D histogram density
        ax = axes_esp[1, 0]
        h = ax.hist2d(esp_true_all, esp_pred_physnet_all, bins=50, cmap='Greens', cmin=1)
        ax.plot([-0.1, 0.9], [-0.1, 0.9], 'r--', alpha=0.5, linewidth=2)
        ax.set_xlabel('True ESP (Ha/e)')
        ax.set_ylabel('PhysNet Pred (Ha/e)')
        ax.set_title('PhysNet - 2D Histogram')
        plt.colorbar(h[3], ax=ax, label='Count')
        
        ax = axes_esp[1, 1]
        h = ax.hist2d(esp_true_all, esp_pred_dcmnet_all, bins=50, cmap='Purples', cmin=1)
        ax.plot([-0.1, 0.9], [-0.1, 0.9], 'r--', alpha=0.5, linewidth=2)
        ax.set_xlabel('True ESP (Ha/e)')
        ax.set_ylabel('DCMNet Pred (Ha/e)')
        ax.set_title('DCMNet - 2D Histogram')
        plt.colorbar(h[3], ax=ax, label='Count')
        
        # Row 1, Col 2-3: Bland-Altman plots (mean vs difference)
        ax = axes_esp[1, 2]
        mean_physnet = (esp_true_all + esp_pred_physnet_all) / 2
        diff_physnet = esp_pred_physnet_all - esp_true_all
        ax.scatter(mean_physnet, diff_physnet, alpha=0.3, s=5, color='green')
        ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Zero bias')
        ax.axhline(diff_physnet.mean(), color='blue', linestyle='-', linewidth=1, alpha=0.5, label=f'Mean: {diff_physnet.mean():.6f}')
        ax.axhline(diff_physnet.mean() + 1.96*diff_physnet.std(), color='orange', linestyle=':', linewidth=1, label='±1.96 SD')
        ax.axhline(diff_physnet.mean() - 1.96*diff_physnet.std(), color='orange', linestyle=':', linewidth=1)
        ax.set_xlabel('Mean of True & Pred (Ha/e)')
        ax.set_ylabel('Difference (Pred - True, Ha/e)')
        ax.set_title('PhysNet - Bland-Altman')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        ax = axes_esp[1, 3]
        mean_dcmnet = (esp_true_all + esp_pred_dcmnet_all) / 2
        diff_dcmnet = esp_pred_dcmnet_all - esp_true_all
        ax.scatter(mean_dcmnet, diff_dcmnet, alpha=0.3, s=5, color='purple')
        ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Zero bias')
        ax.axhline(diff_dcmnet.mean(), color='blue', linestyle='-', linewidth=1, alpha=0.5, label=f'Mean: {diff_dcmnet.mean():.6f}')
        ax.axhline(diff_dcmnet.mean() + 1.96*diff_dcmnet.std(), color='orange', linestyle=':', linewidth=1, label='±1.96 SD')
        ax.axhline(diff_dcmnet.mean() - 1.96*diff_dcmnet.std(), color='orange', linestyle=':', linewidth=1)
        ax.set_xlabel('Mean of True & Pred (Ha/e)')
        ax.set_ylabel('Difference (Pred - True, Ha/e)')
        ax.set_title('DCMNet - Bland-Altman')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Row 2: Q-Q plots and error percentile analysis
        try:
            from scipy import stats
            
            ax = axes_esp[2, 0]
            stats.probplot(error_physnet, dist="norm", plot=ax)
            ax.set_title('PhysNet - Q-Q Plot')
            ax.grid(True, alpha=0.3)
            
            ax = axes_esp[2, 1]
            stats.probplot(error_dcmnet, dist="norm", plot=ax)
            ax.set_title('DCMNet - Q-Q Plot')
            ax.grid(True, alpha=0.3)
        except ImportError:
            # If scipy not available, just plot error histograms
            ax = axes_esp[2, 0]
            ax.hist(error_physnet, bins=50, color='green', alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Error (Ha/e)')
            ax.set_ylabel('Count')
            ax.set_title('PhysNet Error Histogram')
            ax.grid(True, alpha=0.3)
            
            ax = axes_esp[2, 1]
            ax.hist(error_dcmnet, bins=50, color='purple', alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Error (Ha/e)')
            ax.set_ylabel('Count')
            ax.set_title('DCMNet Error Histogram')
            ax.grid(True, alpha=0.3)
        
        # Row 2, Col 2: Error percentiles comparison
        ax = axes_esp[2, 2]
        percentiles = np.arange(0, 101, 5)
        physnet_percentiles = np.percentile(np.abs(error_physnet), percentiles)
        dcmnet_percentiles = np.percentile(np.abs(error_dcmnet), percentiles)
        ax.plot(percentiles, physnet_percentiles, 'o-', color='green', label='PhysNet', linewidth=2)
        ax.plot(percentiles, dcmnet_percentiles, 's-', color='purple', label='DCMNet', linewidth=2)
        ax.set_xlabel('Percentile')
        ax.set_ylabel('|Error| (Ha/e)')
        ax.set_title('Absolute Error Percentiles')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Row 2, Col 3: Cumulative error distribution
        ax = axes_esp[2, 3]
        sorted_err_physnet = np.sort(np.abs(error_physnet))
        sorted_err_dcmnet = np.sort(np.abs(error_dcmnet))
        cdf_physnet = np.arange(1, len(sorted_err_physnet) + 1) / len(sorted_err_physnet)
        cdf_dcmnet = np.arange(1, len(sorted_err_dcmnet) + 1) / len(sorted_err_dcmnet)
        ax.plot(sorted_err_physnet, cdf_physnet, color='green', label='PhysNet', linewidth=2)
        ax.plot(sorted_err_dcmnet, cdf_dcmnet, color='purple', label='DCMNet', linewidth=2)
        ax.set_xlabel('|Error| (Ha/e)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Error Distribution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, max(sorted_err_physnet.max(), sorted_err_dcmnet.max()))
        
        plt.suptitle(f'Comprehensive ESP Analysis (All Samples){epoch_str}', fontsize=14, weight='bold')
        plt.tight_layout()
        esp_analysis_path = save_dir / f'esp_analysis_comprehensive{suffix}.png'
        plt.savefig(esp_analysis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved comprehensive ESP analysis: {esp_analysis_path}")
    
    # Create consolidated ESP example plots - one comprehensive figure per sample
    for idx in range(min(n_esp_examples, len(esp_pred_dcmnet_list))):
        # Get molecule data
        batch_for_plot = prepare_batch_data(valid_data, np.array([idx]), cutoff=cutoff)
        n_atoms = int(batch_for_plot['N'][0])
        atom_positions = np.array(batch_for_plot['R'][:n_atoms])
        atomic_nums = np.array(batch_for_plot['Z'][:n_atoms])
        
        # Create ESP comparison figure (2 rows x 3 cols)
        from ase import Atoms
        from ase.visualize.plot import plot_atoms
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        esp_true = esp_true_list[idx]
        esp_pred_dcmnet = esp_pred_dcmnet_list[idx]
        esp_pred_physnet = esp_pred_physnet_list[idx]
        esp_error_dcmnet = esp_pred_dcmnet - esp_true
        esp_error_physnet = esp_pred_physnet - esp_true
        
        # Compute shared color scales - SYMMETRIC around 0
        # For ESP values: FIXED symmetric scale at ±0.01 au (Hartree/e)
        esp_vmin = -0.01
        esp_vmax = 0.01
        
        # For errors: use symmetric scale around 0
        error_absmax = max(abs(esp_error_dcmnet).max(), abs(esp_error_physnet).max())
        error_vmin = -error_absmax
        error_vmax = error_absmax
        
        # Also compute percentile-based ranges for detailed error views
        p95_max = max(np.percentile(np.abs(esp_error_dcmnet), 95), 
                      np.percentile(np.abs(esp_error_physnet), 95))
        p75_max = max(np.percentile(np.abs(esp_error_dcmnet), 75), 
                      np.percentile(np.abs(esp_error_physnet), 75))
        p50_max = max(np.percentile(np.abs(esp_error_dcmnet), 50), 
                      np.percentile(np.abs(esp_error_physnet), 50))
        
        # True ESP (shared)
        ax = axes[0, 0]
        sc = ax.scatter(range(len(esp_true)), esp_true, c=esp_true, 
                       cmap='RdBu_r', s=5, alpha=0.6, vmin=esp_vmin, vmax=esp_vmax)
        ax.set_xlabel('Grid Point Index')
        ax.set_ylabel('ESP (Hartree/e)')
        ax.set_title(f'True ESP (Sample {idx}){epoch_str}')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='ESP (Ha/e)')
        
        # DCMNet ESP
        ax = axes[0, 1]
        sc = ax.scatter(range(len(esp_pred_dcmnet)), esp_pred_dcmnet, c=esp_pred_dcmnet,
                       cmap='RdBu_r', s=5, alpha=0.6, vmin=esp_vmin, vmax=esp_vmax)
        ax.set_xlabel('Grid Point Index')
        ax.set_ylabel('ESP (Hartree/e)')
        ax.set_title(f'DCMNet ESP{epoch_str}')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='ESP (Ha/e)')
        
        # DCMNet Error
        ax = axes[0, 2]
        sc = ax.scatter(range(len(esp_error_dcmnet)), esp_error_dcmnet, c=esp_error_dcmnet,
                       cmap='RdBu_r', s=5, alpha=0.6, vmin=error_vmin, vmax=error_vmax)
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
        plt.colorbar(sc, ax=ax, label='Error (Ha/e)')
        
        # ASE molecular visualization
        ax = axes[1, 0]
        # Create ASE Atoms object
        atoms_ase = Atoms(numbers=atomic_nums.astype(int), positions=atom_positions)
        plot_atoms(atoms_ase, ax=ax, radii=0.5, rotation=('90x,0y,0z'))
        ax.set_title(f'Molecule (ASE){epoch_str}')
        
        # PhysNet ESP
        ax = axes[1, 1]
        sc = ax.scatter(range(len(esp_pred_physnet)), esp_pred_physnet, c=esp_pred_physnet,
                       cmap='RdBu_r', s=5, alpha=0.6, marker='s', vmin=esp_vmin, vmax=esp_vmax)
        ax.set_xlabel('Grid Point Index')
        ax.set_ylabel('ESP (Hartree/e)')
        ax.set_title(f'PhysNet ESP{epoch_str}')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='ESP (Ha/e)')
        
        # PhysNet Error
        ax = axes[1, 2]
        sc = ax.scatter(range(len(esp_error_physnet)), esp_error_physnet, c=esp_error_physnet,
                       cmap='RdBu_r', s=5, alpha=0.6, marker='s', vmin=error_vmin, vmax=error_vmax)
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
        plt.colorbar(sc, ax=ax, label='Error (Ha/e)')
        
        plt.tight_layout()
        esp_path = save_dir / f'esp_example_{idx}{suffix}.png'
        plt.savefig(esp_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved ESP example {idx}: {esp_path}")
        
        # Create 3D scatter plots for this ESP example
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(18, 6))
        
        # Get grid positions and ESP values
        grid_pos = esp_grid_positions_list[idx]  # (ngrid, 3) - already in Angstroms after conversion
        
        # Get atom positions and atomic numbers for this molecule
        batch_for_atoms = prepare_batch_data(valid_data, np.array([idx]), cutoff=cutoff)
        n_atoms = int(batch_for_atoms['N'][0])
        atom_positions = np.array(batch_for_atoms['R'][:n_atoms])  # (n_atoms, 3) in Angstroms
        atomic_nums = np.array(batch_for_atoms['Z'][:n_atoms])
        
        # No centering needed - grid and atoms should already be aligned after Bohr→Å conversion
        
        # True ESP in 3D
        ax = fig.add_subplot(131, projection='3d')
        sc = ax.scatter(grid_pos[:, 0], grid_pos[:, 1], grid_pos[:, 2],
                       c=esp_true, cmap='RdBu_r', s=3, alpha=0.5, 
                       vmin=esp_vmin, vmax=esp_vmax)
        # Add atom positions
        ax.scatter(atom_positions[:, 0], atom_positions[:, 1], atom_positions[:, 2],
                  c='black', s=300, marker='o', edgecolors='yellow', linewidths=3, 
                  alpha=1.0, label='Atoms')
        # Label atoms
        for i, (pos, Z) in enumerate(zip(atom_positions, atomic_nums)):
            ax.text(pos[0], pos[1], pos[2], f'  {int(Z)}', fontsize=10, 
                   color='black', weight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='yellow', alpha=0.7))
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'True ESP (3D){epoch_str}\nESP range: [{esp_vmin},{esp_vmax}] Ha/e')
        ax.view_init(elev=35, azim=45)  # Isometric view
        plt.colorbar(sc, ax=ax, label='ESP (Ha/e)', shrink=0.6)
        ax.legend()
        
        # PhysNet ESP in 3D
        ax = fig.add_subplot(132, projection='3d')
        sc = ax.scatter(grid_pos[:, 0], grid_pos[:, 1], grid_pos[:, 2],
                       c=esp_pred_physnet, cmap='RdBu_r', s=3, alpha=0.5,
                       vmin=esp_vmin, vmax=esp_vmax)
        # Add atom positions (PhysNet charges are AT atom centers)
        ax.scatter(atom_positions[:, 0], atom_positions[:, 1], atom_positions[:, 2],
                  c='black', s=300, marker='o', edgecolors='lime', linewidths=3, 
                  alpha=1.0, label='Atoms (charge centers)')
        # Label atoms
        for i, (pos, Z) in enumerate(zip(atom_positions, atomic_nums)):
            ax.text(pos[0], pos[1], pos[2], f'  {int(Z)}\n(q)', fontsize=9, 
                   color='black', weight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='lime', alpha=0.7))
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'PhysNet ESP (3D){epoch_str}\nESP range: [{esp_vmin},{esp_vmax}] Ha/e')
        ax.view_init(elev=35, azim=45)  # Isometric view
        plt.colorbar(sc, ax=ax, label='ESP (Ha/e)', shrink=0.6)
        ax.legend()
        
        # DCMNet ESP in 3D
        ax = fig.add_subplot(133, projection='3d')
        sc = ax.scatter(grid_pos[:, 0], grid_pos[:, 1], grid_pos[:, 2],
                       c=esp_pred_dcmnet, cmap='RdBu_r', s=3, alpha=0.5,
                       vmin=esp_vmin, vmax=esp_vmax)
        # Add atom positions
        ax.scatter(atom_positions[:, 0], atom_positions[:, 1], atom_positions[:, 2],
                  c='black', s=300, marker='o', edgecolors='cyan', linewidths=3, 
                  alpha=1.0, label='Atoms')
        # Label atoms
        for i, (pos, Z) in enumerate(zip(atom_positions, atomic_nums)):
            ax.text(pos[0], pos[1], pos[2], f'  {int(Z)}', fontsize=10, 
                   color='black', weight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='cyan', alpha=0.7))
        
        # Add distributed charge positions if available
        if idx < n_esp_examples:
            # Get distributed charges for this molecule (need to recompute)
            _, _, output_dcm = eval_step(
                params=params,
                batch=batch_for_atoms,
                model_apply=model.apply,
                energy_w=energy_w,
                forces_w=forces_w,
                mono_w=mono_w,
                batch_size=1,
                n_dcm=n_dcm,
                dipole_terms=dipole_terms,
                esp_terms=esp_terms,
                esp_min_distance=0.0,
                esp_max_value=1e10,
            )
            # Extract only real atoms (output is already (batch*natoms, n_dcm) format)
            mono_dcm = output_dcm["mono_dist"][:n_atoms]  # (n_atoms, n_dcm)
            dipo_dcm = output_dcm["dipo_dist"][:n_atoms]  # (n_atoms, n_dcm, 3)
            
            # Plot distributed charges
            charges_flat = np.array(mono_dcm).flatten()
            positions_flat = np.array(dipo_dcm).reshape(-1, 3)
            # Only plot charges with significant magnitude
            significant = np.abs(charges_flat) > 0.01
            if np.any(significant):
                ax.scatter(positions_flat[significant, 0], 
                          positions_flat[significant, 1], 
                          positions_flat[significant, 2],
                          c=charges_flat[significant], cmap='RdBu_r', s=50, 
                          marker='^', edgecolors='white', linewidths=1,
                          vmin=-0.5, vmax=0.5, alpha=0.9, label='Dist. charges')
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'DCMNet ESP (3D){epoch_str}\nESP range: [{esp_vmin},{esp_vmax}] Ha/e')
        ax.view_init(elev=35, azim=45)  # Isometric view
        plt.colorbar(sc, ax=ax, label='ESP (Ha/e)', shrink=0.6)
        ax.legend()
        
        plt.tight_layout()
        esp_3d_path = save_dir / f'esp_example_{idx}_3d{suffix}.png'
        plt.savefig(esp_3d_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved 3D ESP example {idx}: {esp_3d_path}")
        
        # Create multi-scale error visualization (3 rows at different percentiles)
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        # Row 1: 100% range (all data)
        ax = axes[0, 0]
        sc = ax.scatter(grid_pos[:, 0], grid_pos[:, 2],
                       c=esp_error_physnet, cmap='RdBu_r', s=5, alpha=0.6,
                       vmin=-error_absmax, vmax=error_absmax)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Z (Å)')
        ax.set_title(f'PhysNet Error (100% range){epoch_str}')
        plt.colorbar(sc, ax=ax, label='Error (Ha/e)')
        ax.grid(True, alpha=0.3)
        # Add atom positions for reference
        ax.scatter(atom_positions[:, 0], atom_positions[:, 2],
                  c='black', s=100, marker='o', edgecolors='lime', linewidths=2, alpha=1.0, zorder=10)
        
        ax = axes[0, 1]
        sc = ax.scatter(grid_pos[:, 0], grid_pos[:, 2],
                       c=esp_error_dcmnet, cmap='RdBu_r', s=20, alpha=0.8,
                       vmin=-error_absmax, vmax=error_absmax)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Z (Å)')
        ax.set_title(f'DCMNet Error (100% range){epoch_str}')
        plt.colorbar(sc, ax=ax, label='Error (Ha/e)')
        ax.grid(True, alpha=0.3)
        # Add atom positions for reference
        ax.scatter(atom_positions[:, 0], atom_positions[:, 2],
                  c='black', s=100, marker='o', edgecolors='cyan', linewidths=2, alpha=1.0, zorder=10)
        
        # Row 2: 95th percentile
        ax = axes[1, 0]
        sc = ax.scatter(grid_pos[:, 0], grid_pos[:, 2],
                       c=esp_error_physnet, cmap='RdBu_r', s=20, alpha=0.8,
                       vmin=-p95_max, vmax=p95_max)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Z (Å)')
        ax.set_title(f'PhysNet Error (95th %-ile){epoch_str}')
        plt.colorbar(sc, ax=ax, label='Error (Ha/e)')
        ax.grid(True, alpha=0.3)
        ax.scatter(atom_positions[:, 0], atom_positions[:, 2],
                  c='black', s=100, marker='o', edgecolors='lime', linewidths=2, alpha=1.0, zorder=10)
        
        ax = axes[1, 1]
        sc = ax.scatter(grid_pos[:, 0], grid_pos[:, 2],
                       c=esp_error_dcmnet, cmap='RdBu_r', s=20, alpha=0.8,
                       vmin=-p95_max, vmax=p95_max)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Z (Å)')
        ax.set_title(f'DCMNet Error (95th %-ile){epoch_str}')
        plt.colorbar(sc, ax=ax, label='Error (Ha/e)')
        ax.grid(True, alpha=0.3)
        ax.scatter(atom_positions[:, 0], atom_positions[:, 2],
                  c='black', s=100, marker='o', edgecolors='cyan', linewidths=2, alpha=1.0, zorder=10)
        
        # Row 3: 75th percentile
        ax = axes[2, 0]
        sc = ax.scatter(grid_pos[:, 0], grid_pos[:, 2],
                       c=esp_error_physnet, cmap='RdBu_r', s=20, alpha=0.8,
                       vmin=-p75_max, vmax=p75_max)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Z (Å)')
        ax.set_title(f'PhysNet Error (75th %-ile){epoch_str}')
        plt.colorbar(sc, ax=ax, label='Error (Ha/e)')
        ax.grid(True, alpha=0.3)
        ax.scatter(atom_positions[:, 0], atom_positions[:, 2],
                  c='black', s=100, marker='o', edgecolors='lime', linewidths=2, alpha=1.0, zorder=10)
        
        ax = axes[2, 1]
        sc = ax.scatter(grid_pos[:, 0], grid_pos[:, 2],
                       c=esp_error_dcmnet, cmap='RdBu_r', s=20, alpha=0.8,
                       vmin=-p75_max, vmax=p75_max)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Z (Å)')
        ax.set_title(f'DCMNet Error (75th %-ile){epoch_str}')
        plt.colorbar(sc, ax=ax, label='Error (Ha/e)')
        ax.grid(True, alpha=0.3)
        ax.scatter(atom_positions[:, 0], atom_positions[:, 2],
                  c='black', s=100, marker='o', edgecolors='cyan', linewidths=2, alpha=1.0, zorder=10)
        
        plt.tight_layout()
        esp_scales_path = save_dir / f'esp_example_{idx}_error_scales{suffix}.png'
        plt.savefig(esp_scales_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved multi-scale ESP errors {idx}: {esp_scales_path}")
        
        # Create radial ESP error plots (error vs distance from each atom)
        batch_for_radial = prepare_batch_data(valid_data, np.array([idx]), cutoff=cutoff)
        n_atoms = int(batch_for_radial['N'][0])
        atom_positions = np.array(batch_for_radial['R'][:n_atoms])  # (natoms, 3)
        atomic_nums = np.array(batch_for_radial['Z'][:n_atoms])
        
        # Create figure with one subplot per atom + combined view
        fig, axes = plt.subplots(1, n_atoms + 1, figsize=(6*(n_atoms+1), 5))
        if n_atoms == 1:
            axes = [axes]
        
        # Get atomic radii for visualization
        import ase.data
        atomic_radii_np = np.array([ase.data.covalent_radii[int(z)] for z in atomic_nums])
        
        # For each atom, plot error vs distance
        for atom_idx in range(n_atoms):
            ax = axes[atom_idx]
            
            # Compute distances from this atom to all grid points
            distances_to_atom = np.linalg.norm(grid_pos - atom_positions[atom_idx], axis=1)
            
            # Plot errors
            ax.scatter(distances_to_atom, esp_error_physnet, alpha=0.4, s=15, 
                      color='green', label='PhysNet', edgecolors='none')
            ax.scatter(distances_to_atom, esp_error_dcmnet, alpha=0.4, s=15, 
                      color='purple', label='DCMNet', edgecolors='none')
            
            # Add horizontal line at zero error
            ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
            
            # Mark the atomic radius zones
            r_cov = atomic_radii_np[atom_idx]
            ax.axvline(r_cov, color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f'r_cov ({r_cov:.2f} Å)')
            ax.axvline(2*r_cov, color='blue', linestyle=':', linewidth=2, alpha=0.7, label=f'2×r_cov ({2*r_cov:.2f} Å)')
            ax.axvspan(0, 2*r_cov, alpha=0.1, color='red', label='Excluded zone')
            
            ax.set_xlabel('Distance from Atom (Å)')
            ax.set_ylabel('ESP Error (Ha/e)')
            ax.set_title(f'Atom {atom_idx} (Z={int(atomic_nums[atom_idx])}){epoch_str}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, min(10, distances_to_atom.max()))
            
        # Combined plot showing all atoms
        ax = axes[n_atoms]
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        for atom_idx in range(n_atoms):
            distances_to_atom = np.linalg.norm(grid_pos - atom_positions[atom_idx], axis=1)
            
            # Plot absolute error
            abs_error_physnet = np.abs(esp_error_physnet)
            abs_error_dcmnet = np.abs(esp_error_dcmnet)
            
            color = colors[atom_idx % len(colors)]
            ax.scatter(distances_to_atom, abs_error_physnet, alpha=0.3, s=10, 
                      color=color, marker='o', label=f'Atom {atom_idx} (Z={int(atomic_nums[atom_idx])})', 
                      edgecolors='none')
            
            # Mark 2×r_cov for this atom
            r_cov = atomic_radii_np[atom_idx]
            ax.axvline(2*r_cov, color=color, linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Distance from Atom (Å)')
        ax.set_ylabel('|ESP Error| (Ha/e)')
        ax.set_title(f'All Atoms - Radial Error Distribution{epoch_str}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(10, np.linalg.norm(grid_pos - atom_positions[0], axis=1).max()))
        ax.set_yscale('log')
        
        plt.suptitle(f'Radial ESP Error Analysis (Sample {idx}){epoch_str}', fontsize=14, weight='bold')
        plt.tight_layout()
        radial_path = save_dir / f'esp_radial_{idx}{suffix}.png'
        plt.savefig(radial_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved radial ESP error plot {idx}: {radial_path}")
        
        # Create distributed charge visualization
        # Reuse atom positions and atomic_nums from above
        batch_for_charges = batch_for_radial
        
        # Run model to get distributed charges
        _, _, output_for_charges = eval_step(
            params=params,
            batch=batch_for_charges,
            model_apply=model.apply,
            energy_w=energy_w,
            forces_w=forces_w,
            mono_w=mono_w,
            batch_size=1,
            n_dcm=n_dcm,
            dipole_terms=dipole_terms,
            esp_terms=esp_terms,
            esp_min_distance=0.0,  # No filtering for plotting
            esp_max_value=1e10,  # No magnitude filtering for plotting
        )
        
        # Extract distributed charges and positions
        natoms_padded = output_for_charges["mono_dist"].shape[0]
        mono_all = output_for_charges["mono_dist"].reshape(natoms_padded, n_dcm)
        dipo_all = output_for_charges["dipo_dist"].reshape(natoms_padded, n_dcm, 3)
        
        # Get only real atoms (not padding)
        charges_dist = mono_all[:n_atoms]  # (n_atoms, n_dcm)
        positions_dist = dipo_all[:n_atoms]  # (n_atoms, n_dcm, 3)
        
        # Flatten for plotting: (natoms*n_dcm,) and (natoms*n_dcm, 3)
        charges_flat = np.array(charges_dist.reshape(-1))
        positions_flat = np.array(positions_dist.reshape(-1, 3))
        
        # Create figure with multiple views
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(18, 12))
        
        # Compute symmetric color scale for charges
        charge_absmax = max(abs(charges_flat.min()), abs(charges_flat.max()))
        
        # View 1: 3D view
        ax = fig.add_subplot(221, projection='3d')
        ax.view_init(elev=35, azim=45)  # Isometric view
        # Plot atoms (larger, dark spheres)
        for i, (pos, Z) in enumerate(zip(atom_positions, atomic_nums)):
            ax.scatter(pos[0], pos[1], pos[2], c='black', s=200, alpha=0.8, marker='o')
            ax.text(pos[0], pos[1], pos[2], f'  {int(Z)}', fontsize=10, color='black')
        
        # Plot distributed charges (smaller, colored by magnitude)
        sc = ax.scatter(positions_flat[:, 0], positions_flat[:, 1], positions_flat[:, 2],
                       c=charges_flat, cmap='RdBu_r', s=50, alpha=0.8,
                       vmin=-charge_absmax, vmax=charge_absmax, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'Distributed Charges (3D){epoch_str}\nBlack = atoms, Colored = charges')
        plt.colorbar(sc, ax=ax, label='Charge (e)', shrink=0.6)
        
        # View 2: XY projection
        ax = fig.add_subplot(222)
        # Plot atoms
        for i, (pos, Z) in enumerate(zip(atom_positions, atomic_nums)):
            ax.scatter(pos[0], pos[1], c='black', s=200, alpha=0.8, marker='o', edgecolors='white', linewidth=2)
            ax.text(pos[0], pos[1], f' {int(Z)}', fontsize=12, color='black', weight='bold')
        
        # Plot distributed charges
        sc = ax.scatter(positions_flat[:, 0], positions_flat[:, 1],
                       c=charges_flat, cmap='RdBu_r', s=50, alpha=0.8,
                       vmin=-charge_absmax, vmax=charge_absmax, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_title(f'XY Projection{epoch_str}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, label='Charge (e)')
        
        # View 3: XZ projection
        ax = fig.add_subplot(223)
        # Plot atoms
        for i, (pos, Z) in enumerate(zip(atom_positions, atomic_nums)):
            ax.scatter(pos[0], pos[2], c='black', s=200, alpha=0.8, marker='o', edgecolors='white', linewidth=2)
            ax.text(pos[0], pos[2], f' {int(Z)}', fontsize=12, color='black', weight='bold')
        
        # Plot distributed charges
        sc = ax.scatter(positions_flat[:, 0], positions_flat[:, 2],
                       c=charges_flat, cmap='RdBu_r', s=50, alpha=0.8,
                       vmin=-charge_absmax, vmax=charge_absmax, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Z (Å)')
        ax.set_title(f'XZ Projection{epoch_str}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, label='Charge (e)')
        
        # View 4: YZ projection
        ax = fig.add_subplot(224)
        # Plot atoms
        for i, (pos, Z) in enumerate(zip(atom_positions, atomic_nums)):
            ax.scatter(pos[1], pos[2], c='black', s=200, alpha=0.8, marker='o', edgecolors='white', linewidth=2)
            ax.text(pos[1], pos[2], f' {int(Z)}', fontsize=12, color='black', weight='bold')
        
        # Plot distributed charges
        sc = ax.scatter(positions_flat[:, 1], positions_flat[:, 2],
                       c=charges_flat, cmap='RdBu_r', s=50, alpha=0.8,
                       vmin=-charge_absmax, vmax=charge_absmax, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Y (Å)')
        ax.set_ylabel('Z (Å)')
        ax.set_title(f'YZ Projection{epoch_str}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, label='Charge (e)')
        
        plt.suptitle(f'DCMNet Distributed Charges (Sample {idx}){epoch_str}', fontsize=14, weight='bold')
        plt.tight_layout()
        charges_path = save_dir / f'charges_example_{idx}{suffix}.png'
        plt.savefig(charges_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved charge distribution {idx}: {charges_path}")
        
        # Create a detailed per-atom charge analysis
        fig, axes = plt.subplots(1, n_atoms, figsize=(6*n_atoms, 6))
        if n_atoms == 1:
            axes = [axes]
        
        for atom_idx in range(n_atoms):
            ax = axes[atom_idx]
            atom_pos = atom_positions[atom_idx]
            atom_Z = int(atomic_nums[atom_idx])
            
            # Get charges for this atom
            charges_atom = np.array(charges_dist[atom_idx])  # (n_dcm,)
            positions_atom = np.array(positions_dist[atom_idx])  # (n_dcm, 3)
            
            # Compute positions relative to this atom
            rel_positions = positions_atom - atom_pos
            
            # 2D projection (XY relative to atom)
            ax.scatter(0, 0, c='black', s=300, marker='o', edgecolors='white', linewidth=3, zorder=10)
            ax.text(0, 0, f'{atom_Z}', fontsize=16, color='white', weight='bold', 
                   ha='center', va='center', zorder=11)
            
            # Plot distributed charges relative to atom
            charge_max_atom = max(abs(charges_atom.min()), abs(charges_atom.max()))
            sc = ax.scatter(rel_positions[:, 0], rel_positions[:, 1],
                           c=charges_atom, cmap='RdBu_r', s=150, alpha=0.9,
                           vmin=-charge_max_atom, vmax=charge_max_atom,
                           edgecolors='black', linewidth=1)
            
            # Draw lines from atom to charges
            for j, (rel_pos, q) in enumerate(zip(rel_positions, charges_atom)):
                ax.plot([0, rel_pos[0]], [0, rel_pos[1]], 'k-', alpha=0.3, linewidth=1)
                # Label with charge value
                ax.text(rel_pos[0], rel_pos[1], f'{q:.3f}', fontsize=8, 
                       ha='center', va='bottom')
            
            ax.set_xlabel('ΔX (Å)')
            ax.set_ylabel('ΔY (Å)')
            ax.set_title(f'Atom {atom_idx} (Z={atom_Z})\nΣq = {charges_atom.sum():.4f} e')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.axhline(0, color='k', linestyle='--', alpha=0.2)
            ax.axvline(0, color='k', linestyle='--', alpha=0.2)
            plt.colorbar(sc, ax=ax, label='Charge (e)')
        
        plt.suptitle(f'Per-Atom Charge Distribution (Sample {idx}){epoch_str}', fontsize=14, weight='bold')
        plt.tight_layout()
        charges_detail_path = save_dir / f'charges_detail_{idx}{suffix}.png'
        plt.savefig(charges_detail_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved detailed charge distribution {idx}: {charges_detail_path}")
    
    print(f"\n✅ All plots saved to: {save_dir}")


def train_model(
    model: JointPhysNetDCMNet,
    train_data: Dict[str, np.ndarray],
    valid_data: Dict[str, np.ndarray],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
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
    dipole_terms: Optional[Sequence[LossTerm]] = None,
    esp_terms: Optional[Sequence[LossTerm]] = None,
    dipole_source: str = 'physnet',
    esp_min_distance: float = 0.0,
    esp_max_value: float = 1e10,
    restart_params: Any = None,
    start_epoch: int = 1,
    optimizer_name: str = 'adamw',
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
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
    # Always initialize a fresh set of parameters for the current model architecture
    dummy_batch = prepare_batch_data(train_data, np.array([0]), cutoff=cutoff)
    
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
    
    # If restarting, merge old parameters with new structure (allows partial loading)
    if restart_params is not None:
        print("\n🔄 Merging checkpoint parameters...")
        
        # Deep merge: copy matching parameters from restart_params to params
        def merge_params(new_tree, old_tree, path=""):
            if isinstance(new_tree, dict):
                merged = {}
                for key in new_tree:
                    new_path = f"{path}/{key}" if path else key
                    if key in old_tree:
                        merged[key] = merge_params(new_tree[key], old_tree[key], new_path)
                    else:
                        print(f"  ⚠️  New parameter (initialized randomly): {new_path}")
                        merged[key] = new_tree[key]
                # Report old parameters not in new model
                for key in old_tree:
                    if key not in new_tree:
                        new_path = f"{path}/{key}" if path else key
                        print(f"  ⚠️  Dropped old parameter: {new_path}")
                return merged
            else:
                # Leaf node - use old value if shapes match
                if hasattr(new_tree, 'shape') and hasattr(old_tree, 'shape'):
                    if new_tree.shape == old_tree.shape:
                        return old_tree
                    else:
                        print(f"  ⚠️  Shape mismatch at {path}: old={old_tree.shape}, new={new_tree.shape} (using new)")
                        return new_tree
                return old_tree
        
        params = merge_params(params, restart_params)
        print(f"✅ Merged checkpoint with {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")
    else:
        print(f"✅ Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")
    
    # Setup optimizer
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    
    optimizer = create_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        **optimizer_kwargs
    )
    opt_state = optimizer.init(params)
    
    # Prepare training indices
    n_train = len(train_data['E'])
    n_valid = len(valid_data['E'])
    
    print(f"\nTraining samples: {n_train}")
    print(f"Validation samples: {n_valid}")
    print(f"Batch size: {batch_size}")

    if not dipole_terms:
        raise ValueError("At least one dipole loss term must be specified")
    if not esp_terms:
        raise ValueError("At least one ESP loss term must be specified")

    print("\nDipole loss terms:")
    for term in dipole_terms:
        print(
            f"  - {term.key}: source={term.source}, metric={term.metric}, weight={term.weight}"
        )
    print("ESP loss terms:")
    for term in esp_terms:
        print(
            f"  - {term.key}: source={term.source}, metric={term.metric}, weight={term.weight}"
        )

    # Training loop with EMA
    best_valid_loss = float('inf')
    ema_decay = 0.999  # EMA decay factor (higher = slower update)
    ema_params = jax.tree_util.tree_map(lambda x: x.copy(), params)  # Initialize EMA params
    
    print(f"\n📊 Using EMA with decay={ema_decay} for validation")
    
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
                mono_w=mono_w,
                batch_size=len(batch_indices),
                n_dcm=n_dcm,
                dipole_terms=dipole_terms,
                esp_terms=esp_terms,
                clip_norm=grad_clip_norm,
                esp_min_distance=esp_min_distance,
                esp_max_value=esp_max_value,
            )
            
            train_losses.append({k: float(v) for k, v in losses.items()})
            
            # Update EMA parameters after each batch
            ema_params = jax.tree_util.tree_map(
                lambda ema, new: ema_decay * ema + (1 - ema_decay) * new,
                ema_params,
                params
            )
        
        # Average training losses
        train_loss_avg = {
            k: np.mean([loss[k] for loss in train_losses])
            for k in train_losses[0].keys()
        }
        
        # Validation phase (use EMA parameters)
        valid_losses = []
        n_valid_batches = (n_valid + batch_size - 1) // batch_size
        
        # Collect predictions for statistics
        all_energy_pred = []
        all_energy_true = []
        all_forces_pred = []
        all_forces_true = []
        all_dipole_physnet_pred = []
        all_dipole_dcmnet_pred = []
        all_dipole_true = []
        all_esp_physnet_pred = []
        all_esp_dcmnet_pred = []
        all_esp_true = []
        
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
                params=ema_params,  # Use EMA parameters for validation
                batch=batch,
                model_apply=model.apply,
                energy_w=energy_w,
                forces_w=forces_w,
                mono_w=mono_w,
                batch_size=len(batch_indices),
                n_dcm=n_dcm,
                dipole_terms=dipole_terms,
                esp_terms=esp_terms,
                esp_min_distance=esp_min_distance,
                esp_max_value=esp_max_value,
            )
            
            valid_losses.append({k: float(v) for k, v in losses.items()})
            
            # Collect for statistics
            all_energy_pred.extend(np.array(output['energy']).flatten())
            all_energy_true.extend(np.array(batch['E']).flatten())
            all_forces_pred.extend(np.array(output['forces']).flatten())
            all_forces_true.extend(np.array(batch['F']).flatten())
            
            # Collect dipole predictions
            all_dipole_physnet_pred.extend(np.array(output['dipoles']).flatten())
            all_dipole_dcmnet_pred.extend(np.array(output['dipoles_dcmnet']).flatten())
            all_dipole_true.extend(np.array(batch['D']).flatten())
            
            # Collect ESP predictions - now computed for all samples in batch
            esp_physnet_batch = np.array(output['esp_physnet'])  # (batch_size, ngrid)
            esp_dcmnet_batch = np.array(output['esp_dcmnet'])    # (batch_size, ngrid)
            esp_true_batch = np.array(batch['esp'])              # (batch_size, ngrid)
            esp_mask_batch = np.array(output['esp_mask'])        # (batch_size, ngrid)
            
            # Flatten and collect only valid (masked) ESP points
            for i in range(len(batch_indices)):
                mask = esp_mask_batch[i] > 0.5
                if np.any(mask):
                    all_esp_physnet_pred.extend(esp_physnet_batch[i][mask])
                    all_esp_dcmnet_pred.extend(esp_dcmnet_batch[i][mask])
                    all_esp_true.extend(esp_true_batch[i][mask])
        
        # Average validation losses
        valid_loss_avg = {
            k: np.mean([loss[k] for loss in valid_losses])
            for k in valid_losses[0].keys()
        }
        
        # Compute validation set statistics and ESP RMSE from collected predictions
        valid_stats = {
            'energy_mean': np.mean(all_energy_true),
            'energy_std': np.std(all_energy_true),
            'forces_mean': np.mean([f for f in all_forces_true if f != 0]),  # Exclude padding
            'forces_std': np.std([f for f in all_forces_true if f != 0]),
            'dipole_mean': np.mean([d for d in all_dipole_true if d != 0]),  # Exclude padding
            'dipole_std': np.std([d for d in all_dipole_true if d != 0]),
            'esp_mean': np.mean(all_esp_true) if all_esp_true else 0.0,
            'esp_std': np.std(all_esp_true) if all_esp_true else 1.0,
        }
        
        # Compute ESP RMSE from all collected ESP predictions (across all validation batches)
        if all_esp_true:
            esp_physnet_arr = np.array(all_esp_physnet_pred)
            esp_dcmnet_arr = np.array(all_esp_dcmnet_pred)
            esp_true_arr = np.array(all_esp_true)
            
            valid_loss_avg['rmse_esp_physnet'] = np.sqrt(np.mean((esp_physnet_arr - esp_true_arr) ** 2))
            valid_loss_avg['rmse_esp_dcmnet'] = np.sqrt(np.mean((esp_dcmnet_arr - esp_true_arr) ** 2))
        
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
                eA_to_Debye = 1.0 / 0.208194  # 1 e·Å = 4.8032 Debye
                
                mae_energy_ev = valid_loss_avg['mae_energy']
                mae_forces_ev = valid_loss_avg['mae_forces']
                mae_dipole_physnet_eA = valid_loss_avg.get('mae_dipole_physnet', 0.0)
                mae_dipole_dcmnet_eA = valid_loss_avg.get('mae_dipole_dcmnet', 0.0)
                
                # Get validation set statistics
                e_mean = valid_stats.get('energy_mean', 0)
                e_std = valid_stats.get('energy_std', 1)
                f_mean = valid_stats.get('forces_mean', 0)
                f_std = valid_stats.get('forces_std', 1)
                d_mean = valid_stats.get('dipole_mean', 0)
                d_std = valid_stats.get('dipole_std', 1)
                esp_mean = valid_stats.get('esp_mean', 0)
                esp_std = valid_stats.get('esp_std', 1)
                
                print(f"    MAE Energy: {mae_energy_ev:.6f} eV  ({mae_energy_ev * eV_to_kcal:.6f} kcal/mol) [μ={e_mean:.3f}, σ={e_std:.3f} eV]")
                print(f"    MAE Forces: {mae_forces_ev:.6f} eV/Å  ({mae_forces_ev * eV_to_kcal:.6f} kcal/mol/Å) [μ={f_mean:.3f}, σ={f_std:.3f} eV/Å]")
                
                if mae_dipole_physnet_eA > 0:
                    print(f"    MAE Dipole (PhysNet): {mae_dipole_physnet_eA:.6f} e·Å  ({mae_dipole_physnet_eA * eA_to_Debye:.6f} D) [μ={d_mean:.3f}, σ={d_std:.3f} e·Å]")
                if mae_dipole_dcmnet_eA > 0:
                    print(f"    MAE Dipole (DCMNet): {mae_dipole_dcmnet_eA:.6f} e·Å  ({mae_dipole_dcmnet_eA * eA_to_Debye:.6f} D) [μ={d_mean:.3f}, σ={d_std:.3f} e·Å]")
                
                # ESP RMSE metrics are only available for batch_size=1
                if 'rmse_esp_physnet' in valid_loss_avg:
                    rmse_esp_physnet_Ha = valid_loss_avg['rmse_esp_physnet']
                    print(f"    RMSE ESP (PhysNet): {rmse_esp_physnet_Ha:.6f} Ha/e  ({rmse_esp_physnet_Ha * Ha_to_kcal:.6f} (kcal/mol)/e) [μ={esp_mean:.6f}, σ={esp_std:.6f} Ha/e]")
                if 'rmse_esp_dcmnet' in valid_loss_avg:
                    rmse_esp_dcmnet_Ha = valid_loss_avg['rmse_esp_dcmnet']
                    print(f"    RMSE ESP (DCMNet): {rmse_esp_dcmnet_Ha:.6f} Ha/e  ({rmse_esp_dcmnet_Ha * Ha_to_kcal:.6f} (kcal/mol)/e) [μ={esp_mean:.6f}, σ={esp_std:.6f} Ha/e]")
                if 'rmse_esp_mixed' in valid_loss_avg:
                    rmse_esp_mixed_Ha = valid_loss_avg['rmse_esp_mixed']
                    print(f"    RMSE ESP (Mixed): {rmse_esp_mixed_Ha:.6f} Ha/e  ({rmse_esp_mixed_Ha * Ha_to_kcal:.6f} (kcal/mol)/e)")
            
            # Print constraint violations if available
            if 'total_charge' in valid_loss_avg:
                print(f"    Total Charge Violation: {valid_loss_avg['total_charge']:.6f}")
            
            # Print charge diagnostics (first validation sample)
            if epoch % (print_freq * 10) == 0 or epoch == 1:  # Every 10 print_freq epochs
                print(f"\n  💡 Charge Diagnostics (first validation sample):")
                diag_batch = prepare_batch_data(valid_data, np.array([0]), cutoff=cutoff)
                _, _, diag_output = eval_step(
                    params=ema_params,  # Use EMA parameters
                    batch=diag_batch,
                    model_apply=model.apply,
                    energy_w=energy_w,
                    forces_w=forces_w,
                    mono_w=mono_w,
                    batch_size=1,
                    n_dcm=n_dcm,
                    dipole_terms=dipole_terms,
                    esp_terms=esp_terms,
                    esp_min_distance=esp_min_distance,
                    esp_max_value=esp_max_value,
                )
                
                n_atoms_diag = int(diag_batch['N'][0])
                charges_physnet_diag = np.array(diag_output['charges_as_mono'][:n_atoms_diag])
                mono_dcm_diag = np.array(diag_output['mono_dist'][:n_atoms_diag].reshape(n_atoms_diag, n_dcm))
                
                print(f"    PhysNet charges: [{charges_physnet_diag.min():.4f}, {charges_physnet_diag.max():.4f}] e, sum={charges_physnet_diag.sum():.4f}")
                print(f"    DCMNet charges:  [{mono_dcm_diag.min():.4f}, {mono_dcm_diag.max():.4f}] e, sum={mono_dcm_diag.sum():.4f}")
                n_pos_dcm = (mono_dcm_diag > 0).sum()
                n_neg_dcm = (mono_dcm_diag < 0).sum()
                print(f"    DCMNet charge signs: {n_pos_dcm} positive, {n_neg_dcm} negative (out of {n_atoms_diag * n_dcm})")
                
                esp_dcmnet_diag = np.array(diag_output['esp_dcmnet'])
                esp_physnet_diag = np.array(diag_output['esp_physnet'])
                esp_target_diag = np.array(diag_batch['esp'][0])
                print(f"    ESP (DCMNet): [{esp_dcmnet_diag.min():.4f}, {esp_dcmnet_diag.max():.4f}] Ha/e")
                print(f"    ESP (PhysNet): [{esp_physnet_diag.min():.4f}, {esp_physnet_diag.max():.4f}] Ha/e")
                print(f"    ESP (Target):  [{esp_target_diag.min():.4f}, {esp_target_diag.max():.4f}] Ha/e")
        
        # Save best model (use EMA parameters)
        if valid_loss_avg['total'] < best_valid_loss:
            best_valid_loss = valid_loss_avg['total']
            save_path = ckpt_dir / name
            save_path.mkdir(exist_ok=True, parents=True)
            
            # Save EMA parameters as the best checkpoint
            with open(save_path / 'best_params.pkl', 'wb') as f:
                pickle.dump(ema_params, f)
            
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
        if plot_freq is not None and plot_freq > 0 and epoch % plot_freq == 0 and HAS_MATPLOTLIB:
            print(f"\n📊 Creating plots at epoch {epoch}...")
            plot_validation_results(
                params=ema_params,  # Use EMA parameters for plotting
                model=model,
                valid_data=valid_data,
                cutoff=cutoff,
                energy_w=energy_w,
                forces_w=forces_w,
                mono_w=mono_w,
                n_dcm=n_dcm,
                save_dir=ckpt_dir / name / 'plots',
                n_samples=plot_samples,
                n_esp_examples=plot_esp_examples,
                epoch=epoch,
                dipole_terms=dipole_terms,
                esp_terms=esp_terms,
            )
    
    return ema_params  # Return EMA parameters as final model


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
    parser.add_argument('--no-subtract-atom-energies', action='store_true', default=False,
                       help='Disable subtraction of reference atomic energies (default: subtract)')
    
    # PhysNet hyperparameters
    parser.add_argument('--physnet-features', type=int, default=64,
                       help='PhysNet: number of features')
    parser.add_argument('--physnet-iterations', type=int, default=3,
                       help='PhysNet: message passing iterations')
    parser.add_argument('--physnet-basis', type=int, default=64,
                       help='PhysNet: number of basis functions')
    parser.add_argument('--physnet-cutoff', type=float, default=6.0,
                       help='PhysNet: cutoff distance (Angstroms)')
    parser.add_argument('--physnet-n-res', type=int, default=3,
                       help='PhysNet: number of residual blocks')
    
    # DCMNet hyperparameters
    parser.add_argument('--dcmnet-features', type=int, default=128,
                       help='DCMNet: number of features')
    parser.add_argument('--dcmnet-iterations', type=int, default=2,
                       help='DCMNet: message passing iterations')
    parser.add_argument('--dcmnet-basis', type=int, default=64,
                       help='DCMNet: number of basis functions')
    parser.add_argument('--dcmnet-cutoff', type=float, default=10.0,
                       help='DCMNet: cutoff distance (Angstroms)')
    parser.add_argument('--n-dcm', type=int, default=3,
                       help='DCMNet: distributed multipoles per atom')
    parser.add_argument('--max-degree', type=int, default=2,
                       help='DCMNet: maximum spherical harmonic degree')
    
    # Non-equivariant model option
    parser.add_argument('--use-noneq-model', action='store_true', default=False,
                       help='Use non-equivariant charge model instead of DCMNet (predicts Cartesian displacements)')
    parser.add_argument('--noneq-features', type=int, default=128,
                       help='Non-equivariant model: hidden layer size')
    parser.add_argument('--noneq-layers', type=int, default=3,
                       help='Non-equivariant model: number of MLP layers')
    parser.add_argument('--noneq-max-displacement', type=float, default=1.0,
                       help='Non-equivariant model: maximum displacement distance (Angstroms)')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (start with 1 for debugging)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'rmsprop', 'muon'],
                       help='Optimizer choice (default: adamw)')
    parser.add_argument('--learning-rate', '--lr', type=float, default=None,
                       help='Learning rate (default: auto-select based on dataset and optimizer)')
    parser.add_argument('--weight-decay', type=float, default=None,
                       help='Weight decay/L2 regularization (default: auto-select based on optimizer)')
    parser.add_argument('--use-recommended-hparams', action='store_true', default=False,
                       help='Use recommended hyperparameters based on dataset properties (overrides manual settings)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Loss weights
    parser.add_argument('--energy-weight', type=float, default=10.0,
                       help='Energy loss weight')
    parser.add_argument('--forces-weight', type=float, default=50.0,
                       help='Forces loss weight')
    parser.add_argument('--dipole-weight', type=float, default=25.0,
                       help='Dipole loss weight')
    parser.add_argument('--esp-weight', type=float, default=10000.0,
                       help='ESP loss weight')
    parser.add_argument('--esp-min-distance', type=float, default=0.0,
                       help='Additional minimum distance (Å) from atoms for ESP grid points (default: 0, uses 2×atomic_radius). Set > 0 to add extra distance constraint.')
    parser.add_argument('--esp-max-value', type=float, default=None,
                       help='Maximum |ESP| value (Hartree/e) to include in loss - filters out high ESP points (default: no limit)')
    parser.add_argument('--mono-weight', type=float, default=100.0,
                       help='Monopole constraint loss weight (enforce distributed charges sum to atomic charges)')
    parser.add_argument('--dipole-source', type=str, default='physnet',
                       choices=LOSS_SOURCE_CHOICES,
                       help='Source for dipole in loss: physnet (from charges) or dcmnet (from distributed multipoles)')
    parser.add_argument('--dipole-loss-sources', type=str, nargs='*', default=None,
                       choices=LOSS_SOURCE_CHOICES,
                       help='Override dipole supervision sources (e.g. physnet dcmnet mixed). Defaults to --dipole-source when omitted.')
    parser.add_argument('--esp-loss-sources', type=str, nargs='*', default=None,
                       choices=LOSS_SOURCE_CHOICES,
                       help='ESP supervision sources (e.g. dcmnet physnet mixed). Defaults to dcmnet when omitted.')
    parser.add_argument('--dipole-metric', type=str, default='l2', choices=LOSS_METRIC_CHOICES,
                       help='Error metric for default dipole loss terms (ignored when --loss-config specified)')
    parser.add_argument('--esp-metric', type=str, default='l2', choices=LOSS_METRIC_CHOICES,
                       help='Error metric for default ESP loss terms (ignored when --loss-config specified)')
    parser.add_argument('--loss-config', type=Path, default=None,
                       help='Optional JSON or YAML file defining dipole/ESP loss terms (overrides individual loss source flags)')
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
    parser.add_argument('--plot-freq', type=int, default=10,
                       help='Create validation plots every N epochs during training (default: 10, set to 0 to disable)')
    parser.add_argument('--plot-samples', type=int, default=100,
                       help='Number of validation samples to plot')
    parser.add_argument('--plot-esp-examples', type=int, default=2,
                       help='Number of ESP examples to visualize')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure loss terms
    dipole_w = args.dipole_weight
    esp_w = args.esp_weight

    cfg_dipole_terms: Tuple[LossTerm, ...] = ()
    cfg_esp_terms: Tuple[LossTerm, ...] = ()

    if args.loss_config is not None:
        cfg_dipole_terms, cfg_esp_terms = load_loss_terms_config(args.loss_config)
        print(f"\n📄 Loaded loss configuration from {args.loss_config}")

    dipole_terms: Tuple[LossTerm, ...]
    esp_terms: Tuple[LossTerm, ...]

    if cfg_dipole_terms:
        dipole_terms = cfg_dipole_terms
    else:
        dipole_sources = args.dipole_loss_sources or [args.dipole_source]
        dipole_terms = tuple(
            LossTerm(source=src, weight=dipole_w, metric=args.dipole_metric, name=src)
            for src in dipole_sources
        )

    if cfg_esp_terms:
        esp_terms = cfg_esp_terms
    else:
        esp_sources = args.esp_loss_sources or ["dcmnet"]
        esp_terms = tuple(
            LossTerm(source=src, weight=esp_w, metric=args.esp_metric, name=src)
            for src in esp_sources
        )

    if not dipole_terms:
        raise ValueError("At least one dipole loss term must be specified")
    if not esp_terms:
        raise ValueError("At least one ESP loss term must be specified")

    args.dipole_terms = dipole_terms
    args.esp_terms = esp_terms
    
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
    subtract_atom_energies = not args.no_subtract_atom_energies
    train_data = load_combined_data(args.train_efd, args.train_esp, 
                                    subtract_atom_energies=subtract_atom_energies, 
                                    verbose=args.verbose)
    
    print("\nLoading validation data...")
    valid_data = load_combined_data(args.valid_efd, args.valid_esp,
                                    subtract_atom_energies=subtract_atom_energies,
                                    verbose=args.verbose)
    
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
    
    # Setup optimizer hyperparameters (before building model to use in recommendations)
    dataset_size = len(train_data['E'])
    total_features = args.physnet_features + args.dcmnet_features
    
    # Get recommended config if requested or if params not specified
    if args.use_recommended_hparams or args.learning_rate is None or args.weight_decay is None:
        recommended_config = get_recommended_optimizer_config(
            dataset_size=dataset_size,
            num_features=total_features,
            num_atoms=args.natoms,
            optimizer_name=args.optimizer
        )
        
        if args.use_recommended_hparams:
            print(f"\n🔧 Using recommended hyperparameters for {args.optimizer.upper()}:")
            for key, value in recommended_config.items():
                print(f"  {key}: {value}")
        
        # Apply defaults only if not manually specified
        if args.learning_rate is None:
            args.learning_rate = recommended_config['learning_rate']
            print(f"\n📊 Auto-selected learning rate: {args.learning_rate}")
        if args.weight_decay is None:
            args.weight_decay = recommended_config.get('weight_decay', 0.0)
            print(f"📊 Auto-selected weight decay: {args.weight_decay}")
        
        # Store additional optimizer-specific params
        optimizer_kwargs = {k: v for k, v in recommended_config.items() 
                          if k not in ['learning_rate', 'weight_decay']}
    else:
        optimizer_kwargs = {}
        # Use defaults if still None
        if args.learning_rate is None:
            args.learning_rate = 0.001
        if args.weight_decay is None:
            args.weight_decay = 1e-4 if args.optimizer == 'adamw' else 0.0
    
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
    
    print("PhysNet configuration:")
    for k, v in physnet_config.items():
        print(f"  {k}: {v}")
    
    if args.use_noneq_model:
        # Use non-equivariant charge model
        noneq_config = {
            'features': args.noneq_features,
            'n_dcm': args.n_dcm,
            'max_atomic_number': args.max_atomic_number,
            'num_layers': args.noneq_layers,
            'max_displacement': args.noneq_max_displacement,
        }
        
        print("\nNon-Equivariant Charge Model configuration:")
        for k, v in noneq_config.items():
            print(f"  {k}: {v}")
        print("  ⚠️  Note: This model is NOT rotationally equivariant!")
        print("     It predicts Cartesian displacements directly.")
        
        model = JointPhysNetNonEquivariant(
            physnet_config=physnet_config,
            noneq_config=noneq_config,
            mix_coulomb_energy=args.mix_coulomb_energy,
        )
        
        print(f"\n✅ Joint PhysNet + Non-Equivariant model created")
    else:
        # Use equivariant DCMNet (default)
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
        
        print("\nDCMNet configuration:")
        for k, v in dcmnet_config.items():
            print(f"  {k}: {v}")
        
        model = JointPhysNetDCMNet(
            physnet_config=physnet_config,
            dcmnet_config=dcmnet_config,
            mix_coulomb_energy=args.mix_coulomb_energy,
        )
        
        print(f"\n✅ Joint PhysNet + DCMNet model created")
    
    # Training setup
    print(f"\n{'#'*70}")
    print("# Training Setup")
    print(f"{'#'*70}\n")
    
    print(f"Training hyperparameters:")
    print(f"  Optimizer: {args.optimizer.upper()}")
    if optimizer_kwargs:
        for k, v in optimizer_kwargs.items():
            print(f"    {k}: {v}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Random seed: {args.seed}")
    
    print(f"\nLoss configuration:")
    print(f"  Energy weight: {args.energy_weight} (mix Coulomb: {args.mix_coulomb_energy})")
    print(f"  Forces weight: {args.forces_weight}")
    print(f"  Monopole weight: {args.mono_weight}")
    print("  Dipole terms:")
    for term in args.dipole_terms:
        print(
            f"    - {term.key}: source={term.source}, metric={term.metric}, weight={term.weight}"
        )
    print("  ESP terms:")
    for term in args.esp_terms:
        print(
            f"    - {term.key}: source={term.source}, metric={term.metric}, weight={term.weight}"
        )
    if args.esp_min_distance > 0:
        print(f"    (additional ESP distance filter: {args.esp_min_distance:.2f} Å)")
    if args.esp_max_value is not None:
        print(
            f"    (ESP magnitude filter: |ESP| <= {args.esp_max_value:.4f} Ha/e)"
        )
    
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
            weight_decay=args.weight_decay,
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
            dipole_terms=args.dipole_terms,
            esp_terms=args.esp_terms,
            dipole_source=args.dipole_source,
            esp_min_distance=args.esp_min_distance,
            esp_max_value=args.esp_max_value if args.esp_max_value is not None else 1e10,
            restart_params=restart_params,
            start_epoch=start_epoch,
            optimizer_name=args.optimizer,
            optimizer_kwargs=optimizer_kwargs,
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
                    mono_w=args.mono_weight,
                    n_dcm=args.n_dcm,
                    save_dir=ckpt_dir / args.name / 'plots',
                    n_samples=args.plot_samples,
                    n_esp_examples=args.plot_esp_examples,
                    dipole_terms=args.dipole_terms,
                    esp_terms=args.esp_terms,
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
