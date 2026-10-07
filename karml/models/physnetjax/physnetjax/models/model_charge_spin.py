"""
PhysNet model with Total Charge and Total Spin conditioning.

.. deprecated::
    Prefer :class:`~mmml.models.physnetjax.physnetjax.models.spooky_model.SpookyPhysNet`
    for Q/S-conditioned training and hybrid MD. This fork remains only for
    loading historical checkpoints; do not extend it.
"""

from __future__ import annotations

import functools
import warnings
from typing import Dict, List, Optional, Tuple

import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

from mmml.models.physnetjax.physnetjax.models.euclidean_fast_attention import fast_attention as efa
from mmml.models.physnetjax.physnetjax.models.physnet_family import PhysNetFamilyMixin
from mmml.models.physnetjax.physnetjax.models.zbl import (
    ZBLRepulsion,
    geometric_pair_distances,
)

EFA = efa.EuclideanFastAttention

# Constants
DTYPE = jnp.float32
HARTREE_TO_KCAL_MOL = 627.509


class PhysNetChargeSpin(PhysNetFamilyMixin, nn.Module):
    """PhysNet with total charge and total spin conditioning (deprecated).

    Prefer :class:`SpookyPhysNet` for new work. This fork remains for loading
    historical checkpoints only.
    
    Key differences from standard PhysNet:
    - Accepts total_charge and total_spin as inputs
    - Embeds molecular properties and broadcasts to atoms
    - Concatenates property embeddings with atomic features
    - Enables multi-state predictions (different charges/spins)
    
    Attributes
    ----------
    features : int
        Number of features in hidden layers
    max_degree : int
        Maximum degree for spherical harmonics
    num_iterations : int
        Number of message passing iterations
    num_basis_functions : int
        Number of radial basis functions
    cutoff : float
        Cutoff distance for interactions (Angstrom)
    max_atomic_number : int
        Maximum atomic number supported
    charges : bool
        Whether to predict atomic charges
    natoms : int
        Maximum number of atoms per molecule
    total_charge : float
        Default total charge (overridden by input)
    n_res : int
        Number of residual blocks
    zbl : bool
        Whether to use ZBL repulsion
    debug : bool | List[str]
        Debug mode flags
    efa : bool
        Whether to use Euclidean Fast Attention
    use_energy_bias : bool
        Whether to use element-wise energy biases
    charge_embed_dim : int
        Dimension of charge embedding
    spin_embed_dim : int
        Dimension of spin embedding
    charge_range : Tuple[int, int]
        Range of charges to support (min, max)
    spin_range : Tuple[int, int]
        Range of spin multiplicities to support (min, max)
    """
    
    # Standard PhysNet parameters
    features: int = 32
    max_degree: int = 3
    num_iterations: int = 2
    num_basis_functions: int = 16
    cutoff: float = 6.0
    max_atomic_number: int = 118
    charges: bool = False
    max_padded_atoms: int = 60
    total_charge: float = 0
    n_refinement_blocks: int = 3
    zbl: bool = True
    trainable_zbl: bool = False
    zbl_cuton: float | None = 0.1
    zbl_cutoff: float = 0.6
    debug: bool | List[str] = False
    efa: bool = False
    use_energy_bias: bool = True
    # Electrostatics/short-vs-long-range switch distances (Angstrom), used by
    # _calc_switches. Defaults reproduce the values every checkpoint trained
    # before this field existed was implicitly hardcoded to -- changing them
    # changes electrostatics behavior and is NOT backward compatible with
    # checkpoints trained under the old defaults. See the identical fix (and
    # full rationale) on mmml.models.physnetjax.physnetjax.models.spooky_model.SpookyPhysNet.
    switch_start: float = 1.0
    switch_end: float = 10.0
    electrostatics_off_start: float = 8.0
    electrostatics_off_end: float = 10.0
    # Euclidean Fast Attention (EFA) hyperparameters, only used when efa=True.
    # Defaults reproduce the values every checkpoint trained before these
    # fields existed was implicitly hardcoded to.
    efa_lebedev_num: int = 194
    efa_max_length: float = 20.0  # maximum distance (Angstrom) for the EPE
    efa_ti_degree_scaling_base: float = 0.5

    @property
    def natoms(self) -> int:
        return self.max_padded_atoms

    @natoms.setter
    def natoms(self, value: int) -> None:
        object.__setattr__(self, "max_padded_atoms", value)

    @property
    def n_res(self) -> int:
        return self.n_refinement_blocks

    @n_res.setter
    def n_res(self, value: int) -> None:
        object.__setattr__(self, "n_refinement_blocks", value)
    
    # Charge and spin conditioning parameters
    charge_embed_dim: int = 16
    spin_embed_dim: int = 16
    charge_range: Tuple[int, int] = (-5, 5)  # Support charges from -5 to +5
    spin_range: Tuple[int, int] = (1, 7)     # Support spin multiplicities 1-7 (singlet to septet)
    
    def setup(self) -> None:
        """Initialize model components including property embeddings."""
        # Standard PhysNet components
        if self.zbl:
            self.repulsion = ZBLRepulsion(
                cutoff=self.zbl_cutoff,
                cuton=self.zbl_cuton,
                trainable=self.trainable_zbl,
            )
        
        self.efa_final = None
        if self.efa:
            b_max = 4 * jnp.pi
            self.efa_final = EFA(
                lebedev_num=self.efa_lebedev_num,
                parametrized=False,
                epe_max_frequency=b_max,
                epe_max_length=self.efa_max_length,
                tensor_integration=True,
                ti_degree_scaling_constants=[
                    self.efa_ti_degree_scaling_base**i for i in range(self.max_degree + 1)
                ],
            )
    
    def return_attributes(self) -> Dict:
        """Return model attributes for checkpointing."""
        warnings.warn(
            "PhysNetChargeSpin is deprecated; use SpookyPhysNet for Q/S-conditioned "
            "models (see docs/mpnn-tool-inventory.md).",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            "features": self.features,
            "max_degree": self.max_degree,
            "num_iterations": self.num_iterations,
            "num_basis_functions": self.num_basis_functions,
            "cutoff": self.cutoff,
            "max_atomic_number": self.max_atomic_number,
            "charges": self.charges,
            "natoms": self.max_padded_atoms,
            "max_padded_atoms": self.max_padded_atoms,
            "total_charge": self.total_charge,
            "n_res": self.n_refinement_blocks,
            "n_refinement_blocks": self.n_refinement_blocks,
            "zbl": self.zbl,
            "trainable_zbl": self.trainable_zbl,
            "zbl_cuton": self.zbl_cuton,
            "zbl_cutoff": self.zbl_cutoff,
            "debug": self.debug,
            "efa": self.efa,
            "use_energy_bias": self.use_energy_bias,
            "charge_embed_dim": self.charge_embed_dim,
            "spin_embed_dim": self.spin_embed_dim,
            "charge_range": self.charge_range,
            "spin_range": self.spin_range,
            "switch_start": self.switch_start,
            "switch_end": self.switch_end,
            "electrostatics_off_start": self.electrostatics_off_start,
            "electrostatics_off_end": self.electrostatics_off_end,
            "efa_lebedev_num": self.efa_lebedev_num,
            "efa_max_length": self.efa_max_length,
            "efa_ti_degree_scaling_base": self.efa_ti_degree_scaling_base,
        }
    
    def _embed_molecular_properties(
        self,
        total_charges: jnp.ndarray,
        total_spins: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Embed molecular total charge and spin.
        
        Parameters
        ----------
        total_charges : jnp.ndarray
            (batch_size,) Total molecular charges
        total_spins : jnp.ndarray
            (batch_size,) Total spin multiplicities (2S+1)
            
        Returns
        -------
        jnp.ndarray
            (batch_size, charge_embed_dim + spin_embed_dim) Embedded properties
        """
        # Create embedding tables for charge and spin
        charge_min, charge_max = self.charge_range
        spin_min, spin_max = self.spin_range
        
        n_charge_states = charge_max - charge_min + 1
        n_spin_states = spin_max - spin_min + 1
        
        # Charge embedding layer
        charge_embed = nn.Embed(
            num_embeddings=n_charge_states,
            features=self.charge_embed_dim,
            dtype=DTYPE,
            name="charge_embed",
        )
        
        # Spin embedding layer  
        spin_embed = nn.Embed(
            num_embeddings=n_spin_states,
            features=self.spin_embed_dim,
            dtype=DTYPE,
            name="spin_embed",
        )
        
        # Convert charges and spins to indices
        charge_indices = jnp.clip(
            (total_charges - charge_min).astype(jnp.int32),
            0,
            n_charge_states - 1,
        )
        spin_indices = jnp.clip(
            (total_spins - spin_min).astype(jnp.int32),
            0,
            n_spin_states - 1,
        )
        
        # Embed
        charge_features = charge_embed(charge_indices)  # (batch_size, charge_embed_dim)
        spin_features = spin_embed(spin_indices)        # (batch_size, spin_embed_dim)
        
        # Concatenate charge and spin embeddings
        mol_features = jnp.concatenate([charge_features, spin_features], axis=-1)
        
        return mol_features  # (batch_size, charge_embed_dim + spin_embed_dim)
    
    def _broadcast_molecular_features(
        self,
        mol_features: jnp.ndarray,
        batch_segments: jnp.ndarray,
        num_atoms: int,
    ) -> jnp.ndarray:
        """
        Broadcast molecular features to all atoms.
        
        Parameters
        ----------
        mol_features : jnp.ndarray
            (batch_size, mol_feature_dim) Molecular features
        batch_segments : jnp.ndarray
            (num_atoms,) Molecule index for each atom
        num_atoms : int
            Total number of atoms
            
        Returns
        -------
        jnp.ndarray
            (num_atoms, mol_feature_dim) Features broadcast to atoms
        """
        return mol_features[batch_segments]
    
    def _calculate_geometric_features(
        self,
        positions: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        batch_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Calculate pairwise geometric features.

        ``batch_mask`` zeroes the message-passing basis for padding edges so
        pad atoms at the origin cannot corrupt real-atom features inside the
        cutoff (translation-invariance bug). Masked pairs are also shifted off
        r=0 before the basis to avoid 0 * NaN forces from coincident padding.
        """
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst

        basis_displacements = displacements
        if batch_mask is not None:
            m = batch_mask.astype(displacements.dtype).reshape(-1, 1)
            basis_displacements = displacements + (1.0 - m)

        basis = e3x.nn.basis(
            basis_displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.exponential_chebyshev,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
        )
        if batch_mask is not None:
            basis = basis * batch_mask.astype(basis.dtype).reshape(
                -1, *([1] * (basis.ndim - 1))
            )
        return basis, displacements
    
    def _process_atomic_features(
        self,
        atomic_numbers: jnp.ndarray,
        basis: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        positions: jnp.ndarray,
        batch_segments: jnp.ndarray,
        graph_mask: jnp.ndarray,
        mol_features_per_atom: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Process atomic features with message passing and molecular conditioning.
        
        Parameters
        ----------
        atomic_numbers : jnp.ndarray
            Atomic numbers
        basis : jnp.ndarray
            Geometric basis functions
        dst_idx : jnp.ndarray
            Destination indices
        src_idx : jnp.ndarray
            Source indices
        positions : jnp.ndarray
            Atomic positions
        batch_segments : jnp.ndarray
            Batch segment indices
        graph_mask : jnp.ndarray
            Graph mask
        mol_features_per_atom : jnp.ndarray
            (num_atoms, mol_feature_dim) Molecular features broadcast to atoms
            
        Returns
        -------
        jnp.ndarray
            Processed atomic features
        """
        # Embed atomic numbers
        embed = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1,
            features=self.features,
            dtype=DTYPE,
        )
        x = embed(atomic_numbers)
        
        # Add molecular property features to initial atomic features
        # Project molecular features and convert to e3x format
        mol_projection = nn.Dense(
            self.features,
            dtype=DTYPE,
            name="mol_feature_projection",
        )
        mol_features_projected = mol_projection(mol_features_per_atom)  # (num_atoms, features)
        
        # Convert to e3x format (add parity and degree dimensions)
        # x has shape (num_atoms, 1, max_degree+1, features) in e3x format
        # We need to match this structure
        mol_features_e3x = jnp.expand_dims(mol_features_projected, axis=(1, 2))  # (num_atoms, 1, 1, features)
        
        # Broadcast to match x's degree dimension
        # Only add to L=0 (scalar) features
        mol_features_e3x = jnp.pad(
            mol_features_e3x,
            ((0, 0), (0, 0), (0, x.shape[2] - 1), (0, 0)),
            mode='constant',
            constant_values=0
        )  # (num_atoms, 1, max_degree+1, features)
        
        # Add to atomic features (only affects L=0 components)
        x = e3x.nn.add(x, mol_features_e3x)
        
        # Message passing iterations (reusing from original PhysNet)
        for i in range(self.num_iterations):
            x = self._message_passing_iteration(
                x, basis, dst_idx, src_idx, i, positions, batch_segments, graph_mask
            )
            x = self._refinement_iteration(x)
        
        # Final processing
        basis = e3x.nn.change_max_degree_or_type(
            basis, max_degree=0, include_pseudotensors=False
        )
        x = e3x.nn.change_max_degree_or_type(
            x, max_degree=0, include_pseudotensors=False
        )
        
        if self.n_refinement_blocks <= -1:
            for i in range(self.num_iterations):
                x = self._attention(
                    x, basis, dst_idx, src_idx, num_heads=self.features // 8
                )
                x = self._refinement_iteration(x)
        
        return x
    
    def _message_passing_iteration(
        self,
        x: jnp.ndarray,
        basis: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        iteration: int,
        positions: jnp.ndarray,
        batch_segments: jnp.ndarray,
        graph_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Single message passing iteration."""
        # Use the same pattern as working PhysNet
        if iteration == self.num_iterations - 1:
            x = e3x.nn.MessagePass(
                max_degree=0,
                include_pseudotensors=False,
            )(x, basis, dst_idx=dst_idx, src_idx=src_idx, indices_are_sorted=False)
            return x
        
        x = e3x.nn.MessagePass(
            include_pseudotensors=False,
        )(x, basis, dst_idx=dst_idx, src_idx=src_idx, indices_are_sorted=False)
        
        return x
    
    def _refinement_iteration(self, x: jnp.ndarray) -> jnp.ndarray:
        """Refinement with residual blocks."""
        for _ in range(self.n_refinement_blocks):
            y = nn.Dense(self.features, dtype=DTYPE)(x)
            y = e3x.nn.silu(y)
            y = nn.Dense(self.features, dtype=DTYPE)(y)
            x = e3x.nn.add(x, y)
            x = e3x.nn.silu(x)
        
        return x
    
    def _attention(
        self,
        x: jnp.ndarray,
        basis: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        num_heads: int = 2,
    ) -> jnp.ndarray:
        """Attention mechanism."""
        # Simplified attention - can be expanded
        return x
    
    # _calc_switches / geometric features / electrostatics / dipole via mixin
    # (electrostatics_damping_sigma defaults to 0 — ChargeSpin never had erf damping)

    def _calculate(
        self,
        x: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        displacements: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        atom_mask: jnp.ndarray,
        batch_mask: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
    ) -> Tuple[Array, Tuple[Array, Array, Array, Array, Array]]:
        """Calculate final energy and related quantities."""
        # Calculate switching functions
        r, off_dist, eshift = self._calc_switches(displacements, batch_mask)
        zbl_distances = geometric_pair_distances(displacements, batch_mask)
        
        # Predict atomic energies
        energy_per_atom = nn.Dense(1, use_bias=False, dtype=DTYPE, name="energy_dense")(x)
        energy_per_atom = jnp.squeeze(energy_per_atom, axis=-1)
        
        # Apply energy bias if enabled
        if self.use_energy_bias:
            energy_bias = self.param(
                "energy_bias",
                nn.initializers.zeros,
                (self.max_atomic_number + 1,),
                DTYPE,
            )
            energy_per_atom = energy_per_atom + jnp.take(
                jnp.asarray(energy_bias), atomic_numbers
            )
        
        # Mask padding atoms
        energy_per_atom = energy_per_atom * atom_mask
        
        # Charges and electrostatics (if enabled)
        charges = jnp.zeros_like(atomic_numbers, dtype=DTYPE)
        electrostatics = jnp.zeros(batch_size, dtype=DTYPE)
        
        if self.charges:
            charges = nn.Dense(1, use_bias=False, dtype=DTYPE, name="charges_dense")(x)
            charges = jnp.squeeze(charges, axis=-1)
            charges = charges * atom_mask
        
        # ZBL repulsion (if enabled)
        repulsion = jnp.zeros_like(atomic_numbers, dtype=DTYPE)
        if self.zbl:
            repulsion = self.repulsion(
                atomic_numbers,
                zbl_distances,
                None,
                1 - eshift,  # Note: 1 - eshift as in working model
                dst_idx,
                src_idx,
                atom_mask,
                batch_mask,
                batch_segments,
                batch_size,
            )
        else:
            repulsion = jnp.zeros_like(atomic_numbers, dtype=DTYPE)
        
        # Sum to get molecular energy
        energy = jax.ops.segment_sum(
            energy_per_atom + repulsion,
            segment_ids=batch_segments,
            num_segments=batch_size,
        )
        
        # Return negative energy for gradient calculation
        total_energy = -jnp.sum(energy)
        
        return total_energy, (energy, charges, electrostatics, repulsion, x)
    
    def energy(
        self,
        atomic_numbers: jnp.ndarray,
        positions: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
        batch_mask: jnp.ndarray,
        atom_mask: jnp.ndarray,
        total_charges: jnp.ndarray,
        total_spins: jnp.ndarray,
    ) -> Tuple[Array, Tuple[Array, Array, Array, Array, Array]]:
        """
        Calculate molecular energy with charge and spin conditioning.
        
        Parameters
        ----------
        atomic_numbers : jnp.ndarray
            Atomic numbers
        positions : jnp.ndarray
            Atomic positions
        dst_idx : jnp.ndarray
            Destination indices for message passing
        src_idx : jnp.ndarray
            Source indices for message passing
        batch_segments : jnp.ndarray
            Batch segment indices
        batch_size : int
            Number of molecules in batch
        batch_mask : jnp.ndarray
            Mask for valid batch elements
        atom_mask : jnp.ndarray
            Mask for valid atoms
        total_charges : jnp.ndarray
            (batch_size,) Total molecular charges
        total_spins : jnp.ndarray
            (batch_size,) Total spin multiplicities
            
        Returns
        -------
        tuple
            (total_energy, (energy, charges, electrostatics, repulsion, features))
        """
        # Calculate geometric features (batch_mask zeroes padding edges before
        # MessagePass — see _calculate_geometric_features).
        basis, displacements = self._calculate_geometric_features(
            positions, dst_idx, src_idx, batch_mask=batch_mask
        )
        
        # Embed molecular properties (charge and spin)
        mol_features = self._embed_molecular_properties(total_charges, total_spins)
        
        # Broadcast molecular features to all atoms
        num_atoms = atomic_numbers.shape[0]
        mol_features_per_atom = self._broadcast_molecular_features(
            mol_features, batch_segments, num_atoms
        )
        
        graph_mask = jnp.ones(batch_size)
        
        # Process atomic features with molecular conditioning
        x = self._process_atomic_features(
            atomic_numbers,
            basis,
            dst_idx,
            src_idx,
            positions,
            batch_segments,
            graph_mask,
            mol_features_per_atom,
        )
        
        return self._calculate(
            x,
            atomic_numbers,
            displacements,
            dst_idx,
            src_idx,
            atom_mask,
            batch_mask,
            batch_segments,
            batch_size,
        )
    
    @nn.compact
    def __call__(
        self,
        atomic_numbers: jnp.ndarray,
        positions: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        total_charges: jnp.ndarray,
        total_spins: jnp.ndarray,
        batch_segments: Optional[jnp.ndarray] = None,
        batch_size: Optional[int] = None,
        batch_mask: Optional[jnp.ndarray] = None,
        atom_mask: Optional[jnp.ndarray] = None,
        predict_forces: bool = True,
        predict_energy: bool = True,
    ) -> Dict[str, Optional[jnp.ndarray]]:
        """
        Forward pass with charge and spin conditioning.
        
        Parameters
        ----------
        atomic_numbers : jnp.ndarray
            Array of atomic numbers
        positions : jnp.ndarray
            Array of atomic positions
        dst_idx : jnp.ndarray
            Destination indices for message passing
        src_idx : jnp.ndarray
            Source indices for message passing
        total_charges : jnp.ndarray
            (batch_size,) Total molecular charges
        total_spins : jnp.ndarray
            (batch_size,) Total spin multiplicities (2S+1)
            Examples: 1=singlet, 2=doublet, 3=triplet, etc.
        batch_segments : Optional[jnp.ndarray], optional
            Batch segment indices
        batch_size : Optional[int], optional
            Batch size
        batch_mask : Optional[jnp.ndarray], optional
            Batch mask
        atom_mask : Optional[jnp.ndarray], optional
            Atom mask
        predict_forces : bool, optional
            Whether to compute forces (default: True)
            Set to False for faster energy-only predictions
        predict_energy : bool, optional
            Whether to compute energy (default: True)
            Set to False for forces-only predictions (rare use case)
            
        Returns
        -------
        Dict[str, Optional[jnp.ndarray]]
            Dictionary containing:
            - energy: Predicted energies (batch_size,) if predict_energy=True
            - forces: Predicted forces (num_atoms, 3) if predict_forces=True
            - charges: Predicted atomic charges (num_atoms,) if self.charges=True
            - electrostatics: Electrostatic energies (batch_size,) if self.charges=True
            - repulsion: ZBL repulsion energies (batch_size,) if self.zbl=True
            - state: Final atomic features
        """
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1
            batch_mask = jnp.ones_like(dst_idx)
            atom_mask = jnp.ones_like(atomic_numbers)
        
        # Branch based on what we need to compute
        if predict_forces and predict_energy:
            # Compute both energy and forces (forces via gradient)
            energy_and_forces = jax.value_and_grad(self.energy, argnums=1, has_aux=True)
            
            (_, (energy, charges, electrostatics, repulsion, state)), gradient = energy_and_forces(
                atomic_numbers,
                positions,
                dst_idx,
                src_idx,
                batch_segments,
                batch_size,
                batch_mask,
                atom_mask,
                total_charges,
                total_spins,
            )
            
            # NOTE: energy() returns -E (negative energy)
            # So gradient = d(-E)/dr = -dE/dr
            # Therefore forces = -gradient = dE/dr
            forces = -gradient
            energy = -energy
            
        elif predict_energy and not predict_forces:
            # Energy only (no gradient computation - faster!)
            _, (energy, charges, electrostatics, repulsion, state) = self.energy(
                atomic_numbers,
                positions,
                dst_idx,
                src_idx,
                batch_segments,
                batch_size,
                batch_mask,
                atom_mask,
                total_charges,
                total_spins,
            )
            energy = -energy
            forces = None
            
        elif predict_forces and not predict_energy:
            # Forces only (still need to compute energy for gradient)
            energy_and_forces = jax.value_and_grad(self.energy, argnums=1, has_aux=True)
            
            (_, (energy_raw, charges, electrostatics, repulsion, state)), gradient = energy_and_forces(
                atomic_numbers,
                positions,
                dst_idx,
                src_idx,
                batch_segments,
                batch_size,
                batch_mask,
                atom_mask,
                total_charges,
                total_spins,
            )
            
            forces = -gradient
            energy = None  # Don't return energy
            
        else:
            # Neither energy nor forces requested - just return state
            _, (_, charges, electrostatics, repulsion, state) = self.energy(
                atomic_numbers,
                positions,
                dst_idx,
                src_idx,
                batch_segments,
                batch_size,
                batch_mask,
                atom_mask,
                total_charges,
                total_spins,
            )
            energy = None
            forces = None
        
        return {
            "energy": energy if predict_energy else None,
            "forces": forces if predict_forces else None,
            "charges": charges if self.charges else None,
            "electrostatics": electrostatics if self.charges else None,
            "repulsion": repulsion if self.zbl else None,
            "state": state,
        }


EF_ChargeSpinConditioned = PhysNetChargeSpin  # deprecated alias

__all__ = ["PhysNetChargeSpin", "EF_ChargeSpinConditioned"]
