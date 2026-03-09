"""
PhysNet model with Total Charge and Total Spin conditioning.

This module implements an enhanced PhysNet that accepts molecular properties
(total charge and total spin) as additional inputs, enabling charge and spin
state-dependent energy and force predictions.
"""

import functools
from typing import Dict, List, Optional, Tuple

import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array
import ase.data

from mmml.physnetjax.physnetjax.models.euclidean_fast_attention import fast_attention as efa
from mmml.physnetjax.physnetjax.models.zbl import ZBLRepulsion

EFA = efa.EuclideanFastAttention

# Constants
DTYPE = jnp.float32
HARTREE_TO_KCAL_MOL = 627.509


class EF_ChargeSpinConditioned(nn.Module):
    """PhysNet Energy and Forces model with charge and spin conditioning.
    
    This model extends the standard PhysNet to accept total molecular charge
    and total spin as additional inputs. These properties are embedded and
    used to condition the atomic feature representations, allowing the model
    to predict charge and spin state-dependent energies and forces.
    
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
    natoms: int = 60
    total_charge: float = 0
    n_res: int = 3
    zbl: bool = True
    debug: bool | List[str] = False
    efa: bool = False
    use_energy_bias: bool = True
    
    # New: Charge and spin conditioning parameters
    charge_embed_dim: int = 16
    spin_embed_dim: int = 16
    charge_range: Tuple[int, int] = (-5, 5)  # Support charges from -5 to +5
    spin_range: Tuple[int, int] = (1, 7)     # Support spin multiplicities 1-7 (singlet to septet)
    
    def setup(self) -> None:
        """Initialize model components including property embeddings."""
        # Standard PhysNet components
        if self.zbl:
            self.repulsion = ZBLRepulsion(
                cutoff=self.cutoff,
                trainable=True,
            )
        
        self.efa_final = None
        if self.efa:
            b_max = 4 * jnp.pi
            self.efa_final = EFA(
                lebedev_num=194,
                parametrized=False,
                epe_max_frequency=b_max,
                epe_max_length=20.0,
                tensor_integration=True,
                ti_degree_scaling_constants=[
                    0.5**i for i in range(self.max_degree + 1)
                ],
            )
    
    def return_attributes(self) -> Dict:
        """Return model attributes for checkpointing."""
        return {
            "features": self.features,
            "max_degree": self.max_degree,
            "num_iterations": self.num_iterations,
            "num_basis_functions": self.num_basis_functions,
            "cutoff": self.cutoff,
            "max_atomic_number": self.max_atomic_number,
            "charges": self.charges,
            "natoms": self.natoms,
            "total_charge": self.total_charge,
            "n_res": self.n_res,
            "zbl": self.zbl,
            "debug": self.debug,
            "efa": self.efa,
            "use_energy_bias": self.use_energy_bias,
            "charge_embed_dim": self.charge_embed_dim,
            "spin_embed_dim": self.spin_embed_dim,
            "charge_range": self.charge_range,
            "spin_range": self.spin_range,
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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Calculate pairwise geometric features."""
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst
        
        basis = e3x.nn.basis(
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.exponential_chebyshev,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
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
        
        if self.n_res <= -1:
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
        for _ in range(self.n_res):
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
    
    def _calc_switches(
        self,
        displacements: jnp.ndarray,
        batch_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Calculate switching functions for electrostatics and repulsion."""
        eps = 1e-6
        min_dist = 0.01
        switch_start = 1.0
        switch_end = 10.0
        
        displacements = displacements + (1 - batch_mask)[..., None]
        squared_distances = jnp.sum(displacements**2, axis=1)
        distances = jnp.sqrt(jnp.maximum(squared_distances, min_dist**2))
        
        switch_dist = e3x.nn.smooth_switch(distances, switch_start, switch_end)
        off_dist = 1.0 - e3x.nn.smooth_switch(distances, 8.0, 10.0)
        one_minus_switch_dist = 1 - switch_dist
        
        safe_distances = distances + eps
        r1 = switch_dist / jnp.sqrt(squared_distances + 1.0)
        r2 = one_minus_switch_dist / safe_distances
        r = r1 + r2
        eshift = safe_distances / (switch_end**2) - 2.0 / switch_end
        
        off_dist *= batch_mask
        eshift *= batch_mask
        
        return r, off_dist, eshift
    
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
            energy_per_atom = energy_per_atom + energy_bias[atomic_numbers]
        
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
                r,
                off_dist,
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
        # Calculate geometric features
        basis, displacements = self._calculate_geometric_features(
            positions, dst_idx, src_idx
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

