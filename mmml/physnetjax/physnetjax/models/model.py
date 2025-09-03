"""
Energy and Forces Neural Network Model implementation.

This module implements a neural network model for predicting molecular energies 
and forces using message passing and equivariant transformations.
"""

import functools
from typing import Union
from typing import Dict, List, Optional, Tuple

import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array
# from jax.experimental import mesh_utils
# from jax.sharding import Mesh
# from jax.sharding import NamedSharding
# from jax.sharding import PartitionSpec as P

from physnetjax.models.euclidean_fast_attention import fast_attention as efa
from physnetjax.models.zbl import ZBLRepulsion

EFA = efa.EuclideanFastAttention
import ase.data

# Constants
DTYPE = jnp.float32
HARTREE_TO_KCAL_MOL = 627.509  # Conversion factor for energy units


class EF(nn.Module):
    """Energy and Forces Neural Network Model.

    A neural network model that predicts molecular energies and forces using message passing
    and equivariant transformations.

    Attributes:
        features: Number of features in the neural network layers
        max_degree: Maximum degree for spherical harmonics
        num_iterations: Number of message passing iterations
        num_basis_functions: Number of radial basis functions
        cutoff: Cutoff distance for interactions
        max_atomic_number: Maximum atomic number supported
        charges: Whether to predict atomic charges
        natoms: Maximum number of atoms in a molecule
        total_charge: Total molecular charge constraint
        n_res: Number of residual blocks
        debug: Debug flags (False or list of debug areas)
    """

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

    def setup(self) -> None:
        """
        Initialize model components.
        
        Sets up the model architecture including ZBL repulsion and
        Euclidean Fast Attention (EFA) if enabled.
        """
        if self.zbl:
            self.repulsion = ZBLRepulsion(
                cutoff=self.cutoff,
                trainable=True,
            )
        self.efa_final = None
        if self.efa:
            b_max = 4 * jnp.pi
            # We now initialize an EFA module.
            self.efa_final = EFA(
                lebedev_num=194,
                parametrized=False,
                epe_max_frequency=b_max,
                epe_max_length=20.0,  # maximum distance in Angstroms for the EPE
                tensor_integration=True,
                ti_degree_scaling_constants=[
                    0.5**i for i in range(self.max_degree + 1)
                ],
            )

    def return_attributes(self) -> Dict:
        """
        Return model attributes for checkpointing.
        
        Returns
        -------
        Dict
            Dictionary containing all model hyperparameters and configuration
        """
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
        }

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
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        """
        Calculate molecular energy and related properties.

        Computes the total energy of molecular systems including atomic energies,
        electrostatic interactions, and repulsion terms.

        Parameters
        ----------
        atomic_numbers : jnp.ndarray
            Array of atomic numbers for each atom
        positions : jnp.ndarray
            Array of atomic coordinates
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

        Returns
        -------
        tuple[Array, tuple[Array, Array, Array, Array]]
            Tuple containing:
            - Total energy (negative sum)
            - Tuple of (energy, charges, electrostatics, repulsion, features)
        """
        # Calculate basic geometric features
        basis, displacements = self._calculate_geometric_features(
            positions, dst_idx, src_idx
        )

        graph_mask = jnp.ones(batch_size)

        # Embed and process atomic features
        x = self._process_atomic_features(
            atomic_numbers,
            basis,
            dst_idx,
            src_idx,
            positions,
            batch_segments,
            graph_mask,
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

    def _calculate_geometric_features(
        self,
        positions: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculate geometric features including displacements and basis functions.
        
        Parameters
        ----------
        positions : jnp.ndarray
            Atomic positions
        dst_idx : jnp.ndarray
            Destination indices for message passing
        src_idx : jnp.ndarray
            Source indices for message passing
            
        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            Tuple of (basis functions, displacements)
        """
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst
        # print(displacements)
        return (
            e3x.nn.basis(
                displacements,
                num=self.num_basis_functions,
                max_degree=self.max_degree,
                radial_fn=e3x.nn.exponential_chebyshev,
                cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
            ),
            displacements,
        )

    def _process_atomic_features(
        self,
        atomic_numbers: jnp.ndarray,
        basis: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        positions: jnp.ndarray,
        batch_segments: jnp.ndarray,
        graph_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Process atomic features through message passing and refinement.
        
        Parameters
        ----------
        atomic_numbers : jnp.ndarray
            Atomic numbers
        basis : jnp.ndarray
            Basis functions
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
            
        Returns
        -------
        jnp.ndarray
            Processed atomic features
        """
        embed = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1,
            features=self.features,
            dtype=DTYPE,
        )
        x = embed(atomic_numbers)

        for i in range(self.num_iterations):
            x = self._message_passing_iteration(
                x, basis, dst_idx, src_idx, i, positions, batch_segments, graph_mask
            )
            x = self._refinement_iteration(x)

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

    def _attention(self, x, basis, dst_idx, src_idx, num_heads=2):
        """
        Apply self-attention mechanism.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input features
        basis : jnp.ndarray
            Basis functions
        dst_idx : jnp.ndarray
            Destination indices
        src_idx : jnp.ndarray
            Source indices
        num_heads : int, optional
            Number of attention heads, by default 2
            
        Returns
        -------
        jnp.ndarray
            Attention output
        """
        return e3x.nn.modules.SelfAttention(
            max_degree=0,
            num_heads=num_heads,
            include_pseudotensors=False,
        )(x, basis, dst_idx=dst_idx, src_idx=src_idx)

    def _multiheadattention(self, x, y, basis, dst_idx, src_idx, num_heads=2):
        """
        Apply multi-head attention mechanism.
        
        Parameters
        ----------
        x : jnp.ndarray
            Query features
        y : jnp.ndarray
            Key/value features
        basis : jnp.ndarray
            Basis functions
        dst_idx : jnp.ndarray
            Destination indices
        src_idx : jnp.ndarray
            Source indices
        num_heads : int, optional
            Number of attention heads, by default 2
            
        Returns
        -------
        jnp.ndarray
            Multi-head attention output
        """
        return e3x.nn.modules.MultiHeadAttention(
            max_degree=self.max_degree,
            num_heads=num_heads,
            include_pseudotensors=False,
        )(x, y, basis, dst_idx=dst_idx, src_idx=src_idx)

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
        """
        Perform one iteration of message passing.
        
        Parameters
        ----------
        x : jnp.ndarray
            Current features
        basis : jnp.ndarray
            Basis functions
        dst_idx : jnp.ndarray
            Destination indices
        src_idx : jnp.ndarray
            Source indices
        iteration : int
            Current iteration number
        positions : jnp.ndarray
            Atomic positions
        batch_segments : jnp.ndarray
            Batch segment indices
        graph_mask : jnp.ndarray
            Graph mask
            
        Returns
        -------
        jnp.ndarray
            Updated features after message passing
        """
        # if it is the last iteration
        if iteration == self.num_iterations - 1:
            x = e3x.nn.MessagePass(
                max_degree=0,
                include_pseudotensors=False,
                # dense_kernel_init=jax.nn.initializers.he_uniform(),
                # dense_bias_init=jax.nn.initializers.zeros,
            )(x, basis, dst_idx=dst_idx, src_idx=src_idx, indices_are_sorted=False)
            return x

        x = e3x.nn.MessagePass(
            include_pseudotensors=False,
            # dense_kernel_init=jax.nn.initializers.he_normal(),
            # dense_bias_init=jax.nn.initializers.zeros,
        )(x, basis, dst_idx=dst_idx, src_idx=src_idx, indices_are_sorted=False)
        if self.efa:
            x1 = self.efa_final(x, positions, batch_segments, graph_mask)
            x = e3x.nn.add(x, x1)
        return x

    def _refinement_iteration(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Perform refinement iterations with residual connections.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input features
            
        Returns
        -------
        jnp.ndarray
            Refined features
        """
        x1 = e3x.nn.silu(x)
        for _ in range(abs(self.n_res)):
            y = e3x.nn.silu(x)
            y = e3x.nn.add(x, y)
            y = e3x.nn.Dense(
                self.features,
            )(y)
            x = e3x.nn.add(x, y)
        y = e3x.nn.Dense(
            self.features,
        )(y)
        y = e3x.nn.silu(y)

        return y

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
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        """
        Calculate energies including charge interactions.
        
        Parameters
        ----------
        x : jnp.ndarray
            Processed atomic features
        atomic_numbers : jnp.ndarray
            Atomic numbers
        displacements : jnp.ndarray
            Interatomic displacements
        dst_idx : jnp.ndarray
            Destination indices
        src_idx : jnp.ndarray
            Source indices
        atom_mask : jnp.ndarray
            Atom mask
        batch_mask : jnp.ndarray
            Batch mask
        batch_segments : jnp.ndarray
            Batch segment indices
        batch_size : int
            Batch size
            
        Returns
        -------
        tuple[Array, tuple[Array, Array, Array, Array]]
            Tuple of (total energy, (atomic energies, charges, electrostatics, repulsion))
        """
        r, off_dist, eshift = self._calc_switches(displacements, batch_mask)

        atomic_energies = self._calculate_atomic_energies(x, atomic_numbers, atom_mask)

        if self.charges:
            atomic_charges = self._calculate_atomic_charges(
                x, atomic_numbers, atom_mask
            )
            electrostatics, batch_electrostatics = self._calculate_electrostatics(
                atomic_charges,
                r,
                off_dist,
                eshift,
                dst_idx,
                src_idx,
                atom_mask,
                batch_mask,
                batch_segments,
                batch_size,
            )
        else:
            atomic_charges = None
            electrostatics = 0.0
            batch_electrostatics = None

        if self.zbl:
            repulsion = self._calculate_repulsion(
                atomic_numbers,
                r,
                off_dist,
                1 - eshift,
                dst_idx,
                src_idx,
                atom_mask,
                batch_mask,
                batch_segments,
                batch_size,
            )
            # # repulsion *= batch_mask[..., None]
            # if not self.debug and "repulsion" in self.debug:
            #     jax.debug.print("Repulsion shape: {x}", x=repulsion.shape)
            #     jax.debug.print("Repulsion: {x}", x=repulsion)

        else:
            repulsion = 0.0

        energy = jax.ops.segment_sum(
            atomic_energies + electrostatics + repulsion,
            segment_ids=batch_segments,
            num_segments=batch_size,
        )

        return -1 * jnp.sum(energy), (
            energy,
            atomic_charges,
            batch_electrostatics,
            repulsion,
            x,
        )

    def _calculate_atomic_charges(
        self, x: jnp.ndarray, atomic_numbers: jnp.ndarray, atom_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate atomic charges from atomic features.
        
        Parameters
        ----------
        x : jnp.ndarray
            Atomic features
        atomic_numbers : jnp.ndarray
            Atomic numbers
        atom_mask : jnp.ndarray
            Atom mask
            
        Returns
        -------
        jnp.ndarray
            Predicted atomic charges
        """
        x = e3x.nn.Dense(1, use_bias=False)(x)

        charge_bias = self.param(
            "charge_bias",
            lambda rng, shape: jnp.zeros(shape),
            (self.max_atomic_number + 1),
        )
        atomic_charges = nn.Dense(
            1, use_bias=False, kernel_init=jax.nn.initializers.zeros, dtype=DTYPE
        )(x)
        atomic_charges += charge_bias[atomic_numbers][..., None, None, None]
        atomic_charges *= atom_mask[..., None, None, None]
        return atomic_charges

    def _calculate_repulsion(
        self,
        atomic_numbers: jnp.ndarray,
        distances: jnp.ndarray,
        off_dist: jnp.ndarray,
        eshift: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        atom_mask: jnp.ndarray,
        batch_mask: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
    ) -> jnp.ndarray:
        """
        Calculate repulsion energies between atoms.
        
        Parameters
        ----------
        atomic_numbers : jnp.ndarray
            Atomic numbers
        distances : jnp.ndarray
            Interatomic distances
        off_dist : jnp.ndarray
            Distance cutoff factors
        eshift : jnp.ndarray
            Energy shift factors
        dst_idx : jnp.ndarray
            Destination indices
        src_idx : jnp.ndarray
            Source indices
        atom_mask : jnp.ndarray
            Atom mask
        batch_mask : jnp.ndarray
            Batch mask
        batch_segments : jnp.ndarray
            Batch segment indices
        batch_size : int
            Batch size
            
        Returns
        -------
        jnp.ndarray
            Repulsion energies per atom
        """
        # add the learnable parameters to the model
        repulsion_energy = self.repulsion(
            atomic_numbers,
            distances,
            off_dist,
            eshift,
            dst_idx,
            src_idx,
            atom_mask,
            batch_mask,
            batch_segments,
            batch_size,
        )
        return repulsion_energy

    def _calculate_atomic_energies(
        self, x: jnp.ndarray, atomic_numbers: jnp.ndarray, atom_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate atomic energies from atomic features.
        
        Parameters
        ----------
        x : jnp.ndarray
            Atomic features
        atomic_numbers : jnp.ndarray
            Atomic numbers
        atom_mask : jnp.ndarray
            Atom mask
            
        Returns
        -------
        jnp.ndarray
            Predicted atomic energies
        """
        x = e3x.nn.Dense(1, use_bias=False)(x)
        energy_bias = self.param(
            "energy_bias",
            lambda rng, shape: jnp.zeros(shape),
            (self.max_atomic_number + 1),
        )
        atomic_energies = nn.Dense(
            1, use_bias=False, kernel_init=jax.nn.initializers.zeros, dtype=DTYPE
        )(x)
        atomic_energies += energy_bias[atomic_numbers][..., None, None, None]
        atomic_energies *= atom_mask[..., None, None, None]

        return atomic_energies

    def _calc_switches(self, displacements: jnp.ndarray, batch_mask: jnp.ndarray):
        """
        Calculate switching functions for smooth interactions.
        
        Parameters
        ----------
        displacements : jnp.ndarray
            Interatomic displacements
        batch_mask : jnp.ndarray
            Batch mask
            
        Returns
        -------
        tuple
            Tuple of (r, off_dist, eshift) switching factors
        """
        # Numerical stability constants
        eps = 1e-6
        min_dist = 0.01  # Minimum distance in Angstroms
        switch_start = 1.0  # Start switching at 2 Angstroms
        switch_end = 10.0  # Complete switch by 10 Angstroms
        # Calculate distances between atom pairs
        displacements = displacements + (1 - batch_mask)[..., None]
        # Safe distance calculation with minimum cutoff
        squared_distances = jnp.sum(displacements**2, axis=1)
        distances = jnp.sqrt(jnp.maximum(squared_distances, min_dist**2))
        # Improved switching function
        switch_dist = e3x.nn.smooth_switch(distances, switch_start, switch_end)
        off_dist = 1.0 - e3x.nn.smooth_switch(distances, 8.0, 10.0)
        one_minus_switch_dist = 1 - switch_dist
        # Calculate interaction potential with improved stability
        safe_distances = distances + eps
        # R1: Short-range regularized potential
        r1 = switch_dist / jnp.sqrt(squared_distances + 1.0)
        # R2: Long-range Coulomb potential with safe distance
        r2 = one_minus_switch_dist / safe_distances
        r = r1 + r2
        eshift = safe_distances / (switch_end**2) - 2.0 / switch_end
        # r *= batch_mask[..., None]
        off_dist *= batch_mask
        eshift *= batch_mask
        return r, off_dist, eshift

    def _calculate_electrostatics(
        self,
        atomic_charges: jnp.ndarray,
        r: jnp.ndarray,
        off_dist: jnp.ndarray,
        eshift: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        atom_mask: jnp.ndarray,
        batch_mask: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
    ) -> Tuple[jnp.ndarray, jnp.array]:
        """
        Calculate electrostatic interactions between atoms.

        Uses a smoothly switched combination of short-range and long-range electrostatics
        to avoid numerical instabilities at zero distance while maintaining accuracy.

        Parameters
        ----------
        atomic_charges : jnp.ndarray
            Predicted atomic charges
        r : jnp.ndarray
            Distance factors
        off_dist : jnp.ndarray
            Distance cutoff factors
        eshift : jnp.ndarray
            Energy shift factors
        dst_idx : jnp.ndarray
            Destination indices for pair interactions
        src_idx : jnp.ndarray
            Source indices for pair interactions
        atom_mask : jnp.ndarray
            Atom mask
        batch_mask : jnp.ndarray
            Batch mask
        batch_segments : jnp.ndarray
            Batch assignment for each atom
        batch_size : int
            Number of molecules in batch

        Returns
        -------
        Tuple[jnp.ndarray, jnp.array]
            Tuple of (atomic electrostatic energies, batch electrostatic energies)
        """
        # Get charges for interacting pairs with safe bounds
        q1 = jnp.clip(jnp.take(atomic_charges, dst_idx, fill_value=0.0), -10.0, 10.0)
        q2 = jnp.clip(jnp.take(atomic_charges, src_idx, fill_value=0.0), -10.0, 10.0)
        # Calculate electrostatic energy (in Hartree)
        # Conversion factor 7.199822675975274 is 1/(4π*ε₀) in atomic units
        electrostatics = 7.199822675975274 * q1 * q2 * r * batch_mask
        # apply shifted force truncation scheme
        electrostatics += eshift * batch_mask
        electrostatics *= off_dist
        # Sum contributions for each atom
        atomic_electrostatics = jax.ops.segment_sum(
            electrostatics,
            segment_ids=dst_idx,
            num_segments=batch_size * self.natoms,
        )
        # atomic_electrostatics *= atom_mask
        batch_electrostatics = jax.ops.segment_sum(
            atomic_electrostatics,
            segment_ids=batch_segments,
            num_segments=batch_size,
        )
        atomic_electrostatics = atomic_electrostatics[..., None, None, None]
        # if not self.debug and "ele" in self.debug:
        #     jax.debug.print(
        #         f"{atomic_electrostatics}", atomic_electrostatics=atomic_electrostatics
        #     )
        return atomic_electrostatics, batch_electrostatics

    def _calculate_dipole(
        self,
        positions: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        charges: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
    ) -> jnp.ndarray:
        """
        Calculate dipoles for a batch of molecules.

        Computes molecular dipole moments from atomic charges and positions
        relative to the center of mass of each molecule.

        Parameters
        ----------
        positions : jnp.ndarray
            Atomic positions
        atomic_numbers : jnp.ndarray
            Atomic numbers
        charges : jnp.ndarray
            Atomic charges
        batch_segments : jnp.ndarray
            Batch segment indices
        batch_size : int
            Number of molecules in the batch

        Returns
        -------
        jnp.ndarray
            Calculated dipoles for each molecule in the batch
        """
        charges = charges.squeeze()
        positions = positions.squeeze()
        atomic_numbers = atomic_numbers.squeeze()
        masses = jnp.take(ase.data.atomic_masses, atomic_numbers)
        bs_masses = jax.ops.segment_sum(
            masses, segment_ids=batch_segments, num_segments=batch_size
        )
        masses_per_atom = jnp.take(bs_masses, batch_segments)
        dis_com = positions * masses[..., None] / masses_per_atom[..., None]
        com = jnp.sum(dis_com, axis=1)
        pos_com = positions - com[..., None]
        dipoles = jax.ops.segment_sum(
            pos_com * charges[..., None],
            segment_ids=batch_segments,
            num_segments=batch_size,
        )
        return dipoles

    @nn.compact
    def __call__(
        self,
        atomic_numbers: jnp.ndarray,
        positions: jnp.ndarray,
        dst_idx: jnp.ndarray,
        src_idx: jnp.ndarray,
        batch_segments: Optional[jnp.ndarray] = None,
        batch_size: Optional[int] = None,
        batch_mask: Optional[jnp.ndarray] = None,
        atom_mask: Optional[jnp.ndarray] = None,
    ) -> Dict[str, Optional[jnp.ndarray]]:
        """
        Forward pass of the model.

        Computes energies, forces, and optionally charges and dipoles
        for molecular systems.

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
        batch_segments : Optional[jnp.ndarray], optional
            Optional batch segment indices, by default None
        batch_size : Optional[int], optional
            Optional batch size, by default None
        batch_mask : Optional[jnp.ndarray], optional
            Optional batch mask, by default None
        atom_mask : Optional[jnp.ndarray], optional
            Optional atom mask, by default None

        Returns
        -------
        Dict[str, Optional[jnp.ndarray]]
            Dictionary containing:
            - energy: Predicted energies
            - forces: Predicted forces
            - charges: Predicted charges (if enabled)
            - electrostatics: Electrostatic energies (if charges enabled)
            - repulsion: Repulsion energies (if ZBL enabled)
            - dipoles: Predicted dipoles (if charges enabled)
            - sum_charges: Sum of charges per molecule (if charges enabled)
            - state: Final atomic features
        """
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1
            batch_mask = jnp.ones_like(dst_idx)
            atom_mask = jnp.ones_like(atomic_numbers)

        # import lovely_jax as lj
        #
        # lj.monkey_patch()
        #
        # jax.debug.print("atomic_numbers {x}", x=atomic_numbers[::])
        # jax.debug.print("positions {x}", x=positions[::])
        # jax.debug.print("dst_idx {x}", x=dst_idx[::])
        # jax.debug.print("src_idx {x}", x=src_idx[::])
        # jax.debug.print("batch_segments {x}", x=batch_segments[::])
        # # jax.debug.print("batch_size {x}", x=batch_size[::1])
        # jax.debug.print("batch_mask {x}", x=batch_mask[::])
        # jax.debug.print("atom_mask {x}", x=atom_mask[::])

        # Since we want to also predict forces, i.e. the gradient of the energy w.r.t. positions (argument 1), we use
        # jax.value_and_grad to create a function for predicting both energy and forces for us.
        energy_and_forces = jax.value_and_grad(self.energy, argnums=1, has_aux=True)

        # Debug input shapes
        # if not self.debug and "idx" in self.debug:
        # # if True:
        #     jax.debug.print("atomic_numbers {x}", x=atomic_numbers.shape)
        #     jax.debug.print("positions {x}", x=positions.shape)
        #     jax.debug.print("dst_idx {x}", x=dst_idx.shape)
        #     jax.debug.print("src_idx {x}", x=src_idx.shape)
        #     jax.debug.print("batch_segments {x}", x=batch_segments.shape)
        #     jax.debug.print("batch_size {x}", x=batch_size)
        #     jax.debug.print("batch_mask {x}", x=batch_mask.shape)
        #     jax.debug.print("atom_mask {x}", x=atom_mask.shape)

        # Calculate energies and forces
        (_, (energy, charges, electrostatics, repulsion, state)), forces = energy_and_forces(
            atomic_numbers,
            positions,
            dst_idx,
            src_idx,
            batch_segments,
            batch_size,
            batch_mask,
            atom_mask,
        )
        forces *= atom_mask[..., None]

        dipoles = (
            self._calculate_dipole(
                positions,
                atomic_numbers,
                charges,
                batch_segments,
                batch_size,
            )
            if self.charges
            else None
        )
        sum_charges = (
            jax.ops.segment_sum(
                charges,
                segment_ids=batch_segments,
                num_segments=batch_size,
            )
            if self.charges
            else None
        )

        # Prepare output dictionary
        output = {
            "energy": energy,
            "forces": forces,
            "charges": charges,
            "electrostatics": electrostatics,
            "repulsion": repulsion,
            "dipoles": dipoles,
            "sum_charges": sum_charges,
            "state": state,
        }
        # Debug output values
        # jax.debug.print("Energy {x}", x=energy)
        # jax.debug.print("Forces {x}", x=forces)
        # batches
        # jax.debug.print("Ref. Energy {x}", x=batch["E"])
        # jax.debug.print("Ref. Forces {x}", x=batch["F"])

        return output
