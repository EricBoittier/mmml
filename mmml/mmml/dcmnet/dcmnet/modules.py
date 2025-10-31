import functools

import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp




class MessagePassingModel(nn.Module):
    """
    E(3)-equivariant message passing model for distributed multipoles.
    
    This model uses E3x to perform equivariant message passing between atoms
    and predicts distributed multipoles and dipoles that reproduce ESP on
    molecular surfaces.
    
    Attributes
    ----------
    features : int
        Number of features per atom, by default 32
    max_degree : int
        Maximum spherical harmonic degree, by default 2
    num_iterations : int
        Number of message passing iterations, by default 3
    num_basis_functions : int
        Number of radial basis functions, by default 8
    cutoff : float
        Distance cutoff for interactions, by default 5.0
    max_atomic_number : int
        Maximum atomic number for embedding, by default 17
    n_dcm : int
        Number of distributed multipoles per atom, by default 4
    include_pseudotensors : bool
        Whether to include pseudotensors, by default False
    """
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 8
    cutoff: float = 5.0
    max_atomic_number: int = 17
    n_dcm: int = 4
    include_pseudotensors: bool = False
    
    def mono(
        self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
    ):
        """
        Forward pass to predict distributed monopoles and dipoles.
        
        Performs E(3)-equivariant message passing between atoms and outputs
        distributed monopoles and dipole positions for each atom.
        
        Parameters
        ----------
        atomic_numbers : array_like
            Atomic numbers, shape (batch_size * natoms,)
        positions : array_like
            Atomic positions in Angstrom, shape (batch_size * natoms, 3)
        dst_idx : array_like
            Destination indices for message passing
        src_idx : array_like
            Source indices for message passing
        batch_segments : array_like
            Batch segment indices
        batch_size : int
            Batch size
            
        Returns
        -------
        tuple
            (atomic_mono, atomic_dipo) where:
            - atomic_mono: Distributed monopoles, shape (batch_size * natoms, n_dcm)
            - atomic_dipo: Distributed dipole positions, shape (batch_size * natoms, n_dcm, 3)
        """
        # 1. Calculate displacement vectors.
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst  # Shape (num_pairs, 3).

        # 2. Expand displacement vectors in basis functions.
        basis = e3x.nn.basis(
            # Shape (num_pairs, 1, (max_degree+1)**2, num_basis_functions).
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
        )

        x = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1, features=self.features
        )(atomic_numbers)

        # 4. Perform iterations (message-passing + atom-wise refinement).
        for i in range(self.num_iterations):
            # Message-pass.
            if i == self.num_iterations - 1:  # Final iteration.
                y = e3x.nn.MessagePass(
                    max_degree=self.max_degree, include_pseudotensors=False
                )(x, basis, dst_idx=dst_idx, src_idx=src_idx)
            else:
                y = e3x.nn.MessagePass(include_pseudotensors=self.include_pseudotensors)(
                    x, basis, dst_idx=dst_idx, src_idx=src_idx
                )
            y = e3x.nn.add(x, y)
            # Atom-wise refinement MLP.
            y = e3x.nn.Dense(self.features)(y)
            y = e3x.nn.silu(y)
            y = e3x.nn.Dense(self.features)(y)
            # Residual connection.
            x = e3x.nn.add(x, y)

        x = e3x.nn.TensorDense(
            features=self.n_dcm,
            max_degree=1,
            include_pseudotensors=False,
        )(x)

        atomic_mono = e3x.nn.change_max_degree_or_type(
            x, max_degree=0, include_pseudotensors=False
        )
        element_bias = self.param(
            "element_bias",
            lambda rng, shape: jnp.zeros(shape),
            (self.max_atomic_number + 1),
        )
        atomic_mono = nn.Dense(
            self.n_dcm,
            use_bias=False,
        )(atomic_mono)

        atomic_mono = atomic_mono.squeeze(axis=1)
        atomic_mono = atomic_mono.squeeze(axis=1)
        atomic_mono += element_bias[atomic_numbers][:, None]

        x = e3x.nn.hard_tanh(x) * 0.175
        
        # Extract dipole components: shape (n_atoms, 3, n_dcm)
        # Then transpose to (n_atoms, n_dcm, 3) for consistency
        n_atoms = x.shape[0]
        atomic_dipo = x[:, 1, 1:4, :].transpose(0, 2, 1)
        
        # Add positions: positions shape is (n_atoms, 3)
        # Expand to (n_atoms, 1, 3) to broadcast with (n_atoms, n_dcm, 3)
        atomic_dipo += positions[:, jnp.newaxis, :]

        return atomic_mono, atomic_dipo

    @nn.compact
    def __call__(
        self,
        atomic_numbers,
        positions,
        dst_idx,
        src_idx,
        batch_segments=None,
        batch_size=None,
    ):
        """
        Main forward pass of the model.
        
        Parameters
        ----------
        atomic_numbers : array_like
            Atomic numbers
        positions : array_like
            Atomic positions
        dst_idx : array_like
            Destination indices
        src_idx : array_like
            Source indices
        batch_segments : array_like, optional
            Batch segment indices, by default None
        batch_size : int, optional
            Batch size, by default None
            
        Returns
        -------
        tuple
            (atomic_mono, atomic_dipo) - Distributed monopoles and dipoles
        """
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1

        return self.mono(
            atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
        )



class MessagePassingModelDEBUG(nn.Module):
    """
    Debug version of MessagePassingModel with additional print statements.
    
    This class is identical to MessagePassingModel but includes debug prints
    to help diagnose issues during model development.
    
    Attributes
    ----------
    features : int
        Number of features per atom, by default 32
    max_degree : int
        Maximum spherical harmonic degree, by default 2
    num_iterations : int
        Number of message passing iterations, by default 3
    num_basis_functions : int
        Number of radial basis functions, by default 8
    cutoff : float
        Distance cutoff for interactions, by default 5.0
    max_atomic_number : int
        Maximum atomic number for embedding, by default 17
    n_dcm : int
        Number of distributed multipoles per atom, by default 4
    include_pseudotensors : bool
        Whether to include pseudotensors, by default False
    """
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 8
    cutoff: float = 5.0
    max_atomic_number: int = 17
    n_dcm: int = 4
    include_pseudotensors: bool = False
    
    def mono(
        self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
    ):
        """
        Debug forward pass with print statements.
        
        Same as MessagePassingModel.mono but with debug prints to track
        tensor shapes and values during execution.
        
        Parameters
        ----------
        atomic_numbers : array_like
            Atomic numbers, shape (batch_size * natoms,)
        positions : array_like
            Atomic positions in Angstrom, shape (batch_size * natoms, 3)
        dst_idx : array_like
            Destination indices for message passing
        src_idx : array_like
            Source indices for message passing
        batch_segments : array_like
            Batch segment indices
        batch_size : int
            Batch size
            
        Returns
        -------
        tuple
            (atomic_mono, atomic_dipo) - Distributed monopoles and dipoles
        """
        # 1. Calculate displacement vectors.
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst  # Shape (num_pairs, 3).

        # 2. Expand displacement vectors in basis functions.
        basis = e3x.nn.basis(
            # Shape (num_pairs, 1, (max_degree+1)**2, num_basis_functions).
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
        )
        jax.debug.print("basis {x}",x=basis.shape)

        x = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1, features=self.features
        )(atomic_numbers)

        # 4. Perform iterations (message-passing + atom-wise refinement).
        for i in range(self.num_iterations):
            jax.debug.print("x {x}",x=x.shape)
            # Message-pass.
            if i == self.num_iterations - 1:  # Final iteration.
                y = e3x.nn.MessagePass(
                    max_degree=self.max_degree, include_pseudotensors=False
                )(x, basis, dst_idx=dst_idx, src_idx=src_idx)
            else:
                y = e3x.nn.MessagePass(include_pseudotensors=self.include_pseudotensors)(
                    x, basis, dst_idx=dst_idx, src_idx=src_idx
                )
            jax.debug.print("y {x}",x=y.shape)
            y = e3x.nn.add(x, y)
            jax.debug.print("y {x}",x=y.shape)
            # Atom-wise refinement MLP.
            y = e3x.nn.Dense(self.features)(y)
            y = e3x.nn.silu(y)
            y = e3x.nn.Dense(self.features)(y)
            jax.debug.print("y {x}",x=y.shape)
            # Residual connection.
            x = e3x.nn.add(x, y)

        non_zero = jnp.nonzero(atomic_numbers)
        jax.debug.print("x {x}",x=x[non_zero])
        jax.debug.print("x {x}",x=x[non_zero].shape)
        x = e3x.nn.TensorDense(
            features=self.n_dcm,
            max_degree=1,
            include_pseudotensors=False,
        )(x)
        jax.debug.print("x {x}",x=x.shape)
        jax.debug.print("x {x}",x=x[non_zero])
        
        atomic_mono = e3x.nn.change_max_degree_or_type(
            x, max_degree=0, include_pseudotensors=False
        )
        element_bias = self.param(
            "element_bias",
            lambda rng, shape: jnp.zeros(shape),
            (self.max_atomic_number + 1),
        )
        jax.debug.print("atomic_mono {x}",x=atomic_mono.shape)
        atomic_mono = nn.Dense(
            self.n_dcm,
            use_bias=False,
        )(atomic_mono)
        jax.debug.print("atomic_mono {x}",x=atomic_mono.shape)
        atomic_mono = atomic_mono.squeeze(axis=1)
        atomic_mono = atomic_mono.squeeze(axis=1)
        atomic_mono += element_bias[atomic_numbers][:, None]
        jax.debug.print("atomic_mono {x}",x=atomic_mono.shape)
        jax.debug.print("x {x}",x=x[non_zero])
        x = e3x.nn.hard_tanh(x)  * 10.0
        jax.debug.print("xx {x}",x=x.shape)
        atomic_dipo = x[:, 1, 1:4, :]
        jax.debug.print("atomic_dipo {x}",x=atomic_dipo[non_zero])
        atomic_dipo += positions[:, :, None]

        return atomic_mono, atomic_dipo

    @nn.compact
    def __call__(
        self,
        atomic_numbers,
        positions,
        dst_idx,
        src_idx,
        batch_segments=None,
        batch_size=None,
    ):
        """
        Debug forward pass with automatic batch handling.
        
        Parameters
        ----------
        atomic_numbers : array_like
            Atomic numbers
        positions : array_like
            Atomic positions
        dst_idx : array_like
            Destination indices
        src_idx : array_like
            Source indices
        batch_segments : array_like, optional
            Batch segment indices, by default None
        batch_size : int, optional
            Batch size, by default None
            
        Returns
        -------
        tuple
            (atomic_mono, atomic_dipo) - Distributed monopoles and dipoles
        """
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1

        return self.mono(
            atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
        )

