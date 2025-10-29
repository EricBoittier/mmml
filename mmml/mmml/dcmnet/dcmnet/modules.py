import functools

import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp

NATOMS = 60


class MessagePassingModel(nn.Module):
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
        
        atomic_dipo = x[:, 1, 1:4, :]
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
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1

        return self.mono(
            atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
        )



class MessagePassingModelDEBUG(nn.Module):
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
        x = e3x.nn.hard_tanh(x)  * 0.3
        jax.debug.print("xx {x}",x=x.shape)
        atomic_dipo = x[:, 1, 1:4, :]
        jax.debug.print("atomic_dipo {x}",x=atomic_dipo[non_zero])
        # atomic_dipo += positions[:, :, None]

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
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1

        return self.mono(
            atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size
        )

