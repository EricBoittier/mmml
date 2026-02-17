import jax
import jax.numpy as jnp
import flax.linen as nn
import functools
import e3x
# -------------------------
# Model
# -------------------------
HARTREE_TO_EV = 27.211386245988

class MessagePassingModel(nn.Module):
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 8
    cutoff: float = 5.0
    max_atomic_number: int = 55
    include_pseudotensors: bool = True
    # Explicit dipole-field coupling:  E_total = E_nn + mu·Ef  (in eV)
    # When False (default) the existing implicit coupling through features is used.
    dipole_field_coupling: bool = False
    field_scale: float = 0.001  # Ef_phys [au] = Ef_input * field_scale

    def EFD(self, atomic_numbers, positions, Ef, dst_idx_flat, src_idx_flat, batch_segments, batch_size, dst_idx=None, src_idx=None):
        """
        Expected shapes:
          atomic_numbers: (B*N,) flattened
          positions:      (B*N, 3) flattened
          Ef:             (B, 3)
          dst_idx_flat/src_idx_flat: (B*E,) pre-computed flattened indices (CUDA-graph-friendly)
          batch_segments: (B*N,) segment ids 0..B-1 repeated per atom
          batch_size:     int (static)
          dst_idx/src_idx: (E,) optional - only needed when batch_segments is None
        Returns:
          proxy_energy: scalar (sum over batch)
          energy: (B,) per-molecule energy


        """
        # Positions and atomic_numbers are already flattened in prepare_batches (like physnetjax)
        # Ensure they're properly shaped (1D for atomic_numbers, 2D for positions)
        positions_flat = positions.reshape(-1, 3)  # Ensure (B*N, 3)
        atomic_numbers_flat = atomic_numbers.reshape(-1)  # Ensure (B*N,) - 1D array
                # Basic dims: use static values (CUDA-graph-friendly)
        B = batch_size  # Static - known at compile time
        N = atomic_numbers_flat.shape[0] // B   # Static - constant number of atoms per molecule
        # Compute displacements using e3x gather operations (CUDA-graph-friendly)
        # Must use flattened indices to get (B*E, 3) displacements matching MessagePass
        positions_dst = e3x.ops.gather_dst(positions_flat, dst_idx=dst_idx_flat)
        positions_src = e3x.ops.gather_src(positions_flat, src_idx=src_idx_flat)
        displacements = positions_src - positions_dst  # (B*E, 3)

        # Build an EF tensor of shape compatible with e3x.nn.Tensor()
        # e3x format is (num_atoms, parity, (lmax+1)^2, features)
        # Start with (B, 4) -> expand to (B*N, 2, 4, features) for parity=2 (pseudotensors)
        pad_ef = jnp.zeros((B, 1), dtype=positions_flat.dtype)
        xEF = jnp.concatenate((pad_ef, Ef), axis=-1)   # (B, 4) - [1, Ef_x, Ef_y, Ef_z]
        xEF = xEF[:, None, :, None]                  # (B, 1, 4, 1)
        xEF = jnp.tile(xEF, (1, 1, 1, self.features)) # (B, 1, 4, features)
        # Expand to per-atom level: (B, 1, 4, features) -> (B, N, 1, 4, features) -> (B*N, 1, 4, features)
        # Insert dimension for N, then repeat: (B, 1, 4, features) -> (B, 1, 1, 4, features) -> (B, N, 1, 4, features)
        xEF = xEF[:, None, :, :]  # (B, 1, 1, 4, features) - add dimension
        xEF = jnp.repeat(xEF, N, axis=1)  # (B, N, 1, 4, features) - repeat N times
        xEF = xEF.reshape(B * N, 1, 4, self.features)  # (B*N, 1, 4, features)
        # Expand parity dimension from 1 to 2 to match x (since include_pseudotensors=True)
        # xEF is (B*N, 1, 4, features), need (B*N, 2, 4, features) - broadcast the parity dimension
        xEF = jnp.broadcast_to(xEF, (B * N, 2, 4, self.features))


        xEF = e3x.nn.change_max_degree_or_type(xEF, max_degree=self.max_degree,
         include_pseudotensors=self.include_pseudotensors)

        # Use pre-computed flattened indices (passed from batch dict, computed outside JIT)
        # This avoids creating traced index arrays inside the JIT function
        basis = e3x.nn.basis(
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff)
        )
        # Embed atoms (flattened) - atomic_numbers_flat is already ensured to be 1D above
        x = e3x.nn.Embed(num_embeddings=self.max_atomic_number + 1, features=self.features)(atomic_numbers_flat)

        # Message passing loop
        for i in range(self.num_iterations):
            y = e3x.nn.MessagePass(
                include_pseudotensors=self.include_pseudotensors,
                max_degree=self.max_degree if i < self.num_iterations - 1 else 0
            )(x, basis, dst_idx=dst_idx_flat, src_idx=src_idx_flat)
            x = e3x.nn.add(x, y)
            x = e3x.nn.silu(x)
            x = e3x.nn.Dense(self.features)(x)
            x = e3x.nn.silu(x)
            # Couple EF - xEF already has correct shape (B*N, 2, 4, features) matching x
            xEF = e3x.nn.Tensor()(x, xEF)
            x = e3x.nn.add(x, xEF)
            x = e3x.nn.TensorDense(max_degree=self.max_degree)(x)
            x = e3x.nn.add(x, y)

        for i in range(4):
            x = e3x.nn.Dense(self.features)(x)
            x = e3x.nn.silu(x)
        x = e3x.nn.Dense(self.features)(x)

        # Save original x before reduction for dipole prediction
        x_orig = x  # (B*N, 2, (max_degree+1)^2, features)
        
        # Reduce to scalars per atom for energy prediction
        x = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)

        # Predict atomic charges (scalar per atom)
        # Use a separate branch from the same features
        x_charge = x  # (B*N, 1, 1, features)

        atomic_charges = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x_charge)
        atomic_charges = jnp.squeeze(atomic_charges, axis=(-1, -2, -3))  # (B*N,)

        # Predict atomic dipoles (3D vector per atom)
        # Use original x_orig and change max_degree to 1
        x_dipole = e3x.nn.change_max_degree_or_type(x_orig, max_degree=1, include_pseudotensors=False)
        # run through a tensor dense layer to get the dipole in the correct shape
        x_dipole = e3x.nn.TensorDense(max_degree=1)(x_dipole)
        # x_dipole shape: (B*N, parity, 4, features) where 4 = (lmax+1)^2 = (1+1)^2
        # Index 0: l=0 (scalar), indices 1-3: l=1 (dipole, 3 components)
        # Apply Dense to reduce features dimension: (B*N, parity, 4, features) -> (B*N, parity, 4, 1)
        x_dipole = e3x.nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.zeros)(x_dipole)
        # Extract l=1 components (indices 1-3) and take real part (first parity dimension, index 0)
        # Shape: (B*N, parity, 3, 1) -> take parity=0 (real part) -> (B*N, 3, 1) -> squeeze -> (B*N, 3)
        atomic_dipoles = x_dipole[:, 0, 1:4, 0]  # (B*N, 3) - take first parity (real), l=1 components, squeeze features
        # Compute molecular dipole: μ = Σ(q_i * (r_i - COM)) + Σ(μ_i)
        # Reshape to (B, N, 3) for positions and dipoles, (B, N) for charges
        positions_batched = positions_flat.reshape(B, N, 3)  # (B, N, 3)
        charges_batched = atomic_charges.reshape(B, N)  # (B, N)
        dipoles_batched = atomic_dipoles.reshape(B, N, 3)  # (B, N, 3)

        # Center of mass (using atomic masses or uniform weighting)
        # For simplicity, use uniform weighting (geometric center)
        com = positions_batched.mean(axis=1, keepdims=True)  # (B, 1, 3)
        positions_centered = positions_batched - com  # (B, N, 3)
        # Charge contribution: Σ(q_i * (r_i - COM))
        charge_dipole = jnp.sum(charges_batched[:, :, None] * positions_centered, axis=1)  # (B, 3)
        # Atomic dipole contribution: Σ(μ_i)
        atomic_dipole_sum = jnp.sum(dipoles_batched, axis=1)  # (B, 3)
        # Total molecular dipole
        dipole = charge_dipole + atomic_dipole_sum  # (B, 3)

        # Store atomic-level properties for downstream use (e.g. AAT)
        self.sow('intermediates', 'atomic_charges', charges_batched)      # (B, N)
        self.sow('intermediates', 'atomic_dipoles', dipoles_batched)      # (B, N, 3)

        # Predict atomic energies
        element_bias = self.param(
            "element_bias",
            lambda rng, shape: jnp.zeros(shape, dtype=positions.dtype),
            (self.max_atomic_number + 1,)
        )
        atomic_energies = nn.Dense(1, use_bias=True, kernel_init=jax.nn.initializers.zeros)(x)
        atomic_energies = jnp.squeeze(atomic_energies, axis=(-1, -2, -3))  # (B*N,)
        atomic_energies = atomic_energies + element_bias[atomic_numbers_flat]
        energy = atomic_energies.reshape(B, N).sum(axis=1)  # (B,)

        # add a Coulomb term to the energy
        # Pairwise Coulomb: E_coul = 0.5 * Σ_{i≠j} q_i * q_j / r_ij
        r_ij = jnp.linalg.norm(displacements, axis=-1)  # (B*E,)
        q_src = atomic_charges[src_idx_flat]  # (B*E,)
        q_dst = atomic_charges[dst_idx_flat]  # (B*E,)
        pair_coulomb = q_src * q_dst / (r_ij + 1e-10)  # (B*E,)
        # Sum per molecule (each pair counted twice in neighbor list, so divide by 2)
        edge_batch = batch_segments[dst_idx_flat]  # (B*E,) batch index per edge
        coulomb_energy = jax.ops.segment_sum(pair_coulomb, edge_batch, num_segments=B) / 2.0  # (B,)
        energy = energy + coulomb_energy * 14.399645  # Coulomb constant in eV·Å/e²


        # Optional explicit dipole-field coupling:
        if self.dipole_field_coupling:

            coupling = jnp.sum( dipole * Ef , axis=-1)  # (B,)  mu·Ef_input
            coupling = coupling * self.field_scale * HARTREE_TO_EV
            energy = energy - coupling

        # Proxy energy for force differentiation
        return -jnp.sum(energy), energy, dipole

    @nn.compact
    def __call__(self, atomic_numbers, positions, Ef, dst_idx_flat=None, src_idx_flat=None, batch_segments=None, batch_size=None, dst_idx=None, src_idx=None):
        """Returns (energy, dipole) - use energy_and_forces() wrapper for forces."""
        if batch_segments is None:
            # atomic_numbers expected (B,N); if B absent, treat as (N,)
            if atomic_numbers.ndim == 1:
                atomic_numbers = atomic_numbers[None, :]
                positions = positions[None, :, :]
                Ef = Ef[None, :]
            B, N = atomic_numbers.shape
            batch_size = B
            # Flatten positions and atomic_numbers to match batch prep pattern
            atomic_numbers = atomic_numbers.reshape(-1)  # (B*N,)
            positions = positions.reshape(-1, 3)          # (B*N, 3)
            batch_segments = jnp.repeat(jnp.arange(B, dtype=jnp.int32), N)
            # Compute flattened indices for this single-batch case
            if dst_idx is None or src_idx is None:
                raise ValueError("dst_idx and src_idx are required when batch_segments is None")
            offsets = jnp.arange(B, dtype=jnp.int32) * N
            dst_idx_flat = (dst_idx[None, :] + offsets[:, None]).reshape(-1)
            src_idx_flat = (src_idx[None, :] + offsets[:, None]).reshape(-1)

        proxy_energy, energy, dipole = self.EFD(
            atomic_numbers, positions, Ef, dst_idx_flat, src_idx_flat, batch_segments, batch_size, dst_idx=dst_idx, src_idx=src_idx
        )
        return energy, dipole


def energy_and_forces(model_apply, params, atomic_numbers, positions, Ef, dst_idx_flat, src_idx_flat, batch_segments, batch_size, dst_idx=None, src_idx=None):
    """Compute energy, forces (negative gradient of energy w.r.t. positions), and dipole."""
    def energy_fn(pos):
        energy, dipole = model_apply(
            params,
            atomic_numbers=atomic_numbers,
            positions=pos,
            Ef=Ef,
            dst_idx_flat=dst_idx_flat,
            src_idx_flat=src_idx_flat,
            batch_segments=batch_segments,
            batch_size=batch_size,
            dst_idx=dst_idx,
            src_idx=src_idx,
        )
        return -jnp.sum(energy), (energy, dipole)
    
    (_, (energy, dipole)), forces = jax.value_and_grad(energy_fn, has_aux=True)(positions)
    return energy, forces, dipole
