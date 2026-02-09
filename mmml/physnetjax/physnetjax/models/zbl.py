"""
JAX/Flax implementation of Ziegler-Biersack-Littmark nuclear repulsion model.

This module provides a model for calculating nuclear repulsion using the ZBL
potential with smooth cutoffs.

Conventions (chosen to match the provided Torch implementation as closely as possible):
- Distances are in Å.
- Energies are in eV.
- Neighborlist includes both (i,j) and (j,i), so a factor 0.5 is applied to avoid
  double counting.
- Switching uses the same quintic polynomial.
- No explicit continuity shift is applied (the switch enforces V(cutoff)=0).
"""

from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.ops import segment_sum

# Constants
BOHR_TO_ANGSTROM = 0.529177249  # Å / Bohr
COULOMB_EV_ANGSTROM = 14.3996454784255  # eV·Å


class ZBLRepulsion(nn.Module):
    """
    Ziegler-Biersack-Littmark nuclear repulsion model (Flax).

    Parameters
    ----------
    cutoff : float
        Upper cutoff distance (Å)
    cuton : Optional[float]
        Lower cutoff distance starting switch-off function (Å).
        If None, uses cuton=0 and switches from 1->0 over [0, cutoff].
    trainable : bool
        If True, repulsion parameters are trainable
    dtype : Any
        Data type for computations
    debug : bool
        Whether to enable debug prints
    """

    cutoff: float
    cuton: Optional[float] = None
    trainable: bool = True
    dtype: Any = jnp.float32
    debug: bool = False

    def setup(self):
        # Default ZBL parameters
        a_coefficient = 0.8854 / BOHR_TO_ANGSTROM # Bohr
        # a_ij = (0.8854 a0) / (Zi^0.23 + Zj^0.23), with a0 in Å because distances are Å
        a_coefficient = 0.8854 * BOHR_TO_ANGSTROM  # Å
        a_exponent = 0.23
        phi_coefficients = [0.18175, 0.50986, 0.28022, 0.02817]
        phi_exponents = [3.19980, 0.94229, 0.40290, 0.20162]

        # Cutoffs
        self.cutoff_dist = jnp.asarray(self.cutoff, dtype=self.dtype)
        if self.cuton is None:
            self.cuton_dist = jnp.asarray(0.0, dtype=self.dtype)
            self.switchoff_range = self.cutoff_dist
            self.use_switch = True
        elif self.cuton < self.cutoff:
            self.cuton_dist = jnp.asarray(self.cuton, dtype=self.dtype)
            self.switchoff_range = jnp.asarray(self.cutoff - self.cuton, dtype=self.dtype)
            self.use_switch = True
        else:
            self.cuton_dist = None
            self.switchoff_range = None
            self.use_switch = False

        # Params (trainable or fixed)
        def make_param(name, value):
            if self.trainable:
                return self.param(name, lambda key: jnp.asarray(value, dtype=self.dtype))
            return jnp.asarray(value, dtype=self.dtype)

        self.a_coefficient = make_param("a_coefficient", a_coefficient)
        self.a_exponent = make_param("a_exponent", a_exponent)
        self.phi_coefficients = make_param("phi_coefficients", phi_coefficients)
        self.phi_exponents = make_param("phi_exponents", phi_exponents)

    def switch_fn(self, distances: jnp.ndarray) -> jnp.ndarray:
        """
        Quintic switch from 1 to 0 between cuton and cutoff (or 0->cutoff if cuton=None).
        """
        if not self.use_switch:
            return jnp.where(distances < self.cutoff_dist, 1.0, 0.0).astype(distances.dtype)

        x = (self.cutoff_dist - distances) / jnp.maximum(self.switchoff_range, 1e-12)
        s = ((6.0 * x - 15.0) * x + 10.0) * x**3
        out = jnp.where(
            distances < self.cuton_dist,
            jnp.ones_like(distances),
            jnp.where(distances >= self.cutoff_dist, jnp.zeros_like(distances), s),
        )
        return jnp.clip(out, 0.0, 1.0)

    def __call__(
        self,
        atomic_numbers: jnp.ndarray,
        distances: jnp.ndarray,
        switch_off: Optional[jnp.ndarray],
        eshift: Optional[jnp.ndarray],  # kept for API compatibility; ignored to match Torch behavior
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        atom_mask: jnp.ndarray,
        batch_mask: jnp.ndarray,
        batch_segments: jnp.ndarray,  # unused, kept for compatibility
        batch_size: int,  # unused, kept for compatibility
    ) -> jnp.ndarray:
        """
        Calculate ZBL nuclear repulsion energies.

        Computes nuclear repulsion using the ZBL potential with improved
        numerical stability and smooth cutoffs.

        Parameters
        ----------
        atomic_numbers : jnp.ndarray
            Array of atomic numbers
        distances : jnp.ndarray
            Array of interatomic distances
        switch_off : jnp.ndarray
            Switch-off factors
        eshift : jnp.ndarray
            Energy shift factors
        idx_i : jnp.ndarray
            Array of indices for first atoms in pairs
        idx_j : jnp.ndarray
            Array of indices for second atoms in pairs
        atom_mask : jnp.ndarray
            Mask for valid atoms
        batch_mask : jnp.ndarray
            Mask for valid batch elements
        batch_segments : jnp.ndarray
            Batch segment indices
        batch_size : int
            Number of molecules in batch

        Returns
        -------
        jnp.ndarray
            Array of repulsion energies per atom
        """

        # Compute atomic number dependent screening length with safe operations
        safe_atomic_numbers = jnp.maximum(atomic_numbers, 1e-6)
        distances = jnp.maximum(distances, 1e-6)
        eshift = jnp.maximum(eshift, 0.0)
        switch_off = jnp.maximum(switch_off, 0.0)
        Returns per-atom repulsion energies in eV with shape (n_atoms, 1, 1, 1).
        """

        # Cast / safety
        Z = jnp.asarray(atomic_numbers, dtype=self.dtype)
        r = jnp.maximum(jnp.asarray(distances, dtype=self.dtype), 1e-8)  # avoid div by 0
        pair_mask = jnp.asarray(batch_mask, dtype=self.dtype)

        # Switch
        if switch_off is None:
            sw = self.switch_fn(r)
        else:
            sw = jnp.clip(jnp.asarray(switch_off, dtype=self.dtype), 0.0, 1.0)

        # Z-dependent screening length
        za = Z ** jnp.abs(self.a_exponent)  # match Torch
        denom = za[idx_i] + za[idx_j]
        denom = jnp.maximum(denom, 1e-12)

        a_ij = jnp.abs(self.a_coefficient) / denom  # Å

        # Screening function phi(x)
        x = r / jnp.maximum(a_ij, 1e-12)  # dimensionless
        coefficients = jnp.abs(self.phi_coefficients)
        coefficients = coefficients / jnp.linalg.norm(coefficients, axis=0, keepdims=True)
        exponents = jnp.abs(self.phi_exponents)

        # phi = sum_k c_k * exp(-a_k * x)
        phi = jnp.sum(coefficients[None, :] * jnp.exp(-exponents[None, :] * x[:, None]), axis=1)

        # Pair energy (eV), includes 0.5 for bidirectional neighborlist
        Zi = Z[idx_i]
        Zj = Z[idx_j]
        Erep_pair = 0.5 * COULOMB_EV_ANGSTROM * (Zi * Zj) / r * phi * sw * pair_mask

        # Accumulate onto atoms i
        n_atoms = Z.shape[0]
        Erep_atom = segment_sum(Erep_pair, idx_i, num_segments=n_atoms)
        Erep_atom = Erep_atom * jnp.asarray(atom_mask, dtype=self.dtype)
        # Erep_atom = jnp.nan_to_num(Erep_atom, nan=0.0, posinf=0.0, neginf=0.0)

        if self.debug:  # print everything for temporary debugging
            jax.debug.print("za_sum {x} {y}", x=za_sum, y=za_sum.shape)
            jax.debug.print("erep {x} {y}", x=erep, y=erep.shape)
            jax.debug.print("dist {x} {y}", x=distances, y=distances.shape)
            jax.debug.print("switch {x} {y}", x=switch_off, y=switch_off.shape)
            jax.debug.print("phi {x} {y}", x=phi, y=phi.shape)
            jax.debug.print("rep {x} {y}", x=repulsion, y=repulsion.shape)
            jax.debug.print("a {x} {y}", x=a_ij, y=a_ij.shape)
            jax.debug.print("za {x} {y}", x=za, y=za.shape)
            jax.debug.print("dist {x} {y}", x=distances, y=distances.shape)
            jax.debug.print("idxi {x} {y}", x=idx_i, y=idx_i.shape)
            jax.debug.print("idxj {x} {y}", x=idx_j, y=idx_j.shape)
            jax.debug.print("atom {x} {y}", x=atomic_numbers, y=atomic_numbers.shape)
            jax.debug.print("rep {x} {y}", x=repulsion, y=repulsion.shape)
        return erep[..., None, None, None]  * (BOHR_TO_ANGSTROM**3) * (1/HARTREE_TO_EV)
        if self.debug:
            jax.debug.print("Erep_pair: {x} {y}", x=Erep_pair, y=Erep_pair.shape)
            jax.debug.print("Erep_atom: {x} {y}", x=Erep_atom, y=Erep_atom.shape)
            jax.debug.print("r: {x} {y}", x=r, y=r.shape)
            jax.debug.print("sw: {x} {y}", x=sw, y=sw.shape)
            jax.debug.print("phi: {x} {y}", x=phi, y=phi.shape)
            jax.debug.print("a_ij: {x} {y}", x=a_ij, y=a_ij.shape)

        return Erep_atom[..., None, None, None]
