"""
JAX/Flax implementation of Ziegler-Biersack-Littmark nuclear repulsion model.

This module provides a neural network model for calculating nuclear repulsion
using the ZBL potential with smooth cutoffs.
"""

from typing import Any, Optional

import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax
from jax.nn import standardize

# Constants
BOHR_TO_ANGSTROM = 0.529177249  # Conversion factor from Bohr to Angstrom
HARTREE_TO_EV = 27.211386245988  # Conversion factor from Hartree to eV


class ZBLRepulsion(nn.Module):
    """Ziegler-Biersack-Littmark nuclear repulsion model.

    Attributes:
        cutoff: Upper cutoff distance
        cuton: Lower cutoff distance starting switch-off function
        trainable: If True, repulsion parameters are trainable
        dtype: Data type for computations
    """

    cutoff: float
    cuton: Optional[float] = None
    trainable: bool = True
    dtype: Any = jnp.float32
    debug: bool = False

    def setup(self):
        """Initialize model parameters."""
        # Default ZBL parameters
        a_coefficient = 0.8854  # Bohr
        a_exponent = 0.23
        phi_coefficients = [0.18175, 0.50986, 0.28022, 0.02817]
        phi_exponents = [3.19980, 0.94229, 0.40290, 0.20162]

        # Setup cutoffs
        self.cutoff_dist = self.cutoff

        if self.cuton is not None and self.cuton < self.cutoff:
            self.cuton_dist = jnp.array([self.cuton], dtype=self.dtype)
            self.switchoff_range = jnp.array(
                [self.cutoff - self.cuton], dtype=self.dtype
            )
            self.use_switch = True
        else:
            self.cuton_dist = 0.0
            self.switchoff_range = self.cutoff
            self.use_switch = True if self.cuton is None else False

        # Initialize parameters
        def make_param(name, value):
            if self.trainable:
                return self.param(name, lambda key: jnp.array(value, dtype=self.dtype))
            return jnp.array(value, dtype=self.dtype)

        self.a_coefficient = make_param("a_coefficient", a_coefficient)
        self.a_exponent = make_param("a_exponent", a_exponent)
        self.phi_coefficients = make_param("phi_coefficients", phi_coefficients)
        self.phi_exponents = make_param("phi_exponents", phi_exponents)

    def switch_fn(self, distances: jnp.ndarray) -> jnp.ndarray:
        """Compute smooth switch factors from 1 to 0.

        Args:
            distances: Array of interatomic distances

        Returns:
            Array of switch factors
        """
        x = (self.cutoff_dist - distances) / self.switchoff_range

        switch = jnp.where(
            distances < self.cuton_dist,
            jnp.ones_like(x),
            jnp.where(
                distances >= self.cutoff_dist,
                jnp.zeros_like(x),
                ((6.0 * x - 15.0) * x + 10.0) * x**3,
            ),
        )
        return switch

    def __call__(
        self,
        atomic_numbers: jnp.ndarray,
        distances: jnp.ndarray,
        switch_off: jnp.ndarray,
        eshift: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        atom_mask: jnp.ndarray,
        batch_mask: jnp.ndarray,
        batch_segments: jnp.ndarray,
        batch_size: int,
    ) -> jnp.ndarray:
        """Calculate ZBL nuclear repulsion energies.

        Args:
            atomic_numbers: Array of atomic numbers
            distances: Array of interatomic distances
            idx_i: Array of indices for first atoms in pairs
            idx_j: Array of indices for second atoms in pairs

        Returns:
            Array of repulsion energies per atom
        """

        # Compute atomic number dependent screening length with safe operations
        # Clip atomic numbers to prevent zero or negative values
        safe_atomic_numbers = jnp.maximum(atomic_numbers, 1e-6)
        distances = jnp.maximum(distances, 1e-6)
        eshift = jnp.maximum(eshift, 0.0)
        switch_off = jnp.maximum(switch_off, 0.0)

        # Use safe power operation
        za = jnp.exp(jnp.log(safe_atomic_numbers) * jnp.abs(self.a_exponent))
        # Compute denominator with better numerical stability
        za_sum = za[idx_i] + za[idx_j]
        za_sum = jnp.maximum(za_sum, 1e-10) * batch_mask
        # Compute screening length
        a_ij = jnp.abs(self.a_coefficient) / za_sum
        # Compute screening function phi with numerical stability
        arguments = jnp.maximum(distances, 1e-10) / jnp.maximum(a_ij, 1e-10)
        # Normalize coefficients directly instead of using softmax
        raw_coefficients = jnp.abs(self.phi_coefficients)
        coeff_sum = jnp.sum(raw_coefficients)
        coefficients = raw_coefficients / jnp.maximum(coeff_sum, 1e-10)
        # Compute phi using log-sum-exp trick for numerical stability
        log_terms = -self.phi_exponents[None, ...] * arguments[..., None]
        max_log = jnp.max(log_terms, axis=1, keepdims=True)
        exp_terms = jnp.exp(log_terms - max_log)
        # Compute phi with coefficient weighting
        phi = (
            jnp.sum(coefficients[None, ...] * exp_terms, axis=1)
            * jnp.exp(max_log)[..., 0]
        )
        # First compute Z_i * Z_j
        charge_product = safe_atomic_numbers[idx_i] * safe_atomic_numbers[idx_j]
        # Compute base repulsion with distance
        base_repulsion = charge_product / distances
        # Apply screening function and switch

        repulsion = base_repulsion * phi + eshift
        # Apply switch-off function
        repulsion *= switch_off * batch_mask

        # Sum contributions for each atom using safe operations
        erep = jax.ops.segment_sum(
            repulsion, segment_ids=idx_i, num_segments=atomic_numbers.shape[0]
        )
        # Apply atom mask and final safety checks
        erep = jnp.multiply(erep, atom_mask)
        erep = jnp.nan_to_num(erep, nan=0.0, posinf=0.0, neginf=0.0)

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

        return erep[..., None, None, None]
