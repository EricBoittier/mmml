from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jaxtyping
from e3x import so3
from e3x.config import Config
from e3x.nn.features import _extract_max_degree_and_check_shape
from e3x.nn.modules import (
    _duplication_indices_for_max_degree,
    _make_tensor_product_mask,
    initializers,
)
from flax import linen as nn
from flax.linen.dtypes import promote_dtype

InitializerFn = initializers.InitializerFn
Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Integer = jaxtyping.Integer
UInt32 = jaxtyping.UInt32
Shape = Sequence[Union[int, Any]]
Dtype = Any  # This could be a real type if support for that is added.
PRNGKey = UInt32[Array, "2"]
PrecisionLike = jax.lax.PrecisionLike

default_tensor_kernel_init = initializers.tensor_lecun_normal()


class TensorIntegration(nn.Module):
    r"""Tensor product of two equivariant representations expanded on a grid with numerical
    integration over grid points.

    Attributes:

    max_degree: Maximum degree of the output. If not given, ``max_degree`` is
      chosen as the maximum of the maximum degrees of inputs1 and inputs2.
    include_pseudotensors: If ``False``, all coupling paths that produce
      pseudotensors are omitted.
    cartesian_order: If ``True``, Cartesian order is assumed.
    dtype: The dtype of the computation.
    param_dtype: The dtype passed to parameter initializers.
    precision: Numerical precision of the computation, see `jax.lax.Precision`
      for details.
    kernel_init: Initializer function for the weight matrix.
    parametrize_coupling_paths: Trainable parameters for valid coupling paths between the degrees.
    """

    max_degree: Optional[int] = None
    include_pseudotensors: bool = True
    cartesian_order: bool = Config.cartesian_order
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: InitializerFn = default_tensor_kernel_init
    parametrize_coupling_paths: bool = False

    @nn.compact
    def __call__(
        self,
        inputs1: Union[
            Float[Array, "... K 1 (max_degree1+1)**2 num_features"],
            Float[Array, "... K 2 (max_degree1+1)**2 num_features"],
        ],
        inputs2: Union[
            Float[Array, "... K 1 (max_degree2+1)**2 num_features"],
            Float[Array, "... K 2 (max_degree2+1)**2 num_features"],
        ],
        integration_weights: Float[Array, "K"],
    ) -> Union[
        Float[Array, "... 1 (max_degree3+1)**2 num_features"],
        Float[Array, "... 2 (max_degree3+1)**2 num_features"],
    ]:
        """Computes the tensor product of inputs1 and inputs2 expanded on K grid points and integrates numerically.

        Args:
          inputs1: The first factor of the tensor product evaluated at the grid points.
          inputs2: The second factor of the tensor product evaluated at the grid points.
          integration_weights: Integration weight associated with every grid point.

        Returns:
          The tensor product of inputs1 and inputs2, where each output irrep is a
          weighted linear combination with learnable weights of all valid coupling
          paths.
        """

        # Determine max_degree of inputs and output.
        max_degree1 = _extract_max_degree_and_check_shape(inputs1.shape)
        max_degree2 = _extract_max_degree_and_check_shape(inputs2.shape)
        max_degree3 = (
            max(max_degree1, max_degree2)
            if self.max_degree is None
            else self.max_degree
        )

        # Check that max_degree3 is not larger than is sensible.
        if max_degree3 > max_degree1 + max_degree2:
            raise ValueError(
                "max_degree for the tensor product of inputs with max_degree"
                f" {max_degree1} and {max_degree2} can be at most"
                f" {max_degree1 + max_degree2}, received max_degree={max_degree3}"
            )

        # Check that axis -1 (number of features) of both inputs matches in size.
        if inputs1.shape[-1] != inputs2.shape[-1]:
            raise ValueError(
                "axis -1 of inputs1 and input2 must have the same size, "
                f"received shapes {inputs1.shape} and {inputs2.shape}"
            )

        # Check that integration grids are of equal size.
        if inputs1.shape[-4] != inputs1.shape[-4]:
            raise ValueError(
                "number of integration points must be equal for inputs1 and inputs2, "
                f"received shapes {inputs1.shape[-4]} and {inputs2.shape[-4]}"
            )

        # Extract number of features from size of axis -1.
        features = inputs1.shape[-1]

        # If both inputs contain no pseudotensors and at least one input or the
        # output has max_degree == 0, the tensor product will not produce
        # pseudotensors, in this case, the output will be returned with no
        # pseudotensor channel, regardless of whether self.include_pseudotensors is
        # True or False.
        if (inputs1.shape[-3] == inputs2.shape[-3] == 1) and (
            max_degree1 == 0 or max_degree2 == 0 or max_degree3 == 0
        ):
            include_pseudotensors = False
        else:
            include_pseudotensors = self.include_pseudotensors

        # Determine number of parity channels.
        num_parity1 = inputs1.shape[-3]
        num_parity2 = inputs2.shape[-3]
        num_parity3 = 2 if include_pseudotensors else 1

        # Initialize parameters.
        kernel_shape = (
            num_parity1,
            max_degree1 + 1,
            num_parity2,
            max_degree2 + 1,
            num_parity3,
            max_degree3 + 1,
            features,
        )

        # Trainable kernel weights if coupling paths are parametrizable.
        if self.parametrize_coupling_paths:
            kernel = self.param(
                "kernel", self.kernel_init, kernel_shape, self.param_dtype
            )
        # Otherwise constant ones.
        else:
            with jax.ensure_compile_time_eval():
                kernel = jnp.ones(kernel_shape, dtype=self.param_dtype)

        (kernel,) = promote_dtype(kernel, dtype=self.dtype)

        # If any of the two inputs or the output do not contain pseudotensors, the
        # forbidded coupling paths correspond to "mixed entries within array
        # slices". However, if all inputs and the output contain pseudotensors, the
        # forbidden coupling paths all correspond to "whole slices" of the arrays.
        # Instead of masking specific entries, it is then more efficient to slice
        # the arrays and compute the allowed paths separately, effectively cutting
        # the number of necessary computations in half.
        mixed_coupling_paths = not num_parity1 == num_parity2 == num_parity3 == 2

        # Initialize constants.
        with jax.ensure_compile_time_eval():
            # Clebsch-Gordan tensor.
            cg = so3.clebsch_gordan(
                max_degree1,
                max_degree2,
                max_degree3,
                cartesian_order=self.cartesian_order,
            )

            # Mask for zeroing out forbidden (parity violating) coupling paths.
            if mixed_coupling_paths:
                mask = _make_tensor_product_mask(kernel_shape[:-1])
            else:
                mask = 1

            # Indices for expanding shape of kernel.
            idx1 = _duplication_indices_for_max_degree(max_degree1)
            idx2 = _duplication_indices_for_max_degree(max_degree2)
            idx3 = _duplication_indices_for_max_degree(max_degree3)

            # Mask kernel (only necessary for mixed coupling paths)
            if mixed_coupling_paths:
                kernel *= mask

        # Expand shape (necessary for correct broadcasting).
        kernel = jnp.take(kernel, idx1, axis=1, indices_are_sorted=True)
        kernel = jnp.take(kernel, idx2, axis=3, indices_are_sorted=True)
        kernel = jnp.take(kernel, idx3, axis=5, indices_are_sorted=True)

        if mixed_coupling_paths:
            return jnp.einsum(
                "k, ...kplf,...kqmf,plqmrnf,lmn->...rnf",
                integration_weights,
                inputs1,
                inputs2,
                kernel,
                cg,
                precision=self.precision,
                optimize="optimal",
            )
        else:
            # Compute all allowed even/odd + even/odd -> even/odd coupling paths.
            def _couple_and_integrate_slices(
                i: int, j: int, k: int
            ) -> Float[Array, "... (max_degree3+1)**2 num_features"]:
                """Helper function for coupling slice (i, j, k)."""
                return jnp.einsum(
                    "k, ...klf,...kmf,lmnf,lmn->...nf",
                    integration_weights,
                    inputs1[..., i, :, :],
                    inputs2[..., j, :, :],
                    kernel[i, :, j, :, k, :, :],
                    cg,
                    precision=self.precision,
                    optimize="optimal",
                )

            eee = _couple_and_integrate_slices(0, 0, 0)  # even + even -> even
            ooe = _couple_and_integrate_slices(1, 1, 0)  # odd + odd -> even
            eoo = _couple_and_integrate_slices(0, 1, 1)  # even + odd -> odd
            oeo = _couple_and_integrate_slices(1, 0, 1)  # odd + even -> odd

            # Combine same parities and return stacked features.
            return jnp.stack((eee + ooe, eoo + oeo), axis=-3)
