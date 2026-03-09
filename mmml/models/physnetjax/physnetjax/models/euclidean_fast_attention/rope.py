"""Euclidean Rope attention."""

import logging
from typing import Optional

import e3x
import jax.numpy as jnp
import numpy as np
from jax import ops
from jaxtyping import Array

LEBEDEV_FREQUENCY_LOOKUP = {
    50: np.pi,
    86: 2 * np.pi,
    110: 2.5 * np.pi,
    146: 3 * np.pi,
    194: 4 * np.pi,
    230: 4.5 * np.pi,
    266: 5 * np.pi,
    302: 5.5 * np.pi,
    350: 6.5 * np.pi,
    434: 7.5 * np.pi,
    590: 9 * np.pi,
    770: 11 * np.pi,
    974: 12.5 * np.pi,
    6000: 35 * np.pi,
}


def apply_rotary_position_embedding(x: Array, sin: Array, cos: Array):
    """Applies rotary position embedding."""
    if not (x.shape[-1] == sin.shape[-1] == cos.shape[-1] and x.shape[-1] % 2 == 0):
        raise ValueError(
            "x, sin, and cos must have the same (even) size in the last dimension,"
            f" received shapes {x.shape}, {sin.shape}, and {cos.shape}"
        )
    x = jnp.expand_dims(x, axis=-4)  # Add axis for integration grid.
    sin = jnp.expand_dims(sin, axis=(-2, -3))  # Add axes for parity/degree.
    cos = jnp.expand_dims(cos, axis=(-2, -3))  # Add axes for parity/degree.
    y = jnp.reshape(jnp.stack((-x[..., 1::2], x[..., ::2]), axis=-1), x.shape)
    return x * cos + y * sin


def calculate_rotary_position_embedding(x: Array, theta: Array):
    """Calculates rotary position embeddings.

    Args:
      x: RoPe input, (..., N)
      theta: RoPe frequencies, (M)

    Returns:
      RoPe embedded input, (..., N, 2*M)
    """
    angle = x[..., :, None] * theta[None, :]
    sin = jnp.sin(angle)
    cos = jnp.cos(angle)
    sin, cos = (
        jnp.repeat(x, 2, axis=-1, total_repeat_length=2 * theta.shape[-1])
        for x in (sin, cos)
    )
    return sin, cos


def apply(
    q: Array,
    k: Array,
    v: Array,
    pos: Array,
    theta: Array,
    grid_u: Array,
    grid_w: Array,
    batch_segments: Array,
    graph_mask: Array,
    include_pseudotensors_qk: Optional[bool] = None,
    include_pseudotensors_v: Optional[bool] = None,
    max_degree_qk: Optional[int] = None,
    max_degree_v: Optional[int] = None,
    do_integration: bool = True,
):
    """Calculate linear attention with Euclidean RoPE.

    Args:
      q: Query, following E3x convention, (N, 1 or 2, (max_degree_qk+1)**2, num_features_qk)
      k: Key, following E3x convention, (N, 1 or 2, (max_degree_qk+1)**2, num_features_qk)
      v: Value, following E3x convention, (N, 1 or 2, (max_degree_v+1)**2, num_features_v)
      pos: Node positions, (N, dim)
      theta: Frequencies, (num_features_qk/2)
      grid_u: Lebedev grid points, (M, dim)
      grid_w: Lebedev integration weights, (M)
      batch_segments: Batch segments, (N)
      graph_mask: Graph mask, (max_num_graphs)
      include_pseudotensors_qk: Include pseudotensors from query and key.
      include_pseudotensors_v: Include pseudotensors from value.
      max_degree_qk: Max degree to use in query and key.
      max_degree_v: Max degree to use in value. Changes the output degree.
      do_integration: Already perform the integration by summing over the grid points.

    Returns:
      Euclidean RoPe attended features.
    """
    assert (
        pos.ndim == 2
    )  # needs to be ensured, since we perform segment_sum along the 0-th axis

    num_parity_qk_in, num_degrees_qk_in = q.shape[-3], q.shape[-2]
    num_parity_v_in, num_degrees_v_in = v.shape[-3], v.shape[-2]
    num_features_v = v.shape[-1]

    max_degree_v_present = int(np.rint(np.sqrt(num_degrees_v_in) - 1).item())
    if max_degree_v is not None:
        if max_degree_v > max_degree_v_present:
            raise ValueError(
                f"`max_degree_v = {max_degree_v}` must not be larger than maximal degree present in the value "
                f"vector = {max_degree_v_present}. "
            )

    max_degree_qk_present = int(np.rint(np.sqrt(num_degrees_qk_in) - 1).item())
    if max_degree_qk is not None:
        if max_degree_qk > max_degree_qk_present:
            raise ValueError(
                f"`max_degree_qk = {max_degree_qk}` must not be larger than maximal degree present in the value "
                f"vector = {max_degree_qk_present}. "
            )

    # is max_degree_qk is not present, default to maximal degree of query and key
    if max_degree_qk is None:
        max_degree_qk = max_degree_qk_present

    # is max_degree_v is not present, default to maximal degree of value
    if max_degree_v is None:
        max_degree_v = max_degree_v_present

    # check if query, key and value have pseudotensors
    pseudotensors_qk_present = num_parity_qk_in == 2
    pseudotensors_v_present = num_parity_v_in == 2

    if not pseudotensors_qk_present and include_pseudotensors_qk:
        logging.warning(
            f"include_pseudotensors_qk={include_pseudotensors_qk} in rope.apply() but query and key have no "
            f"pseudotensors. This still gives correct results but makes computations unnecessarily expensive "
            f"due to padding with zeros."
        )

    if not pseudotensors_v_present and include_pseudotensors_v:
        logging.warning(
            f"include_pseudotensors_v={include_pseudotensors_v} in rope.apply() but value has no pseudotensors. "
            f"This still gives correct results but makes computations unnecessarily expensive due to "
            f"padding with zeros."
        )

    q = e3x.nn.change_max_degree_or_type(
        q, include_pseudotensors=include_pseudotensors_qk, max_degree=max_degree_qk
    )

    k = e3x.nn.change_max_degree_or_type(
        k, include_pseudotensors=include_pseudotensors_qk, max_degree=max_degree_qk
    )

    v = e3x.nn.change_max_degree_or_type(
        v, include_pseudotensors=include_pseudotensors_v, max_degree=max_degree_v
    )

    # They might have changed compared to the input due to the application of change_max_degree_or_type.
    num_parity_v, num_degrees_v = v.shape[-3], v.shape[-2]

    # Calculate projection of positions on directions of integration grid.
    x = jnp.einsum("nd,md->nm", pos, grid_u)

    # Calculate sin/cos for RoPE.
    sin, cos = calculate_rotary_position_embedding(x, theta)  # (N, M, num_features_qk)

    # Apply RoPE to queries and keys.
    q = apply_rotary_position_embedding(q, sin, cos)  # (N, M, P, L, num_features_qk)
    k = apply_rotary_position_embedding(k, sin, cos)  # (N, M, P, L, num_features_qk)

    # Flatten parity, degree, and feature channels into one axis.
    q = jnp.reshape(
        q, (*q.shape[:-3], -1)
    )  # (N, M, Dqk) with Dqk=num_parity_qk*num_degrees_qk*num_features_qk
    k = jnp.reshape(
        k, (*k.shape[:-3], -1)
    )  # (N, M, Dqk) with Dqk=num_parity_qk*num_degrees_qk*num_features_qk
    v = jnp.reshape(
        v, (*v.shape[:-3], -1)
    )  # (N, Dv) with Dv=num_parity_v*num_degrees_v*num_features_v

    # Scale q for keeping variance of dot product in check.
    q /= jnp.sqrt(q.shape[-1])

    if len(graph_mask) > 1:

        # Compute outer product of keys and values. Batch axis must be kept explicitly.
        kv = jnp.einsum("nmk,nv->nmkv", k, v)  # (N, M, Dqk, Dv)

        # Compute structure wise sum over nodes, and broadcast.
        kv = ops.segment_sum(kv, batch_segments, num_segments=len(graph_mask))[
            batch_segments
        ]  # (N, M, Dqk, Dv)

        if do_integration:
            # Calculate the result of the linear scaling attention operation and perform the numerical integration.
            y = jnp.einsum("nmd,nmdv,m->nv", q, kv, grid_w)  # (N, M, Dv)
        # Calculate the result of the linear scaling attention operation and keep the evaluations per grid point.
        else:
            y = jnp.einsum("nmd,nmdv->nmv", q, kv)  # (N, M, Dv)

    else:
        # Compute outer product of keys and values and sum over all atoms.
        kv = jnp.einsum("nmk,nv->mkv", k, v)  # (M, Dqk, Dv)

        # Calculate the result of the linear scaling attention operation and perform the numerical integration.
        if do_integration:
            y = jnp.einsum("nmd,mdv,m->nv", q, kv, grid_w)  # (N, Dv)
        # Calculate the result of the linear scaling attention operation and keep the evaluations per grid point.
        else:
            y = jnp.einsum("nmd,mdv->nmv", q, kv)  # (N, M, Dv)

    num_parity_v_out = 2 if include_pseudotensors_v else 1
    # Bring back into E3x convention.
    y = jnp.reshape(y, (*y.shape[:-1], num_parity_v, num_degrees_v, num_features_v))
    # (N, P, L, F)    if do_integration = True
    # (N, M, P, L, F) if do_integration = False

    return y

