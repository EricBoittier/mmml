"""Conversions between packed E3x irreps and traceless Cartesian tensors."""

from __future__ import annotations

from collections.abc import Mapping

import e3x
import jax.numpy as jnp


def num_irrep_components(max_degree: int) -> int:
    """Return the packed irrep width through ``max_degree``."""
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")
    return (max_degree + 1) ** 2


def split_irrep_blocks(
    multipoles: jnp.ndarray,
    max_degree: int = 3,
) -> dict[str, jnp.ndarray]:
    """Split ``[..., (max_degree + 1) ** 2]`` into ``2*l + 1`` blocks."""
    multipoles = jnp.asarray(multipoles)
    expected = num_irrep_components(max_degree)
    if multipoles.shape[-1] != expected:
        raise ValueError(
            f"Expected {expected} multipole components through l={max_degree}, "
            f"got {multipoles.shape[-1]}"
        )

    blocks = {}
    start = 0
    for degree in range(max_degree + 1):
        width = 2 * degree + 1
        blocks[f"l{degree}_irrep"] = multipoles[..., start : start + width]
        start += width
    return blocks


def traceless_tensors_from_irreps(
    blocks: Mapping[str, jnp.ndarray],
    max_degree: int = 3,
) -> dict[str, jnp.ndarray]:
    """Convert irrep blocks to traceless symmetric Cartesian tensors."""
    tensors = {"l0_monopole": jnp.asarray(blocks["l0_irrep"])[..., 0]}
    names = {1: "dipole", 2: "quadrupole", 3: "octupole"}
    for degree in range(1, max_degree + 1):
        name = names.get(degree, f"degree_{degree}")
        tensors[f"l{degree}_{name}_tensor"] = e3x.so3.irreps_to_tensor(
            jnp.asarray(blocks[f"l{degree}_irrep"]),
            degree=degree,
        )
    return tensors


def irrep_blocks_to_traceless(
    multipoles: jnp.ndarray,
    max_degree: int = 3,
) -> dict[str, jnp.ndarray]:
    """Return packed irrep blocks and their traceless Cartesian equivalents."""
    blocks = split_irrep_blocks(multipoles, max_degree=max_degree)
    return {**blocks, **traceless_tensors_from_irreps(blocks, max_degree=max_degree)}
