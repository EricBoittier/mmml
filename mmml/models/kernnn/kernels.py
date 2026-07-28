"""JAX 1D kernel functions for KerNN descriptors.

Port of ``scripts/kernn/utils/kernels.py``. Elementwise in the last dimension.
``xi`` is the reference (usually min-energy) distance vector.
"""

from __future__ import annotations

import jax.numpy as jnp


def get_1d_kernels_k20(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (2.0 / xl - 2.0 / 3.0 * xs / xl**2)


def get_1d_kernels_k21(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (2.0 / (3.0 * xl**2) - 1.0 / 3.0 * xs / xl**3)


def get_1d_kernels_k22(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (1.0 / (3.0 * xl**3) - 1.0 / 5.0 * xs / xl**4)


def get_1d_kernels_k23(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (1.0 / (5.0 * xl**4) - 2.0 / 15.0 * xs / xl**5)


def get_1d_kernels_k24(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (2.0 / (15.0 * xl**5) - 2.0 / 21.0 * xs / xl**6)


def get_1d_kernels_k25(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (2.0 / (21.0 * xl**6) - 1.0 / 14.0 * xs / xl**7)


def get_1d_kernels_k26(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (1.0 / (14.0 * xl**7) - 1.0 / 18.0 * xs / xl**8)


def get_1d_kernels_k30(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (
        3.0 / xl - 3.0 / 2.0 * xs / xl**2 + 3.0 / 10.0 * xs**2 / xl**3
    )


def get_1d_kernels_k31(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (
        3.0 / (4.0 * xl**2)
        - 3.0 / 5.0 * xs / xl**3
        + 3.0 / 20.0 * xs**2 / xl**4
    )


def get_1d_kernels_k32(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (
        3.0 / (10.0 * xl**3)
        - 3.0 / 10.0 * xs / xl**4
        + 3.0 / 35.0 * xs**2 / xl**5
    )


def get_1d_kernels_k33(x, xi, scale=1.0):
    """1D k33 kernel used by the H2CO KerNN models."""
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (
        3.0 / (20.0 * xl**4)
        - 6.0 / 35.0 * xs / xl**5
        + 3.0 / 56.0 * xs**2 / xl**6
    )


def get_1d_kernels_k34(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (
        3.0 / (35.0 * xl**5)
        - 3.0 / 28.0 * xs / xl**6
        + 1.0 / 28.0 * xs**2 / xl**7
    )


def get_1d_kernels_k35(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (
        3.0 / (56.0 * xl**6)
        - 1.0 / 14.0 * xs / xl**7
        + 1.0 / 40.0 * xs**2 / xl**8
    )


def get_1d_kernels_k36(x, xi, scale=1.0):
    xl = jnp.maximum(x, xi)
    xs = jnp.minimum(x, xi)
    return scale * (
        1.0 / (28.0 * xl**7)
        - 1.0 / 20.0 * xs / xl**8
        + 1.0 / 55.0 * xs**2 / xl**9
    )


KERNEL_FNS = {
    "k20": get_1d_kernels_k20,
    "k21": get_1d_kernels_k21,
    "k22": get_1d_kernels_k22,
    "k23": get_1d_kernels_k23,
    "k24": get_1d_kernels_k24,
    "k25": get_1d_kernels_k25,
    "k26": get_1d_kernels_k26,
    "k30": get_1d_kernels_k30,
    "k31": get_1d_kernels_k31,
    "k32": get_1d_kernels_k32,
    "k33": get_1d_kernels_k33,
    "k34": get_1d_kernels_k34,
    "k35": get_1d_kernels_k35,
    "k36": get_1d_kernels_k36,
}
