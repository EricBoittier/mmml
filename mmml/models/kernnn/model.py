"""Flax KerNN model: Softplus MLP on standardized 1D kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp

from mmml.models.kernnn.distances import get_bond_length_abcc
from mmml.models.kernnn.kernels import KERNEL_FNS, get_1d_kernels_k33


@dataclass(frozen=True)
class KerNNStats:
    """Normalization / reference statistics bundled with a checkpoint."""

    mean_e: float
    std_e: float
    min_r: Any  # (n_input,)
    mean_k: Any  # (n_input,)
    std_k: Any  # (n_input,)

    def as_arrays(self, dtype=jnp.float32) -> dict[str, jnp.ndarray]:
        return {
            "mean_e": jnp.asarray(self.mean_e, dtype=dtype),
            "std_e": jnp.asarray(self.std_e, dtype=dtype),
            "min_r": jnp.asarray(self.min_r, dtype=dtype),
            "mean_k": jnp.asarray(self.mean_k, dtype=dtype),
            "std_k": jnp.asarray(self.std_k, dtype=dtype),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "KerNNStats":
        return cls(
            mean_e=float(data["mean_e"]),
            std_e=float(data["std_e"]),
            min_r=data["min_r"],
            mean_k=data["mean_k"],
            std_k=data["std_k"],
        )


@dataclass(frozen=True)
class KerNNConfig:
    n_input: int = 6
    n_hidden: int = 20
    n_out: int = 1
    kernel: str = "k33"
    distance_scheme: str = "abcc"
    model_type: str = "kernnn"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_input": int(self.n_input),
            "n_hidden": int(self.n_hidden),
            "n_out": int(self.n_out),
            "kernel": str(self.kernel),
            "distance_scheme": str(self.distance_scheme),
            "model_type": str(self.model_type),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "KerNNConfig":
        data = dict(data or {})
        return cls(
            n_input=int(data.get("n_input", 6)),
            n_hidden=int(data.get("n_hidden", 20)),
            n_out=int(data.get("n_out", 1)),
            kernel=str(data.get("kernel", "k33")),
            distance_scheme=str(data.get("distance_scheme", "abcc")),
            model_type=str(data.get("model_type", "kernnn")),
        )


class FFNet(nn.Module):
    """Softplus feed-forward network matching ``scripts/kernn`` FFNet."""

    n_input: int = 6
    n_hidden: int = 20
    n_out: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_hidden, name="dense_0")(x)
        x = nn.softplus(x)
        x = nn.Dense(self.n_hidden, name="dense_1")(x)
        x = nn.softplus(x)
        x = nn.Dense(self.n_hidden, name="dense_2")(x)
        x = nn.softplus(x)
        x = nn.Dense(self.n_out, name="dense_3")(x)
        return x


def _kernel_fn(name: str):
    try:
        return KERNEL_FNS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown kernel {name!r}; choose one of {sorted(KERNEL_FNS)}"
        ) from exc


def descriptor_from_positions(
    positions,
    stats: KerNNStats | Mapping[str, Any],
    *,
    config: KerNNConfig | None = None,
):
    """ABCC distances → standardized k33 (or configured) kernel features."""
    cfg = config or KerNNConfig()
    if cfg.distance_scheme != "abcc":
        raise ValueError(
            f"unsupported distance_scheme {cfg.distance_scheme!r} (v1 supports 'abcc')"
        )
    st = stats.as_arrays() if isinstance(stats, KerNNStats) else KerNNStats.from_mapping(stats).as_arrays()
    r = get_bond_length_abcc(positions, cfg.n_input)
    k_fn = _kernel_fn(cfg.kernel)
    k = k_fn(r, st["min_r"], 1.0)
    return (k - st["mean_k"]) / st["std_k"]


def energy_from_params(
    params: Mapping[str, Any],
    positions,
    stats: KerNNStats | Mapping[str, Any],
    *,
    config: KerNNConfig | None = None,
):
    """Scalar or batched energy (eV) from Flax params and positions."""
    cfg = config or KerNNConfig()
    st = stats.as_arrays() if isinstance(stats, KerNNStats) else KerNNStats.from_mapping(stats).as_arrays()
    model = FFNet(n_input=cfg.n_input, n_hidden=cfg.n_hidden, n_out=cfg.n_out)
    features = descriptor_from_positions(positions, st, config=cfg)
    # Squeeze batch dim handling: model expects (..., n_input)
    if features.ndim == 1:
        raw = model.apply({"params": params}, features[None, :])[0, 0]
    else:
        raw = model.apply({"params": params}, features)[..., 0]
    return raw * st["std_e"] + st["mean_e"]


def energy_and_forces(
    params: Mapping[str, Any],
    positions,
    stats: KerNNStats | Mapping[str, Any],
    *,
    config: KerNNConfig | None = None,
):
    """Energy (eV) and forces (eV/Å) via ``jax.value_and_grad``."""
    cfg = config or KerNNConfig()

    def _energy(pos):
        e = energy_from_params(params, pos, stats, config=cfg)
        if e.ndim == 0:
            return e
        return jnp.sum(e)

    if positions.ndim == 2:
        energy, neg_forces = jax.value_and_grad(_energy)(positions)
        return energy, -neg_forces

    if positions.ndim == 3:
        # Per-structure energies and forces
        def one(pos):
            e, f = energy_and_forces(params, pos, stats, config=cfg)
            return e, f

        return jax.vmap(one)(positions)

    raise ValueError(f"positions must be (N,3) or (B,N,3); got {positions.shape}")


# Re-export commonly used symbol
__all__ = [
    "FFNet",
    "KerNNConfig",
    "KerNNStats",
    "descriptor_from_positions",
    "energy_and_forces",
    "energy_from_params",
    "get_1d_kernels_k33",
]
