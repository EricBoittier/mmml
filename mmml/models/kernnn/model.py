"""Flax KerNN model: Softplus MLP on standardized 1D kernels (+ optional dual/dihedral)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp

from mmml.models.kernnn.dihedrals import h2co_hcoh_dihedral
from mmml.models.kernnn.distances import DISTANCE_FNS, n_features_for_scheme
from mmml.models.kernnn.kernels import KERNEL_FNS, get_1d_kernels_k33


@dataclass(frozen=True)
class KerNNStats:
    """Normalization / reference statistics bundled with a checkpoint."""

    mean_e: float
    std_e: float
    min_r: Any  # (n_input,)
    mean_k: Any  # (n_input,)
    std_k: Any  # (n_input,)
    mean_dihedral: float = 0.0
    std_dihedral: float = 1.0

    def as_arrays(self, dtype=jnp.float32) -> dict[str, jnp.ndarray]:
        return {
            "mean_e": jnp.asarray(self.mean_e, dtype=dtype),
            "std_e": jnp.asarray(self.std_e, dtype=dtype),
            "min_r": jnp.asarray(self.min_r, dtype=dtype),
            "mean_k": jnp.asarray(self.mean_k, dtype=dtype),
            "std_k": jnp.asarray(self.std_k, dtype=dtype),
            "mean_dihedral": jnp.asarray(self.mean_dihedral, dtype=dtype),
            "std_dihedral": jnp.asarray(self.std_dihedral, dtype=dtype),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "KerNNStats":
        return cls(
            mean_e=float(data["mean_e"]),
            std_e=float(data["std_e"]),
            min_r=data["min_r"],
            mean_k=data["mean_k"],
            std_k=data["std_k"],
            mean_dihedral=float(data.get("mean_dihedral", 0.0)),
            std_dihedral=float(data.get("std_dihedral", 1.0)),
        )


@dataclass(frozen=True)
class KerNNConfig:
    n_input: int = 6
    n_hidden: int = 20
    n_out: int = 1
    kernel: str = "k33"
    distance_scheme: str = "abcc"
    architecture: str = "ffnet"  # "ffnet" | "dual"
    dual_dropout: float = 0.0  # match Torch Dual dropout only when > 0 (train)
    model_type: str = "kernnn"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_input": int(self.n_input),
            "n_hidden": int(self.n_hidden),
            "n_out": int(self.n_out),
            "kernel": str(self.kernel),
            "distance_scheme": str(self.distance_scheme),
            "architecture": str(self.architecture),
            "dual_dropout": float(self.dual_dropout),
            "model_type": str(self.model_type),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "KerNNConfig":
        data = dict(data or {})
        scheme = str(data.get("distance_scheme", "abcc"))
        default_n = n_features_for_scheme(scheme) if scheme in DISTANCE_FNS else 6
        return cls(
            n_input=int(data.get("n_input", default_n)),
            n_hidden=int(data.get("n_hidden", 20)),
            n_out=int(data.get("n_out", 1)),
            kernel=str(data.get("kernel", "k33")),
            distance_scheme=scheme,
            architecture=str(data.get("architecture", "ffnet")),
            dual_dropout=float(data.get("dual_dropout", 0.0)),
            model_type=str(data.get("model_type", "kernnn")),
        )


class FFNet(nn.Module):
    """Softplus feed-forward network matching the classic KerNN FFNet."""

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


class DualFFNet(nn.Module):
    """Dual-branch Softplus net: kernel features + dihedral (Torch Dual port).

    Deeper dihedral branch + optional dropout matches ``FFNet_Dual.py``.
    """

    n_input_kernel: int = 6
    n_hidden: int = 20
    n_out: int = 1
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, kernel_input, dihedral_input, *, deterministic: bool = True):
        # Kernel branch: 3× Dense+Softplus
        k = nn.Dense(self.n_hidden, name="kernel_0")(kernel_input)
        k = nn.softplus(k)
        k = nn.Dense(self.n_hidden, name="kernel_1")(k)
        k = nn.softplus(k)
        k = nn.Dense(self.n_hidden, name="kernel_2")(k)
        k = nn.softplus(k)

        # Dihedral branch (deeper)
        d = nn.Dense(self.n_hidden, name="dihedral_0")(dihedral_input)
        d = nn.softplus(d)
        d = nn.Dense(self.n_hidden, name="dihedral_1")(d)
        d = nn.softplus(d)

        x = jnp.concatenate([k, d], axis=-1)
        x = nn.Dense(self.n_hidden, name="out_0")(x)
        x = nn.softplus(x)
        if self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        x = nn.Dense(self.n_out, name="out_1")(x)
        return x


def _kernel_fn(name: str):
    try:
        return KERNEL_FNS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown kernel {name!r}; choose one of {sorted(KERNEL_FNS)}"
        ) from exc


def _distance_fn(scheme: str):
    try:
        return DISTANCE_FNS[scheme]
    except KeyError as exc:
        raise ValueError(
            f"unknown distance_scheme {scheme!r}; choose one of {sorted(DISTANCE_FNS)}"
        ) from exc


def _coerce_stats_arrays(stats: KerNNStats | Mapping[str, Any], dtype=jnp.float32) -> dict[str, jnp.ndarray]:
    """Normalize KerNNStats or a mapping into JAX arrays (JIT-safe)."""
    if isinstance(stats, KerNNStats):
        return stats.as_arrays(dtype=dtype)
    out = {
        "mean_e": jnp.asarray(stats["mean_e"], dtype=dtype),
        "std_e": jnp.asarray(stats["std_e"], dtype=dtype),
        "min_r": jnp.asarray(stats["min_r"], dtype=dtype),
        "mean_k": jnp.asarray(stats["mean_k"], dtype=dtype),
        "std_k": jnp.asarray(stats["std_k"], dtype=dtype),
        "mean_dihedral": jnp.asarray(stats.get("mean_dihedral", 0.0), dtype=dtype),
        "std_dihedral": jnp.asarray(stats.get("std_dihedral", 1.0), dtype=dtype),
    }
    return out


def descriptor_from_positions(
    positions,
    stats: KerNNStats | Mapping[str, Any],
    *,
    config: KerNNConfig | None = None,
):
    """Distances → standardized kernel features."""
    cfg = config or KerNNConfig()
    st = _coerce_stats_arrays(stats)
    # Raw pair count for ABCC helpers is always 6; feature length may be 7 for sym
    r = _distance_fn(cfg.distance_scheme)(positions, 6)
    k_fn = _kernel_fn(cfg.kernel)
    k = k_fn(r, st["min_r"], 1.0)
    return (k - st["mean_k"]) / st["std_k"]


def dihedral_feature_from_positions(
    positions,
    stats: KerNNStats | Mapping[str, Any],
):
    """Standardized H–C–O–H dihedral feature with trailing dim ``(..., 1)``."""
    st = _coerce_stats_arrays(stats)
    phi = h2co_hcoh_dihedral(positions)
    feat = (phi - st["mean_dihedral"]) / st["std_dihedral"]
    return feat[..., None]


def _build_model(cfg: KerNNConfig):
    if cfg.architecture == "dual":
        return DualFFNet(
            n_input_kernel=cfg.n_input,
            n_hidden=cfg.n_hidden,
            n_out=cfg.n_out,
            dropout_rate=cfg.dual_dropout,
        )
    if cfg.architecture == "ffnet":
        return FFNet(n_input=cfg.n_input, n_hidden=cfg.n_hidden, n_out=cfg.n_out)
    raise ValueError(
        f"unknown architecture {cfg.architecture!r}; choose 'ffnet' or 'dual'"
    )


def energy_from_params(
    params: Mapping[str, Any],
    positions,
    stats: KerNNStats | Mapping[str, Any],
    *,
    config: KerNNConfig | None = None,
    deterministic: bool = True,
):
    """Scalar or batched energy (eV) from Flax params and positions."""
    cfg = config or KerNNConfig()
    st = _coerce_stats_arrays(stats)
    model = _build_model(cfg)
    features = descriptor_from_positions(positions, st, config=cfg)

    if cfg.architecture == "dual":
        dih = dihedral_feature_from_positions(positions, st)
        if features.ndim == 1:
            raw = model.apply(
                {"params": params},
                features[None, :],
                dih[None, :],
                deterministic=deterministic,
            )[0, 0]
        else:
            raw = model.apply(
                {"params": params},
                features,
                dih,
                deterministic=deterministic,
            )[..., 0]
    else:
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
        e = energy_from_params(
            params, pos, stats, config=cfg, deterministic=True
        )
        if e.ndim == 0:
            return e
        return jnp.sum(e)

    if positions.ndim == 2:
        energy, neg_forces = jax.value_and_grad(_energy)(positions)
        return energy, -neg_forces

    if positions.ndim == 3:

        def one(pos):
            e, f = energy_and_forces(params, pos, stats, config=cfg)
            return e, f

        return jax.vmap(one)(positions)

    raise ValueError(f"positions must be (N,3) or (B,N,3); got {positions.shape}")


# Re-export commonly used symbol and helpers used by checkpoint I/O
__all__ = [
    "DualFFNet",
    "FFNet",
    "KerNNConfig",
    "KerNNStats",
    "descriptor_from_positions",
    "dihedral_feature_from_positions",
    "energy_and_forces",
    "energy_from_params",
    "get_1d_kernels_k33",
]

# Used by checkpoint.init_params / load_kernnn_model
def build_model(cfg: KerNNConfig):
    return _build_model(cfg)
