"""Per-CGenFF-type LJ scale parameters -> per-atom CHARMM Rmin/2 and epsilon.

theta = {"log_eps": (T,), "log_sig": (T,)} over the fitted type names. A type's
epsilon is scaled by exp(log_eps) and its Rmin/2 by exp(log_sig); the pair
combining rules (eps_ij = sqrt(eps_i eps_j), Rmin_ij = Rmin_i/2 + Rmin_j/2)
then scale pairs geometrically / arithmetically. Bounds match
:mod:`mmml.models.mm_lj_scales`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from mmml.models.mm_lj_scales import (
    MM_LJ_EPSILON_SCALE_BOUNDS,
    MM_LJ_SIGMA_SCALE_BOUNDS,
)

Theta = dict[str, jnp.ndarray]


@dataclass(frozen=True)
class LjTypeMap:
    """Which fitted type each atom belongs to (-1 = not fitted, scale 1)."""

    type_names: tuple[str, ...]
    atom_type: np.ndarray  # (n_atoms,) int, index into type_names or -1

    @classmethod
    def from_atc(
        cls,
        at_codes: Sequence[int],
        atc_names: Sequence[str],
        fit_types: Sequence[str] | None = None,
    ) -> LjTypeMap:
        names = [str(atc_names[int(c)]) for c in at_codes]
        types = tuple(sorted(set(names))) if fit_types is None else tuple(fit_types)
        index = {t: k for k, t in enumerate(types)}
        return cls(types, np.asarray([index.get(n, -1) for n in names], dtype=np.int32))


def init_theta(type_map: LjTypeMap) -> Theta:
    n = len(type_map.type_names)
    return {"log_eps": jnp.zeros(n), "log_sig": jnp.zeros(n)}


def project_theta(theta: Theta) -> Theta:
    """Clip log-scales into the MM LJ scale bounds."""
    lo_e, hi_e = (float(np.log(b)) for b in MM_LJ_EPSILON_SCALE_BOUNDS)
    lo_s, hi_s = (float(np.log(b)) for b in MM_LJ_SIGMA_SCALE_BOUNDS)
    return {
        **theta,
        "log_eps": jnp.clip(theta["log_eps"], lo_e, hi_e),
        "log_sig": jnp.clip(theta["log_sig"], lo_s, hi_s),
    }


def per_atom_lj(
    theta: Theta,
    type_map: LjTypeMap,
    base_rmins: jnp.ndarray,
    base_epsilons: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scaled per-atom (Rmin/2, epsilon) for ``update_mm_pairs.energy_with_lj``."""
    idx = jnp.asarray(type_map.atom_type)
    fitted = idx >= 0
    safe = jnp.where(fitted, idx, 0)
    log_eps = jnp.where(fitted, theta["log_eps"][safe], 0.0)
    log_sig = jnp.where(fitted, theta["log_sig"][safe], 0.0)
    return base_rmins * jnp.exp(log_sig), base_epsilons * jnp.exp(log_eps)
