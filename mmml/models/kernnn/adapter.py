"""PhysNet-shaped apply adapter so KerNN plugs into DMC / umbrella / MBAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp

from mmml.models.kernnn.model import KerNNConfig, KerNNStats, energy_and_forces, energy_from_params


@dataclass
class KerNNApplyAdapter:
    """Expose ``.apply(...)`` matching PhysNet energy/force call sites.

    Ignores Z / sparse indices; expects ABCC geometries with ``n_atoms`` atoms
    (default 4). Packed umbrella positions ``(K * N, 3)`` are reshaped using
    ``batch_size`` / ``n_atoms``.
    """

    stats: KerNNStats
    config: KerNNConfig
    n_atoms: int = 4

    def apply(
        self,
        params: Mapping[str, Any],
        *,
        positions,
        atomic_numbers=None,
        dst_idx=None,
        src_idx=None,
        batch_segments=None,
        batch_size=None,
        batch_mask=None,
        atom_mask=None,
        compute_forces: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        del atomic_numbers, dst_idx, src_idx, batch_segments, batch_mask, atom_mask, kwargs
        pos = jnp.asarray(positions)
        n = int(self.n_atoms)
        if pos.ndim == 2 and pos.shape[0] == n:
            if compute_forces:
                e, f = energy_and_forces(params, pos, self.stats, config=self.config)
                return {"energy": e, "forces": f}
            e = energy_from_params(params, pos, self.stats, config=self.config)
            return {"energy": e}

        if pos.ndim == 2:
            if batch_size is None:
                if pos.shape[0] % n != 0:
                    raise ValueError(
                        f"KerNN packed positions length {pos.shape[0]} not divisible "
                        f"by n_atoms={n}"
                    )
                k = pos.shape[0] // n
            else:
                k = int(batch_size)
            r = pos.reshape(k, n, 3)
            if compute_forces:
                e, f = energy_and_forces(params, r, self.stats, config=self.config)
                return {"energy": e, "forces": f.reshape(k * n, 3)}
            e = energy_from_params(params, r, self.stats, config=self.config)
            return {"energy": e}

        if pos.ndim == 3:
            if compute_forces:
                e, f = energy_and_forces(params, pos, self.stats, config=self.config)
                return {"energy": e, "forces": f}
            e = energy_from_params(params, pos, self.stats, config=self.config)
            return {"energy": e}

        raise ValueError(f"unsupported positions shape {pos.shape}")
