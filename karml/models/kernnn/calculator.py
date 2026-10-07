"""ASE calculator for KerNN (JAX)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("KerNNCalculator requires ASE.") from exc

from mmml.models.kernnn.checkpoint import load_checkpoint
from mmml.models.kernnn.model import KerNNConfig, KerNNStats, energy_and_forces


class KerNNCalculator(Calculator):
    """ASE calculator wrapping a KerNN JSON checkpoint."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        label: str = "kernnn",
        **kwargs: Any,
    ):
        super().__init__(label=label, **kwargs)
        params, config, stats, metadata = load_checkpoint(checkpoint)
        self.checkpoint = Path(checkpoint).expanduser()
        self.params = params
        self.config = config
        self.stats = stats
        self.metadata = metadata

    @classmethod
    def from_components(
        cls,
        params: dict[str, Any],
        stats: KerNNStats | dict[str, Any],
        config: KerNNConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "KerNNCalculator":
        """Build a calculator from in-memory params (no file required)."""
        obj = cls.__new__(cls)
        Calculator.__init__(obj, label=kwargs.pop("label", "kernnn"), **kwargs)
        obj.checkpoint = None
        obj.params = params
        obj.config = (
            config
            if isinstance(config, KerNNConfig)
            else KerNNConfig.from_mapping(config)
        )
        obj.stats = (
            stats if isinstance(stats, KerNNStats) else KerNNStats.from_mapping(stats)
        )
        obj.metadata = {}
        return obj

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        properties = properties or ["energy", "forces"]
        Calculator.calculate(self, atoms, properties, system_changes)
        positions = jnp.asarray(self.atoms.get_positions(), dtype=jnp.float32)
        n_atoms = int(self.config.n_atoms)
        if positions.shape[0] != n_atoms:
            raise ValueError(
                f"KerNN calculator expects {n_atoms} atoms "
                f"(scheme={self.config.distance_scheme}); got {positions.shape[0]}"
            )
        energy, forces = energy_and_forces(
            self.params,
            positions,
            self.stats,
            config=self.config,
        )
        self.results["energy"] = float(np.asarray(energy))
        self.results["forces"] = np.asarray(forces, dtype=np.float64)
