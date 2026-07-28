"""Configuration for batched umbrella sampling (NVT + MBAR)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


SeedMode = Literal["stretch", "tile", "frames"]


@dataclass(frozen=True)
class UmbrellaConfig:
    """Inputs for :func:`mmml.umbrella.sample.run_umbrella_nvt`."""

    checkpoint: Path
    structure: Path
    output_dir: Path
    atom_i: int
    atom_j: int
    targets_A: tuple[float, ...] = ()
    xi_min: float | None = None
    xi_max: float | None = None
    n_windows: int | None = None
    k_ev_A2: float | tuple[float, ...] = 10.0
    temperature_K: float = 300.0
    timestep_fs: float = 0.5
    nsteps: int = 1000
    printfreq: int = 100
    savefreq: int | None = None
    seed: int = 42
    use_ema: bool = True
    overwrite: bool = False
    structure_index: int = 0
    seed_mode: SeedMode = "stretch"

    def __post_init__(self) -> None:
        if self.atom_i == self.atom_j or min(self.atom_i, self.atom_j) < 0:
            raise ValueError("atom_i and atom_j must be distinct non-negative indices")
        if self.temperature_K <= 0:
            raise ValueError(f"temperature_K must be > 0 (got {self.temperature_K})")
        if self.timestep_fs <= 0:
            raise ValueError(f"timestep_fs must be > 0 (got {self.timestep_fs})")
        if self.nsteps < 1:
            raise ValueError(f"nsteps must be >= 1 (got {self.nsteps})")
        if self.printfreq < 1:
            raise ValueError(f"printfreq must be >= 1 (got {self.printfreq})")
        if self.savefreq is not None and self.savefreq < 1:
            raise ValueError(f"savefreq must be >= 1 (got {self.savefreq})")
        if self.structure_index < 0:
            raise ValueError(f"structure_index must be >= 0 (got {self.structure_index})")
        if self.seed_mode not in ("stretch", "tile", "frames"):
            raise ValueError(
                f"seed_mode must be stretch|tile|frames (got {self.seed_mode!r})"
            )
        targets = self.resolve_targets()
        if len(targets) < 1:
            raise ValueError("need at least one umbrella window target")
        ks = self.resolve_force_constants()
        if len(ks) != len(targets):
            raise ValueError(
                f"k_ev_A2 length {len(ks)} must match number of windows {len(targets)}"
            )
        if any(k < 0 for k in ks):
            raise ValueError("force constants must be non-negative")

    def resolve_targets(self) -> tuple[float, ...]:
        """Return window centers ξ₀ from ``targets_A`` or a linear grid."""
        if self.targets_A:
            return tuple(float(x) for x in self.targets_A)
        if (
            self.xi_min is None
            or self.xi_max is None
            or self.n_windows is None
        ):
            return ()
        if self.n_windows < 1:
            raise ValueError(f"n_windows must be >= 1 (got {self.n_windows})")
        if self.n_windows == 1:
            return (float(self.xi_min),)
        import numpy as np

        grid = np.linspace(float(self.xi_min), float(self.xi_max), int(self.n_windows))
        return tuple(float(x) for x in grid)

    def resolve_force_constants(self) -> tuple[float, ...]:
        """Per-window force constants matching :meth:`resolve_targets`."""
        n = len(self.resolve_targets())
        if isinstance(self.k_ev_A2, (int, float)):
            return tuple(float(self.k_ev_A2) for _ in range(n))
        return tuple(float(k) for k in self.k_ev_A2)

    def effective_savefreq(self) -> int:
        return int(self.savefreq if self.savefreq is not None else self.printfreq)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UmbrellaConfig:
        raw = dict(data)
        for key in ("checkpoint", "structure", "output_dir"):
            if key in raw and raw[key] is not None:
                raw[key] = Path(raw[key])
        if "targets_A" in raw and raw["targets_A"] is not None:
            raw["targets_A"] = tuple(float(x) for x in raw["targets_A"])
        if "k_ev_A2" in raw and raw["k_ev_A2"] is not None:
            k = raw["k_ev_A2"]
            if isinstance(k, (list, tuple)):
                raw["k_ev_A2"] = tuple(float(x) for x in k)
            else:
                raw["k_ev_A2"] = float(k)
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("checkpoint", "structure", "output_dir"):
            out[key] = str(out[key])
        out["targets_A"] = list(out["targets_A"])
        if isinstance(self.k_ev_A2, tuple):
            out["k_ev_A2"] = list(self.k_ev_A2)
        return out


@dataclass(frozen=True)
class UmbrellaMbarConfig:
    """Inputs for :func:`mmml.umbrella.mbar.run_umbrella_mbar`."""

    run_dir: Path
    checkpoint: Path | None = None
    temperature_K: float | None = None
    mbar_verbose: bool = False
    ml_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.temperature_K is not None and self.temperature_K <= 0:
            raise ValueError(f"temperature_K must be > 0 (got {self.temperature_K})")
        if self.ml_batch_size < 1:
            raise ValueError(f"ml_batch_size must be >= 1 (got {self.ml_batch_size})")
