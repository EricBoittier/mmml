"""Configuration for MMML NEB sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping


OptimizerName = Literal["BFGS", "FIRE", "MDMin"]
InterpolateMethod = Literal["idpp", "linear"]
NebMethod = Literal["improvedtangent", "aseneb", "eb", "spline", "string"]


@dataclass(frozen=True)
class NebConfig:
    """Inputs for :func:`mmml.neb.run.run_neb`."""

    initial: Path
    final: Path
    checkpoint: Path
    output_dir: Path
    n_images: int = 11
    fmax: float = 0.05
    climb: bool = False
    interpolate: InterpolateMethod = "idpp"
    optimizer: OptimizerName = "BFGS"
    neb_method: NebMethod = "improvedtangent"
    spring_k: float = 0.1
    shared_calculator: bool = True
    max_steps: int | None = None
    plot: bool = True
    overwrite: bool = False
    calculator: str | None = None
    """Optional calculator backend (``physnet`` / ``kernnn``); auto-detect if unset."""
    pair_indices: tuple[tuple[int, int], ...] = field(
        default_factory=lambda: ((1, 2), (0, 2))
    )
    """Optional atom-index pairs (0-based) logged as distances along the path.

    Defaults match NH₃–CH₃Cl ordering ``Cl, N, C, …`` → N–C and Cl–C.
    """

    def __post_init__(self) -> None:
        if self.n_images < 3:
            raise ValueError(f"n_images must be >= 3 (got {self.n_images})")
        if self.fmax <= 0.0:
            raise ValueError(f"fmax must be > 0 (got {self.fmax})")
        if self.spring_k <= 0.0:
            raise ValueError(f"spring_k must be > 0 (got {self.spring_k})")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1 (got {self.max_steps})")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NebConfig:
        raw = dict(data)
        for key in ("initial", "final", "checkpoint", "output_dir"):
            if key in raw and raw[key] is not None:
                raw[key] = Path(raw[key])
        if "pair_indices" in raw and raw["pair_indices"] is not None:
            pairs = []
            for item in raw["pair_indices"]:
                if len(item) != 2:
                    raise ValueError(f"pair_indices entries must be length-2, got {item!r}")
                pairs.append((int(item[0]), int(item[1])))
            raw["pair_indices"] = tuple(pairs)
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("initial", "final", "checkpoint", "output_dir"):
            out[key] = str(out[key])
        out["pair_indices"] = [list(pair) for pair in self.pair_indices]
        return out
