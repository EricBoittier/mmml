"""Serializable scientific configuration for rigid one-dimensional dimer scans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Literal


RESULT_SCHEMA_VERSION = "1.0"
ORIENTATION_SCHEMA_VERSION = "1.0"

EnergyDefinition = Literal["total", "interaction"]
FailurePolicy = Literal["fail", "allow_partial"]


@dataclass(frozen=True)
class DimerScanConfig:
    """Complete scientific definition of a rigid 1D dimer scan."""

    residues: tuple[str, str]
    calculator: str
    distances_angstrom: tuple[float, ...]
    checkpoint: Path | None = None
    orientation: str = "campaign-default"
    orientation_schema_version: str = ORIENTATION_SCHEMA_VERSION
    distance_definition: str = "oriented-anchor-separation"
    energy_definition: EnergyDefinition = "interaction"
    failure_policy: FailurePolicy = "fail"
    charge: float | None = None
    spin: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        residues = tuple(str(value).upper() for value in self.residues)
        distances = tuple(float(value) for value in self.distances_angstrom)
        if len(residues) != 2:
            raise ValueError("residues must contain exactly two residue names")
        if not all(residues):
            raise ValueError("residue names must not be empty")
        if not distances:
            raise ValueError("distances_angstrom must contain at least one point")
        if any(not math.isfinite(value) or value <= 0.0 for value in distances):
            raise ValueError("all scan distances must be positive and finite")
        if len(set(distances)) != len(distances):
            raise ValueError("scan distances must be unique")
        if tuple(sorted(distances)) != distances:
            raise ValueError("scan distances must be strictly increasing")
        if self.energy_definition not in ("total", "interaction"):
            raise ValueError("energy_definition must be 'total' or 'interaction'")
        if self.failure_policy not in ("fail", "allow_partial"):
            raise ValueError("failure_policy must be 'fail' or 'allow_partial'")
        object.__setattr__(self, "residues", residues)
        object.__setattr__(self, "distances_angstrom", distances)
        object.__setattr__(self, "calculator", str(self.calculator).lower())
        if self.checkpoint is not None:
            object.__setattr__(self, "checkpoint", Path(self.checkpoint).expanduser())

    def to_dict(self, *, resolve_paths: bool = False) -> dict[str, Any]:
        """Return a JSON-safe representation including all resolved defaults."""

        data = asdict(self)
        data["residues"] = list(self.residues)
        data["distances_angstrom"] = list(self.distances_angstrom)
        if self.checkpoint is not None:
            path = self.checkpoint.resolve() if resolve_paths else self.checkpoint
            data["checkpoint"] = str(path)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DimerScanConfig:
        values = dict(data)
        values["residues"] = tuple(values["residues"])
        values["distances_angstrom"] = tuple(values["distances_angstrom"])
        if values.get("checkpoint") is not None:
            values["checkpoint"] = Path(values["checkpoint"])
        return cls(**values)
