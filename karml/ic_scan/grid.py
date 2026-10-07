"""Expand configured DoFs into concrete scan points."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from .config import DegreeOfFreedom, IcScanConfig


@dataclass(frozen=True)
class ScanPoint:
    """One prepared geometry request inside a named scan job."""

    scan_name: str
    point_id: str
    global_index: int
    local_index: int
    coordinates: dict[str, float]
    active_dofs: tuple[str, ...]

    def to_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "scan_name": self.scan_name,
            "point_id": self.point_id,
            "global_index": self.global_index,
            "local_index": self.local_index,
            "active_dofs": list(self.active_dofs),
        }
        for name, value in self.coordinates.items():
            info[f"coord_{name}"] = float(value)
        return info


def reference_coordinates(
    config: IcScanConfig,
    base_values: dict[str, float],
) -> dict[str, float]:
    """Merge measured structure values with optional explicit reference overrides."""

    coords = dict(base_values)
    coords.update(config.reference)
    return coords


def expand_scan_points(
    config: IcScanConfig,
    *,
    base_values: dict[str, float],
) -> list[ScanPoint]:
    """Expand all scan jobs into an ordered list of coordinate assignments.

    ``product`` (default when ``scans`` omitted with ``scan_mode=product``):
    cartesian product of the selected DoF grids.

    ``individual`` (or one-DoF scan specs): hold inactive DoFs at the reference
    geometry while sweeping one coordinate at a time.
    """

    dof_map = config.dof_map()
    ref = reference_coordinates(config, base_values)
    missing = [name for name in dof_map if name not in ref]
    if missing:
        raise ValueError(
            "could not resolve reference values for DoFs: "
            f"{missing}; provide config.reference or ensure the structure defines them"
        )

    points: list[ScanPoint] = []
    global_index = 0
    for scan in config.resolved_scans():
        active = tuple(scan.dofs)
        grids = [dof_map[name].values for name in active]
        for local_index, combo in enumerate(product(*grids)):
            coordinates = dict(ref)
            for name, value in zip(active, combo, strict=True):
                coordinates[name] = float(value)
            point_id = f"{scan.name}-{local_index:06d}"
            points.append(
                ScanPoint(
                    scan_name=scan.name,
                    point_id=point_id,
                    global_index=global_index,
                    local_index=local_index,
                    coordinates=coordinates,
                    active_dofs=active,
                )
            )
            global_index += 1
    return points


def dof_units(kind: str) -> str:
    return {"bond": "angstrom", "angle": "degree", "dihedral": "degree"}[kind]


def summarize_dofs(dofs: tuple[DegreeOfFreedom, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": dof.name,
            "kind": dof.kind,
            "atoms": list(dof.atoms),
            "n_points": len(dof.values),
            "unit": dof_units(dof.kind),
        }
        for dof in dofs
    ]
