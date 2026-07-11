#!/usr/bin/env python3
"""Plot a 1D or 2D free-energy surface from an ASE-readable trajectory.

Coordinate syntax uses zero-based atom indices:
  distance:i,j | angle:i,j,k | dihedral:i,j,k,l | x:i | y:i | z:i
  com-distance:i,j,...;k,l,... | rg:i,j,... | info:key
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

from mmml.utils.plotting.fes import calculate_fes, evaluate_coordinates, plot_fes


def _indices(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def parse_coordinate(specification: str):
    """Convert a CLI coordinate specification into an ASE frame callable."""
    try:
        kind, arguments = specification.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"coordinate lacks a ':' separator: {specification!r}") from exc
    kind = kind.lower()
    if kind == "distance":
        indices = _indices(arguments)
        _require_count(kind, indices, 2)
        return lambda atoms: float(atoms.get_distance(*indices, mic=True))
    if kind == "angle":
        indices = _indices(arguments)
        _require_count(kind, indices, 3)
        return lambda atoms: float(atoms.get_angle(*indices, mic=True))
    if kind == "dihedral":
        indices = _indices(arguments)
        _require_count(kind, indices, 4)
        return lambda atoms: float(atoms.get_dihedral(*indices, mic=True))
    if kind in {"x", "y", "z"}:
        atom_index = int(arguments)
        axis = {"x": 0, "y": 1, "z": 2}[kind]
        return lambda atoms: float(atoms.positions[atom_index, axis])
    if kind == "com-distance":
        groups = arguments.split(";")
        if len(groups) != 2:
            raise ValueError("com-distance requires two ';'-separated index groups")
        first, second = map(_indices, groups)
        return lambda atoms: float(
            np.linalg.norm(atoms[first].get_center_of_mass() - atoms[second].get_center_of_mass())
        )
    if kind == "rg":
        selected = _indices(arguments)
        return lambda atoms: _radius_of_gyration(atoms, selected)
    if kind == "info":
        return lambda atoms: float(atoms.info[arguments])
    raise ValueError(f"unknown coordinate type {kind!r}")


def _require_count(kind: str, indices: list[int], count: int) -> None:
    if len(indices) != count:
        raise ValueError(f"{kind} requires {count} atom indices")


def _radius_of_gyration(atoms: Atoms, indices: list[int]) -> float:
    selected = atoms[indices] if indices else atoms
    positions = selected.positions
    masses = selected.get_masses()
    center = np.average(positions, axis=0, weights=masses)
    return float(np.sqrt(np.average(np.sum((positions - center) ** 2, axis=1), weights=masses)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--coordinate", "-c", action="append", required=True)
    parser.add_argument("--label", action="append")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument("--range", dest="ranges", action="append", nargs=2, type=float)
    parser.add_argument("--weights", type=Path, help="One text weight per trajectory frame")
    parser.add_argument("--energy-unit", choices=("kcal/mol", "kJ/mol", "eV"), default="kcal/mol")
    parser.add_argument("--max-free-energy", type=float)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--data-output", type=Path, help="Optional NPZ with samples and FES arrays")
    args = parser.parse_args()
    if not 1 <= len(args.coordinate) <= 2:
        parser.error("provide one or two --coordinate options")
    frames = read(args.trajectory, index=":")
    coordinates = [parse_coordinate(specification) for specification in args.coordinate]
    samples = evaluate_coordinates(frames, coordinates)
    weights = np.loadtxt(args.weights) if args.weights else None
    surface = calculate_fes(
        samples, temperature_k=args.temperature, bins=args.bins, ranges=args.ranges,
        weights=weights, energy_unit=args.energy_unit,
    )
    figure, _ = plot_fes(
        surface, labels=args.label or args.coordinate, max_free_energy=args.max_free_energy
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    if args.data_output:
        args.data_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.data_output, samples=samples, free_energy=surface.free_energy,
            probability=surface.probability, **{
                f"coordinate_{index}": values for index, values in enumerate(surface.coordinates)
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
