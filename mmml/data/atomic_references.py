"""Utilities for loading atomic reference energies from JSON data."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
from ase.data import atomic_numbers

DEFAULT_REFERENCE_LEVEL = "wb97m-d3(bj)/def2-tzvp"
DEFAULT_CHARGE_STATE = 0
DEFAULT_UNIT = "hartree"

_DATA_PATH = Path(__file__).resolve().with_name("atomic_reference_energies.json")

_UNIT_FACTORS = {
    "hartree": 1.0,
    "ev": 27.211386245988,
    "kcal/mol": 627.509474,
    "kj/mol": 2625.499638,
}


@lru_cache(maxsize=None)
def _load_reference_data(data_path: Optional[Path | str] = None) -> Mapping[str, Dict[str, float]]:
    """Load the atomic reference energy table from JSON."""

    path = Path(data_path) if data_path is not None else _DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Atomic reference energy table not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Atomic reference energy table must be a mapping")

    return data


def list_reference_levels(data_path: Optional[Path | str] = None) -> Iterable[str]:
    """Return the available reference levels contained in the JSON table."""

    return tuple(_load_reference_data(data_path).keys())


def _normalise_unit(unit: str) -> str:
    normalised = unit.lower()
    if normalised not in _UNIT_FACTORS:
        raise ValueError(
            f"Unknown energy unit '{unit}'. Expected one of {tuple(_UNIT_FACTORS)}"
        )
    return normalised


def _convert_value(energy_hartree: float, to_unit: str) -> float:
    unit = _normalise_unit(to_unit)
    factor = _UNIT_FACTORS[unit]
    return float(energy_hartree * factor)


def get_atomic_reference_dict(
    *,
    level: str = DEFAULT_REFERENCE_LEVEL,
    charge_state: int = DEFAULT_CHARGE_STATE,
    unit: str = DEFAULT_UNIT,
    fallback_to_neutral: bool = True,
    data_path: Optional[Path | str] = None,
) -> Dict[int, float]:
    """Return a mapping from atomic number to reference energy.

    Parameters
    ----------
    level
        Level of theory / basis entry inside the JSON table.
    charge_state
        Charge state to select (e.g. ``0`` for neutral atoms).
    unit
        Desired output energy unit. Supported units are ``hartree``, ``eV``,
        ``kcal/mol`` and ``kJ/mol``.
    fallback_to_neutral
        If ``True`` and the requested charge state is missing for an element,
        fall back to the neutral value when available.
    data_path
        Optional path to an alternative JSON table.
    """

    table = _load_reference_data(data_path)

    if level not in table:
        raise ValueError(
            f"Unknown atomic reference level '{level}'. Available levels: {tuple(table.keys())}"
        )

    selected: Dict[int, float] = {}
    by_element: Dict[int, Dict[int, float]] = {}

    for entry, energy in table[level].items():
        try:
            symbol, charge_text = entry.split(":")
        except ValueError as exc:
            raise ValueError(f"Invalid entry '{entry}' in atomic reference table") from exc

        if symbol not in atomic_numbers:
            raise ValueError(f"Unknown chemical symbol '{symbol}' in atomic reference table")

        charge = int(charge_text)
        atomic_number = atomic_numbers[symbol]

        by_element.setdefault(atomic_number, {})[charge] = float(energy)
        if charge == charge_state:
            selected[atomic_number] = float(energy)

    if fallback_to_neutral and charge_state != 0:
        for atomic_number, charges in by_element.items():
            if atomic_number not in selected and 0 in charges:
                selected[atomic_number] = charges[0]

    if not selected:
        raise ValueError(
            f"No atomic reference energies found for charge {charge_state} at level '{level}'"
        )

    unit_normalised = _normalise_unit(unit)

    return {
        atomic_number: _convert_value(energy, unit_normalised)
        for atomic_number, energy in selected.items()
    }


def get_atomic_reference_array(
    *,
    level: str = DEFAULT_REFERENCE_LEVEL,
    charge_state: int = DEFAULT_CHARGE_STATE,
    unit: str = DEFAULT_UNIT,
    size: Optional[int] = None,
    fallback_to_neutral: bool = True,
    data_path: Optional[Path | str] = None,
) -> np.ndarray:
    """Return reference energies as an array indexed by atomic number."""

    reference_dict = get_atomic_reference_dict(
        level=level,
        charge_state=charge_state,
        unit=unit,
        fallback_to_neutral=fallback_to_neutral,
        data_path=data_path,
    )

    max_z = max(reference_dict.keys(), default=0)
    array_length = max(size or 0, max_z + 1, 119)

    reference_array = np.zeros(array_length, dtype=float)
    for atomic_number, energy in reference_dict.items():
        reference_array[atomic_number] = energy

    return reference_array


