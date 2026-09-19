"""Shared synthetic SPICE-α HDF5 writers. No remote downloads."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from mmml.data.spice_alpha import SPICE_ALPHA_CANONICAL_UNITS


def write_spice_h5(
    path: Path,
    *,
    units: dict[str, str] | None = SPICE_ALPHA_CANONICAL_UNITS,
    include_dipole: bool = True,
    include_polar: bool = True,
    include_charge: bool = True,
    charged: bool = False,
    extra_group: bool = True,
    nan_energy: bool = False,
    nan_gradient: bool = False,
    file_level_units: bool = True,
) -> Path:
    """Two groups: 9-atom ethanol-like (2 confs) and 3-atom water (1 conf)."""
    with h5py.File(path, "w") as handle:
        if units is not None and file_level_units:
            handle.attrs["units_map"] = json.dumps(units)
        ethanol = handle.create_group("CCO")
        if units is not None and not file_level_units:
            ethanol.attrs["units_map"] = json.dumps(units)
        ethanol.create_dataset(
            "atomic_numbers", data=np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], np.int32)
        )
        coords = np.zeros((2, 9, 3), np.float64)
        coords[0, :, 0] = np.arange(9) * 1.1
        coords[1, :, 0] = np.arange(9) * 1.1 + 0.05
        ethanol.create_dataset("conformations", data=coords)
        energy = np.array([-100.0, -99.5])
        if nan_energy:
            energy[1] = np.nan
        ethanol.create_dataset("dft_total_energy", data=energy)
        gradient = np.zeros((2, 9, 3), np.float64)
        gradient[0, 0, 0] = 1.5
        if nan_gradient:
            gradient[1, 0, 0] = np.nan
        ethanol.create_dataset("dft_total_gradient", data=gradient)
        if include_dipole:
            ethanol.create_dataset(
                "scf_dipole", data=np.array([[0.2, 0.0, 0.0], [0.21, 0.0, 0.0]])
            )
        if include_polar:
            ethanol.create_dataset(
                "polarizability", data=np.eye(3)[None].repeat(2, 0) * 1.2
            )
        if include_charge:
            q = np.zeros((2, 9, 1))
            if charged:
                q[1, 0, 0] = 1.0
            ethanol.create_dataset("mbis_charges", data=q)
        if extra_group:
            water = handle.create_group("O")
            water.create_dataset("atomic_numbers", data=np.array([8, 1, 1], np.int32))
            water_r = np.zeros((1, 3, 3), np.float64)
            water_r[0, 0, 0] = 0.0
            water_r[0, 1, 0] = 0.96
            water_r[0, 2, 0] = -0.24
            water.create_dataset("conformations", data=water_r)
            water.create_dataset("dft_total_energy", data=np.array([-14.0]))
            water.create_dataset("dft_total_gradient", data=np.zeros((1, 3, 3)))
            if include_dipole:
                water.create_dataset("scf_dipole", data=np.array([[0.0, 0.0, 0.4]]))
            if include_charge:
                water.create_dataset("mbis_charges", data=np.zeros((1, 3, 1)))
        skip = handle.create_group("metadata_only")
        skip.create_dataset("note", data=np.array([1]))
    return path
