from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[2] / "scripts" / "analyze_des_geometry_coverage.py"
SPEC = importlib.util.spec_from_file_location("analyze_des_geometry_coverage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frame_separations_for_two_hydrogen_molecules():
    z = np.array([1, 1, 1, 1], dtype=np.int32)
    r = np.array([[0, 0, 0], [0.74, 0, 0], [4, 0, 0], [4.74, 0, 0]], dtype=float)

    pair, com_distance, closest_distance = MODULE.frame_separations(z, r)

    assert pair == "H2 + H2"
    assert np.isclose(com_distance, 4.0)
    assert np.isclose(closest_distance, 3.26)


def test_analyze_hdf5_reports_per_pair_quantiles(tmp_path):
    import h5py

    path = tmp_path / "dimers.h5"
    with h5py.File(path, "w") as fh:
        for i, separation in enumerate((3.0, 4.0, 5.0)):
            g = fh.create_group(f"frame_{i}")
            g["atomic_numbers"] = np.array([1, 1], dtype=np.int32)
            g["positions"] = np.array([[0, 0, 0], [separation, 0, 0]], dtype=float)

    result = MODULE.analyze(path)

    assert result["frames_analyzed"] == 3
    assert result["frames_skipped_not_two_components"] == 0
    pair = result["pairs"]["H + H"]
    assert pair["com_distance_A"]["median"] == 4.0
    assert np.isclose(pair["closest_distance_A"]["q90_span"], 1.8)
