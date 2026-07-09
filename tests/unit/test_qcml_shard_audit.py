from __future__ import annotations

import numpy as np
import orbax.checkpoint as ocp

from scripts.audit_qcml_shards import audit_cache


def test_multipole_shard_audit_detects_consistent_data(tmp_path) -> None:
    shard = {
        "R": np.array([[[0, 0, 0], [2, 0, 0]]], dtype=np.float32),
        "Z": np.array([[1, 1]], dtype=np.int32),
        "Q": np.array([0], dtype=np.float32),
        "S": np.array([1], dtype=np.float32),
        "atom_mask": np.ones((1, 2), dtype=np.float32),
        "multipoles": np.zeros((1, 16), dtype=np.float32),
    }
    ocp.PyTreeCheckpointer().save(tmp_path / "cache", shard)

    report = audit_cache(tmp_path / "cache")

    assert report["integrity"]["padding_violations"] == 0
    assert report["multipoles"]["monopole_charge_max_abs_error"] == 0
    assert report["multipoles"]["quadrupole_max_abs_trace"] == 0
    assert report["pair_distance_bohr"]["median"] == 2


def test_mbd_shard_audit_checks_force_and_positive_targets(tmp_path) -> None:
    shard = {
        "R": np.array([[[-1, 0, 0], [1, 0, 0]]], dtype=np.float32),
        "Z": np.array([[6, 6]], dtype=np.int32),
        "Q": np.array([0], dtype=np.float32),
        "S": np.array([1], dtype=np.float32),
        "atom_mask": np.ones((1, 2), dtype=np.float32),
        "E_mbd": np.array([-0.01]),
        "F_mbd": np.array([[[0.1, 0, 0], [-0.1, 0, 0]]], dtype=np.float32),
        "C6_mbd": np.ones((1, 2), dtype=np.float32),
        "alpha_mbd": np.ones((1, 2), dtype=np.float32),
    }
    ocp.PyTreeCheckpointer().save(tmp_path / "cache", shard)

    report = audit_cache(tmp_path / "cache")

    assert report["mbd"]["positive_energy_fraction"] == 0
    assert report["mbd"]["nonpositive_c6_count"] == 0
    assert report["mbd"]["nonpositive_polarizability_count"] == 0
    assert report["mbd"]["relative_net_force_residual"]["max"] == 0
    assert report["mbd"]["relative_torque_residual_bohr"]["max"] == 0
