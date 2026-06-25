"""Unit tests for hybrid_reference NPZ loaders and cutoff grid helpers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.cutoffs import (
    cutoff_grids_from_args,
    cutoff_search_result_dict,
    parse_comma_float_grid,
)
from mmml.interfaces.pycharmmInterface.hybrid_reference import (
    compute_com_distances,
    load_geometry_npz,
    load_reference_trajectory_npz,
    run_cutoff_grid_search,
)


def test_parse_comma_float_grid() -> None:
    assert parse_comma_float_grid("1.5, 2.0 ,3") == [1.5, 2.0, 3.0]


def test_cutoff_search_result_dict_canonical_and_legacy() -> None:
    row = cutoff_search_result_dict(
        ml_switch_width=2.0,
        mm_switch_on=5.0,
        mm_switch_width=1.0,
        mse_energy=0.1,
        mse_forces=0.2,
        objective=0.3,
    )
    assert row["ml_switch_width"] == 2.0
    assert row["ml_cutoff"] == 2.0
    assert row["mm_cutoff"] == 1.0
    assert row["objective"] == pytest.approx(0.3)


def test_cutoff_grids_from_args_legacy_aliases() -> None:
    args = Namespace(
        ml_cutoff_grid="1.0,2.0",
        mm_switch_on_grid="4.0",
        mm_cutoff_grid="0.5",
    )
    ml_g, mm_on_g, mm_w_g = cutoff_grids_from_args(args)
    assert ml_g == [1.0, 2.0]
    assert mm_on_g == [4.0]
    assert mm_w_g == [0.5]


def test_load_geometry_npz_handoff_keys(tmp_path: Path) -> None:
    pos = np.zeros((4, 3))
    z = np.array([6, 1, 1, 1], dtype=np.int32)
    path = tmp_path / "geom.npz"
    np.savez_compressed(path, positions=pos, atomic_numbers=z, pbc=False, metadata="{}")

    payload = load_geometry_npz(path)
    np.testing.assert_allclose(payload.handoff.positions, pos)
    assert payload.charges is None


def test_load_geometry_npz_positions_only(tmp_path: Path) -> None:
    pos = np.random.default_rng(0).random((10, 3))
    path = tmp_path / "geom.npz"
    np.savez_compressed(path, positions=pos, pbc=False, metadata="{}")

    payload = load_geometry_npz(path)
    np.testing.assert_allclose(payload.handoff.positions, pos)
    assert payload.handoff.atomic_numbers.size == 0


def test_evaluate_hybrid_mse_converts_hartree_reference() -> None:
    from mmml.interfaces.pycharmmInterface.hybrid_reference import evaluate_hybrid_mse_on_frames

    class _FakeAtoms:
        def set_positions(self, _pos) -> None:
            return None

        def get_potential_energy(self) -> float:
            return -27.211386

        def get_forces(self) -> np.ndarray:
            return np.zeros((2, 3))

    metrics = evaluate_hybrid_mse_on_frames(
        _FakeAtoms(),
        R_all=np.zeros((1, 2, 3)),
        frame_indices=np.array([0], dtype=int),
        E_all=np.array([-1.0]),
        F_all=None,
        has_E=True,
        has_F=False,
        reference_energy_unit="hartree",
    )
    assert metrics["mse_energy"] == pytest.approx(0.0, abs=1e-6)


def test_load_geometry_npz_trajectory_frame(tmp_path: Path) -> None:
    R = np.random.default_rng(1).random((3, 10, 3))
    Z = np.broadcast_to(np.array([6, 1, 1, 17, 17] * 2, dtype=np.int32), (3, 10))
    path = tmp_path / "traj.npz"
    np.savez_compressed(path, R=R, Z=Z, N=np.full(3, 10, dtype=int))

    payload = load_geometry_npz(path, frame=2)
    np.testing.assert_allclose(payload.handoff.positions, R[2])
    np.testing.assert_array_equal(payload.handoff.atomic_numbers, Z[2])


def test_load_reference_trajectory_npz_selects_frames(tmp_path: Path) -> None:
    n_atoms = 4
    n_frames = 6
    R = np.random.default_rng(0).random((n_frames, n_atoms, 3))
    E = np.linspace(-1.0, -0.5, n_frames)
    F = np.random.default_rng(1).random((n_frames, n_atoms, 3))
    z = np.array([6, 1, 1, 1], dtype=np.int32)
    path = tmp_path / "traj.npz"
    np.savez_compressed(path, R=R, E=E, F=F, Z=z, N=np.full(n_frames, n_atoms))

    ref = load_reference_trajectory_npz(
        path,
        z_fallback=z,
        n_atoms_monomer=2,
        n_monomers=2,
        max_frames=3,
    )
    assert ref.has_E
    assert ref.has_F
    assert len(ref.frame_indices) <= 3
    assert ref.com_distances.shape[0] == n_frames


def test_load_reference_trajectory_rejects_handoff_npz(tmp_path: Path) -> None:
    path = tmp_path / "handoff.npz"
    np.savez_compressed(path, positions=np.zeros((3, 3)), atomic_numbers=np.ones(3, dtype=int))
    with pytest.raises(ValueError, match="single-frame handoff"):
        load_reference_trajectory_npz(
            path,
            z_fallback=np.ones(3, dtype=int),
            n_atoms_monomer=3,
            n_monomers=1,
            max_frames=10,
        )


def test_compute_com_distances_dimer() -> None:
    R = np.array([[[0, 0, 0], [1, 0, 0], [5, 0, 0], [6, 0, 0]]], dtype=float)
    dist = compute_com_distances(R, n_atoms_monomer=2, n_monomers=2, center_frames=False)
    assert dist[0] == pytest.approx(5.0)


def test_run_cutoff_grid_search_picks_best() -> None:
    from mmml.interfaces.pycharmmInterface.hybrid_reference import ReferenceTrajectory

    reference = ReferenceTrajectory(
        path=Path("x.npz"),
        R=np.zeros((1, 2, 3)),
        Z=np.array([1, 1]),
        E=np.array([-1.0]),
        F=np.zeros((1, 2, 3)),
        frame_indices=np.array([0]),
        com_distances=np.array([1.0]),
        has_E=True,
        has_F=True,
        n_frames=1,
    )

    def fake_evaluate(**kwargs):
        obj = 10.0 if kwargs["ml_switch_width"] == 1.0 else 1.0
        return cutoff_search_result_dict(
            ml_switch_width=kwargs["ml_switch_width"],
            mm_switch_on=kwargs["mm_switch_on"],
            mm_switch_width=kwargs["mm_switch_width"],
            mse_energy=obj,
            mse_forces=0.0,
            objective=obj,
        )

    with patch(
        "mmml.interfaces.pycharmmInterface.hybrid_reference.evaluate_cutoff_triple",
        side_effect=fake_evaluate,
    ):
        results, best = run_cutoff_grid_search(
            ml_grid=[1.0, 2.0],
            mm_on_grid=[5.0],
            mm_w_grid=[1.0],
            atoms=MagicMock(),
            attach_calculator=MagicMock(),
            reference=reference,
            energy_weight=1.0,
            force_weight=0.0,
            verbose=False,
        )

    assert len(results) == 2
    assert best is not None
    assert best["ml_switch_width"] == 2.0
