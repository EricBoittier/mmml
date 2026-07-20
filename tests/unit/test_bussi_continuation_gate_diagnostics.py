"""Unit tests for Bussi continuation-gate force/bond diagnostic dumps."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.bussi_continuation_gate_diagnostics import (
    BUSSI_GATE_RESTART_LIVE_RMSD_EQUAL_A,
    DIAGNOSTICS_SCHEMA,
    bond_type_from_atomic_numbers,
    build_bussi_continuation_gate_diagnostics,
    build_monomer_grms_outlier_records,
    build_restart_vs_live_record,
    build_top_force_atom_records,
    build_worst_bond_records,
    dump_bussi_continuation_gate_diagnostics,
    nearest_neighbor_distance_angstrom,
    resolve_bussi_gate_diagnostics_path,
    write_bussi_continuation_gate_diagnostics,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    _bussi_subchunk_grms_blocks_continuation,
)


def test_bond_type_from_atomic_numbers():
    assert bond_type_from_atomic_numbers(8, 1) == "OH"
    assert bond_type_from_atomic_numbers(1, 8) == "OH"
    assert bond_type_from_atomic_numbers(1, 1) == "HH"
    assert bond_type_from_atomic_numbers(8, 8) == "OO"
    assert bond_type_from_atomic_numbers(6, 1) == "other"


def test_nearest_neighbor_distance_mic():
    # Atom 0 at origin; atom 1 near +L image of origin along x.
    side = 10.0
    pos = np.array([[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=np.float64)
    nj, dist = nearest_neighbor_distance_angstrom(pos, 0, box_side_A=side)
    assert nj == 1
    assert dist == pytest.approx(0.5)


def test_build_top_force_atoms_and_worst_oh_bonds():
    # Two TIP3-like waters: monomer 1 has collapsed OH.
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
            [5.0, 0.0, 0.0],
            [5.30, 0.0, 0.0],  # collapsed O–H ~0.30 Å
            [4.76, 0.93, 0.0],
        ],
        dtype=np.float64,
    )
    z = np.array([8, 1, 1, 8, 1, 1], dtype=int)
    forces = np.zeros((6, 3), dtype=np.float64)
    forces[4] = [100.0, 0.0, 0.0]  # hottest atom
    forces[3] = [40.0, 0.0, 0.0]
    mono = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    top = build_top_force_atom_records(
        forces,
        positions,
        atomic_numbers=z,
        atom_to_monomer=mono,
        top_n=2,
    )
    assert len(top) == 2
    assert top[0]["atom_index"] == 4
    assert top[0]["element"] == "H"
    assert top[0]["monomer_index"] == 1
    assert top[0]["nearest_neighbor_atom_index"] == 3
    assert top[0]["nearest_neighbor_distance_A"] == pytest.approx(0.30)

    bonds = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    worst = build_worst_bond_records(
        positions,
        bonds,
        atomic_numbers=z,
        atom_to_monomer=mono,
        top_n=3,
        bond_types=("OH", "HH"),
    )
    assert worst[0]["bond_type"] == "OH"
    assert worst[0]["atom_i"] == 3
    assert worst[0]["atom_j"] == 4
    assert worst[0]["distance_A"] == pytest.approx(0.30)
    assert worst[0]["monomer_index"] == 1


def test_monomer_grms_outliers_vs_median():
    # 3 monomers × 1 atom: one extreme outlier.
    forces = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [50.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    report = build_monomer_grms_outlier_records(
        forces,
        [1, 1, 1],
        outlier_factor=3.0,
        floor_kcalmol_A=5.0,
    )
    # Single-atom GRMS is RMS of force components: |F|/sqrt(3).
    median = 1.0 / np.sqrt(3.0)
    assert report["median_grms_kcalmol_A"] == pytest.approx(median)
    assert report["outlier_threshold_kcalmol_A"] == pytest.approx(5.0)
    assert len(report["outliers"]) == 1
    assert report["outliers"][0]["monomer_index"] == 2


def test_restart_vs_live_identical_and_differing():
    live = np.zeros((3, 3), dtype=np.float64)
    same = build_restart_vs_live_record(live, live.copy(), restart_path="heat.0013.res")
    assert same["coords_differ"] is False
    assert same["rmsd_angstrom"] <= BUSSI_GATE_RESTART_LIVE_RMSD_EQUAL_A

    restart = live.copy()
    restart[0, 0] = 1.0
    diff = build_restart_vs_live_record(live, restart, restart_path="heat.0013.res")
    assert diff["coords_differ"] is True
    assert diff["rmsd_angstrom"] == pytest.approx(np.sqrt(1.0 / 3.0))


def test_build_payload_schema_and_write(tmp_path: Path):
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        dtype=np.float64,
    )
    z = np.array([8, 1, 1], dtype=int)
    forces = np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    payload = build_bussi_continuation_gate_diagnostics(
        overlap_context="overlap heat chunk 15/16",
        global_step=7500,
        gate_grms_kcalmol_A=70.6,
        gate_limit_kcalmol_A=50.0,
        forces_kcalmol_A=forces,
        positions_A=positions,
        atomic_numbers=z,
        atoms_per_list=[3],
        bond_pairs=[(0, 1), (0, 2), (1, 2)],
        microchunk_series=[
            {
                "global_step": 7000,
                "grms_kcalmol_A": 12.0,
                "temperature_K": 280.0,
                "energy_kcalmol": -25000.0,
            },
            {
                "global_step": 7500,
                "grms_kcalmol_A": 70.6,
                "temperature_K": 310.0,
                "energy_kcalmol": -24000.0,
            },
        ],
        restart_positions_A=positions.copy(),
        restart_path="heat.0013.res",
        box_side_A=30.0,
    )
    assert payload["schema"] == DIAGNOSTICS_SCHEMA
    assert payload["gate_grms_kcalmol_A"] == pytest.approx(70.6)
    assert len(payload["microchunk_series"]) == 2
    assert payload["restart_vs_live"]["coords_differ"] is False
    assert payload["worst_bonds"][0]["bond_type"] in {"OH", "HH"}

    path = write_bussi_continuation_gate_diagnostics(
        payload, tmp_path / "cleanup" / "bussi_continuation_gate_step7500.json"
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["global_step"] == 7500
    assert loaded["top_force_atoms"][0]["atom_index"] == 0


def test_resolve_path_uses_cleanup_dir(tmp_path: Path):
    args = SimpleNamespace(output_dir=str(tmp_path), no_recovery_artifacts=False)
    path = resolve_bussi_gate_diagnostics_path(args, global_step=100)
    assert path == (tmp_path / "cleanup" / "bussi_continuation_gate_step100.json").resolve()


def test_dump_respects_no_heat_abort_force_dump(tmp_path: Path):
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        no_recovery_artifacts=False,
        no_heat_abort_force_dump=True,
    )
    ctx = SimpleNamespace(workflow_args=args, ml_Z=None)
    assert (
        dump_bussi_continuation_gate_diagnostics(
            ctx,
            overlap_context="test",
            global_step=1,
            gate_grms_kcalmol_A=80.0,
            gate_limit_kcalmol_A=50.0,
        )
        is None
    )


def test_gate_calls_dump_when_grms_exceeds_limit():
    series = [{"global_step": 100, "grms_kcalmol_A": 70.0}]
    ctx = SimpleNamespace(workflow_args=SimpleNamespace(output_dir="/tmp/x"))
    with (
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms",
            return_value=70.0,
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.bussi_continuation_gate_diagnostics."
            "dump_bussi_continuation_gate_diagnostics",
            return_value=Path("/tmp/x/cleanup/bussi_continuation_gate_step100.json"),
        ) as dump,
    ):
        blocked = _bussi_subchunk_grms_blocks_continuation(
            overlap_context="heat",
            global_step=100,
            mlpot_ctx=ctx,
            microchunk_series=series,
            restart_path="heat.0001.res",
        )
    assert blocked is True
    dump.assert_called_once()
    kwargs = dump.call_args.kwargs
    assert kwargs["global_step"] == 100
    assert kwargs["gate_grms_kcalmol_A"] == pytest.approx(70.0)
    assert kwargs["microchunk_series"] is series


def test_gate_skips_dump_when_grms_ok():
    with (
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.cli_common.charmm_grms",
            return_value=12.0,
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.bussi_continuation_gate_diagnostics."
            "dump_bussi_continuation_gate_diagnostics"
        ) as dump,
    ):
        blocked = _bussi_subchunk_grms_blocks_continuation(
            overlap_context="heat",
            global_step=100,
        )
    assert blocked is False
    dump.assert_not_called()
