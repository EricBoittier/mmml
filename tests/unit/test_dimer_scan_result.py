from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mmml.analysis.dimer_scans import distance_scan_geometries
from mmml.dimer_scan import (
    ORIENTATION_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    DimerScanConfig,
    Provenance,
    ScanResult,
    run_dimer_scan,
)
from mmml.dimer_scan.evaluate import evaluate_geometries


class DistanceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, *, fail_above: float | None = None):
        super().__init__()
        self.fail_above = fail_above

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        span = float(np.ptp(atoms.positions[:, 0])) if len(atoms) > 1 else 0.0
        if self.fail_above is not None and span > self.fail_above:
            raise RuntimeError("deliberate scan-point failure")
        self.results["energy"] = float(len(atoms) ** 2 + span)
        self.results["forces"] = np.full((len(atoms), 3), float(len(atoms)))


def provenance() -> Provenance:
    return Provenance(
        created_utc="2026-01-01T00:00:00+00:00",
        software={"mmml": "test"},
        platform={"system": "test"},
        git={"commit": "abc", "dirty": False},
        checkpoint=None,
    )


def config(*, distances=(2.0, 3.0)) -> DimerScanConfig:
    return DimerScanConfig(
        residues=("H", "H"),
        calculator="fake",
        distances_angstrom=distances,
        energy_definition="total",
    )


def geometries(distances=(2.0, 3.0)):
    return distance_scan_geometries(
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        distances,
        pair=("H", "H"),
    )


def test_every_requested_point_has_success_or_failure_record():
    result = evaluate_geometries(
        config(),
        geometries(),
        lambda: DistanceCalculator(fail_above=2.5),
        provenance=provenance(),
    )

    assert len(result.records) == 2
    assert len(result.frames) == 2
    assert [record.status for record in result.records] == ["success", "failed"]
    assert result.records[1].error_type == "RuntimeError"
    assert result.frames[1].info["status"] == "failed"
    assert result.has_failures


def test_interaction_energy_and_forces_use_same_monomer_reference():
    scan_config = DimerScanConfig(
        residues=("H", "H"),
        calculator="fake",
        distances_angstrom=(2.0,),
        energy_definition="interaction",
    )
    result = evaluate_geometries(
        scan_config,
        geometries((2.0,)),
        DistanceCalculator,
        provenance=provenance(),
    )

    [record] = result.records
    [frame] = result.frames
    assert record.energy_ev == pytest.approx(4.0)
    np.testing.assert_allclose(frame.get_forces(), np.ones((2, 3)))


def test_manifest_and_extxyz_round_trip(tmp_path: Path):
    result = evaluate_geometries(
        config(),
        geometries(),
        DistanceCalculator,
        provenance=provenance(),
    )

    paths = result.write(tmp_path / "scan")
    loaded = ScanResult.read(tmp_path / "scan")

    assert paths["plot"].is_file()
    assert paths["ase_trajectory"].is_file()
    assert loaded.config == result.config
    assert loaded.records == result.records
    assert len(loaded.frames) == len(result.frames)
    np.testing.assert_allclose(loaded.frames[0].get_forces(), result.frames[0].get_forces())
    assert loaded.frames[0].info["point_id"] == "point-000000"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        result.write(tmp_path / "scan")


def test_bundle_detects_modified_artifact(tmp_path: Path):
    result = evaluate_geometries(
        config(distances=(2.0,)),
        geometries((2.0,)),
        DistanceCalculator,
        provenance=provenance(),
    )
    result.write(tmp_path / "scan")
    with (tmp_path / "scan" / "data.csv").open("a") as handle:
        handle.write("tampered\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        ScanResult.read(tmp_path / "scan")


def test_schema_versions_are_public_and_serialized(tmp_path: Path):
    assert RESULT_SCHEMA_VERSION == "1.0"
    assert ORIENTATION_SCHEMA_VERSION == "1.0"
    scan_config = config(distances=(2.0,))
    assert scan_config.to_dict()["orientation_schema_version"] == "1.0"


def test_public_run_api_is_importable():
    assert callable(run_dimer_scan)
