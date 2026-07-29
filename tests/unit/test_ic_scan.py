"""Unit tests for the internal-coordinate scan package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import write

from mmml.cli.__main__ import main as mmml_main
from mmml.cli.misc.ic_scan import build_parser, main as ic_scan_main
from mmml.cli.registry import command_by_name
from mmml.ic_scan import (
    DegreeOfFreedom,
    IcScanConfig,
    ScanSpec,
    build_grid,
    expand_scan_points,
    prepare_geometries,
    run_ic_scan,
)


def _butane_like() -> Atoms:
    """Four-atom chain suitable for bond/angle/dihedral tests."""

    return Atoms(
        symbols=["C", "C", "C", "C"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.4, 0.0],
            [3.5, 1.4, 0.0],
        ],
    )


@pytest.fixture
def structure_xyz(tmp_path: Path) -> Path:
    path = tmp_path / "mol.xyz"
    write(path, _butane_like())
    return path


def test_build_grid_linspace_inclusive():
    grid = build_grid(start=-180.0, stop=180.0, n_points=5, kind="dihedral")
    assert grid[0] == pytest.approx(-180.0)
    assert grid[-1] == pytest.approx(180.0)
    assert len(grid) == 5


def test_dof_from_dict_accepts_n_points():
    dof = DegreeOfFreedom.from_dict(
        {
            "name": "phi",
            "kind": "dihedral",
            "atoms": [0, 1, 2, 3],
            "start": -90,
            "stop": 90,
            "n_points": 3,
        }
    )
    assert dof.values == pytest.approx((-90.0, 0.0, 90.0))


def test_product_vs_individual_point_counts(structure_xyz: Path):
    dofs = (
        DegreeOfFreedom(
            name="r",
            kind="bond",
            atoms=(0, 1),
            values=(1.4, 1.5, 1.6),
        ),
        DegreeOfFreedom(
            name="phi",
            kind="dihedral",
            atoms=(0, 1, 2, 3),
            values=(-60.0, 60.0),
        ),
    )
    product = IcScanConfig(
        structure=structure_xyz,
        dofs=dofs,
        scan_mode="product",
        evaluate="none",
    )
    individual = IcScanConfig(
        structure=structure_xyz,
        dofs=dofs,
        scan_mode="individual",
        evaluate="none",
    )
    assert len(expand_scan_points(product, base_values={"r": 1.5, "phi": 0.0})) == 6
    assert len(expand_scan_points(individual, base_values={"r": 1.5, "phi": 0.0})) == 5


def test_explicit_scans_subset(structure_xyz: Path):
    config = IcScanConfig(
        structure=structure_xyz,
        dofs=(
            DegreeOfFreedom("r", "bond", (0, 1), (1.4, 1.6)),
            DegreeOfFreedom("a", "angle", (0, 1, 2), (100.0, 110.0)),
            DegreeOfFreedom("phi", "dihedral", (0, 1, 2, 3), (-60.0, 60.0)),
        ),
        scans=(
            ScanSpec("bond_only", ("r",)),
            ScanSpec("ramachandran", ("a", "phi")),
        ),
        evaluate="none",
    )
    points = expand_scan_points(
        config, base_values={"r": 1.5, "a": 105.0, "phi": 0.0}
    )
    assert len(points) == 2 + 4
    assert {p.scan_name for p in points} == {"bond_only", "ramachandran"}


def test_prepare_geometries_sets_requested_dihedral(structure_xyz: Path):
    config = IcScanConfig(
        structure=structure_xyz,
        dofs=(
            DegreeOfFreedom(
                name="phi",
                kind="dihedral",
                atoms=(0, 1, 2, 3),
                values=(0.0, 90.0),
            ),
        ),
        evaluate="none",
    )
    _, prepared = prepare_geometries(config)
    assert len(prepared) == 2
    for point, atoms in prepared:
        actual = float(atoms.get_dihedral(0, 1, 2, 3))
        # ASE dihedral convention can wrap; compare absolute circular distance.
        target = point.coordinates["phi"]
        delta = (actual - target + 180.0) % 360.0 - 180.0
        assert abs(delta) < 1.0


def test_dihedral_mask_must_include_a4(structure_xyz: Path):
    config = IcScanConfig(
        structure=structure_xyz,
        dofs=(
            DegreeOfFreedom(
                name="phi",
                kind="dihedral",
                atoms=(0, 1, 2, 3),
                values=(0.0,),
                mask=(0, 1, 2),  # missing a4=3
            ),
        ),
        evaluate="none",
    )
    with pytest.raises(ValueError, match="must include a4"):
        prepare_geometries(config)


def test_nma_omega_and_2d_methyl_product():
    from mmml.ic_scan.topology import angles_match

    nma = Path(__file__).resolve().parents[1].parent / "examples" / "ic_scan" / "nma.xyz"
    if not nma.is_file():
        pytest.skip("examples/ic_scan/nma.xyz not present")
    config = IcScanConfig(
        structure=nma,
        evaluate="none",
        dofs=(
            DegreeOfFreedom(
                "omega",
                "dihedral",
                (0, 4, 6, 8),
                values=(-90.0, 0.0, 90.0),
            ),
            DegreeOfFreedom(
                "n_methyl",
                "dihedral",
                (4, 6, 8, 9),
                values=(-180.0, 0.0, 90.0),
            ),
        ),
        scans=(ScanSpec("omega_methyl_2d", ("omega", "n_methyl")),),
    )
    _, prepared = prepare_geometries(config)
    assert len(prepared) == 9
    for point, atoms in prepared:
        for name, dof in config.dof_map().items():
            assert angles_match(
                float(atoms.get_dihedral(*dof.atoms)),
                point.coordinates[name],
                atol_deg=1.0,
            )


def test_nma_bad_omega_mask_hr_only_raises():
    nma = Path(__file__).resolve().parents[1].parent / "examples" / "ic_scan" / "nma.xyz"
    if not nma.is_file():
        pytest.skip("examples/ic_scan/nma.xyz not present")
    config = IcScanConfig(
        structure=nma,
        evaluate="none",
        dofs=(
            DegreeOfFreedom(
                "omega",
                "dihedral",
                (0, 4, 6, 8),
                values=(0.0, 90.0),
                mask=(9, 10, 11),  # HR* only — excludes a4=CR
            ),
        ),
    )
    with pytest.raises(ValueError, match="must include a4"):
        prepare_geometries(config)


def test_run_ic_scan_with_emt_and_roundtrip(structure_xyz: Path, tmp_path: Path):
    config = IcScanConfig(
        structure=structure_xyz,
        dofs=(
            DegreeOfFreedom(
                name="r",
                kind="bond",
                atoms=(0, 1),
                values=(1.45, 1.55),
            ),
        ),
        calculator="xtb",  # placeholder; overridden by factory injection
        evaluate="energy",
    )
    result = run_ic_scan(config, calculator=lambda: EMT())
    assert len(result.records) == 2
    assert all(record.status == "success" for record in result.records)
    assert all(record.energy_ev is not None for record in result.records)
    assert all(record.max_force_ev_A is not None for record in result.records)
    assert all(record.max_force_ev_A > 0 for record in result.records)
    out = tmp_path / "bundle"
    result.write(out)
    loaded = type(result).read(out)
    assert len(loaded.records) == 2
    assert loaded.records[0].energy_ev == pytest.approx(result.records[0].energy_ev)
    assert loaded.records[0].max_force_ev_A == pytest.approx(
        result.records[0].max_force_ev_A
    )
    assert (out / "energy_r.png").is_file() or any(
        p.name.startswith("energy_") for p in out.glob("*.png")
    )
    assert any(p.name.startswith("maxforce_") for p in out.glob("*.png"))


def test_plot_model_comparison(structure_xyz: Path, tmp_path: Path):
    from mmml.ic_scan.plotting import plot_model_comparison

    def _scan(tag: str):
        config = IcScanConfig(
            structure=structure_xyz,
            dofs=(
                DegreeOfFreedom(
                    name="phi",
                    kind="dihedral",
                    atoms=(0, 1, 2, 3),
                    values=(-60.0, 0.0, 60.0),
                ),
            ),
            calculator="xtb",
            evaluate="energy",
        )
        return run_ic_scan(config, calculator=lambda: EMT())

    a = _scan("a")
    b = _scan("b")
    out = tmp_path / "compare"
    paths = plot_model_comparison({"A": a, "B": b}, out)
    assert paths
    assert any("compare_energy" in p.name for p in paths)
    assert any("compare_maxforce" in p.name for p in paths)


def test_config_yaml_roundtrip(structure_xyz: Path, tmp_path: Path):
    config_path = tmp_path / "ic_scan.yaml"
    payload = {
        "structure": "mol.xyz",
        "calculator": "xtb",
        "evaluate": "none",
        "scan_mode": "individual",
        "dofs": [
            {
                "name": "phi",
                "kind": "dihedral",
                "atoms": [0, 1, 2, 3],
                "start": -180,
                "stop": 180,
                "n_points": 4,
            }
        ],
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    # structure path relative to YAML
    write(tmp_path / "mol.xyz", _butane_like())
    data = yaml.safe_load(config_path.read_text())
    data["structure"] = str(tmp_path / "mol.xyz")
    config = IcScanConfig.from_dict(data)
    assert config.scan_mode == "individual"
    assert len(config.dofs[0].values) == 4
    dumped = config.to_dict()
    assert dumped["dofs"][0]["kind"] == "dihedral"


def test_ic_scan_cli_registered_and_prepare_only(structure_xyz: Path, tmp_path: Path, monkeypatch, capsys):
    spec = command_by_name("ic-scan")
    assert spec is not None
    assert spec.module == "mmml.cli.misc.ic_scan"
    assert build_parser().prog == "mmml ic-scan"

    monkeypatch.setattr("sys.argv", ["mmml", "ic-scan", "--help"])
    with pytest.raises(SystemExit) as exc:
        mmml_main()
    assert exc.value.code == 0
    assert "bond/angle/dihedral" in capsys.readouterr().out.lower()

    config_path = tmp_path / "scan.yaml"
    write(tmp_path / "mol.xyz", _butane_like())
    config_path.write_text(
        yaml.safe_dump(
            {
                "structure": "mol.xyz",
                "evaluate": "energy",
                "calculator": "xtb",
                "dofs": [
                    {
                        "name": "r",
                        "kind": "bond",
                        "atoms": [0, 1],
                        "values": [1.5, 1.6],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert (
        ic_scan_main(
            [
                "--config",
                str(config_path),
                "--prepare-only",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"]["prepared"] == 2
    assert manifest["counts"]["successful"] == 0
