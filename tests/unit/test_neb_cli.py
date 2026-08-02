from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator

from mmml.cli.__main__ import main as mmml_main
from mmml.cli.misc.neb import build_parser, main
from mmml.cli.registry import command_by_name
from mmml.neb import NebConfig, run_neb
from mmml.neb.run import path_length_coordinate, relative_energies_kcal


class _ZeroForceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
        Calculator.calculate(self, atoms, properties, system_changes)
        n = len(atoms)
        # Mild harmonic well so NEB forces are finite but converge quickly.
        pos = atoms.get_positions()
        self.results = {
            "energy": float(0.5 * np.sum(pos**2)),
            "forces": -pos.copy(),
        }


def test_neb_cli_is_registered_and_help_is_reachable(monkeypatch, capsys):
    spec = command_by_name("neb")
    assert spec is not None
    assert spec.module == "mmml.cli.misc.neb"
    assert build_parser().prog == "mmml neb"

    monkeypatch.setattr("sys.argv", ["mmml", "neb", "--help"])
    with pytest.raises(SystemExit) as exc:
        mmml_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Nudged elastic band" in out


def test_neb_config_rejects_too_few_images():
    with pytest.raises(ValueError, match="n_images"):
        NebConfig(
            initial=Path("a.xyz"),
            final=Path("b.xyz"),
            checkpoint=Path("c.json"),
            output_dir=Path("out"),
            n_images=2,
        )


def test_path_helpers_match_relative_profile():
    images = []
    for x in (0.0, 1.0, 2.0):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [x, 0.0, 0.0]])
        atoms.calc = _ZeroForceCalculator()
        images.append(atoms)
    rc = path_length_coordinate(images)
    assert rc.shape == (3,)
    assert rc[0] == pytest.approx(0.0)
    assert rc[-1] == pytest.approx(2.0)
    e = relative_energies_kcal(images)
    assert e[0] == pytest.approx(0.0)
    assert e[1] > 0.0


def test_run_neb_with_custom_calculator(tmp_path: Path):
    initial = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
    final = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]])
    init_path = tmp_path / "initial.xyz"
    final_path = tmp_path / "final.xyz"
    from ase.io import write

    write(str(init_path), initial)
    write(str(final_path), final)

    out = tmp_path / "neb_out"
    config = NebConfig(
        initial=init_path,
        final=final_path,
        checkpoint=tmp_path / "unused.json",
        output_dir=out,
        n_images=5,
        fmax=0.2,
        max_steps=20,
        plot=False,
        pair_indices=((0, 1),),
        overwrite=True,
    )
    result = run_neb(config, calculator_factory=_ZeroForceCalculator)
    assert result.paths["xyz"].is_file()
    assert result.paths["profile"].is_file()
    assert result.paths["summary"].is_file()
    assert len(result.images) == 5
    assert "d_0_1" in result.pair_distance_angstrom


def test_neb_cli_accepts_yaml_config(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "neb.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "checkpoint": "kl.json",
                "initial": "reag.xyz",
                "final": "prod.xyz",
                "output_dir": "out",
                "n_images": 7,
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    class _Result:
        barrier_kcal_mol = 1.23
        paths = {"summary": tmp_path / "summary.json"}

    def fake_run(config):
        captured["config"] = config
        return _Result()

    monkeypatch.setattr("mmml.cli.misc.neb.run_neb", fake_run)
    assert main(["--config", str(config_path), "--overwrite"]) == 0
    cfg = captured["config"]
    assert cfg.n_images == 7
    assert cfg.checkpoint == config_path.parent / "kl.json"
    assert cfg.overwrite is True
