from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mmml.cli.__main__ import main as mmml_main
from mmml.cli.misc.dimer_scan import _distance_grid, build_parser, main
from mmml.cli.registry import command_by_name


EXPECTED_CALCULATORS = {
    "physnet",
    "spookynet",
    "mbd",
    "multipoles",
    "efield",
    "xtb",
    "dftb3-d4",
    "pyscf",
}


def test_dimer_scan_cli_is_registered_and_help_is_reachable(monkeypatch, capsys):
    spec = command_by_name("dimer-scan")
    assert spec is not None
    assert spec.module == "mmml.cli.misc.dimer_scan"
    assert build_parser().prog == "mmml dimer-scan"

    monkeypatch.setattr("sys.argv", ["mmml", "dimer-scan", "--help"])
    with pytest.raises(SystemExit) as exc:
        mmml_main()
    assert exc.value.code == 0
    assert "reproducible rigid 1D dimer" in capsys.readouterr().out


def test_distance_grid_requires_an_exact_inclusive_stop():
    assert _distance_grid("2.0:3.0:0.5") == (2.0, 2.5, 3.0)
    with pytest.raises(Exception, match="land exactly"):
        _distance_grid("2.0:3.0:0.3")


def test_documented_calculator_choices_are_parser_choices():
    action = next(item for item in build_parser()._actions if item.dest == "calculator")
    assert set(action.choices) == EXPECTED_CALCULATORS


def test_dimer_scan_accepts_validated_yaml_config_with_policy_provenance(
    tmp_path: Path, monkeypatch
):
    config_path = tmp_path / "dimer_scan.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "residues": ["TIP3", "TIP3"],
                "calculator": "xtb",
                "distances_angstrom": [2.5, 3.0],
                "interaction_policy": "interaction_policy.yaml",
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    class _Result:
        records = (1, 2)
        has_failures = False

        def write(self, output, *, overwrite=False):
            return {"manifest": Path(output) / "manifest.json"}

    def fake_run(config):
        captured["config"] = config
        return _Result()

    monkeypatch.setattr("mmml.cli.misc.dimer_scan.run_dimer_scan", fake_run)
    assert main(["--config", str(config_path), "--output", str(tmp_path / "out")]) == 0
    assert captured["config"].interaction_policy == tmp_path / "interaction_policy.yaml"
