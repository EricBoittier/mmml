from __future__ import annotations

import pytest

from mmml.cli.__main__ import main as mmml_main
from mmml.cli.misc.dimer_scan import _distance_grid, build_parser
from mmml.cli.registry import command_by_name


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
