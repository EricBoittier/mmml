"""CLI wiring tests for ``mmml dmc``."""

from __future__ import annotations

import pytest

from mmml.cli.__main__ import main as mmml_main
from mmml.cli.parser_utils import parser_available
from mmml.cli.registry import command_by_name
from mmml.generate.dmc.dmc import build_parser


def test_dmc_cli_is_registered():
    spec = command_by_name("dmc")
    assert spec is not None
    assert spec.module == "mmml.generate.dmc.dmc"
    assert parser_available("dmc")
    assert build_parser().prog == "mmml dmc"


def test_dmc_help_is_reachable(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["mmml", "dmc", "--help"])
    with pytest.raises(SystemExit) as exc:
        mmml_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Diffusion Monte Carlo" in out
    assert "jax.vmap" in out


def test_dmc_masses_support_nh3_ch3cl_elements():
    from mmml.generate.dmc.dmc import _masses_and_charges
    import numpy as np

    symbols = np.array(["Cl", "N", "C", "H", "H", "H", "H", "H", "H"], dtype=str)
    mass, z = _masses_and_charges(symbols)
    assert mass.shape == (9,)
    assert z.tolist() == [17, 7, 6, 1, 1, 1, 1, 1, 1]


def test_dmc_parser_requires_core_flags():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--natm",
            "10",
            "--nwalker",
            "8",
            "--stepsize",
            "1e-3",
            "--nstep",
            "20",
            "--eqstep",
            "5",
            "--alpha",
            "100.0",
            "--checkpoint",
            "/tmp/ckpt",
            "--input",
            "geo.xyz",
            "--max-batch",
            "4",
            "--seed",
            "0",
        ]
    )
    assert args.natm == 10
    assert args.nwalker == 8
    assert args.max_batch == 4
    assert args.seed == 0
