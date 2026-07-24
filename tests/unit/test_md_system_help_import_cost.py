"""``mmml md-system -h`` must not import JAX / heavy MLpot runtime modules."""

from __future__ import annotations

import contextlib
import io
import sys

import pytest


def test_md_system_module_import_does_not_load_jax():
    # Fresh attribute check in this process: if a prior test already imported jax,
    # skip the hard assertion and only check the md_system module itself is light.
    jax_already = "jax" in sys.modules
    from mmml.cli.run import md_system

    assert md_system.build_parser is not None
    if not jax_already:
        assert "jax" not in sys.modules
        assert "mmml.interfaces.pycharmmInterface.mlpot.cli_common" not in sys.modules


def test_md_system_help_does_not_import_jax_or_cli_common():
    from mmml.cli.run import md_system

    jax_before = "jax" in sys.modules
    cli_common_before = "mmml.interfaces.pycharmmInterface.mlpot.cli_common" in sys.modules
    pandas_before = "pandas" in sys.modules

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        with pytest.raises(SystemExit) as excinfo:
            md_system.build_parser().parse_args(["-h"])
    assert excinfo.value.code == 0
    help_text = buf.getvalue()
    # Default -h is a category index (not the full flag wall).
    assert "Help is split into categories" in help_text
    assert "-h1" in help_text
    assert "--setup" in help_text  # listed under common starting flags
    assert "--help-all" in help_text

    if not jax_before:
        assert "jax" not in sys.modules
    if not cli_common_before:
        assert "mmml.interfaces.pycharmmInterface.mlpot.cli_common" not in sys.modules
    if not pandas_before:
        assert "pandas" not in sys.modules


def test_md_system_main_help_short_circuits(monkeypatch, capsys):
    from mmml.cli.run import md_system

    monkeypatch.setattr(sys, "argv", ["mmml md-system", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        md_system.main()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "Help is split into categories" in out
    assert "--help-all" in out


def test_ml_dtypes_add_args_does_not_import_jax():
    import argparse

    jax_before = "jax" in sys.modules
    from mmml.interfaces.pycharmmInterface.ml_dtypes import add_ml_compute_dtype_args

    if not jax_before:
        assert "jax" not in sys.modules
    parser = argparse.ArgumentParser()
    add_ml_compute_dtype_args(parser)
    args = parser.parse_args(["--ml-compute-dtype", "float32"])
    assert args.ml_compute_dtype == "float32"
    if not jax_before:
        assert "jax" not in sys.modules
