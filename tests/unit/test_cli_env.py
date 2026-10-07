"""Tests for ``karml env``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from karml.cli import env as env_cli


def test_collect_env_report_includes_ckpt_and_presets(monkeypatch, tmp_path):
    ckpt = tmp_path / "test_portable.json"
    ckpt.write_text('{"params": {}, "config": {}, "metadata": {}}')
    monkeypatch.setenv("KARML_CKPT", str(ckpt))
    report = env_cli.collect_env_report()
    assert report["KARML_CKPT"] == str(ckpt.resolve())
    assert report["KARML_CKPT_source"] == "KARML_CKPT"
    assert report["KARML_CKPT_set"] is True
    assert report.get("presets_dir")


def test_export_lines_suggest_ckpt_when_unset(monkeypatch, tmp_path):
    ckpt = tmp_path / "model.json"
    ckpt.write_text("{}")
    monkeypatch.delenv("KARML_CKPT", raising=False)
    monkeypatch.delenv("KARML_CHECKPOINT", raising=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            env_cli,
            "_resolve_karml_ckpt",
            lambda: (ckpt.resolve(), "test"),
        )
        lines = env_cli.export_lines({"KARML_CKPT": str(ckpt.resolve()), "KARML_CKPT_set": False})
    assert any("export KARML_CKPT=" in line for line in lines)


def test_main_json(monkeypatch, tmp_path, capsys):
    ckpt = tmp_path / "ckpt.json"
    ckpt.write_text("{}")
    monkeypatch.setenv("KARML_CKPT", str(ckpt))
    rc = env_cli.main(["--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["KARML_CKPT"] == str(ckpt.resolve())
    assert "model_defaults" in data
    assert set(data["model_defaults"].keys()) == {"physnet", "spookynet", "mbd", "multipoles"}


def test_model_defaults_resolution(monkeypatch, tmp_path):
    spooky = tmp_path / "spooky.json"
    mbd = tmp_path / "mbd.json"
    mult = tmp_path / "mult.json"
    for p in (spooky, mbd, mult):
        p.write_text("{}")

    monkeypatch.setenv("SPOOKYNET_CKPT", str(spooky))
    monkeypatch.setenv("MBD_CKPT", str(mbd))
    monkeypatch.setenv("MULTIPOLES_CKPT", str(mult))

    report = env_cli.collect_env_report()
    assert report["SPOOKYNET_CKPT"] == str(spooky.resolve())
    assert report["MBD_CKPT"] == str(mbd.resolve())
    assert report["MULTIPOLES_CKPT"] == str(mult.resolve())

    models = report["model_defaults"]
    assert models["spookynet"]["available"] is True
    assert models["mbd"]["available"] is True
    assert models["multipoles"]["available"] is True
