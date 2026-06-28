"""Tests for ``mmml env``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmml.cli import env as env_cli


def test_collect_env_report_includes_ckpt_and_presets(monkeypatch, tmp_path):
    ckpt = tmp_path / "test_portable.json"
    ckpt.write_text('{"params": {}, "config": {}, "metadata": {}}')
    monkeypatch.setenv("MMML_CKPT", str(ckpt))
    report = env_cli.collect_env_report()
    assert report["MMML_CKPT"] == str(ckpt.resolve())
    assert report["MMML_CKPT_source"] == "MMML_CKPT"
    assert report["MMML_CKPT_set"] is True
    assert report.get("presets_dir")


def test_export_lines_suggest_ckpt_when_unset(monkeypatch, tmp_path):
    ckpt = tmp_path / "model.json"
    ckpt.write_text("{}")
    monkeypatch.delenv("MMML_CKPT", raising=False)
    monkeypatch.delenv("MMML_CHECKPOINT", raising=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            env_cli,
            "_resolve_mmml_ckpt",
            lambda: (ckpt.resolve(), "test"),
        )
        lines = env_cli.export_lines({"MMML_CKPT": str(ckpt.resolve()), "MMML_CKPT_set": False})
    assert any("export MMML_CKPT=" in line for line in lines)


def test_main_json(monkeypatch, tmp_path, capsys):
    ckpt = tmp_path / "ckpt.json"
    ckpt.write_text("{}")
    monkeypatch.setenv("MMML_CKPT", str(ckpt))
    rc = env_cli.main(["--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["MMML_CKPT"] == str(ckpt.resolve())
