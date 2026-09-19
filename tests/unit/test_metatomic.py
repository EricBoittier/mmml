"""Unit tests for optional metatomic checkpoint helpers (no torch required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmml.interfaces.calculators.metatomic import (
    DEFAULT_METATOMIC_DEVICE,
    METATOMIC_DEVICE_ENV,
    have_metatomic,
    is_metatomic_checkpoint,
    load_metatomic_calculator,
    metatomic_device_name,
    resolve_metatomic_model_path,
)


def test_is_metatomic_checkpoint_suffix_and_missing_path() -> None:
    assert is_metatomic_checkpoint("export.pt")
    assert is_metatomic_checkpoint("export.pth")
    assert not is_metatomic_checkpoint("params.json")
    assert not is_metatomic_checkpoint(None)
    assert not is_metatomic_checkpoint("unused-path")


def test_is_metatomic_checkpoint_export_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not is_metatomic_checkpoint(empty)

    export = tmp_path / "export"
    export.mkdir()
    (export / "model.pt").write_bytes(b"stub")
    assert is_metatomic_checkpoint(export)
    assert resolve_metatomic_model_path(export) == (export / "model.pt").resolve()


def test_resolve_metatomic_model_path_file(tmp_path: Path) -> None:
    pt = tmp_path / "atomistic.pt"
    pt.write_bytes(b"stub")
    assert resolve_metatomic_model_path(pt) == pt.resolve()


def test_resolve_metatomic_model_path_missing_dir(tmp_path: Path) -> None:
    d = tmp_path / "no-model"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="none of"):
        resolve_metatomic_model_path(d)


def test_metatomic_device_name_env_and_override(monkeypatch) -> None:
    monkeypatch.delenv(METATOMIC_DEVICE_ENV, raising=False)
    assert metatomic_device_name() == DEFAULT_METATOMIC_DEVICE
    monkeypatch.setenv(METATOMIC_DEVICE_ENV, "cuda")
    assert metatomic_device_name() == "cuda"
    assert metatomic_device_name(device="cpu") == "cpu"


def test_have_metatomic_does_not_raise() -> None:
    assert have_metatomic() in (True, False)


def test_load_metatomic_calculator_without_extra_or_invalid_file(tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"not-a-torchscript-atomistic-model")
    if not have_metatomic():
        with pytest.raises(ModuleNotFoundError, match="uv sync --extra metatomic"):
            load_metatomic_calculator(pt)
        return
    with pytest.raises(Exception):
        load_metatomic_calculator(pt)
