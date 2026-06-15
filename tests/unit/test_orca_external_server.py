"""Unit tests for the persistent MMML ORCA server/client."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.interfaces.orca_external.client import send_to_server
from mmml.interfaces.orca_external.runner import clear_calculator_cache
from mmml.interfaces.orca_external.server import MmmlOrcaServer
from mmml.interfaces.orca_external.settings import MmmlOrcaSettings


@pytest.fixture
def orca_server_kwargs() -> dict:
    """Fast, deterministic batching for unit tests."""
    return {"max_batch_size": 8, "batch_wait_ms": 0.0}


def _write_orca_job(tmp_path: Path, basename: str = "job_EXT") -> Path:
    xyz_path = tmp_path / f"{basename}.xyz"
    xyz_path.write_text(
        "\n".join(
            [
                "2",
                "water",
                "O 0.0 0.0 0.0",
                "H 0.96 0.0 0.0",
            ]
        )
    )
    extinp_path = tmp_path / f"{basename}.extinp.tmp"
    extinp_path.write_text(
        "\n".join(
            [
                f"{basename}.xyz",
                "0",
                "1",
                "1",
                "1",
            ]
        )
    )
    return extinp_path


def _dummy_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "dummy.pkl"
    checkpoint.write_bytes(b"")
    return checkpoint


def _install_mock_calculator(monkeypatch) -> None:
    class _MockCalc:
        def calculate(self, atoms=None, properties=None, system_changes=None):
            self.results = {
                "energy": -100.0,
                "forces": np.zeros((len(atoms), 3)),
            }

        def get_potential_energy(self, atoms=None):
            return self.results["energy"]

        def get_forces(self, atoms=None):
            return self.results["forces"]

    monkeypatch.setattr(
        "mmml.interfaces.orca_external.runner.get_calculator",
        lambda settings: _MockCalc(),
    )


def test_server_handle_writes_engrad(
    tmp_path: Path, monkeypatch, orca_server_kwargs: dict
) -> None:
    clear_calculator_cache()
    _install_mock_calculator(monkeypatch)
    extinp_path = _write_orca_job(tmp_path)

    settings = MmmlOrcaSettings(checkpoint=_dummy_checkpoint(tmp_path))
    server = MmmlOrcaServer(default_settings=settings, **orca_server_kwargs)
    try:
        result = server.handle(
            [extinp_path.name],
            directory=str(tmp_path),
        )

        assert result["status"] == "Success"
        engrad_path = tmp_path / "job_EXT.engrad"
        assert engrad_path.is_file()
        assert "2\n" in engrad_path.read_text()
    finally:
        server.shutdown()
        clear_calculator_cache()


def test_server_inherits_default_checkpoint(
    tmp_path: Path, monkeypatch, orca_server_kwargs: dict
) -> None:
    clear_calculator_cache()
    _install_mock_calculator(monkeypatch)
    extinp_path = _write_orca_job(tmp_path, basename="inherit_EXT")

    settings = MmmlOrcaSettings(checkpoint=_dummy_checkpoint(tmp_path), cutoff=8.5)
    server = MmmlOrcaServer(default_settings=settings, **orca_server_kwargs)
    try:
        server.handle([extinp_path.name], directory=str(tmp_path))
        assert (tmp_path / "inherit_EXT.engrad").is_file()
    finally:
        server.shutdown()
        clear_calculator_cache()


def test_client_forwards_to_server(
    tmp_path: Path, monkeypatch, orca_server_kwargs: dict
) -> None:
    clear_calculator_cache()
    _install_mock_calculator(monkeypatch)
    extinp_path = _write_orca_job(tmp_path, basename="client_EXT")

    settings = MmmlOrcaSettings(checkpoint=_dummy_checkpoint(tmp_path))
    server = MmmlOrcaServer(default_settings=settings, **orca_server_kwargs)

    def _fake_urlopen(request, timeout=None):
        payload = json.loads(request.data.decode("utf-8"))
        result = server.handle(payload["arguments"], payload["directory"])

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(result).encode("utf-8")

        return _Resp()

    monkeypatch.setattr("mmml.interfaces.orca_external.client.urllib.request.urlopen", _fake_urlopen)

    try:
        send_to_server(
            "127.0.0.1:8888",
            [extinp_path.name],
            working_directory=str(tmp_path),
        )

        assert (tmp_path / "client_EXT.engrad").is_file()
    finally:
        server.shutdown()
        clear_calculator_cache()
