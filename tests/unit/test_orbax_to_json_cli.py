from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_orbax_to_json_cli_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "mmml.cli.__main__", "orbax-to-json", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--output" in proc.stdout
    assert "checkpoint" in proc.stdout.lower()


def test_orbax_to_json_cli_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("orbax")

    import orbax.checkpoint as ocp

    from mmml.utils.model_checkpoint import json_to_params, to_jsonable

    params = {
        "embedding": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "dense": {
            "kernel": np.random.randn(4, 8).astype(np.float32) * 0.01,
            "bias": np.zeros(8, dtype=np.float32),
        },
    }
    orbax_dir = tmp_path / "orbax_ckpt"
    ocp.PyTreeCheckpointer().save(str(orbax_dir), params)

    json_path = tmp_path / "exported_params.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mmml.cli.__main__",
            "orbax-to-json",
            str(orbax_dir),
            "-o",
            str(json_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json_path.exists()
    assert "Wrote portable checkpoint" in proc.stdout

    with open(json_path) as f:
        data = json.load(f)
    assert "params" in data

    loaded = json_to_params(json_path)["params"]
    np.testing.assert_allclose(loaded["embedding"], params["embedding"])


def test_orbax_to_json_cli_rejects_json_checkpoint(tmp_path: Path) -> None:
    from mmml.utils.model_checkpoint import to_jsonable

    json_path = tmp_path / "already.json"
    with open(json_path, "w") as f:
        json.dump({"params": to_jsonable({"x": np.array([1.0], dtype=np.float32)})}, f)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mmml.cli.__main__",
            "orbax-to-json",
            str(json_path),
            "-o",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "already JSON" in proc.stderr
