"""
Unit tests for orbax-to-JSON checkpoint conversion.

Verifies that:
- orbax checkpoints can be converted to JSON
- JSON params can be loaded on CPU with float64 precision
- Round-trip preserves values (within float precision)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mmml.utils.model_checkpoint import (
    json_to_params,
    load_model_checkpoint,
    orbax_to_json,
    to_jsonable,
)

# Path to DESdimers final2 checkpoint (may not exist or be loadable on CPU)
DESDIMERS_CKPT = Path(__file__).parent.parent / "mmml" / "physnetjax" / "ckpts" / "DESdimers" / "final2"


def _create_synthetic_params():
    """Create minimal synthetic params for testing."""
    return {
        "embedding": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "dense": {
            "kernel": np.random.randn(4, 8).astype(np.float32) * 0.01,
            "bias": np.zeros(8, dtype=np.float32),
        },
    }


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_to_jsonable_roundtrip():
    """to_jsonable converts arrays to lists."""
    params = _create_synthetic_params()
    jsonable = to_jsonable(params)
    assert isinstance(jsonable, dict)
    assert "embedding" in jsonable
    assert isinstance(jsonable["embedding"], list)
    assert len(jsonable["embedding"]) == 2
    assert len(jsonable["embedding"][0]) == 2


def test_orbax_to_json_and_json_to_params_roundtrip(temp_dir):
    """orbax -> JSON -> params round-trip preserves structure and values."""
    pytest.importorskip("orbax")

    import orbax.checkpoint as ocp

    # Create synthetic params and save as orbax (CPU)
    params = _create_synthetic_params()
    orbax_dir = temp_dir / "orbax_ckpt"
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(str(orbax_dir), params)

    # Convert to JSON
    json_path = temp_dir / "params.json"
    orbax_to_json(orbax_checkpoint_dir=orbax_dir, output_path=json_path)

    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
    assert "params" in data
    assert "embedding" in data["params"]

    # Load back as numpy (default float32)
    loaded = json_to_params(json_path)
    assert "params" in loaded
    loaded_params = loaded["params"]
    assert isinstance(loaded_params["embedding"], np.ndarray)
    np.testing.assert_allclose(loaded_params["embedding"], params["embedding"])
    np.testing.assert_allclose(
        loaded_params["dense"]["kernel"], params["dense"]["kernel"]
    )


def test_json_to_params_float64(temp_dir):
    """json_to_params with dtype=float64 produces float64 arrays."""
    params = _create_synthetic_params()
    json_path = temp_dir / "params.json"
    with open(json_path, "w") as f:
        json.dump({"params": to_jsonable(params)}, f, indent=2)

    loaded = json_to_params(json_path, dtype="float64")
    loaded_params = loaded["params"]
    assert loaded_params["embedding"].dtype == np.float64
    assert loaded_params["dense"]["kernel"].dtype == np.float64


def test_load_model_checkpoint_json_with_dtype(temp_dir):
    """load_model_checkpoint loads params.json with dtype conversion."""
    params = _create_synthetic_params()
    json_path = temp_dir / "params.json"
    with open(json_path, "w") as f:
        json.dump({"params": to_jsonable(params)}, f, indent=2)

    ckpt = load_model_checkpoint(temp_dir, dtype="float64")
    assert "params" in ckpt
    assert ckpt["params"]["embedding"].dtype == np.float64


def test_orbax_to_json_with_config_and_metadata(temp_dir):
    """orbax_to_json includes config and metadata when provided."""
    pytest.importorskip("orbax")

    import orbax.checkpoint as ocp

    params = _create_synthetic_params()
    orbax_dir = temp_dir / "orbax_ckpt"
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(str(orbax_dir), params)

    json_path = temp_dir / "params.json"
    orbax_to_json(
        orbax_checkpoint_dir=orbax_dir,
        output_path=json_path,
        config={"features": 64, "cutoff": 6.0},
        metadata={"epoch": 100},
    )

    loaded = json_to_params(json_path)
    # orbax_to_json with config/metadata adds them to the payload
    with open(json_path) as f:
        data = json.load(f)
    assert "config" in data
    assert data["config"]["features"] == 64
    assert "metadata" in data
    assert data["metadata"]["epoch"] == 100


@pytest.mark.skipif(
    not DESDIMERS_CKPT.exists(),
    reason="DESdimers final2 checkpoint not found",
)
def test_desdimers_orbax_to_json_if_loadable(temp_dir):
    """
    Convert DESdimers final2 to JSON when loadable.

    This test is skipped if the checkpoint doesn't exist.
    It may fail on CPU-only machines if the checkpoint was saved with GPU sharding.
    """
    pytest.importorskip("orbax")

    json_path = temp_dir / "desdimers_params.json"
    try:
        orbax_to_json(
            orbax_checkpoint_dir=DESDIMERS_CKPT,
            output_path=json_path,
        )
        assert json_path.exists()
        # Verify we can load it back with float64
        loaded = json_to_params(json_path, dtype="float64")
        assert "params" in loaded
        # Check that params have expected structure (nested dict of arrays)
        assert isinstance(loaded["params"], dict)
    except (ValueError, OSError) as e:
        if "cuda" in str(e).lower() or "sharding" in str(e).lower():
            pytest.skip(
                f"DESdimers checkpoint requires GPU to load: {e}"
            )
        raise
