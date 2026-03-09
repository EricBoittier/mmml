from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


ORBAX_EPOCH_1985 = Path("mmml/models/physnetjax/ckpts/DESdimers/epoch-1985").resolve()


def _assert_tree_allclose(a, b, path: str = "root") -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict), f"{path}: type mismatch {type(a)} != {type(b)}"
        assert set(a.keys()) == set(b.keys()), f"{path}: key mismatch"
        for k in a:
            _assert_tree_allclose(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, (list, tuple)):
        assert isinstance(b, type(a)), f"{path}: type mismatch {type(a)} != {type(b)}"
        assert len(a) == len(b), f"{path}: length mismatch {len(a)} != {len(b)}"
        for i, (ai, bi) in enumerate(zip(a, b)):
            _assert_tree_allclose(ai, bi, f"{path}[{i}]")
        return
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)


def test_epoch1985_orbax_to_json_roundtrip_matches_params(tmp_path: Path):
    """
    Convert epoch-1985 Orbax checkpoint to JSON and compare parameters.

    This validates the conversion itself by checking the restored Orbax params
    against the params loaded back from the generated JSON.
    """
    if not _can_import("orbax"):
        pytest.skip("orbax not available in this environment")
    if not ORBAX_EPOCH_1985.exists():
        pytest.skip(f"Missing orbax checkpoint: {ORBAX_EPOCH_1985}")

    import orbax.checkpoint as ocp
    from mmml.utils.model_checkpoint import orbax_to_json, json_to_params

    converted_json = tmp_path / "epoch1985_params.json"
    orbax_to_json(
        orbax_checkpoint_dir=ORBAX_EPOCH_1985,
        output_path=converted_json,
    )
    assert converted_json.exists()

    restored_orbax = ocp.PyTreeCheckpointer().restore(str(ORBAX_EPOCH_1985))
    params_orbax = restored_orbax["params"] if isinstance(restored_orbax, dict) and "params" in restored_orbax else restored_orbax

    restored_json = json_to_params(converted_json, backend="numpy")
    params_json = restored_json["params"]

    _assert_tree_allclose(params_orbax, params_json)

