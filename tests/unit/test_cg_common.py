from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


def _load_cg_common():
    path = Path(__file__).resolve().parents[2] / "examples" / "cg_common.py"
    spec = importlib.util.spec_from_file_location("cg_common_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cg_common = _load_cg_common()


def test_load_cg_config_applies_json_then_cli(tmp_path: Path) -> None:
    config = tmp_path / "cg.json"
    config.write_text(json.dumps({"n_waters": 8, "temperature": 250.0}))

    settings = cg_common.load_cg_config(
        {
            "checkpoint": "old.json",
            "peptide_checkpoint": "old.json",
            "water_checkpoint": "old.json",
            "n_waters": 4,
            "temperature": 200.0,
        },
        description="test",
        argv=[
            "--config",
            str(config),
            "--checkpoint",
            "new.json",
            "--temperature",
            "275",
        ],
    )

    assert settings.n_waters == 8
    assert settings.temperature == 275.0
    assert settings.checkpoint == "new.json"
    assert settings.peptide_checkpoint == "new.json"
    assert settings.water_checkpoint == "new.json"


def test_validate_supported_elements_has_no_size_or_order_restriction() -> None:
    model = type("Model", (), {"max_atomic_number": 8})()
    numbers = np.array([8, 1, 6, 1] * 20)
    cg_common.validate_supported_elements(model, numbers, label="model")


def test_validate_supported_elements_rejects_out_of_range_element() -> None:
    model = type("Model", (), {"max_atomic_number": 8})()
    with pytest.raises(ValueError, match=r"contains \[17\]"):
        cg_common.validate_supported_elements(model, [1, 17], label="model")


def test_probe_charge_output_accepts_finite_per_atom_charges() -> None:
    class ChargeModel:
        charges = True
        max_atomic_number = 8

        def apply(self, params, **kwargs):
            del params
            return {"charges": jnp.zeros(kwargs["atomic_numbers"].shape[0])}

    result = cg_common.probe_charge_output(
        ChargeModel(),
        {},
        [8, 1, 1],
        np.zeros((3, 3)),
        charge=0.0,
        spin=1.0,
        label="water",
    )
    assert result.shape == (3,)


def test_probe_charge_output_rejects_missing_charges() -> None:
    class NoChargeModel:
        charges = True
        max_atomic_number = 8

        def apply(self, params, **kwargs):
            del params, kwargs
            return {"energy": jnp.array(0.0)}

    with pytest.raises(ValueError, match="contains no atomic charges"):
        cg_common.probe_charge_output(
            NoChargeModel(),
            {},
            [8, 1, 1],
            np.zeros((3, 3)),
            charge=0.0,
            spin=1.0,
            label="water",
        )
