"""Unit tests for the Mode D (latent_mean) precomputed charge template."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmml.models.latent_charge_template import (
    LatentChargeTemplate,
    load_latent_charge_template,
    save_latent_charge_template,
    tile_latent_charge_template,
)


def _template(charges: np.ndarray) -> LatentChargeTemplate:
    return LatentChargeTemplate(
        atomic_numbers=np.array([6, 17, 17, 1, 1]),
        charges=charges,
        charges_std=np.zeros_like(charges),
        n_samples=42,
        resid="DCM",
        source_checkpoint="ckpts/mp2_nms",
        source_data="mp2_nms15_clean_train.npz",
    )


def test_save_load_roundtrip(tmp_path: Path):
    charges = np.array([0.3, -0.1, -0.1, -0.05, -0.05])
    path = tmp_path / "template.npz"
    save_latent_charge_template(path, _template(charges))
    loaded = load_latent_charge_template(path)
    assert np.allclose(loaded.charges, charges)
    assert np.array_equal(loaded.atomic_numbers, [6, 17, 17, 1, 1])
    assert loaded.resid == "DCM"
    assert loaded.n_samples == 42
    assert loaded.source_checkpoint == "ckpts/mp2_nms"


def test_load_rejects_non_neutral_template(tmp_path: Path):
    charges = np.array([0.3, -0.1, -0.1, -0.05, 0.0])  # net +0.05 e
    path = tmp_path / "bad_template.npz"
    save_latent_charge_template(path, _template(charges))
    with pytest.raises(ValueError, match="net charge"):
        load_latent_charge_template(path)


def test_tile_across_monomers():
    charges = np.array([0.3, -0.1, -0.1, -0.05, -0.05])
    tiled = tile_latent_charge_template(_template(charges), 4)
    assert tiled.shape == (20,)
    assert np.allclose(tiled[:5], charges)
    assert np.allclose(tiled[5:10], charges)
    assert np.allclose(tiled[15:20], charges)
    # A homogeneous tiled liquid box stays net-neutral.
    assert tiled.sum() == pytest.approx(0.0, abs=1e-10)


def test_tile_accepts_raw_array():
    charges = np.array([0.2, -0.2])
    tiled = tile_latent_charge_template(charges, 3)
    assert np.allclose(tiled, [0.2, -0.2, 0.2, -0.2, 0.2, -0.2])
