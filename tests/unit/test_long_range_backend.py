"""Unit tests for long-range electrostatic backend selection."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.long_range_backend import (
    LongRangeCoulombResult,
    collect_lr_solver_mapping,
    create_lr_solver,
    describe_lr_solver,
    pick_lr_solver,
    resolve_lr_solver,
)


def test_resolve_lr_solver_accepts_known_names():
    assert resolve_lr_solver("mic") == "mic"
    assert resolve_lr_solver("scafacos") == "scafacos"
    assert resolve_lr_solver(None) == "auto"


def test_resolve_lr_solver_rejects_unknown():
    with pytest.raises(ValueError, match="lr_solver must be"):
        resolve_lr_solver("ewald")


def test_pick_lr_solver_auto_prefers_scafacos():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_scafacos",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_jax_pme",
        return_value=True,
    ):
        assert pick_lr_solver("auto") == "scafacos"


def test_pick_lr_solver_auto_falls_back_to_jax_pme():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_scafacos",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_jax_pme",
        return_value=True,
    ):
        assert pick_lr_solver("auto") == "jax_pme"


def test_pick_lr_solver_auto_falls_back_to_mic():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_scafacos",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_jax_pme",
        return_value=False,
    ):
        assert pick_lr_solver("auto") == "mic"


def test_pick_lr_solver_requested_scafacos_missing_falls_back():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_scafacos",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_jax_pme",
        return_value=False,
    ):
        assert pick_lr_solver("scafacos") == "mic"


def test_create_lr_solver_mic():
    solver = create_lr_solver("mic")
    assert solver.name == "mic"


def test_create_lr_solver_jax_pme():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.pick_lr_solver",
        return_value="jax_pme",
    ):
        solver = create_lr_solver("jax_pme")
    assert solver.name == "jax_pme"


def test_create_lr_solver_scafacos_delegates():
    fake = LongRangeCoulombResult(
        energy_kcalmol=1.0,
        forces_kcalmol_A=np.zeros((2, 3)),
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.pick_lr_solver",
        return_value="scafacos",
    ), mock.patch(
        "mmml.interfaces.scafacosInterface.scafacos_session.compute_scafacos_coulomb",
        return_value=mock.Mock(energy_kcalmol=1.0, forces_kcalmol_A=np.zeros((2, 3))),
    ) as compute:
        solver = create_lr_solver("scafacos")
        out = solver.compute(
            np.zeros((2, 3)),
            np.array([1.0, -1.0]),
            box_length_A=20.0,
        )
    compute.assert_called_once()
    assert out.energy_kcalmol == pytest.approx(1.0)


def test_describe_lr_solver_includes_flags():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.pick_lr_solver",
        return_value="mic",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_scafacos",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_jax_pme",
        return_value=True,
    ):
        text = describe_lr_solver()
    assert "lr_solver=mic" in text
    assert "scafacos=no" in text
    assert "jax_pme=yes" in text


def test_collect_lr_solver_mapping_jax_pme():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.pick_lr_solver",
        return_value="jax_pme",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.resolve_lr_solver",
        return_value="jax_pme",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_scafacos",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_jax_pme",
        return_value=True,
    ):
        mapping = collect_lr_solver_mapping(
            lr_solver="jax_pme",
            jax_pme_method="pme",
            jax_pme_sr_cutoff_A=7.5,
        )
    assert mapping["lr_solver"] == "jax_pme"
    assert mapping["jax_pme_method"] == "pme"
    assert mapping["jax_pme_sr_cutoff_Å"] == "7.5"
    assert mapping["coulomb_mode"] == "jax-pme k-space + pair SR"
    assert "lr_solver_requested" not in mapping


def test_collect_lr_solver_mapping_auto_fallback():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.pick_lr_solver",
        return_value="mic",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.resolve_lr_solver",
        return_value="auto",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_scafacos",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.have_jax_pme",
        return_value=False,
    ):
        mapping = collect_lr_solver_mapping(lr_solver="auto")
    assert mapping["lr_solver"] == "mic"
    assert mapping["lr_solver_requested"] == "auto"
    assert mapping["coulomb_mode"] == "truncated MIC (pair loop)"
