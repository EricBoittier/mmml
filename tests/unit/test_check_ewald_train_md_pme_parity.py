"""Unit tests for train↔MD native Ewald (+ optional switched LJ) parity helpers."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from mmml.data.units import KCAL_MOL_TO_EV
from mmml.interfaces.pycharmmInterface.long_range_backend import (
    LongRangeInteractionResult,
)


def _load_script_module():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / (
        "scripts/check_ewald_train_md_pme_parity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_ewald_train_md_pme_parity", path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dimer_data(*, sep: float = 3.5) -> dict:
    n = 4
    return {
        "Z": np.array([[6, 1, 6, 1]], dtype=np.int32),
        "R": np.array(
            [[[0.0, 0, 0], [1.0, 0, 0], [sep, 0, 0], [sep + 1.0, 0, 0]]],
            dtype=np.float64,
        ),
        "mol_id": np.array([[0, 0, 1, 1]], dtype=np.int32),
        "cgenff_charge": np.array([[0.5, -0.5, -0.5, 0.5]], dtype=np.float64),
        "cgenff_type_idx": np.array([[0, 1, 0, 1]], dtype=np.int32),
        "cgenff_master_sigmas": np.array([3.5, 2.5]),
        "cgenff_master_epsilons": np.array([0.1, 0.05]),
        "N": np.array([n]),
    }


def test_compare_ewald_kernel_reports_zero_diff_when_paths_agree():
    mod = _load_script_module()
    pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
    q = np.array([1.0, -1.0], dtype=np.float64)
    mid = np.array([0, 1], dtype=np.int32)
    e_kcal = 12.5

    with mock.patch(
        "mmml.models.ewald_hybrid_coulomb.hybrid_ewald_coulomb_energy",
        return_value=e_kcal,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.compute_native_ewald_coulomb",
        return_value=LongRangeInteractionResult(
            energy_kcalmol=e_kcal,
            forces_kcalmol_A=np.zeros((2, 3)),
        ),
    ):
        out = mod.compare_ewald_kernel_kcalmol(
            pos, q, mid, box_length_A=30.0, accuracy=1e-4
        )
    assert out["abs_diff_kcalmol"] == pytest.approx(0.0)
    assert out["e_train_kcalmol"] == pytest.approx(e_kcal)


def test_compare_hybrid_emm_coulomb_only_with_stub_model():
    mod = _load_script_module()
    data = _dimer_data(sep=6.0)
    e_kcal = -3.25
    n = 4

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.compute_native_ewald_coulomb",
        return_value=LongRangeInteractionResult(
            energy_kcalmol=e_kcal,
            forces_kcalmol_A=np.zeros((n, 3)),
        ),
    ), mock.patch(
        # hybrid_forward binds the symbol at import time
        "mmml.models.hybrid_energy.hybrid_ewald_coulomb_energy",
        return_value=e_kcal,
    ):
        out = mod.compare_hybrid_emm_eV(
            data, 0, box_length_A=30.0, accuracy=1e-4, include_lj=False
        )
    assert out["e_mm_train_eV"] == pytest.approx(e_kcal * KCAL_MOL_TO_EV, rel=1e-10)
    assert out["abs_diff_eV"] == pytest.approx(0.0, abs=1e-12)


def test_compare_hybrid_emm_ewald_plus_lj_real_agreement():
    """Unmocked tiny dimer: hybrid_forward e_mm == composed MD Ewald+LJ ref."""
    mod = _load_script_module()
    # Separation inside the MM-on window for defaults (handoff 6.5→8.0).
    data = _dimer_data(sep=7.0)
    out = mod.compare_hybrid_emm_eV(
        data,
        0,
        box_length_A=30.0,
        accuracy=1e-6,
        include_lj=True,
        mm_switch_on=8.0,
        mm_switch_width=5.0,
        ml_switch_width=1.5,
        complementary_handoff=True,
    )
    assert out["e_md_lj_kcalmol"] != 0.0
    assert out["abs_diff_eV"] < 1e-6


def test_cli_help_mentions_lj_layer():
    mod = _load_script_module()
    doc = (mod.__doc__ or "").lower()
    assert "switched lj" in doc or "include-lj" in doc
    assert "ewald" in doc
