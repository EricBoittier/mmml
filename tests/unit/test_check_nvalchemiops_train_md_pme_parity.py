"""Unit tests for train↔MD nvalchemiops PME parity helpers."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from mmml.data.units import KCAL_MOL_TO_EV
from mmml.interfaces.pycharmmInterface.long_range_backend import (
    CHARMM_COULOMB_KCAL,
    LongRangeInteractionResult,
)


def _load_script_module():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / (
        "scripts/check_nvalchemiops_train_md_pme_parity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_nvalchemiops_train_md_pme_parity", path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compare_pme_kernel_reports_zero_diff_when_paths_agree():
    mod = _load_script_module()
    pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
    q = np.array([1.0, -1.0], dtype=np.float64)
    mid = np.array([0, 1], dtype=np.int32)
    e_kcal = 12.5

    with mock.patch(
        "mmml.models.nvalchemiops_hybrid_coulomb.hybrid_nvalchemiops_pme_coulomb_energy",
        return_value=e_kcal,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.compute_nvalchemiops_pme_coulomb",
        return_value=LongRangeInteractionResult(
            energy_kcalmol=e_kcal,
            forces_kcalmol_A=np.zeros((2, 3)),
        ),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.estimate_nvalchemiops_pme_real_space_cutoff",
        return_value=10.0,
    ):
        out = mod.compare_pme_kernel_kcalmol(
            pos, q, mid, box_length_A=30.0, accuracy=1e-4
        )
    assert out["abs_diff_kcalmol"] == pytest.approx(0.0)
    assert out["e_train_kcalmol"] == pytest.approx(e_kcal)
    assert out["e_md_kcalmol"] == pytest.approx(e_kcal)


def test_compare_hybrid_emm_matches_md_coulomb_with_stub_model(tmp_path):
    mod = _load_script_module()
    n = 4
    data = {
        "Z": np.array([[6, 1, 6, 1]], dtype=np.int32),
        "R": np.array(
            [[[0.0, 0, 0], [1.0, 0, 0], [6.0, 0, 0], [7.0, 0, 0]]],
            dtype=np.float64,
        ),
        "mol_id": np.array([[0, 0, 1, 1]], dtype=np.int32),
        "cgenff_charge": np.array([[0.5, -0.5, -0.5, 0.5]], dtype=np.float64),
        "cgenff_type_idx": np.array([[0, 1, 0, 1]], dtype=np.int32),
        "cgenff_master_sigmas": np.array([3.5, 2.5]),
        "cgenff_master_epsilons": np.array([0.1, 0.05]),
        "N": np.array([n]),
    }
    e_kcal = -3.25

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.compute_nvalchemiops_pme_coulomb",
        return_value=LongRangeInteractionResult(
            energy_kcalmol=e_kcal,
            forces_kcalmol_A=np.zeros((n, 3)),
        ),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.long_range_backend.estimate_nvalchemiops_pme_real_space_cutoff",
        return_value=9.0,
    ), mock.patch(
        "mmml.models.nvalchemiops_hybrid_coulomb.nvalchemiops_pme_coulomb_energy_jax",
        return_value=e_kcal,
    ):
        out = mod.compare_hybrid_emm_eV(
            data,
            0,
            box_length_A=30.0,
            accuracy=1e-4,
            checkpoint=None,
        )
    assert out["e_mm_train_eV"] == pytest.approx(e_kcal * KCAL_MOL_TO_EV, rel=1e-10)
    assert out["e_md_coulomb_eV"] == pytest.approx(e_kcal * KCAL_MOL_TO_EV, rel=1e-10)
    assert out["abs_diff_eV"] == pytest.approx(0.0, abs=1e-12)


def test_cli_help_mentions_layers():
    mod = _load_script_module()
    assert "kernel" in mod.__doc__.lower()
    assert "e_mm" in mod.__doc__
    assert CHARMM_COULOMB_KCAL > 0
