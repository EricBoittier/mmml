"""Unit tests for CHARMM ↔ MLpot atom identity helpers (no CHARMM runtime)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
    DecomposedMlpotModel,
    _DeferredDecomposedMlpotCalculator,
)
from mmml.interfaces.pycharmmInterface.mlpot.setup import (
    MlpotContext,
    _calculator_atomic_numbers,
    _masses_consistent_with_z,
    verify_mlpot_charmm_atom_consistency,
)


def test_masses_consistent_with_z_accepts_matching_masses():
    import ase.data

    z = np.array([6, 1, 17], dtype=int)
    masses = ase.data.atomic_masses_common[z]
    assert _masses_consistent_with_z(masses, z) == []


def test_masses_consistent_with_z_accepts_cgenff_chlorine_mass():
    """CGenFF Cl mass (35.453) differs from ASE (34.969) but still maps to Z=17."""
    z = np.array([6, 1, 17, 17], dtype=int)
    masses = np.array([12.011, 1.008, 35.453, 35.453], dtype=float)
    assert _masses_consistent_with_z(masses, z) == []


def test_masses_consistent_with_z_flags_wrong_z_assignment():
    z = np.array([6, 1], dtype=int)
    masses = np.array([12.0, 5.0], dtype=float)
    issues = _masses_consistent_with_z(masses, z)
    assert len(issues) == 1
    assert "atom 1" in issues[0]
    assert "best matches" in issues[0]


def test_deferred_calculator_exposes_atomic_numbers_before_first_ener():
    z = np.array([6, 1, 1, 1, 6, 1, 1, 1], dtype=int)
    model = DecomposedMlpotModel(
        None,
        CutoffParameters(),
        2,
        z,
        pending_factory=MagicMock(),
        pending_factory_z=z,
        defer_jax_until_after_sd=True,
    )
    deferred = _DeferredDecomposedMlpotCalculator(model, ml_atomic_numbers=z)
    np.testing.assert_array_equal(deferred.ml_atomic_numbers, z)
    np.testing.assert_array_equal(deferred.atomic_numbers, z.astype(np.int32))


def test_calculator_atomic_numbers_reads_deferred_wrapper():
    z = np.array([6, 1, 1, 1], dtype=int)
    model = DecomposedMlpotModel(
        None,
        CutoffParameters(),
        1,
        z,
        pending_factory=MagicMock(),
        pending_factory_z=z,
        defer_jax_until_after_sd=True,
    )
    deferred = _DeferredDecomposedMlpotCalculator(model, ml_atomic_numbers=z)
    mlpot = MagicMock()
    mlpot.calculator = deferred
    ctx = MlpotContext(
        mlpot=mlpot,
        pyCModel=model,
        params=None,
        model=None,
        ml_Z=z,
    )
    z_calc = _calculator_atomic_numbers(ctx)
    assert z_calc is not None
    np.testing.assert_array_equal(z_calc, z)


def test_verify_mlpot_charmm_atom_consistency_accepts_deferred_calculator(monkeypatch):
    import sys
    import types

    z = np.array([6, 1, 1, 1, 6, 1, 1, 1], dtype=int)
    n = len(z)
    model = DecomposedMlpotModel(
        None,
        CutoffParameters(),
        2,
        z,
        pending_factory=MagicMock(),
        pending_factory_z=z,
        defer_jax_until_after_sd=True,
    )
    deferred = _DeferredDecomposedMlpotCalculator(model, ml_atomic_numbers=z)

    mlpot = MagicMock()
    mlpot.calculator = deferred
    mlpot.ml_Z = z
    mlpot.ml_indices = np.arange(n, dtype=int)
    mlpot.ml_Natoms = n

    ctx = MlpotContext(
        mlpot=mlpot,
        pyCModel=model,
        params=None,
        model=None,
        ml_Z=z,
    )

    import ase.data

    fake_psf = types.ModuleType("pycharmm.psf")
    fake_psf.get_amass = lambda: ase.data.atomic_masses_common[z]
    fake_psf.get_atype = lambda: np.array(["C", "H", "H", "H"] * 2, dtype=str)

    fake_coor = types.ModuleType("pycharmm.coor")
    fake_coor.get_natom = lambda: n

    fake_pycharmm = types.ModuleType("pycharmm")
    fake_pycharmm.psf = fake_psf
    fake_pycharmm.coor = fake_coor

    monkeypatch.setitem(sys.modules, "pycharmm", fake_pycharmm)
    monkeypatch.setitem(sys.modules, "pycharmm.psf", fake_psf)
    monkeypatch.setitem(sys.modules, "pycharmm.coor", fake_coor)
    monkeypatch.setitem(
        sys.modules,
        "mmml.interfaces.pycharmmInterface.import_pycharmm",
        types.ModuleType("mmml.interfaces.pycharmmInterface.import_pycharmm"),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.utils.get_Z_from_psf",
        lambda: z.copy(),
    )

    verify_mlpot_charmm_atom_consistency(
        ctx,
        context="staged dynamics",
        quiet=True,
    )
