"""Unit tests for long-range solver hybrid GRMS comparison."""

from __future__ import annotations

import math

import pytest

from mmml.interfaces.pycharmmInterface.lr_solver_grms_compare import (
    LrSolverGrmsRow,
    _best_grms_from_steps,
    validate_lr_solver_hybrid_grms,
)


def test_best_grms_from_steps_prefers_lowest() -> None:
    steps = [
        {"hybrid_grms_kcalmol_A": 42.0},
        {"grms_kcalmol_A": 12.5},
        {"hybrid_grms_kcalmol_A": 18.0},
    ]
    assert _best_grms_from_steps(steps) == pytest.approx(12.5)


def test_validate_jax_pme_methods_agree() -> None:
    rows = [
        LrSolverGrmsRow("jax_pme", jax_pme_method="ewald", status="ok", hybrid_grms_kcalmol_A=48.0),
        LrSolverGrmsRow("jax_pme", jax_pme_method="pme", status="ok", hybrid_grms_kcalmol_A=49.0),
        LrSolverGrmsRow("jax_pme", jax_pme_method="p3m", status="ok", hybrid_grms_kcalmol_A=47.5),
    ]
    result = validate_lr_solver_hybrid_grms(rows, jax_pme_rtol=0.10)
    assert result.ok
    assert any("jax_pme methods agree" in msg for msg in result.messages)


def test_validate_jax_pme_methods_fail_when_divergent() -> None:
    rows = [
        LrSolverGrmsRow("jax_pme", jax_pme_method="ewald", status="ok", hybrid_grms_kcalmol_A=48.0),
        LrSolverGrmsRow("jax_pme", jax_pme_method="pme", status="ok", hybrid_grms_kcalmol_A=80.0),
    ]
    result = validate_lr_solver_hybrid_grms(rows, jax_pme_rtol=0.10)
    assert not result.ok
    assert any("jax_pme pme" in msg for msg in result.messages)


def test_validate_reports_mic_vs_jax_pme_without_strict_match() -> None:
    rows = [
        LrSolverGrmsRow("mic", status="ok", hybrid_grms_kcalmol_A=35.0),
        LrSolverGrmsRow("jax_pme", jax_pme_method="ewald", status="ok", hybrid_grms_kcalmol_A=48.0),
    ]
    result = validate_lr_solver_hybrid_grms(rows)
    assert result.ok
    assert any("MIC GRMS=" in msg for msg in result.messages)


def test_validate_requires_finite_grms_on_successful_runs() -> None:
    rows = [
        LrSolverGrmsRow("mic", status="ok", hybrid_grms_kcalmol_A=None),
        LrSolverGrmsRow("jax_pme", jax_pme_method="ewald", status="ok", hybrid_grms_kcalmol_A=math.nan),
    ]
    result = validate_lr_solver_hybrid_grms(rows)
    assert not result.ok
    assert any("missing finite hybrid GRMS" in msg for msg in result.messages)
