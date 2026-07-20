"""Unit tests for pressure-targeted cubic box optimization (no CHARMM)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.box_pressure_opt import (
    BoxPressureOptConfig,
    build_cpt_box_refine_dynamics_kw,
    pressure_objective,
    refine_box_side_1d,
    run_box_pressure_opt,
    run_mc_pressure_box_moves,
    write_box_pressure_opt_json,
)


def _synthetic_pressure_fn(k_atm_A3: float = 2.7e4):
    """``P = k / L^3`` (atm) — unique L for any positive target pressure."""

    def _fn(_positions: np.ndarray, box_side_A: float) -> float:
        L = float(box_side_A)
        return float(k_atm_A3) / (L**3)

    return _fn


def _two_monomer_positions(box_A: float = 12.0) -> tuple[np.ndarray, list[int]]:
    # Two well-separated monomers (3 atoms each) in a cubic cell.
    half = 0.5 * float(box_A)
    pos = np.array(
        [
            [half - 1.5, half, half],
            [half - 0.5, half, half],
            [half - 1.5, half + 0.9, half],
            [half + 1.5, half, half],
            [half + 2.5, half, half],
            [half + 1.5, half + 0.9, half],
        ],
        dtype=np.float64,
    )
    return pos, [3, 3]


def test_pressure_objective_scale():
    assert pressure_objective(3.0, target_pressure_atm=1.0, scale_atm=2.0) == pytest.approx(
        1.0
    )


def test_mc_moves_box_toward_pressure_target():
    k = 2.7e4
    target_p = 1.0
    target_L = (k / target_p) ** (1.0 / 3.0)
    pos, apl = _two_monomer_positions(20.0)
    cfg = BoxPressureOptConfig(
        target_pressure_atm=target_p,
        mc_steps=120,
        mc_step_scale=0.05,
        mc_temperature=0.5,
        mc_min_scale=0.4,
        mc_max_scale=1.2,
        run_1d_refine=False,
        seed=11,
        min_intermonomer_distance_A=0.1,
    )
    new_pos, new_L, summary = run_mc_pressure_box_moves(
        pos,
        atoms_per_list=apl,
        box_side_A=20.0,
        pressure_fn=_synthetic_pressure_fn(k),
        config=cfg,
    )
    assert summary.ran
    assert new_pos.shape == pos.shape
    assert abs(new_L - target_L) < abs(20.0 - target_L)
    assert abs(summary.final_pressure_atm - target_p) < abs(
        summary.initial_pressure_atm - target_p
    )


def test_1d_refine_improves_pressure_match():
    k = 2.7e4
    target_p = 1.0
    target_L = (k / target_p) ** (1.0 / 3.0)
    # Start slightly off the MC-ish optimum.
    L0 = target_L * 1.05
    pos, apl = _two_monomer_positions(L0)
    cfg = BoxPressureOptConfig(
        target_pressure_atm=target_p,
        run_1d_refine=True,
        refine_1d_bracket_frac=0.12,
        refine_1d_max_iters=30,
        refine_1d_tol_A=1.0e-4,
        min_intermonomer_distance_A=0.1,
    )
    new_pos, new_L, summary = refine_box_side_1d(
        pos,
        atoms_per_list=apl,
        box_side_A=L0,
        pressure_fn=_synthetic_pressure_fn(k),
        config=cfg,
    )
    assert summary.ran
    assert new_pos.shape == pos.shape
    assert abs(new_L - target_L) < abs(L0 - target_L)
    assert abs(summary.final_pressure_atm - target_p) < abs(
        summary.initial_pressure_atm - target_p
    )


def test_full_opt_writes_certified_box_json(tmp_path: Path):
    k = 2.7e4
    target_p = 1.0
    pos, apl = _two_monomer_positions(18.0)
    cfg = BoxPressureOptConfig(
        target_pressure_atm=target_p,
        mc_steps=80,
        mc_step_scale=0.04,
        seed=3,
        run_1d_refine=True,
        run_cpt_refine=True,
        min_intermonomer_distance_A=0.1,
    )

    def _fake_cpt(p, L):
        # CPT "refine" nudges L 1% toward the analytic optimum.
        target_L = (k / target_p) ** (1.0 / 3.0)
        L2 = 0.99 * float(L) + 0.01 * target_L
        return p, L2, {"ran": True, "mean_box_A": L2, "reason": "fake_cpt"}

    new_pos, new_L, result = run_box_pressure_opt(
        pos,
        atoms_per_list=apl,
        box_side_A=18.0,
        pressure_fn=_synthetic_pressure_fn(k),
        config=cfg,
        composition="TIP3:2",
        cpt_refine_fn=_fake_cpt,
        output_dir=tmp_path,
    )
    assert result.status == "pass"
    assert "mc_pressure" in result.steps_applied
    assert "refine_1d" in result.steps_applied
    assert "cpt_refine" in result.steps_applied
    assert result.box_json_path is not None
    payload = json.loads(result.box_json_path.read_text(encoding="utf-8"))
    assert payload["box_side_A"] == pytest.approx(new_L)
    assert payload["final_cubic_side_A"] == pytest.approx(new_L)
    assert payload["target_pressure_atm"] == pytest.approx(1.0)
    assert abs(payload["final_pressure_atm"] - target_p) < 0.2


def test_cpt_plan_includes_pref():
    cfg = BoxPressureOptConfig(target_pressure_atm=1.5, cpt_nstep=250)
    kw = build_cpt_box_refine_dynamics_kw(cfg)
    assert kw["nstep"] == 250
    assert kw["cpt"] is True
    assert "pint pconst pref" in kw
    assert kw["pint pconst pref"] == pytest.approx(1.5)


def test_write_box_json_roundtrip(tmp_path: Path):
    from mmml.interfaces.pycharmmInterface.mlpot.box_pressure_opt import (
        BoxPressureOptResult,
    )

    result = BoxPressureOptResult(
        status="pass",
        composition="TIP3:90",
        n_molecules=90,
        n_atoms=270,
        box_side_A=29.5,
        final_cubic_side_A=29.5,
        target_pressure_atm=1.0,
        final_pressure_atm=1.2,
        temperature_K=300.0,
        steps_applied=["mc_pressure"],
    )
    path = write_box_pressure_opt_json(result, tmp_path / "box.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["final_cubic_side_A"] == pytest.approx(29.5)
    assert data["composition"] == "TIP3:90"
