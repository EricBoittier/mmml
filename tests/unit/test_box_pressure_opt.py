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
        mc_steps=160,
        mc_step_scale=0.06,
        mc_temperature=0.3,
        mc_min_scale=0.5,
        mc_max_scale=1.2,
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
    assert abs(payload["final_pressure_atm"] - target_p) < abs(
        _synthetic_pressure_fn(k)(pos, 18.0) - target_p
    )


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


def test_from_box_json_offline_smoke(tmp_path: Path):
    from mmml.interfaces.pycharmmInterface.mlpot.box_pressure_opt import (
        run_box_pressure_opt_from_box_json,
    )

    liquid = tmp_path / "liquid_box"
    liquid.mkdir()
    (liquid / "box.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "composition": "TIP3:8",
                "n_molecules": 8,
                "n_atoms": 24,
                "box_side_A": 12.0,
                "final_cubic_side_A": 12.0,
            }
        ),
        encoding="utf-8",
    )
    cfg = BoxPressureOptConfig(
        target_pressure_atm=1.0,
        mc_steps=40,
        seed=5,
        run_1d_refine=True,
        min_intermonomer_distance_A=0.05,
    )
    result = run_box_pressure_opt_from_box_json(
        liquid,
        output_dir=tmp_path / "box_pressure_opt",
        config=cfg,
    )
    assert result.status == "pass"
    assert result.box_json_path is not None
    assert result.box_json_path.is_file()
    assert result.pressure_source == "synthetic_inverse_cube"
    # Synthetic model is calibrated to certified L → stay near 12 Å.
    assert result.box_side_A == pytest.approx(12.0, rel=0.05)


def test_from_box_json_use_charmm_dispatches(monkeypatch, tmp_path: Path):
    from mmml.interfaces.pycharmmInterface.mlpot import box_pressure_opt as bpo
    from mmml.interfaces.pycharmmInterface.mlpot.box_pressure_opt import (
        BoxPressureOptResult,
        run_box_pressure_opt_from_box_json,
    )

    called = {}

    def _fake_live(liquid_box_dir, *, output_dir=None, config=None):
        called["liquid"] = str(liquid_box_dir)
        called["output"] = str(output_dir) if output_dir is not None else None
        called["cpt"] = bool(config.run_cpt_refine) if config is not None else None
        out = Path(output_dir or tmp_path / "opt")
        out.mkdir(parents=True, exist_ok=True)
        path = out / "box.json"
        result = BoxPressureOptResult(
            status="pass",
            composition="TIP3:2",
            n_molecules=2,
            n_atoms=6,
            box_side_A=30.0,
            final_cubic_side_A=30.0,
            target_pressure_atm=1.0,
            final_pressure_atm=1.1,
            temperature_K=300.0,
            pressure_source="charmm_prsi",
            steps_applied=["mc_pressure", "cpt_refine"],
            artifacts={"model_psf": str(out / "model.psf"), "model_crd": str(out / "model.crd")},
        )
        path.write_text(json.dumps(result.to_box_json()), encoding="utf-8")
        result.box_json_path = path
        return result

    monkeypatch.setattr(bpo, "run_box_pressure_opt_charmm_live", _fake_live)
    liquid = tmp_path / "liquid_box"
    liquid.mkdir()
    (liquid / "box.json").write_text("{}", encoding="utf-8")
    cfg = BoxPressureOptConfig(run_cpt_refine=True, mc_steps=8)
    result = run_box_pressure_opt_from_box_json(
        liquid,
        output_dir=tmp_path / "box_pressure_opt",
        config=cfg,
        use_charmm_pressure=True,
    )
    assert called["cpt"] is True
    assert result.pressure_source == "charmm_prsi"
    assert "cpt_refine" in result.steps_applied


def test_make_charmm_cpt_refine_returns_mean_L(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.box_pressure_opt import (
        make_charmm_cpt_box_refine_fn,
    )
    import mmml.interfaces.pycharmmInterface.mlpot.dynamics as dyn
    import mmml.interfaces.pycharmmInterface.mlpot.pbc_env as pbc
    import mmml.interfaces.pycharmmInterface.mlpot.setup as setup

    sides = [30.0, 29.8, 29.9, 30.1, 30.0]
    calls = {"dyn": 0, "idx": 0}
    pos0, apl = _two_monomer_positions(30.0)

    def _push(side, quiet=True):
        calls["pushed"] = float(side)

    def _sync(pos):
        calls["synced"] = np.asarray(pos).shape

    def _get_pos():
        return pos0.copy()

    def _get_side(*, fallback_side_A=None):
        i = min(calls["idx"], len(sides) - 1)
        calls["idx"] += 1
        return float(sides[i])

    def _build_cpt(**kwargs):
        nstep = max(1, int(round(float(kwargs["duration_ps"]) / float(kwargs["timestep_ps"]))))
        return {
            "nstep": nstep,
            "timestep": float(kwargs["timestep_ps"]),
            "finalt": float(kwargs["temp"]),
            "cpt": True,
            "nsavc": 0,
        }

    def _run_dyn(kw):
        calls["dyn"] += 1
        assert int(kw["nstep"]) >= 1
        if calls["dyn"] == 1:
            assert kw.get("start") is True
            assert int(kw.get("iasvel", 0)) == 1
        else:
            assert kw.get("start") is False
            assert int(kw.get("iasvel", 0)) == 0

    monkeypatch.setattr(dyn, "build_cpt_equilibration_dynamics", _build_cpt)
    monkeypatch.setattr(dyn, "run_dynamics", _run_dyn)
    monkeypatch.setattr(pbc, "push_charmm_cubic_box_side_A", _push)
    monkeypatch.setattr(pbc, "get_charmm_cubic_box_side_A", _get_side)
    monkeypatch.setattr(setup, "sync_charmm_positions", _sync)
    monkeypatch.setattr(setup, "get_charmm_positions_array", _get_pos)

    cfg = BoxPressureOptConfig(
        cpt_nstep=25,
        cpt_timestep_ps=0.001,
        cpt_l_samples=5,
        target_pressure_atm=1.0,
        temperature_K=300.0,
    )
    refine = make_charmm_cpt_box_refine_fn(cfg, atoms_per_list=apl)
    new_pos, mean_L, summary = refine(pos0, 30.0)
    assert calls["dyn"] == 5
    assert new_pos.shape == pos0.shape
    assert mean_L == pytest.approx(float(np.mean(sides)))
    assert summary["ran"] is True
    assert summary["mean_box_A"] == pytest.approx(mean_L)
    assert len(summary["box_samples_A"]) == 5
