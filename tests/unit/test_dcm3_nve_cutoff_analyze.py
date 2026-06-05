"""Unit tests for DCM:3 NVE cutoff sweep analysis (no CHARMM)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_ANALYZE = _REPO / "workflows" / "dcm3_nve_cutoff_sweep" / "scripts" / "analyze_nve_energy.py"


def _load_analyze():
    spec = importlib.util.spec_from_file_location("analyze_nve_energy", _ANALYZE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_dyna_lines() -> None:
    mod = _load_analyze()
    text = "\n".join(
        [
            "DYNA>         100       0.0250   -1234.5678901    12.3456789   -1246.9135690   298.1234",
            "DYNA>         200       0.0500   -1234.5600000    12.3400000   -1246.9000000   297.9000",
        ]
    )
    rows = mod.parse_dyna_lines(text)
    assert len(rows) == 2
    assert rows[0]["step"] == 100.0
    assert rows[1]["total_energy_kcal"] == -1234.56


def test_summarize_nve_energy_smoothness() -> None:
    mod = _load_analyze()
    smooth = [
        {
            "total_energy_kcal": -100.0 + 1e-4 * i,
            "potential_energy_kcal": -110.0,
            "kinetic_energy_kcal": 10.0,
            "temperature_K": 300.0,
            "time_ps": 0.001 * i,
            "step": float(i),
        }
        for i in range(100)
    ]
    noisy = [dict(r) for r in smooth]
    noisy[50]["total_energy_kcal"] += 5.0

    s_smooth = mod.summarize_nve_energy(smooth)
    s_noisy = mod.summarize_nve_energy(noisy)
    assert s_smooth["smoothness_score"] < s_noisy["smoothness_score"]
    assert s_noisy["max_abs_etot_step_delta_kcal"] >= 4.9
    assert s_smooth["status"] == "pass"


def test_classify_energy_catastrophe() -> None:
    mod = _load_analyze()
    rows = [
        {
            "total_energy_kcal": -100.0,
            "potential_energy_kcal": -110.0,
            "kinetic_energy_kcal": 10.0,
            "temperature_K": 300.0,
            "time_ps": 0.0,
            "step": 0.0,
        },
        {
            "total_energy_kcal": 1.0e9,
            "potential_energy_kcal": 1.0e9,
            "kinetic_energy_kcal": 0.0,
            "temperature_K": 300.0,
            "time_ps": 0.001,
            "step": 1.0,
        },
    ]
    summary = mod.summarize_nve_energy(rows, catastrophe_score=10000.0)
    assert summary["status"] == "fail_energy"
    assert "catastrophe threshold" in summary["notes"]


def test_cutoff_lib_preset_geometry_keys() -> None:
    sys.path.insert(0, str(_REPO / "workflows" / "dcm3_nve_cutoff_sweep" / "scripts"))
    from cutoff_lib import (
        expected_nve_nstep,
        geometry_ids,
        load_config,
        preset_ids,
        workflow_root,
    )

    cfg = load_config(workflow_root() / "config.yaml")
    presets = preset_ids(cfg)
    assert "extended_mm5" in presets
    assert "dcm9_stability" in presets
    assert "dcm9_ml01" in presets
    assert "dcm9_ml001" in presets
    assert "handoff_7p5" in presets
    assert "extended_mm5_ml05" in presets
    assert "handoff_7p5_ml15" in presets
    assert "code_default" not in presets
    assert "mid" in geometry_ids(cfg)
    assert len(presets) == 7
    assert len(geometry_ids(cfg)) == 4
    assert expected_nve_nstep(cfg) == 20000
    assert int(cfg["dynamics_overlap_check_interval"]) >= expected_nve_nstep(cfg)
    assert float(cfg["energy_catastrophe_score"]) == 10000.0


def test_collect_sweep_flags_fail_energy(tmp_path) -> None:
    sys.path.insert(0, str(_REPO / "workflows" / "dcm3_nve_cutoff_sweep" / "scripts"))
    from collect_sweep import _effective_status, _preset_mean_scores

    row = {"status": "pass", "smoothness_score": 500.0, "preset_id": "good"}
    assert _effective_status(row, 10000.0) == "pass"
    bad = {"status": "pass", "smoothness_score": 1.0e9, "preset_id": "bad"}
    assert _effective_status(bad, 10000.0) == "fail_energy"

    rows = [
        {"preset_id": "a", "status": "pass", "smoothness_score": 100.0},
        {"preset_id": "a", "status": "fail_energy", "smoothness_score": 1.0e9},
        {"preset_id": "b", "status": "pass", "smoothness_score": 200.0},
    ]
    sane = _preset_mean_scores(rows, exclude_statuses=frozenset({"fail_energy"}))
    assert sane["a"][0] == 100.0
    assert sane["b"][0] == 200.0


def test_collect_sweep_write_report(tmp_path) -> None:
    sys.path.insert(0, str(_REPO / "workflows" / "dcm3_nve_cutoff_sweep" / "scripts"))
    from collect_sweep import _write_report

    rows = [
        {
            "preset_id": "dcm9_stability",
            "geom_id": "far",
            "mm_switch_on": 7.0,
            "mm_switch_width": 5.0,
            "ml_switch_width": 0.1,
            "etot_std_kcal": 100.0,
            "max_abs_etot_step_delta_kcal": 20.0,
            "etot_drift_kcal": 50.0,
            "smoothness_score": 125.0,
            "status": "pass",
        },
        {
            "preset_id": "handoff_7p5",
            "geom_id": "far",
            "mm_switch_on": 7.5,
            "mm_switch_width": 5.0,
            "ml_switch_width": 0.1,
            "etot_std_kcal": 1.0e9,
            "max_abs_etot_step_delta_kcal": 1.0e9,
            "etot_drift_kcal": 1.0e9,
            "smoothness_score": 1.0e9,
            "status": "fail_energy",
        },
    ]
    cfg = {
        "composition": "DCM:3",
        "ps_nve": 5.0,
        "dt_fs": 0.25,
        "geometry_variants": {"far": {}},
    }
    md_path = tmp_path / "report.md"
    _write_report(rows, md_path, cfg, catastrophe=10000.0)
    text = md_path.read_text(encoding="utf-8")
    assert "excluding `fail_energy`" in text
    assert "**Suggested preset (lowest sane mean, all 1 geoms):** `dcm9_stability`" in text


def test_collect_sweep_suggestion_requires_all_geoms(tmp_path) -> None:
    sys.path.insert(0, str(_REPO / "workflows" / "dcm3_nve_cutoff_sweep" / "scripts"))
    from collect_sweep import _write_report

    rows = [
        {
            "preset_id": "partial",
            "geom_id": "close",
            "mm_switch_on": 8.0,
            "mm_switch_width": 5.0,
            "ml_switch_width": 0.5,
            "etot_std_kcal": 10.0,
            "max_abs_etot_step_delta_kcal": 1.0,
            "etot_drift_kcal": 1.0,
            "smoothness_score": 11.1,
            "status": "pass",
        },
        {
            "preset_id": "partial",
            "geom_id": "mid",
            "mm_switch_on": 8.0,
            "mm_switch_width": 5.0,
            "ml_switch_width": 0.5,
            "etot_std_kcal": 1.0e9,
            "max_abs_etot_step_delta_kcal": 1.0e9,
            "etot_drift_kcal": 1.0e9,
            "smoothness_score": 1.0e9,
            "status": "fail_energy",
        },
        {
            "preset_id": "robust",
            "geom_id": "close",
            "mm_switch_on": 8.0,
            "mm_switch_width": 5.0,
            "ml_switch_width": 1.5,
            "etot_std_kcal": 100.0,
            "max_abs_etot_step_delta_kcal": 20.0,
            "etot_drift_kcal": 50.0,
            "smoothness_score": 125.0,
            "status": "pass",
        },
        {
            "preset_id": "robust",
            "geom_id": "mid",
            "mm_switch_on": 8.0,
            "mm_switch_width": 5.0,
            "ml_switch_width": 1.5,
            "etot_std_kcal": 100.0,
            "max_abs_etot_step_delta_kcal": 20.0,
            "etot_drift_kcal": 50.0,
            "smoothness_score": 125.0,
            "status": "pass",
        },
    ]
    cfg = {
        "composition": "DCM:3",
        "ps_nve": 5.0,
        "dt_fs": 0.25,
        "geometry_variants": {"close": {}, "mid": {}},
        "cutoff_presets": {"partial": {}, "robust": {}},
    }
    md_path = tmp_path / "report.md"
    _write_report(rows, md_path, cfg, catastrophe=10000.0)
    text = md_path.read_text(encoding="utf-8")
    assert "**Suggested preset (lowest sane mean, all 2 geoms):** `robust`" in text
    assert "failed at least one geometry" in text


def test_prepare_geometry_shell_uses_mpirun_wrapper_for_script() -> None:
    sh = (
        _REPO
        / "workflows"
        / "dcm3_nve_cutoff_sweep"
        / "scripts"
        / "prepare_geometry_shell.sh"
    )
    text = sh.read_text(encoding="utf-8")
    assert "mmml_resolve_env" in text
    assert 'exec "$MPIRUN" "$WORKFLOW_ROOT/scripts/prepare_geometry.py"' in text
    assert '"$MPIRUN" "$PY"' not in text


def test_prepare_geometry_does_not_setup_nbonds_before_cluster_build() -> None:
    py = (
        _REPO
        / "workflows"
        / "dcm3_nve_cutoff_sweep"
        / "scripts"
        / "prepare_geometry.py"
    )
    text = py.read_text(encoding="utf-8")
    build_idx = text.index("build_cluster_from_args_with_tag")
    assert "setup_default_nbonds" not in text[:build_idx]
