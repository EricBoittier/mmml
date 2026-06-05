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
    assert "dcm9_stability" in presets
    assert "extended_handoff" in presets
    assert "extended_mm5" in presets
    assert "handoff_7p5" in presets
    assert "code_default" not in presets
    assert "wide_ml_taper" not in presets
    assert "mid" in geometry_ids(cfg)
    assert len(presets) == 4
    assert len(geometry_ids(cfg)) == 4
    assert expected_nve_nstep(cfg) == 4000
    assert int(cfg["dynamics_overlap_check_interval"]) >= expected_nve_nstep(cfg)


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
