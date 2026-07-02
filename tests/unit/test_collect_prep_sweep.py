"""GRMS extraction for dcm_density_setup_compare prep sweep collector."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "workflows" / "dcm_density_setup_compare" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from collect_prep_sweep import _extract_grms_metrics  # noqa: E402


def test_extract_grms_from_plain_dashboard() -> None:
    text = """
Post MLpot SD pass 1
[State]
  Hybrid GRMS: 42.1250 kcal/mol/Å (0.1825 eV/Å)
  USER energy: -1234.5 kcal/mol
"""
    m = _extract_grms_metrics(text)
    assert m["post_mini_grms"] == "42.1250"
    assert m["gate_grms"] == "42.1250"
    assert m["under_50"] is True


def test_extract_grms_from_rich_stripped() -> None:
    text = (
        "\x1b[1;36mPost MLpot SD pass 1\x1b[0m\n"
        "│ Hybrid GRMS │ 88.5000 kcal/mol/Å (0.3836 eV/Å) │\n"
    )
    m = _extract_grms_metrics(text)
    assert m["post_mini_grms"] == "88.5000"
    assert m["under_50"] is False


def test_extract_grms_from_sd_partial() -> None:
    text = (
        "MLpot SD pass 1 partial: watchdog stopped further chunks; "
        "geometry restored (GRMS≈406.9370 kcal/mol/Å)\n"
    )
    m = _extract_grms_metrics(text)
    assert m["post_mini_grms"] == "406.9370"
    assert m["gate_grms"] == "406.9370"


def test_extract_grms_from_rich_table_row() -> None:
    text = """
╭────────────────────────────── Post MLpot mini ───────────────────────────────╮
│ │ ┃ Hybrid GRMS ┃ CHARMM GRMS ┃ USER energy ┃ Diag        ┃ Note         ┃ │ │
│ │ │ 41.2500 kcal/mol/Å (0.1788 eV/Å) │ 193.4801 kcal/mol/Å (0.8391 eV/Å) │ │ │
╰──────────────────────────────────────────────────────────────────────────────╯
"""
    m = _extract_grms_metrics(text)
    assert m["post_mini_grms"] == "41.2500"
    assert m["gate_grms"] == "41.2500"


def test_extract_grms_baseline_stall_partial() -> None:
    text = """
MLpot SD pass 1 stalled: hybrid GRMS plateaued (GRMS≈452.6729 kcal/mol/Å); CHARMM SD ineffective
MLpot SD pass 1 partial: watchdog stopped further chunks; geometry restored (GRMS≈262.1532 kcal/mol/Å)
╭────────────────────────────── Post MLpot mini ───────────────────────────────╮
│ │ ┃ Hybrid GRMS ┃ CHARMM GRMS ┃ USER energy ┃ Diag        ┃ Note         ┃ │ │
│ │ │ 262.1532 kcal/mol/Å (1.1356 eV/Å) │ 0.0000 kcal/mol/Å (0.0000 eV/Å) │ │ │
╰──────────────────────────────────────────────────────────────────────────────╯
"""
    m = _extract_grms_metrics(text)
    assert m["sd_stall_grms"] == "452.6729"
    assert m["sd_partial_grms"] == "262.1532"
    assert m["post_mini_grms"] == "262.1532"
    assert m["under_50"] is False

def test_extract_grms_prefers_post_mini_over_partial() -> None:
    text = """
MLpot SD pass 1 partial: geometry restored (GRMS≈500.0000 kcal/mol/Å)
Post MLpot SD pass 1
  Hybrid GRMS: 35.0000 kcal/mol/Å (0.1515 eV/Å)
"""
    m = _extract_grms_metrics(text)
    assert m["post_mini_grms"] == "35.0000"


def test_extract_grms_pre_dynamics_gate() -> None:
    text = "Pre-dynamics GRMS OK: 12.3400 kcal/mol/Å (limit 50.0)\n"
    m = _extract_grms_metrics(text)
    assert m["pre_dynamics_grms"] == "12.3400"
    assert m["gate_grms"] == "12.3400"
    assert m["under_50"] is True
