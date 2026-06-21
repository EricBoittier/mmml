"""Campaign monitor / failure taxonomy for pbc_solvent_burst."""

from __future__ import annotations

import sys
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows" / "pbc_solvent_burst"
SCRIPTS = WORKFLOW / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from monitor_lib import classify_failure, extract_campaign_markers  # noqa: E402


def test_classify_failure_grms_gate() -> None:
    text = "Post MLpot mini GRMS: 5.55\nPre-dynamics GRMS 175.00 > 50.00"
    assert "A_grms_gate" in classify_failure(text)


def test_classify_failure_handoff_fortran() -> None:
    text = "Fortran runtime error: Bad value during floating point read\ncontinue_seed.res"
    assert "C_handoff_fortran" in classify_failure(text)


def test_extract_campaign_markers_heat_memory() -> None:
    text = (
        "tier=12M max_Npr=240\n"
        "mini-nstep scaled 300 -> 515\n"
        "max_grms_before_dyn scaled 50.0 -> 75.0\n"
        "HEAT: (in-memory coords after mini) | memory handoff\n"
    )
    markers = extract_campaign_markers(text)
    assert markers["tier"] == "12M"
    assert markers["heat_handoff"] == "memory"
    assert "515" in markers["mini_nstep"]
