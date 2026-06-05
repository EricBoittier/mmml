"""Scaling workflow inbfrq helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_WORKFLOW = Path(__file__).resolve().parents[2] / "workflows" / "dcm_nve_scaling" / "scripts"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from scaling_lib import (  # noqa: E402
    inbfrq_from_slug,
    inbfrq_slug,
    nve_inbfrq_values,
    run_variant_dir,
)


def test_inbfrq_slug_roundtrip():
    for v in (-1, 0, 1, 10, 50):
        s = inbfrq_slug(v)
        assert inbfrq_from_slug(s) == v


def test_run_variant_dir_inbfrq_zero():
    cfg = {"composition_prefix": "DCM", "output_root": "results"}
    p = run_variant_dir(cfg, 5, 0)
    assert p.name == "inbfrq_0"


def test_run_variant_dir_layout():
    cfg = {"composition_prefix": "DCM", "output_root": "results", "nve_inbfrq_values": [-1, 50]}
    p = run_variant_dir(cfg, 5, -1)
    assert p.name == "inbfrq_neg1"
    assert p.parent.name == "dcm_5_nve"


def test_nve_inbfrq_values_default():
    assert nve_inbfrq_values({}) == [-1]
