"""Packmol solvation must pack into the volume that sized N from density.

Regression: ``determine_n_molecules_from_density`` sizes N from the cubic cell
volume L**3, but ``run_packmol_solvation`` packed the solvent into the inscribed
sphere, which holds only pi/6 = 52% of that volume.  Every density-sized build
therefore over-requested by ~1.9x and Packmol exited 173 ("failed to converge")
after burning its whole nloop budget -- independent of solvent.
"""

from __future__ import annotations

import math
from pathlib import Path

SETUP_BOX = Path("mmml/interfaces/pycharmmInterface/setupBox.py")


def _solvation_body() -> str:
    src = SETUP_BOX.read_text(encoding="utf-8")
    return src.split("def run_packmol_solvation")[1].split("\ndef ")[0]


def test_sphere_region_cannot_hold_a_cube_sized_count() -> None:
    """The geometric factor behind the bug: a sphere is ~52% of its bounding cube."""
    side = 30.0
    cube = side**3
    inscribed_sphere = (4.0 / 3.0) * math.pi * (side / 2.0) ** 3
    assert inscribed_sphere / cube < 0.53
    # ~1.9x over-request before the -0.5 A shell shrink is even applied.
    assert cube / inscribed_sphere > 1.9


def test_default_region_is_the_box() -> None:
    body = _solvation_body()
    assert 'region: str = PACKMOL_REGION' in body
    src = SETUP_BOX.read_text(encoding="utf-8")
    assert 'PACKMOL_REGION = "box"' in src


def test_box_region_fills_cell_outside_solute() -> None:
    """region='box' must emit `inside box` + a solute `outside sphere` exclusion."""
    body = _solvation_body()
    assert "inside box 0.0 0.0 0.0 {side_length} {side_length} {side_length}" in body
    assert "outside sphere {cx} {cy} {cz} {inner_radius}" in body


def test_count_is_clamped_to_region_capacity() -> None:
    """N must never exceed what the region can hold, or Packmol exits 173."""
    body = _solvation_body()
    assert "solvent_capacity(" in body
    assert "n_molecules = capacity" in body
    # The clamped count is returned so callers build a matching PSF.
    assert "return int(n_molecules)" in body


def test_callers_use_the_returned_count() -> None:
    cli = Path("mmml/cli/make/make_box.py").read_text(encoding="utf-8")
    assert "n_molecules = setupBox.run_packmol_solvation(" in cli
    src = SETUP_BOX.read_text(encoding="utf-8")
    main_body = src.split("\ndef main(density")[1].split("\ndef ")[0]
    assert "n_molecules = run_packmol_solvation(" in main_body


def test_periodic_packing_uses_packmol_pbc() -> None:
    """`pbc` keeps the tolerance valid across cell faces at bulk density."""
    body = _solvation_body()
    assert "pbc 0.0 0.0 0.0 {side_length} {side_length} {side_length}" in body


def test_nloop_raised_above_packmol_default() -> None:
    """Packmol's own default is 50; the failure log recommends raising it."""
    src = SETUP_BOX.read_text(encoding="utf-8")
    assert "PACKMOL_NLOOP = 200" in src
    assert "nloop {int(nloop)}" in _solvation_body()


def test_fill_fraction_leaves_headroom_below_bulk_density() -> None:
    src = SETUP_BOX.read_text(encoding="utf-8")
    line = next(
        ln for ln in src.splitlines() if ln.startswith("PACKMOL_FILL_FRACTION")
    )
    value = float(line.split("=")[1].strip())
    assert 0.90 <= value < 1.0


def test_capacity_scales_with_region_volume() -> None:
    """solvent_capacity must return ~cube/sphere = 1.9x more for the box region."""
    body = SETUP_BOX.read_text(encoding="utf-8").split("def solvent_capacity")[1]
    body = body.split("\ndef ")[0]
    # box branch uses the full cube; sphere branch uses the shell.
    assert "available = float(side_length) ** 3 - solute_vol" in body
    assert "(4.0 / 3.0) * np.pi * (float(outer_radius) ** 3) - solute_vol" in body
