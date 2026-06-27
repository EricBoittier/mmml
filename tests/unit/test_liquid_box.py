"""Unit tests for mmml liquid-box (profiles, certification, report)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.liquid_box_build import (
    BOX_JSON,
    REPORT_MD,
    LiquidBoxBuildResult,
    apply_liquid_box_profile,
    certify_intermonomer_geometry,
    estimate_density_g_cm3,
    render_liquid_box_report,
    write_liquid_box_artifacts,
)


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        composition="DCM:10",
        output_dir=Path("/tmp/liquid_box_test"),
        profile="dense",
        box_auto=None,
        box_size=None,
        target_density_g_cm3=None,
        bulk_density_fraction=None,
        liquid_prep=None,
        charmm_pre_minimize=None,
        save=None,
        setup=None,
        pre_mlpot_overlap_min_distance=None,
        dynamics_overlap_min_distance=1.5,
        quiet=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_apply_liquid_box_profile_dense_enables_liquid_prep():
    args = _args(profile="dense")
    name = apply_liquid_box_profile(args)
    assert name == "dense"
    assert args.liquid_prep is True
    assert args.setup == "pbc_nvt"
    assert args.box_auto == "density"
    assert int(args.mini_lattice_abnr_steps) >= 200


def test_apply_liquid_box_profile_standard_skips_liquid_prep():
    args = _args(profile="standard")
    name = apply_liquid_box_profile(args)
    assert name == "standard"
    assert args.liquid_prep is False


def test_apply_liquid_box_profile_conservative_caps_bulk_fraction():
    args = _args(profile="conservative")
    apply_liquid_box_profile(args)
    assert args.liquid_prep is True
    assert float(args.bulk_density_fraction) <= 0.55


def test_estimate_density_g_cm3_round_trip():
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
        cubic_box_side_from_target_density,
        total_mass_g_for_composition,
    )

    comp = {"DCM": 206}
    target = 1.326
    mass = total_mass_g_for_composition(comp)
    side = cubic_box_side_from_target_density(
        n_molecules=206,
        total_mass_g=mass,
        target_density_g_cm3=target,
    )
    rho = estimate_density_g_cm3(
        composition=comp,
        box_side_A=side,
        n_molecules=206,
    )
    assert rho is not None
    assert rho == pytest.approx(target, rel=1e-6)


def test_certify_intermonomer_geometry_pass_and_fail():
    atoms_per = [2, 2]
    pos_ok = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    args = _args(pre_mlpot_overlap_min_distance=1.0)
    worst, passed, msg = certify_intermonomer_geometry(
        pos_ok,
        atoms_per,
        args=args,
        box_side=30.0,
        use_pbc=False,
    )
    assert passed is True
    assert worst >= 1.0
    assert "prep floor" in msg

    pos_bad = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    worst_bad, passed_bad, _ = certify_intermonomer_geometry(
        pos_bad,
        atoms_per,
        args=args,
        box_side=30.0,
        use_pbc=False,
    )
    assert passed_bad is False
    assert worst_bad < 1.0


def test_write_liquid_box_artifacts(tmp_path: Path):
    out = tmp_path / "box"
    result = LiquidBoxBuildResult(
        status="pass",
        out_dir=out,
        profile="dense",
        composition="DCM:10",
        n_molecules=10,
        n_atoms=50,
        box_side_A=20.0,
        density_g_cm3=1.32,
        worst_intermonomer_A=1.2,
        prep_overlap_floor_A=1.0,
        mm_grms_kcalmol_A=8.5,
        model_psf=out / "model.psf",
        model_crd=out / "model.crd",
        message="ok",
        steps_applied=["packmol_cluster", "charmm_mm_pre_minimize"],
    )
    write_liquid_box_artifacts(result, args=_args())
    assert (out / BOX_JSON).is_file()
    assert (out / REPORT_MD).is_file()
    payload = json.loads((out / BOX_JSON).read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["n_molecules"] == 10
    report = (out / REPORT_MD).read_text(encoding="utf-8")
    assert "Liquid box report" in report
    assert "mmml md-system" in report


def test_render_liquid_box_report_fail_includes_message():
    result = LiquidBoxBuildResult(
        status="fail",
        out_dir=Path("/tmp/x"),
        profile="dense",
        composition="DCM:10",
        n_molecules=10,
        n_atoms=50,
        message="overlap persists",
    )
    text = render_liquid_box_report(result)
    assert "FAIL" in text
    assert "overlap persists" in text


def test_liquid_box_cli_parser_requires_composition():
    from mmml.cli.run.liquid_box import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "/tmp/x"])

    ns = parser.parse_args(
        ["--composition", "DCM:10", "--output-dir", "/tmp/x", "--profile", "standard"]
    )
    assert ns.composition == "DCM:10"
    assert ns.profile == "standard"


def test_composition_tag_from_liquid_box_args_without_residue():
    """liquid-box uses --composition only; cluster tag must not require --residue."""
    from mmml.cli.run.liquid_box import build_parser
    from mmml.cli.run.md_pbc_suite.ase import _parse_composition
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import composition_tag

    parser = build_parser()
    args = parser.parse_args(
        ["--composition", "DCM:206", "--output-dir", "/tmp/x", "--profile", "dense"]
    )
    composition = _parse_composition(args.composition)
    n_mol = sum(count for _, count in composition)
    residue = getattr(args, "residue", composition[0][0]).upper()
    tag = composition_tag(composition, residue, n_mol)
    assert tag == "dcm_206"
    assert not hasattr(args, "residue")
