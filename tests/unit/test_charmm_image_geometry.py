"""Unit tests for CHARMM IMAGE (MKIMAT2) min-distance parsing and gates."""

from __future__ import annotations

import argparse

import pytest

from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
    assert_charmm_image_min_distance,
    assert_charmm_image_min_distance_after_update,
    parse_mkimat2_min_distances,
    summarize_mkimat2_min_distances,
)

_SAMPLE_LOG = """
 SELECTED IMAGES ATOMS BEING CENTERED ABOUT  0.000000  0.000000  0.000000

 <MKIMAT2>: updating the image atom lists and remapping
 Transformation   Atoms  Groups  Residues  Min-Distance
    1  Z0Z0N1R1 has      80      16      16        6.18
    2  N1Z0Z0R1 has      55      11      11        7.17
    3  Z0N1Z0R1 has      55      11      11        8.93
 Total of  435 atoms and   87 groups and   87 residues were included
"""

_SAMPLE_BAD_LOG = """
 <MKIMAT2>: updating the image atom lists and remapping
 Transformation   Atoms  Groups  Residues  Min-Distance
    5  Z0Z0N1R1 has     125      25      25        0.00
    8  N1Z0Z0R1 has     150      30      30        0.00
   10  Z0N1Z0R1 has     120      24      24        0.81
"""


def test_parse_mkimat2_min_distances_reads_transformation_rows():
    distances = parse_mkimat2_min_distances(_SAMPLE_LOG)
    assert distances == pytest.approx([6.18, 7.17, 8.93])


def test_parse_mkimat2_uses_latest_block():
    combined = _SAMPLE_LOG + _SAMPLE_BAD_LOG
    report = summarize_mkimat2_min_distances(combined)
    assert report.worst == pytest.approx(0.00)
    assert len(report.distances) == 3


def test_assert_charmm_image_min_distance_accepts_safe_contacts():
    worst = assert_charmm_image_min_distance(
        _SAMPLE_LOG,
        min_distance_A=2.3,
        context="test",
    )
    assert worst == pytest.approx(6.18)


def test_assert_charmm_image_min_distance_aborts_on_zero_contact():
    with pytest.raises(RuntimeError, match="Min-Distance 0.00"):
        assert_charmm_image_min_distance(
            _SAMPLE_BAD_LOG,
            min_distance_A=2.3,
            context="test",
        )


def test_assert_charmm_image_min_distance_aborts_on_subfloor_contact():
    log = """
 <MKIMAT2>: updating the image atom lists and remapping
 Transformation   Atoms  Groups  Residues  Min-Distance
    9  N1P1Z0R1 has      40       8       8        1.37
"""
    with pytest.raises(RuntimeError, match="1.37 Å < prep floor 2.30 Å"):
        assert_charmm_image_min_distance(log, min_distance_A=2.3, context="test")


def test_assert_charmm_image_min_distance_aborts_when_mkimat2_missing():
    with pytest.raises(RuntimeError, match="no <MKIMAT2>"):
        assert_charmm_image_min_distance("ENER only output", context="test")


def test_assert_charmm_image_min_distance_after_update_uses_provided_log(monkeypatch):
    called = {"update": 0}

    def _fail_update():
        called["update"] += 1
        raise AssertionError("UPDATE should not run when charmm_log is provided")

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_image_geometry.run_charmm_update_capture_image_log",
        _fail_update,
    )
    worst = assert_charmm_image_min_distance_after_update(
        charmm_log=_SAMPLE_LOG,
        min_distance_A=2.3,
        workflow_args=argparse.Namespace(),
        context="test gate",
    )
    assert worst == pytest.approx(6.18)
    assert called["update"] == 0


def test_run_charmm_post_bimag_probe_prefers_fd_capture(monkeypatch):
    calls: list[str] = []

    def _fake_capture(script: str, *, replay: bool = True) -> str:
        calls.append(script)
        if script == "UPDATE":
            return _SAMPLE_LOG
        return ""

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_image_geometry._run_charmm_script_capture_fortran",
        _fake_capture,
    )
    def _fail_outu_probe(_cmd: str) -> str:
        raise AssertionError("OUTU probe should not run")

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_image_geometry._probe_command_via_charmm_log_file",
        _fail_outu_probe,
    )
    from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
        run_charmm_post_bimag_image_probe_log,
    )

    log = run_charmm_post_bimag_image_probe_log()
    assert calls == ["UPDATE"]
    assert parse_mkimat2_min_distances(log) == pytest.approx([6.18, 7.17, 8.93])


def test_run_charmm_image_probe_log_falls_back_to_fd_capture(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_image_geometry._force_charmm_image_remap_for_probe",
        lambda: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_image_geometry._probe_command_via_charmm_log_file",
        lambda _cmd: "",
    )

    def _fake_capture(script: str, *, replay: bool = True) -> str:
        calls.append(script)
        if script == "UPDATE":
            return "UPDATE only output\n"
        return _SAMPLE_LOG

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_image_geometry._run_charmm_script_capture_fortran",
        _fake_capture,
    )
    from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
        run_charmm_image_probe_log,
    )

    log = run_charmm_image_probe_log()
    assert calls == ["UPDATE", "ENER"]
    assert parse_mkimat2_min_distances(log) == pytest.approx([6.18, 7.17, 8.93])


def test_assert_charmm_image_mic_fallback_uses_psf_elements(monkeypatch):
    import numpy as np

    captured: dict[str, object] = {}

    def _fake_prep_geometry(
        positions,
        atoms_per_list,
        *,
        min_distance_A,
        box_side,
        use_pbc,
        context="",
        args=None,
        atomic_numbers=None,
    ):
        captured.update(
            {
                "atomic_numbers": None if atomic_numbers is None else np.asarray(atomic_numbers).tolist(),
                "context": context,
            }
        )
        return 2.577

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        lambda: np.zeros((10, 3)),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_image_geometry._resolve_atoms_per_for_image_gate",
        lambda _args: [5, 5],
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_image_geometry._resolve_atomic_numbers_for_image_gate",
        lambda _args: np.array([1, 1, 1, 1, 1, 6, 6, 6, 6, 17], dtype=int),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder.assert_pre_mlpot_intermonomer_geometry",
        _fake_prep_geometry,
    )
    monkeypatch.setattr(
        "mmml.utils.intermonomer_geometry.summarize_worst_intermonomer_contact",
        lambda *a, **k: type(
            "S",
            (),
            {
                "format_log_line": lambda self: (
                    "worst inter-monomer contact 2.577 Å "
                    "(monomers 50/51, atoms H–H; prep floor 2.30 Å)"
                )
            },
        )(),
    )
    from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
        assert_charmm_image_mic_fallback,
    )

    worst = assert_charmm_image_mic_fallback(
        workflow_args=argparse.Namespace(solvents=["DCM"]),
        box_side_A=28.0,
        min_distance_A=2.3,
        context="test gate",
    )
    assert worst == pytest.approx(2.577)
    assert captured["atomic_numbers"] == [1, 1, 1, 1, 1, 6, 6, 6, 6, 17]


def test_assert_charmm_image_mic_fallback_calls_mic_geometry(monkeypatch):
    import numpy as np

    captured: dict[str, object] = {}

    def _fake_mic_geometry(pos, atoms_per, *, box_side, use_pbc, args=None, atomic_numbers=None, context=""):
        captured.update(
            {
                "box_side": box_side,
                "use_pbc": use_pbc,
                "context": context,
                "n_atoms": len(pos),
            }
        )
        return 3.42

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        lambda: np.zeros((10, 3)),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_image_geometry._resolve_atoms_per_for_image_gate",
        lambda _args: [5, 5],
    )
    monkeypatch.setattr(
        "mmml.utils.intermonomer_geometry.assert_pre_mlpot_mic_geometry",
        _fake_mic_geometry,
    )
    from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
        assert_charmm_image_mic_fallback,
    )

    worst = assert_charmm_image_mic_fallback(
        workflow_args=argparse.Namespace(),
        box_side_A=27.993,
        min_distance_A=2.3,
        context="test gate",
    )
    assert worst == pytest.approx(3.42)
    assert captured["box_side"] == pytest.approx(27.993)
    assert captured["use_pbc"] is True
    assert "MIC fallback" in str(captured["context"])


def test_assert_charmm_image_min_distance_aborts_on_dense_pbc_margin():
    log = """
 <MKIMAT2>: updating the image atom lists and remapping
 Transformation   Atoms  Groups  Residues  Min-Distance
    5  Z0Z0N1R1 has     120      24      24        3.00
    8  N1Z0Z0R1 has     145      29      29        3.00
"""
    with pytest.raises(RuntimeError, match="3.00 Å < prep floor 3.50 Å"):
        assert_charmm_image_min_distance(log, min_distance_A=3.5, context="test")


def test_resolve_mkimat2_min_distance_uses_mlpot_margin():
    from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
        resolve_mkimat2_min_distance_A,
    )

    assert resolve_mkimat2_min_distance_A(None) == pytest.approx(3.5)
    assert resolve_mkimat2_min_distance_A(argparse.Namespace()) == pytest.approx(3.5)


def test_assert_charmm_image_min_distance_after_update_uses_mkimat_floor(monkeypatch):
    dense_log = """
 <MKIMAT2>: updating the image atom lists and remapping
 Transformation   Atoms  Groups  Residues  Min-Distance
    5  Z0Z0N1R1 has     120      24      24        3.00
"""
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_image_geometry.run_charmm_image_probe_log",
        lambda **kwargs: dense_log,
    )
    with pytest.raises(RuntimeError, match="3.00 Å < prep floor 3.50 Å"):
        assert_charmm_image_min_distance_after_update(
            workflow_args=argparse.Namespace(),
            context="test gate",
            post_bimag=True,
        )


def test_assert_charmm_image_min_distance_after_update_mic_fallback(monkeypatch):
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_image_geometry.run_charmm_image_probe_log",
        lambda **kwargs: "no mkimat here",
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_image_geometry.assert_charmm_image_mic_fallback",
        lambda **kwargs: 2.5,
    )
    worst = assert_charmm_image_min_distance_after_update(
        workflow_args=argparse.Namespace(),
        context="test",
        cubic_box_side_A=28.0,
    )
    assert worst == pytest.approx(2.5)
