"""Unit tests for CHARMM IMAGE (MKIMAT2) min-distance parsing and gates."""

from __future__ import annotations

import argparse

import pytest

from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
    assert_charmm_image_min_distance,
    assert_charmm_image_min_distance_after_update,
    parse_mkimat2_min_distances,
    resolve_mic_registration_fallback_min_A,
    resolve_mkimat2_min_distance_A,
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


def test_assert_charmm_image_mic_fallback_uses_registration_floor(monkeypatch):
    import numpy as np

    args = argparse.Namespace(
        solvents=["DCM"],
        _cluster_atoms_per_list=[5] * 52,
        pre_mlpot_overlap_min_distance=2.3,
    )

    def _fail_prep_geometry(*a, **k):
        raise RuntimeError("2.100 Å < required 2.40 Å (H–Cl)")

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
        lambda _args: np.array([17, 17, 17, 17, 17, 17, 17, 17, 17, 17], dtype=int),
    )
    monkeypatch.setattr(
        "mmml.utils.intermonomer_geometry.assert_pre_mlpot_mic_geometry",
        _fail_prep_geometry,
    )
    from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
        assert_charmm_image_min_distance_after_update,
    )

    with pytest.raises(RuntimeError, match="2.40 Å"):
        assert_charmm_image_min_distance_after_update(
            workflow_args=args,
            context="MLpot PBC registration (post-MLpot)",
            cubic_box_side_A=28.0,
            charmm_log="no mkimat",
            post_bimag=False,
        )


def test_assert_charmm_image_mic_fallback_aborts_below_registration_floor(monkeypatch):
    import numpy as np

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
        lambda _args: np.full(10, 17, dtype=int),
    )

    def _prep_fail(*a, **k):
        raise RuntimeError(
            "registration gate (MIC prep element-pair floors): "
            "monomers 3/20, atoms Cl–Cl, distance=2.1000 Å < required 2.9000 Å"
        )

    monkeypatch.setattr(
        "mmml.utils.intermonomer_geometry.assert_pre_mlpot_mic_geometry",
        _prep_fail,
    )
    from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
        assert_charmm_image_mic_fallback,
    )

    with pytest.raises(RuntimeError, match="2.1000"):
        assert_charmm_image_mic_fallback(
            workflow_args=argparse.Namespace(
                solvents=["DCM"],
                _cluster_atoms_per_list=[5] * 52,
                pre_mlpot_overlap_min_distance=2.3,
            ),
            box_side_A=32.0,
            min_distance_A=2.3,
            context="registration gate",
        )


def test_assert_charmm_image_mic_fallback_uses_psf_elements(monkeypatch):
    import numpy as np

    captured: dict[str, object] = {}

    def _fake_prep(*a, **k):
        captured["context"] = k.get("context", "")
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
        "mmml.utils.intermonomer_geometry.assert_pre_mlpot_mic_geometry",
        _fake_prep,
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
        workflow_args=argparse.Namespace(solvents=["DCM"], pre_mlpot_overlap_min_distance=2.3),
        box_side_A=28.0,
        min_distance_A=2.3,
        context="test gate",
    )
    assert worst == pytest.approx(2.577)
    assert "MIC prep element-pair floors" in str(captured["context"])


def test_assert_charmm_image_mic_fallback_calls_registration_floor(monkeypatch):
    import numpy as np

    captured: dict[str, object] = {}

    def _fake_prep(*a, **k):
        captured.update({"box_side": k.get("box_side"), "n_atoms": len(a[0])})
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
        "mmml.interfaces.pycharmmInterface.charmm_image_geometry._resolve_atomic_numbers_for_image_gate",
        lambda _args: np.ones(10, dtype=int),
    )
    monkeypatch.setattr(
        "mmml.utils.intermonomer_geometry.assert_pre_mlpot_mic_geometry",
        _fake_prep,
    )
    monkeypatch.setattr(
        "mmml.utils.intermonomer_geometry.summarize_worst_intermonomer_contact",
        lambda *a, **k: type("S", (), {"format_log_line": lambda self: "worst inter-monomer contact 3.42 Å"})(),
    )
    from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
        assert_charmm_image_mic_fallback,
    )

    worst = assert_charmm_image_mic_fallback(
        workflow_args=argparse.Namespace(pre_mlpot_overlap_min_distance=2.3),
        box_side_A=27.993,
        min_distance_A=2.3,
        context="test gate",
    )
    assert worst == pytest.approx(3.42)
    assert captured["box_side"] == pytest.approx(27.993)
    assert captured["n_atoms"] == 10


def test_assert_charmm_image_min_distance_aborts_on_dense_pbc_margin():
    log = """
 <MKIMAT2>: updating the image atom lists and remapping
 Transformation   Atoms  Groups  Residues  Min-Distance
    5  Z0Z0N1R1 has     120      24      24        3.00
    8  N1Z0Z0R1 has     145      29      29        3.00
"""
    with pytest.raises(RuntimeError, match="3.00 Å < prep floor 3.50 Å"):
        assert_charmm_image_min_distance(log, min_distance_A=3.5, context="test")


def test_resolve_mkimat2_min_distance_default():
    from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
        resolve_mkimat2_min_distance_A,
    )

    assert resolve_mkimat2_min_distance_A(None) == pytest.approx(1.0)
    assert resolve_mkimat2_min_distance_A(argparse.Namespace()) == pytest.approx(1.0)


def test_resolve_mkimat2_min_distance_dense_dcm_uses_same_default():
    from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
        resolve_mkimat2_min_distance_A,
    )

    args = argparse.Namespace(
        solvents=["DCM"],
        _cluster_atoms_per_list=[5] * 52,
    )
    assert resolve_mkimat2_min_distance_A(args) == pytest.approx(1.0)
    assert resolve_mkimat2_min_distance_A(None) == pytest.approx(1.0)
    assert resolve_mkimat2_min_distance_A(argparse.Namespace()) == pytest.approx(1.0)


def test_assert_charmm_image_min_distance_aborts_dense_dcm_mkimat_margin(monkeypatch):
    log = """
 <MKIMAT2>: updating the image atom lists and remapping
 Transformation   Atoms  Groups  Residues  Min-Distance
    5  Z0Z0N1R1 has     120      24      24        4.40
"""
    args = argparse.Namespace(
        solvents=["DCM"],
        _cluster_atoms_per_list=[5] * 52,
        charmm_image_mlpot_min_distance=4.5,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_image_geometry.run_charmm_image_probe_log",
        lambda **kwargs: log,
    )
    with pytest.raises(RuntimeError, match="4.40 Å < prep floor 4.50 Å"):
        assert_charmm_image_min_distance_after_update(
            workflow_args=args,
            context="test gate",
            post_bimag=True,
        )


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
            workflow_args=argparse.Namespace(charmm_image_mlpot_min_distance=3.5),
            context="test gate",
            post_bimag=True,
        )


def test_resolve_mic_registration_fallback_uses_prep_floor_not_mkimat():
    args = argparse.Namespace(
        solvents=["DCM"],
        _cluster_atoms_per_list=[5] * 52,
        pre_mlpot_overlap_min_distance=2.3,
    )
    assert resolve_mkimat2_min_distance_A(args) == pytest.approx(1.0)
    assert resolve_mic_registration_fallback_min_A(args) == pytest.approx(2.3)
    assert resolve_mic_registration_fallback_min_A(None) == pytest.approx(2.3)
    sparse = argparse.Namespace(solvents=["DCM"], _cluster_atoms_per_list=[5] * 10)
    assert resolve_mic_registration_fallback_min_A(sparse) == pytest.approx(2.3)


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


def test_format_charmm_image_nb_stats_tight_buffer():
    from mmml.interfaces.pycharmmInterface.charmm_image_geometry import (
        CharmmImageNbStats,
        format_charmm_image_nb_stats,
    )

    stats = CharmmImageNbStats(
        natom=260,
        natim=690,
        ntrans=9,
        nnb=33670,
        niminb=50000,
        iminb_capacity=50001,
        nimnb=2774,
        imjnb_capacity=9714,
        niming=120,
        mlpot_active=True,
    )
    text = format_charmm_image_nb_stats(stats)
    assert "niminb=50000/50001" in text
    assert " tight" in text
    assert " MLpot" in text
    assert stats.iminb_headroom == 1


def test_get_iminb_stats_uses_ctypes_out_pointers(monkeypatch) -> None:
    import importlib.util
    import sys
    import types
    from pathlib import Path

    calls: list[tuple] = []

    def _fake_getter(*args) -> int:
        calls.append(args)
        return 1

    fake_lib = types.SimpleNamespace(
        charmm=types.SimpleNamespace(image_get_iminb_stats=_fake_getter)
    )
    fake_pycharmm = types.ModuleType("pycharmm")
    fake_pycharmm.lib = fake_lib  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pycharmm", fake_pycharmm)
    monkeypatch.setitem(sys.modules, "pycharmm.lib", fake_lib)

    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "pycharmm_image_isolated",
        repo_root / "pycharmm" / "image.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    raw = mod.get_iminb_stats()
    assert raw is not None
    assert set(raw) == {
        "natom",
        "natim",
        "ntrans",
        "nnb",
        "niminb",
        "iminb_capacity",
        "nimnb",
        "imjnb_capacity",
        "niming",
        "mlpot_active",
    }
    assert len(calls) == 1
    assert len(calls[0]) == 10
