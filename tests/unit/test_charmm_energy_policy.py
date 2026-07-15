"""Unit tests for CHARMM energy-term PSF/.prm enforcement."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest


def test_resolve_charmm_energy_term_policies_no_periodic_vdw_implies_vdw():
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
        resolve_charmm_energy_term_policies,
    )

    args = argparse.Namespace(
        periodic_charmm_vdw=False,
        charmm_zero_energy_terms=None,
    )
    policies = resolve_charmm_energy_term_policies(args)
    assert [p.name for p in policies] == ["vdw"]


def test_resolve_charmm_energy_term_policies_custom_terms():
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
        resolve_charmm_energy_term_policies,
    )

    # periodic_external keeps CHARMM IMAGE VDW on by default, so only the
    # explicitly requested terms are enforced.
    args = argparse.Namespace(
        mm_nonbond_mode="periodic_external",
        periodic_charmm_vdw=True,
        charmm_zero_energy_terms="elec,bonded",
    )
    policies = resolve_charmm_energy_term_policies(args)
    assert [p.name for p in policies] == ["elec", "bonded"]


def test_resolve_charmm_energy_term_policies_jax_mic_adds_vdw():
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
        resolve_charmm_energy_term_policies,
    )

    # jax_mic: CHARMM IMAGE VDW must be zeroed to avoid double-counting;
    # vdw policy is added implicitly even when periodic_charmm_vdw=True (not explicit).
    args = argparse.Namespace(
        mm_nonbond_mode="jax_mic",
        periodic_charmm_vdw=True,
        charmm_zero_energy_terms="elec,bonded",
    )
    policies = resolve_charmm_energy_term_policies(args)
    assert [p.name for p in policies] == ["elec", "bonded", "vdw"]


def test_nonbond_only_prm_text_removes_vdw_sections():
    from mmml.interfaces.pycharmmInterface.charmm_prm_zero import (
        nonbond_only_prm_text,
    )

    sample = (
        "NONBONDED nbxmod 5\n"
        "CTCL   0.0\n"
        "CL     0.0       -0.1200     2.4700\n"
        "NBFIX\n"
        "CL   CTCL    -0.1200     2.4700\n"
    )
    out = nonbond_only_prm_text(sample)
    assert "VDW term removed" in out
    assert "NONBONDED" not in out
    assert "NBFIX" not in out
    assert "nbxmod" not in out.lower()
    assert "CL" not in out


def test_write_prm_policy_overlay_nonbond(tmp_path: Path):
    from mmml.interfaces.pycharmmInterface.charmm_prm_zero import (
        write_prm_policy_overlay,
    )

    src = tmp_path / "src.prm"
    src.write_text(
        "BONDS\n"
        "CT   CL    300.0       1.76\n"
        "NONBONDED nbxmod 5\n"
        "CL     0.0       -0.1200     2.4700\n",
        encoding="utf-8",
    )
    dst = tmp_path / "overlay.prm"
    write_prm_policy_overlay(src, dst, zero_bonded=False, zero_nonbond=True)
    text = dst.read_text(encoding="utf-8")
    assert "MMML energy-policy overlay" in text
    # Append overlay must emit ε=0 NONBONDED rows (not omit the section).
    assert "NONBONDED" in text
    assert "nbxmod" not in text.lower()
    assert "CL" in text
    assert "-0.1200" not in text
    assert "300.0" not in text
    assert "2.4700" in text


def test_apply_before_pbc_writes_epsilon_zero_overlay(tmp_path: Path, monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot import charmm_energy_policy as cep

    args = argparse.Namespace(
        mm_nonbond_mode="jax_mic",
        periodic_charmm_vdw=False,
        charmm_zero_energy_terms=None,
        quiet=True,
        output_dir=tmp_path,
    )
    src = tmp_path / "par.prm"
    src.write_text(
        "NONBONDED nbxmod 5\n"
        "CL     0.0       -0.1200     2.4700 ! comment\n"
        "END\n",
        encoding="utf-8",
    )
    read_calls: list[Path] = []

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap.cgenff_prm_path",
        lambda: src,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.nbonds_config.read_cgenff_prm",
        lambda path, append=False: read_calls.append(Path(path)),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.import_pycharmm",
        object(),
        raising=False,
    )

    class _Silent:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    scripts: list[str] = []
    monkeypatch.setattr(
        cep,
        "charmm_silent_command",
        lambda: _Silent(),
        raising=False,
    )
    # Patch at the import sites used inside the function
    import types
    import sys

    fake_pycharmm = types.ModuleType("pycharmm")
    fake_lingo = types.ModuleType("pycharmm.lingo")
    fake_lingo.charmm_script = lambda s: scripts.append(s)
    fake_pycharmm.lingo = fake_lingo
    monkeypatch.setitem(sys.modules, "pycharmm", fake_pycharmm)
    monkeypatch.setitem(sys.modules, "pycharmm.lingo", fake_lingo)
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_silent_command",
        lambda: _Silent(),
    )

    applied = cep.apply_charmm_energy_term_policies_before_pbc_finalize(
        args,
        ml_selection=object(),
        verbose=False,
    )
    assert applied == ["vdw"]
    assert len(read_calls) == 1
    overlay = read_calls[0]
    assert overlay.name == "zeroed_vdw_pre_pbc.prm"
    text = overlay.read_text(encoding="utf-8")
    assert "NONBONDED" in text
    assert "-0.1200" not in text
    assert any("scalar vdw" in s.lower() for s in scripts)


def test_policy_violation_detects_imnb():
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
        POLICY_REGISTRY,
        _policy_violation,
    )

    policy = POLICY_REGISTRY["vdw"]
    bad, hits = _policy_violation(
        policy,
        {"VDW": 0.0, "IMNB": -1.0528, "USER": -1000.0},
    )
    assert bad
    assert hits == {"IMNB": pytest.approx(-1.0528)}


def test_policy_violation_detects_small_imnb():
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
        POLICY_REGISTRY,
        _policy_violation,
    )

    policy = POLICY_REGISTRY["vdw"]
    bad, hits = _policy_violation(
        policy,
        {"VDW": 0.0, "IMNB": -3.0e-4, "USER": -1000.0},
    )
    assert bad
    assert hits == {"IMNB": pytest.approx(-3.0e-4)}


def test_nonbond_policy_overlay_emits_epsilon_zero_rows(tmp_path: Path):
    from mmml.interfaces.pycharmmInterface.charmm_prm_zero import (
        write_prm_policy_overlay,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import cgenff_prm_path

    dst = tmp_path / "overlay.prm"
    write_prm_policy_overlay(
        cgenff_prm_path(),
        dst,
        zero_bonded=False,
        zero_nonbond=True,
    )
    text = dst.read_text(encoding="utf-8")
    assert "NONBONDED" in text
    assert "nbxmod" not in text.lower()
    assert "MMML energy-policy overlay" in text
    assert "HBOND" not in "\n".join(
        line for line in text.splitlines() if not line.startswith("*")
    )

def test_enforce_skips_when_terms_already_zero(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot import charmm_energy_policy as cep

    sel = object()
    args = argparse.Namespace(
        periodic_charmm_vdw=False,
        charmm_zero_energy_terms=None,
        quiet=True,
    )

    monkeypatch.setattr(
        cep,
        "measure_charmm_energy_terms",
        lambda: {"VDW": 0.0, "IMNB": 0.0, "USER": -1.0},
    )
    monkeypatch.setattr(
        cep,
        "_run_silent_ener",
        lambda: None,
    )

    applied = cep.enforce_charmm_energy_term_policies(
        args,
        ml_selection=sel,
        use_pbc=False,
        cubic_box_side_A=None,
        verbose=False,
    )
    assert applied == []


def test_enforce_can_verify_without_late_reload(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot import charmm_energy_policy as cep

    args = argparse.Namespace(
        periodic_charmm_vdw=False,
        charmm_zero_energy_terms=None,
        quiet=True,
    )

    monkeypatch.setattr(
        cep,
        "measure_charmm_energy_terms",
        lambda: {"VDW": 0.0, "IMNB": -1.324845, "USER": 0.0},
    )
    monkeypatch.setattr(cep, "_run_silent_ener", lambda: None)

    with pytest.raises(RuntimeError, match="IMNB=-1.32485"):
        cep.enforce_charmm_energy_term_policies(
            args,
            ml_selection=object(),
            use_pbc=True,
            cubic_box_side_A=50.0,
            verbose=False,
            reload_on_violation=False,
        )


def test_enforce_tolerates_small_imnb_after_pre_remediation(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot import charmm_energy_policy as cep

    args = argparse.Namespace(
        periodic_charmm_vdw=False,
        charmm_zero_energy_terms=None,
        quiet=True,
    )

    monkeypatch.setattr(
        cep,
        "measure_charmm_energy_terms",
        lambda: {"VDW": 0.0, "IMNB": -0.0921032, "USER": 0.0},
    )
    monkeypatch.setattr(cep, "_run_silent_ener", lambda: None)

    applied = cep.enforce_charmm_energy_term_policies(
        args,
        ml_selection=object(),
        use_pbc=True,
        cubic_box_side_A=50.0,
        verbose=False,
        reload_on_violation=False,
    )
    assert applied == ["vdw"]


def test_enforce_tolerates_image_imnb_after_pre_remediation(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot import charmm_energy_policy as cep

    args = argparse.Namespace(
        periodic_charmm_vdw=False,
        charmm_zero_energy_terms=None,
        quiet=True,
    )

    monkeypatch.setattr(
        cep,
        "measure_charmm_energy_terms",
        lambda: {"VDW": 0.0, "IMNB": -0.296236, "USER": 0.0},
    )
    monkeypatch.setattr(cep, "_run_silent_ener", lambda: None)

    applied = cep.enforce_charmm_energy_term_policies(
        args,
        ml_selection=object(),
        use_pbc=True,
        cubic_box_side_A=50.0,
        verbose=False,
        reload_on_violation=False,
    )
    assert applied == ["vdw"]
