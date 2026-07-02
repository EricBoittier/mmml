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

    args = argparse.Namespace(
        periodic_charmm_vdw=True,
        charmm_zero_energy_terms="elec,bonded",
    )
    policies = resolve_charmm_energy_term_policies(args)
    assert [p.name for p in policies] == ["elec", "bonded"]


def test_nonbond_only_prm_text_zeros_eps():
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
    assert "0.0" in out
    assert "NONBONDED" not in out
    assert "CL" in out


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
    assert "0.0" in text
    assert "300.0" not in text


def test_policy_violation_detects_imnb():
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
        POLICY_REGISTRY,
        _policy_violation,
    )

    policy = POLICY_REGISTRY["vdw"]
    bad, hits = _policy_violation(
        policy,
        {"VDW": 0.0, "IMNB": -0.0528, "USER": -1000.0},
    )
    assert bad
    assert hits == {"IMNB": pytest.approx(-0.0528)}


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
