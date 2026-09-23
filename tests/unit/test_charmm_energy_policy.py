"""Unit tests for CHARMM energy-term PSF/.prm enforcement."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from unittest import mock

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
    # eval_charmm_script does no case folding: lowercase commands are
    # "Unrecognized" and silently skipped, so every script must be uppercase.
    assert all(s == s.upper() for s in scripts), scripts
    # The ε=0 APPEND overlay does not replace the live VDW table; SKIPE does.
    assert "SKIPE VDW IMNB" in scripts


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


def test_all_ml_registration_combined_skipe_bonded_and_vdw_keeps_user(
    tmp_path: Path, monkeypatch
):
    """All-ML CHARMM registration: both policies SKIPE, USER stays on."""
    from mmml.interfaces.pycharmmInterface.mlpot import block_terms
    from mmml.interfaces.pycharmmInterface.mlpot import charmm_energy_policy as cep

    scripts: list[str] = []
    sel = mock.Mock()
    sel.get_atom_indexes.return_value = [0, 1, 2]

    fake_pycharmm = types.ModuleType("pycharmm")
    fake_lingo = types.ModuleType("pycharmm.lingo")
    fake_lingo.charmm_script = lambda s: scripts.append(s)
    fake_pycharmm.lingo = fake_lingo
    fake_pycharmm.coor = mock.Mock()
    fake_pycharmm.coor.get_natom.return_value = 3
    fake_pycharmm.psf = mock.Mock()
    fake_pycharmm.psf.get_charges.return_value = [0.1, 0.2, 0.3]
    monkeypatch.setitem(sys.modules, "pycharmm", fake_pycharmm)
    monkeypatch.setitem(sys.modules, "pycharmm.lingo", fake_lingo)

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap.apply_zeroed_cgenff_params"
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap.assert_psf_bonds_present",
        return_value=400,
    ), mock.patch.object(block_terms, "_import_pycharmm", return_value=fake_pycharmm):
        block_terms.zero_mlpot_psf_mm_terms(sel)

    src = tmp_path / "par.prm"
    src.write_text(
        "NONBONDED nbxmod 5\nCL     0.0       -0.1200     2.4700\nEND\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap.cgenff_prm_path",
        lambda: src,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.nbonds_config.read_cgenff_prm",
        lambda path, append=False: None,
    )

    class _Silent:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(cep, "charmm_silent_command", lambda: _Silent(), raising=False)
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_silent_command",
        lambda: _Silent(),
    )
    applied = cep.apply_charmm_energy_term_policies_before_pbc_finalize(
        argparse.Namespace(
            mm_nonbond_mode="jax_mic",
            periodic_charmm_vdw=False,
            charmm_zero_energy_terms=None,
            quiet=True,
            output_dir=tmp_path,
        ),
        ml_selection=sel,
        verbose=False,
    )

    joined = "\n".join(scripts)
    assert "SKIPE " + " ".join(block_terms.ALL_ML_SKIPE_BONDED) in scripts
    assert "SKIPE VDW IMNB" in scripts
    assert applied == ["vdw"]
    assert "USER" not in joined


def test_unknown_zero_energy_term_is_rejected():
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
        resolve_charmm_energy_term_policies,
    )

    args = argparse.Namespace(
        periodic_charmm_vdw=True,
        charmm_zero_energy_terms="elec,not-a-term",
    )
    with pytest.raises(ValueError, match="Unknown"):
        resolve_charmm_energy_term_policies(args)


def test_empty_term_list_and_periodic_vdw_keeps_no_policies():
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
        resolve_charmm_energy_term_policies,
    )

    args = argparse.Namespace(
        mm_nonbond_mode="periodic_external",
        periodic_charmm_vdw=True,
        charmm_zero_energy_terms="  ,  ",
    )
    assert resolve_charmm_energy_term_policies(args) == []
    assert resolve_charmm_energy_term_policies(None) == []


def test_summarize_policy_energy_terms_keeps_finite_keys_only():
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
        POLICY_REGISTRY,
        summarize_policy_energy_terms,
    )

    out = summarize_policy_energy_terms(
        [POLICY_REGISTRY["vdw"], POLICY_REGISTRY["elec"]],
        {"VDW": 0.0, "IMNB": float("nan"), "ELEC": 1.25, "USER": -9.0},
    )
    assert out == {"ELEC": 1.25, "VDW": 0.0}


def test_enforce_skips_probe_when_mlpot_user_is_active(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot import charmm_energy_policy as cep

    called = []
    monkeypatch.setattr(cep, "_run_silent_ener", lambda: called.append("ener"))
    applied = cep.enforce_charmm_energy_term_policies(
        argparse.Namespace(
            periodic_charmm_vdw=False,
            charmm_zero_energy_terms=None,
            quiet=True,
        ),
        ml_selection=object(),
        use_pbc=False,
        cubic_box_side_A=None,
        skip_ener_probe=True,
    )
    assert applied == []
    assert called == []


def test_policy_scratch_dir_defaults_and_uses_output_dir(tmp_path: Path):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
        _policy_scratch_dir,
    )

    assert _policy_scratch_dir(None) == Path("charmm_energy_policy")
    args = argparse.Namespace(output_dir=tmp_path)
    assert _policy_scratch_dir(args) == tmp_path / "charmm_energy_policy"


def test_post_remediation_policy_loosens_only_vdw():
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
        POLICY_REGISTRY,
        _post_remediation_policy,
    )

    vdw = _post_remediation_policy(POLICY_REGISTRY["vdw"])
    assert vdw.tolerance_kcal == pytest.approx(1.0)
    elec = _post_remediation_policy(POLICY_REGISTRY["elec"])
    assert elec is POLICY_REGISTRY["elec"]


def test_enforce_hbond_has_no_prm_remediation(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot import charmm_energy_policy as cep

    monkeypatch.setattr(
        cep,
        "measure_charmm_energy_terms",
        lambda: {"HBON": 2.0, "IMHB": 0.0},
    )
    monkeypatch.setattr(cep, "_run_silent_ener", lambda: None)
    with pytest.raises(RuntimeError, match="no PSF/.prm remediation"):
        cep.enforce_charmm_energy_term_policies(
            argparse.Namespace(
                periodic_charmm_vdw=True,
                charmm_zero_energy_terms="hbond",
                quiet=True,
            ),
            ml_selection=object(),
            use_pbc=False,
            cubic_box_side_A=None,
        )


# --- SKIPE ELEC IMEL when every CHARMM charge is zero (all-ML jax_mic) ---------


@pytest.fixture
def skipe_registry(monkeypatch):
    """Fresh SKIPE registry (it mirrors process-global CHARMM state)."""
    from mmml.interfaces.pycharmmInterface.mlpot import charmm_energy_policy as cep

    monkeypatch.setattr(cep, "_SKIPPED_CHARMM_TERMS", set())
    monkeypatch.delenv(cep.KEEP_CHARMM_ELEC_ENV, raising=False)
    return cep


def _fake_pycharmm_with_charges(monkeypatch, charges):
    scripts: list[str] = []
    fake_pycharmm = types.ModuleType("pycharmm")
    fake_lingo = types.ModuleType("pycharmm.lingo")
    fake_lingo.charmm_script = lambda s: scripts.append(s)
    fake_pycharmm.lingo = fake_lingo
    fake_pycharmm.psf = mock.Mock()
    fake_pycharmm.psf.get_charges.return_value = list(charges)
    monkeypatch.setitem(sys.modules, "pycharmm", fake_pycharmm)
    monkeypatch.setitem(sys.modules, "pycharmm.lingo", fake_lingo)
    return scripts


@pytest.mark.parametrize(
    ("mode", "charges", "expected"),
    [
        ("jax_mic", [0.0, 0.0, 0.0], True),  # all-ML jax_mic: ELEC/IMEL are exactly zero
        ("jax_mic", [0.0, -0.1, 0.1], False),  # MM atoms keep charges: CHARMM ELEC is live
        ("jax_mic", [], False),
        ("periodic_external", [0.0, 0.0, 0.0], False),  # CHARMM owns VDW: no vdw policy
    ],
)
def test_charmm_elec_redundant_policy(mode, charges, expected):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
        charmm_elec_redundant,
        resolve_charmm_energy_term_policies,
    )

    args = argparse.Namespace(
        mm_nonbond_mode=mode, periodic_charmm_vdw=True, charmm_zero_energy_terms=None
    )
    policies = resolve_charmm_energy_term_policies(args)
    assert charmm_elec_redundant(policies, charges) is expected


def test_charmm_elec_redundant_needs_charges():
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
        POLICY_REGISTRY,
        charmm_elec_redundant,
    )

    assert charmm_elec_redundant([POLICY_REGISTRY["vdw"]], None) is False
    assert charmm_elec_redundant([POLICY_REGISTRY["elec"]], [0.0]) is False


def test_skip_redundant_charmm_elec_issues_uppercase_skipe(skipe_registry, monkeypatch):
    cep = skipe_registry
    scripts = _fake_pycharmm_with_charges(monkeypatch, [0.0] * 9)
    skipped = cep.skip_redundant_charmm_elec([cep.POLICY_REGISTRY["vdw"]])
    assert skipped == ["ELEC", "IMEL"]
    # Lowercase commands are silent no-ops through eval_charmm_script.
    assert scripts == ["SKIPE ELEC IMEL"]
    assert cep.charmm_skipped_terms() == frozenset({"ELEC", "IMEL"})


def test_skip_redundant_charmm_elec_keeps_live_charges(skipe_registry, monkeypatch):
    cep = skipe_registry
    scripts = _fake_pycharmm_with_charges(monkeypatch, [0.0, -0.2, 0.2])
    assert cep.skip_redundant_charmm_elec([cep.POLICY_REGISTRY["vdw"]]) == []
    assert scripts == []
    assert cep.charmm_skipped_terms() == frozenset()


def test_skip_redundant_charmm_elec_env_opt_out(skipe_registry, monkeypatch):
    cep = skipe_registry
    scripts = _fake_pycharmm_with_charges(monkeypatch, [0.0] * 3)
    monkeypatch.setenv(cep.KEEP_CHARMM_ELEC_ENV, "1")
    assert cep.skip_redundant_charmm_elec([cep.POLICY_REGISTRY["vdw"]]) == []
    assert scripts == []


def test_apply_before_pbc_skips_elec_after_vdw_for_zero_charges(
    skipe_registry, tmp_path: Path, monkeypatch
):
    cep = skipe_registry
    scripts = _fake_pycharmm_with_charges(monkeypatch, [0.0] * 9)
    src = tmp_path / "par.prm"
    src.write_text("NONBONDED nbxmod 5\nCL     0.0       -0.1200     2.4700\nEND\n")
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap.cgenff_prm_path",
        lambda: src,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.nbonds_config.read_cgenff_prm",
        lambda path, append=False: None,
    )

    class _Silent:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_silent_command",
        lambda: _Silent(),
    )
    applied = cep.apply_charmm_energy_term_policies_before_pbc_finalize(
        argparse.Namespace(
            mm_nonbond_mode="jax_mic",
            periodic_charmm_vdw=False,
            charmm_zero_energy_terms=None,
            quiet=True,
            output_dir=tmp_path,
        ),
        ml_selection=object(),
    )
    assert applied == ["vdw"]
    skipes = [s for s in scripts if s.startswith("SKIPE")]
    assert skipes == ["SKIPE VDW IMNB", "SKIPE ELEC IMEL"]
    assert "USER" not in "\n".join(scripts)
    assert cep.charmm_skipped_terms() == frozenset({"VDW", "IMNB", "ELEC", "IMEL"})


def test_route_keeps_skipped_term_buckets_in_user(skipe_registry, monkeypatch):
    """CHARMM adds a routed bucket only to active terms: skipped ones stay in USER."""
    from mmml.interfaces.pycharmmInterface.mlpot import charmm_eterm_routing as r

    cep = skipe_registry
    pushed: list[dict] = []
    monkeypatch.setattr(r, "push_mlpot_nb_components_to_charmm", lambda **kw: pushed.append(kw))
    monkeypatch.delenv("MMML_MLPOT_ROUTE_MM_ETERMS", raising=False)
    comps = {
        "vdw_primary": -3.0,
        "vdw_image": -1.0,
        "elec_primary": -2.0,
        "elec_image": -0.5,
        "mm_total": -6.5,
    }
    # Nothing skipped: the whole MM bucket leaves USER.
    assert r.route_mlpot_callback_energy_kcalmol(-100.0, dict(comps)) == pytest.approx(-93.5)
    # VDW/IMNB skipped: only ELEC/IMEL are routed.
    cep._SKIPPED_CHARMM_TERMS.update({"VDW", "IMNB"})
    assert r.route_mlpot_callback_energy_kcalmol(-100.0, dict(comps)) == pytest.approx(-97.5)
    assert pushed[-1]["vdw_primary_kcal"] == 0.0 and pushed[-1]["vdw_image_kcal"] == 0.0
    # All four skipped: USER keeps the full energy.
    cep._SKIPPED_CHARMM_TERMS.update({"ELEC", "IMEL"})
    assert r.route_mlpot_callback_energy_kcalmol(-100.0, dict(comps)) == pytest.approx(-100.0)
