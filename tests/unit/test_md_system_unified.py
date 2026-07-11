"""Tests for the ``mmml.cli.run.md_system_unified`` opt-in unified-stack path."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from mmml.cli.run.md_system_unified import (
    check_md_system_args_supported,
    run_unified_jaxmd,
)

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "examples" / "sppoky-epoch-0010_params.json"


def _args(**overrides):
    base = dict(
        setup="pbc_nve",
        dt_fs=1.0,
        ps=0.01,
        temperature=300.0,
        pressure=1.0,
        composition="TIP3:4",
        n_molecules=None,
        box_size=15.0,
        builder=None,
        template_pdb=None,
        continue_from=None,
        seed=1,
        checkpoint=str(CKPT),
        output_dir=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# --- config validation (fast, no CHARMM) ------------------------------------


def test_check_supported_accepts_default():
    check_md_system_args_supported(_args())  # no error


def test_check_supported_rejects_pyxtal_builder():
    with pytest.raises(NotImplementedError, match="packmol composition builder"):
        check_md_system_args_supported(_args(builder="pyxtal"))


def test_check_supported_rejects_template_pdb():
    with pytest.raises(NotImplementedError, match="template-pdb"):
        check_md_system_args_supported(_args(template_pdb="foo.pdb"))


def test_check_supported_rejects_continue_from():
    with pytest.raises(NotImplementedError, match="continue-from"):
        check_md_system_args_supported(_args(continue_from="run1"))


def test_check_supported_requires_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        check_md_system_args_supported(_args(checkpoint=None))


def test_run_unified_jaxmd_fails_fast_on_unsupported():
    """The unsupported-combo check must run before any CHARMM build."""
    with pytest.raises(NotImplementedError, match="template-pdb"):
        run_unified_jaxmd(_args(template_pdb="foo.pdb"))


# --- end-to-end integration (real CHARMM build + real checkpoint) ----------


def _pycharmm_or_skip():
    try:
        import pycharmm  # noqa: F401  (triggers libcharmm load)
    except OSError:
        pytest.skip("libcharmm not available")
    if not CKPT.exists():
        pytest.skip(f"checkpoint {CKPT.name} not present")


def test_end_to_end_pbc_nve(capsys):
    _pycharmm_or_skip()
    rc = run_unified_jaxmd(_args(setup="pbc_nve", seed=21))
    assert rc == 0
    out = capsys.readouterr().out
    assert "jaxmd-unified" in out


def test_end_to_end_pbc_nvt(capsys):
    _pycharmm_or_skip()
    rc = run_unified_jaxmd(_args(setup="pbc_nvt", seed=22, ps=0.02))
    assert rc == 0
    out = capsys.readouterr().out
    assert "jaxmd-unified" in out


def test_end_to_end_builds_ffparams():
    """The packmol+PSF helper must produce FFParams, or mm_nonbonded can't run."""
    _pycharmm_or_skip()
    from mmml.cli.run.md_system_unified import build_packmol_system_with_ffparams
    from mmml.md.lowering import runconfig_from_md_system_args

    run_config = runconfig_from_md_system_args(_args(setup="pbc_nve", seed=23))
    system = build_packmol_system_with_ffparams(run_config.system)
    assert system.ff_params is not None
    assert system.n_atoms == 12  # 4 TIP3 waters
    # TIP3 charges: O=-0.834, H=+0.417
    assert np.allclose(sorted(system.ff_params.charges), sorted([-0.834, 0.417, 0.417] * 4))
