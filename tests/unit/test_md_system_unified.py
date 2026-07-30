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
        sampler="md",
        ff=None,
        mbd_checkpoint=None,
        mbd_weight=1.0,
        multipole_checkpoint=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# --- config validation (fast, no CHARMM) ------------------------------------


def test_check_supported_accepts_default():
    check_md_system_args_supported(_args())  # no error


def test_check_supported_rejects_pyxtal_builder():
    # from_pdb joined packmol as a supported builder; pyxtal is still rejected.
    with pytest.raises(NotImplementedError, match="pyxtal"):
        check_md_system_args_supported(_args(builder="pyxtal"))


def test_check_supported_rejects_template_pdb():
    with pytest.raises(NotImplementedError, match="template-pdb"):
        check_md_system_args_supported(_args(template_pdb="foo.pdb"))


def test_check_supported_rejects_continue_from():
    with pytest.raises(NotImplementedError, match="continue-from"):
        check_md_system_args_supported(_args(continue_from="run1"))


def test_check_supported_requires_checkpoint_for_ml_intra():
    with pytest.raises(ValueError, match="checkpoint"):
        check_md_system_args_supported(_args(checkpoint=None, sampler="md", ff=None))


def test_check_supported_cgenff_rigid_without_checkpoint():
    check_md_system_args_supported(
        _args(checkpoint=None, sampler="rigid", ff="cgenff")
    )


def test_check_supported_zbl_mbd_multipoles_without_spooky_checkpoint():
    check_md_system_args_supported(
        _args(checkpoint=None, sampler="rigid", ff="zbl-mbd-multipoles")
    )


def test_run_unified_jaxmd_fails_fast_on_unsupported():
    """The unsupported-combo check must run before any CHARMM build."""
    with pytest.raises(NotImplementedError, match="template-pdb"):
        run_unified_jaxmd(_args(template_pdb="foo.pdb"))


def test_run_unified_pins_mlpot_device_context(monkeypatch):
    """Energy/MD must run under mlpot_jax_device_context (GPU by default)."""
    from contextlib import contextmanager
    from unittest import mock

    entered = {"n": 0}

    @contextmanager
    def _fake_ctx():
        entered["n"] += 1
        yield mock.Mock(platform="cpu", id=0)

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_context",
        _fake_ctx,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.apply_mlpot_jax_platform_env",
        lambda quiet=True: "cpu",
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.print_jax_device_banner",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.reset_mlpot_device_fallback_flag",
        lambda: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_device_context_fell_back_to_cpu",
        lambda: False,
    )

    with pytest.raises(NotImplementedError, match="template-pdb"):
        # Fails before context — prove early validation still first.
        run_unified_jaxmd(_args(template_pdb="foo.pdb"))
    assert entered["n"] == 0

    # Patch past validation + CHARMM into the energy path.
    monkeypatch.setattr(
        "mmml.cli.run.md_system_unified.check_md_system_args_supported",
        lambda args: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.import_pycharmm.ensure_pycharmm_loaded",
        lambda: True,
    )

    class _Sys:
        n_atoms = 3
        Z = np.array([8, 1, 1], dtype=np.int32)
        R = np.zeros((3, 3), dtype=np.float64)
        monomer_indices = [[0, 1, 2]]
        mol_id = np.array([0, 0, 0], dtype=np.int32)
        residue_names = ["TIP3"]
        ff_params = None
        box = None

    monkeypatch.setattr(
        "mmml.cli.run.md_system_unified.build_packmol_system_with_ffparams",
        lambda *a, **k: _Sys(),
    )
    monkeypatch.setattr(
        "mmml.md.lowering.runconfig_from_md_system_args",
        lambda args: mock.Mock(
            terms=("ml_intra", "mm_nonbonded"),
            system=mock.Mock(builder="packmol"),
        ),
    )
    monkeypatch.setattr(
        "mmml.cli.run.md_system_unified.build_energy_context",
        lambda *a, **k: mock.Mock(),
    )

    class _Traj:
        n_frames = 2
        metadata = {"energies": np.array([-1.0, -1.1])}

    monkeypatch.setattr(
        "mmml.md.assemble.assemble_and_run",
        lambda *a, **k: _Traj(),
    )

    rc = run_unified_jaxmd(_args(composition="TIP3:1", checkpoint=str(CKPT)))
    assert rc == 0
    assert entered["n"] == 1


# --- end-to-end integration (real CHARMM build + real checkpoint) ----------


def _pycharmm_or_skip():
    try:
        import pycharmm  # noqa: F401  (triggers libcharmm load)
    except OSError:
        pytest.skip("libcharmm not available")
    if not CKPT.exists():
        pytest.skip(f"checkpoint {CKPT.name} not present")


@pytest.mark.pycharmm
def test_end_to_end_pbc_nve(capsys):
    _pycharmm_or_skip()
    rc = run_unified_jaxmd(_args(setup="pbc_nve", seed=21))
    assert rc == 0
    out = capsys.readouterr().out
    assert "jaxmd-unified" in out


@pytest.mark.pycharmm
def test_end_to_end_pbc_nvt(capsys):
    _pycharmm_or_skip()
    rc = run_unified_jaxmd(_args(setup="pbc_nvt", seed=22, ps=0.02))
    assert rc == 0
    out = capsys.readouterr().out
    assert "jaxmd-unified" in out


@pytest.mark.pycharmm
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
