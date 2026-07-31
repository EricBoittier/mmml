"""Tests for the ``mmml.cli.run.md_system_unified`` opt-in unified-stack path."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from mmml.cli.run.md_system_unified import (
    check_md_system_args_supported,
    format_npt_volume_pressure_line,
    npt_volume_ratio_ok,
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


def test_check_supported_allows_continue_from():
    check_md_system_args_supported(_args(continue_from="artifacts/nvt/handoff/state.npz"))


def test_apply_incoming_handoff_overlays_geometry(monkeypatch):
    from mmml.cli.run.md_handoff import MdHandoffState
    from mmml.cli.run.md_system_unified import _apply_incoming_handoff
    from mmml.md.system import MolecularSystem

    system = MolecularSystem(
        R=np.zeros((3, 3), dtype=np.float64),
        Z=np.array([8, 1, 1], dtype=np.int32),
        box=np.eye(3) * 10.0,
        mol_id=np.array([0, 0, 0], dtype=np.int32),
    )
    handoff = MdHandoffState(
        positions=np.ones((3, 3), dtype=np.float64),
        atomic_numbers=np.array([8, 1, 1], dtype=np.int32),
        cell=np.eye(3) * 12.0,
        pbc=True,
        metadata={"source": "unit-test"},
    )
    monkeypatch.setattr(
        "mmml.cli.run.md_system_unified._resolve_handoff_in",
        lambda args: handoff,
    )
    out, from_h = _apply_incoming_handoff(argparse.Namespace(continue_from=None), system)
    assert from_h is True
    assert np.allclose(out.R, 1.0)
    assert np.allclose(out.box, np.eye(3) * 12.0)


def test_publish_unified_handoff_sets_context():
    from mmml.cli.run.md_handoff import clear_handoff_context, get_handoff_out
    from mmml.cli.run.md_system_unified import _publish_unified_handoff
    from mmml.md.system import MolecularSystem

    clear_handoff_context()
    system = MolecularSystem(
        R=np.zeros((2, 3), dtype=np.float64),
        Z=np.array([1, 1], dtype=np.int32),
        box=np.eye(3) * 8.0,
        mol_id=np.array([0, 0], dtype=np.int32),
    )

    class _Traj:
        metadata = {
            "positions": np.stack([np.zeros((2, 3)), np.ones((2, 3))], axis=0),
            "boxes": np.stack([np.eye(3) * 8.0, np.eye(3) * 9.0], axis=0),
        }

    _publish_unified_handoff(argparse.Namespace(temperature=310.0), system, _Traj())
    out = get_handoff_out()
    assert out is not None
    assert np.allclose(out.positions, 1.0)
    assert np.allclose(out.cell, np.eye(3) * 9.0)
    assert out.velocities is None
    clear_handoff_context()


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


def test_npt_volume_ratio_ok_and_format_line():
    ok, ratio = npt_volume_ratio_ok([1000.0, 1100.0])
    assert ok and abs(ratio - 1.1) < 1e-12
    ok, ratio = npt_volume_ratio_ok([1000.0, 3000.0])
    assert not ok and abs(ratio - 3.0) < 1e-12
    ok, ratio = npt_volume_ratio_ok([1000.0, float("nan")])
    assert not ok

    line = format_npt_volume_pressure_line(
        {
            "volumes_A3": np.array([8000.0, 8100.0]),
            "pressures_bar": np.array([1.2, 0.9]),
            "pressures_kin_bar": np.array([100.0, 110.0]),
            "pressures_vir_bar": np.array([-98.8, -109.1]),
            "target_pressure_bar": 1.0,
        }
    )
    assert line is not None
    assert "V0=8000" in line
    assert "Vfinal/V0=" in line
    assert "P0=" in line
    assert "Pkin0=" in line
    assert "Pvir0=" in line
    assert "P_target=1" in line


def test_fire_minimize_picks_best_energy_frame(monkeypatch):
    """Packmol cold-start premin must restore the lowest-energy FIRE frame."""
    from mmml.cli.run.md_system_unified import _fire_minimize_system
    from mmml.md.config import EnsembleSpec, RunConfig
    from mmml.md.system import MolecularSystem, SystemSpec

    system = MolecularSystem(
        R=np.zeros((2, 3), dtype=np.float64),
        Z=np.array([1, 1], dtype=np.int32),
        box=np.eye(3) * 10.0,
        mol_id=np.array([0, 0], dtype=np.int32),
    )
    positions = np.stack(
        [
            np.zeros((2, 3)),
            np.ones((2, 3)),
            2.0 * np.ones((2, 3)),
        ],
        axis=0,
    )
    energies = np.array([10.0, -5.0, 1.0])

    class _Traj:
        metadata = {"positions": positions, "energies": energies}

    monkeypatch.setattr(
        "mmml.md.assemble.assemble_and_run",
        lambda *a, **k: _Traj(),
    )
    run_config = RunConfig(
        system=SystemSpec(builder="packmol"),
        terms=("ml_intra",),
        ensemble=EnsembleSpec(
            ensemble="nvt",
            space="pbc",
            temperature_K=300.0,
            dt_fs=0.25,
            n_steps=10,
            thermostat="langevin",
            params={"seed": 0, "float64": True},
        ),
    )
    out = _fire_minimize_system(
        argparse.Namespace(jaxmd_minimize_steps=50, seed=0),
        run_config,
        system,
        ctx=object(),
        term_kwargs=None,
    )
    assert np.allclose(out.R, positions[1])


def test_run_unified_jaxmd_fails_fast_on_unsupported():
    """The unsupported-combo check must run before any CHARMM build."""
    with pytest.raises(NotImplementedError, match="template-pdb"):
        run_unified_jaxmd(_args(template_pdb="foo.pdb"))


def test_run_unified_pins_mlpot_device_context(monkeypatch):
    """Energy/MD must run under mlpot_jax_device_context (GPU by default)."""
    from contextlib import contextmanager
    from unittest import mock

    from mmml.md.system import MolecularSystem

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

    toy = MolecularSystem(
        R=np.zeros((3, 3), dtype=np.float64),
        Z=np.array([8, 1, 1], dtype=np.int32),
        box=None,
        mol_id=np.array([0, 0, 0], dtype=np.int32),
        monomer_indices=[np.array([0, 1, 2], dtype=np.int32)],
    )

    monkeypatch.setattr(
        "mmml.cli.run.md_system_unified.build_packmol_system_with_ffparams",
        lambda *a, **k: toy,
    )
    ens = mock.Mock(
        ensemble="nve",
        thermostat=None,
        dt_fs=1.0,
        n_steps=10,
        temperature_K=300.0,
        pressure_bar=1.0,
        params={"float64": True},
    )
    monkeypatch.setattr(
        "mmml.md.lowering.runconfig_from_md_system_args",
        lambda args: mock.Mock(
            terms=("ml_intra", "mm_nonbonded"),
            system=mock.Mock(builder="packmol"),
            ensemble=ens,
        ),
    )
    monkeypatch.setattr(
        "mmml.cli.run.md_system_unified.build_energy_context",
        lambda *a, **k: mock.Mock(),
    )

    class _Traj:
        n_frames = 2
        metadata = {
            "energies": np.array([-1.0, -1.1]),
            "positions": np.stack([toy.R, toy.R], axis=0),
        }

    monkeypatch.setattr(
        "mmml.md.assemble.assemble_and_run",
        lambda *a, **k: _Traj(),
    )

    rc = run_unified_jaxmd(_args(composition="TIP3:1", checkpoint=str(CKPT)))
    assert rc == 0
    assert entered["n"] == 1


def test_run_unified_handoff_skips_fire_and_applies_R(monkeypatch):
    """Campaign continue-from must overlay R/box and skip cold-start FIRE."""
    from contextlib import contextmanager
    from unittest import mock

    from mmml.cli.run.md_handoff import MdHandoffState, clear_handoff_context, get_handoff_out
    from mmml.md.system import MolecularSystem

    clear_handoff_context()
    toy = MolecularSystem(
        R=np.zeros((3, 3), dtype=np.float64),
        Z=np.array([8, 1, 1], dtype=np.int32),
        box=np.eye(3) * 10.0,
        mol_id=np.array([0, 0, 0], dtype=np.int32),
    )
    handoff = MdHandoffState(
        positions=2.0 * np.ones((3, 3), dtype=np.float64),
        atomic_numbers=toy.Z.copy(),
        cell=np.eye(3) * 11.0,
        pbc=True,
    )
    captured: dict = {}
    fire_calls = {"n": 0}

    @contextmanager
    def _fake_ctx():
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
    monkeypatch.setattr(
        "mmml.cli.run.md_system_unified.check_md_system_args_supported",
        lambda args: None,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.import_pycharmm.ensure_pycharmm_loaded",
        lambda: True,
    )
    monkeypatch.setattr(
        "mmml.cli.run.md_system_unified.build_packmol_system_with_ffparams",
        lambda *a, **k: toy,
    )
    ens = mock.Mock(
        ensemble="nvt",
        thermostat="langevin",
        dt_fs=0.25,
        n_steps=10,
        temperature_K=300.0,
        pressure_bar=1.0,
        params={"float64": True},
    )
    monkeypatch.setattr(
        "mmml.md.lowering.runconfig_from_md_system_args",
        lambda args: mock.Mock(
            terms=("ml_intra", "mm_nonbonded"),
            system=mock.Mock(builder="packmol"),
            ensemble=ens,
        ),
    )
    monkeypatch.setattr(
        "mmml.cli.run.md_system_unified.build_energy_context",
        lambda *a, **k: mock.Mock(),
    )
    monkeypatch.setattr(
        "mmml.cli.run.md_system_unified._resolve_handoff_in",
        lambda args: handoff,
    )
    def _fire(*_a, **_k):
        fire_calls["n"] += 1
        return toy

    monkeypatch.setattr(
        "mmml.cli.run.md_system_unified._fire_minimize_system",
        _fire,
    )

    class _Traj:
        n_frames = 2
        metadata = {
            "energies": np.array([-1.0, -1.1]),
            "positions": np.stack(
                [2.0 * np.ones((3, 3)), 3.0 * np.ones((3, 3))], axis=0
            ),
            "boxes": np.stack([np.eye(3) * 11.0, np.eye(3) * 11.5], axis=0),
        }

    def _assemble(run_config, *, system=None, ctx=None, term_kwargs=None, **_k):
        captured["system"] = system
        return _Traj()

    monkeypatch.setattr("mmml.md.assemble.assemble_and_run", _assemble)

    rc = run_unified_jaxmd(
        _args(
            composition="TIP3:1",
            checkpoint=str(CKPT),
            continue_from="dummy.npz",
            jaxmd_minimize_steps=250,
            handoff_pre_minimize=False,
        )
    )
    assert rc == 0
    assert fire_calls["n"] == 0
    assert np.allclose(captured["system"].R, 2.0)
    assert np.allclose(captured["system"].box, np.eye(3) * 11.0)
    out = get_handoff_out()
    assert out is not None
    assert np.allclose(out.positions, 3.0)
    assert np.allclose(out.cell, np.eye(3) * 11.5)
    clear_handoff_context()


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
