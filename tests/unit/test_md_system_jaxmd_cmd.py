"""Unit tests for md-system → JAX-MD argv forwarding."""

from __future__ import annotations

from argparse import Namespace

import pytest

from mmml.interfaces.pycharmmInterface.cutoffs import (
    DEFAULT_ML_SWITCH_WIDTH,
    DEFAULT_MM_SWITCH_ON,
    DEFAULT_MM_SWITCH_WIDTH,
)


def _jaxmd_args(**overrides) -> Namespace:
    base = dict(
        backend="jaxmd",
        setup="pbc_nvt",
        composition="DCM:20",
        spacing=5.0,
        ps=50.0,
        dt_fs=0.25,
        temperature=130.0,
        pressure=1.0,
        traj_chunk_frames=0,
        n_molecules=20,
        box_size=32.0,
        checkpoint="/tmp/ckpt.json",
        output_dir="/tmp/out",
        template_pdb=None,
        seed=123,
        min_intermonomer_atom_distance=0.1,
        packmol=None,
        packmol_placement=None,
        packmol_sphere=None,
        packmol_radius=None,
        packmol_tolerance=2.0,
        packmol_center=None,
        flat_bottom_radius=None,
        flat_bottom_k=1.0,
        flat_bottom_mode="system",
        nvt_integrator="nhc",
        traj_export_molecular_wrap=False,
        skip_jit_warmup=False,
        extra_args=[],
        mm_switch_on=9.0,
        mm_switch_width=1.5,
        ml_switch_width=1.0,
        handoff_pre_minimize=True,
        continue_velocities=True,
        handoff_quality_gate=True,
        handoff_quality_fmax_eVA=1.0,
        handoff_quality_action="minimize",
        handoff_velocity_remove_drift=True,
        handoff_require_cell=False,
        jaxmd_minimize_steps=500,
        jaxmd_pbc_minimize_steps=300,
        calculator_pre_minimize=True,
        charmm_pre_minimize=True,
        pre_min_fmax=0.1,
        pre_min_steps=50,
    )
    base.update(overrides)
    return Namespace(**base)


def test_build_command_jaxmd_forwards_handoff_and_cutoff_flags() -> None:
    from mmml.cli.run.md_system import build_command

    backend, argv = build_command(_jaxmd_args())
    assert backend == "jaxmd"
    assert "--mm-switch-on" in argv
    assert argv[argv.index("--mm-switch-on") + 1] == "9.0"
    assert "--ml-switch-width" in argv
    assert argv[argv.index("--ml-switch-width") + 1] == "1.0"
    assert "--handoff-pre-minimize" in argv
    assert "--handoff-quality-gate" in argv
    assert "--jaxmd-minimize-steps" in argv
    assert argv[argv.index("--jaxmd-minimize-steps") + 1] == "500"
    assert "--continue-velocities" in argv


def test_build_command_jaxmd_forwards_default_cutoffs_from_namespace() -> None:
    from mmml.cli.run.md_system import build_command

    backend, argv = build_command(
        _jaxmd_args(
            mm_switch_on=DEFAULT_MM_SWITCH_ON,
            mm_switch_width=DEFAULT_MM_SWITCH_WIDTH,
            ml_switch_width=DEFAULT_ML_SWITCH_WIDTH,
            handoff_pre_minimize=False,
            handoff_quality_gate=False,
        )
    )
    assert backend == "jaxmd"
    assert argv[argv.index("--mm-switch-on") + 1] == str(DEFAULT_MM_SWITCH_ON)
    assert argv[argv.index("--mm-switch-width") + 1] == str(DEFAULT_MM_SWITCH_WIDTH)
    assert argv[argv.index("--ml-switch-width") + 1] == str(DEFAULT_ML_SWITCH_WIDTH)
    assert "--handoff-pre-minimize" not in argv
