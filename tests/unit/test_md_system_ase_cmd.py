"""Unit tests for md-system → ASE argv forwarding."""

from __future__ import annotations

from argparse import Namespace

from mmml.interfaces.pycharmmInterface.cutoffs import (
    DEFAULT_ML_SWITCH_WIDTH,
    DEFAULT_MM_SWITCH_ON,
    DEFAULT_MM_SWITCH_WIDTH,
)


def _ase_args(**overrides) -> Namespace:
    base = dict(
        backend="ase",
        setup="pbc_nve",
        composition="DCM:5",
        spacing=5.0,
        ps=2.0,
        dt_fs=0.25,
        traj_chunk_frames=0,
        n_molecules=5,
        box_size=25.0,
        checkpoint="/tmp/ckpt.json",
        output_dir="/tmp/out",
        template_pdb=None,
        seed=42,
        min_intermonomer_atom_distance=0.1,
        packmol=None,
        packmol_placement="cube",
        packmol_sphere=None,
        packmol_radius=6.9,
        packmol_tolerance=1.0,
        packmol_center=None,
        flat_bottom_radius=None,
        flat_bottom_k=1.0,
        flat_bottom_mode="system",
        nvt_integrator="auto",
        traj_export_molecular_wrap=False,
        skip_jit_warmup=False,
        extra_args=["--lr-solver", "mic", "--include-mm", "--nve-temp-K", "300"],
        mm_switch_on=DEFAULT_MM_SWITCH_ON,
        mm_switch_width=DEFAULT_MM_SWITCH_WIDTH,
        ml_switch_width=DEFAULT_ML_SWITCH_WIDTH,
        handoff_pre_minimize=False,
        continue_velocities=True,
        handoff_quality_gate=False,
        handoff_quality_fmax_eVA=1.0,
        handoff_quality_action="minimize",
        handoff_velocity_remove_drift=True,
        handoff_require_cell=False,
        jaxmd_minimize_steps=200,
        jaxmd_pbc_minimize_steps=200,
        calculator_pre_minimize=True,
        charmm_pre_minimize=True,
        include_mm=True,
        pre_min_fmax=0.1,
        pre_min_steps=50,
    )
    base.update(overrides)
    return Namespace(**base)


def test_build_command_ase_uses_mm_cutoff_not_switch_width() -> None:
    from mmml.cli.run.md_system import build_command

    backend, argv = build_command(_ase_args())
    assert backend == "ase"
    assert "--mm-cutoff" in argv
    assert argv[argv.index("--mm-cutoff") + 1] == str(DEFAULT_MM_SWITCH_WIDTH)
    assert "--ml-cutoff" in argv
    assert argv[argv.index("--ml-cutoff") + 1] == str(DEFAULT_ML_SWITCH_WIDTH)
    assert "--mm-switch-width" not in argv
    assert "--ml-switch-width" not in argv
    assert "--jaxmd-minimize-steps" not in argv
    assert "--calculator-pre-minimize" not in argv
    assert "--lr-solver" not in argv
    assert "--include-mm" in argv
    assert "--nve-temp-K" in argv
