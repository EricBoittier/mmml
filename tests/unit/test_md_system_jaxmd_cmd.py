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
        steps_per_recording=800,
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
    assert "--jaxmd-pbc-minimize-steps" in argv
    assert argv[argv.index("--jaxmd-pbc-minimize-steps") + 1] == "300"
    assert "--steps-per-recording" in argv
    assert argv[argv.index("--steps-per-recording") + 1] == "800"
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


def test_build_command_jaxmd_forwards_lr_solver_and_mm_charge_mode() -> None:
    """Regression: these were silently dropped for jaxmd/ase (only pycharmm
    got --lr-solver; --mm-charge-mode was forwarded nowhere at all), so a
    YAML defaults: {lr_solver: ewald, mm_charge_mode: latent} campaign job
    silently ran with mic + fixed CGenFF charges instead."""
    from mmml.cli.run.md_system import build_command

    backend, argv = build_command(
        _jaxmd_args(lr_solver="ewald", mm_charge_mode="latent", mm_charge_correction=False)
    )
    assert backend == "jaxmd"
    assert "--lr-solver" in argv
    assert argv[argv.index("--lr-solver") + 1] == "ewald"
    assert "--mm-charge-mode" in argv
    assert argv[argv.index("--mm-charge-mode") + 1] == "latent"
    assert "--mm-charge-correction" not in argv
    assert "--ewald-omit-self" not in argv

    backend, argv = build_command(
        _jaxmd_args(lr_solver="ewald", ewald_omit_self=True, mm_charge_mode="fixed")
    )
    assert "--ewald-omit-self" in argv
    assert argv[argv.index("--mm-charge-mode") + 1] == "fixed"

    backend, argv = build_command(_jaxmd_args(mm_charge_correction=True))
    assert "--mm-charge-correction" in argv


def test_jaxmd_jargs_forwards_mm_charge_mode_into_runner() -> None:
    """NVE Hellmann–Feynman preflight reads args.mm_charge_mode on the
    SimpleNamespace passed to set_up_nhc_sim_routine. If missing, it defaults
    to fixed and never freezes q_MM — the gate then fails for q0/latent*."""
    from pathlib import Path

    src = Path("mmml/cli/run/md_pbc_suite/jaxmd.py").read_text(encoding="utf-8")
    assert "jargs = SimpleNamespace(" in src
    jargs_start = src.index("jargs = SimpleNamespace(")
    # End at the call that consumes jargs (avoids nested-paren fragility).
    jargs_end = src.index("set_up_nhc_sim_routine(", jargs_start)
    jargs_block = src[jargs_start:jargs_end]
    assert 'mm_charge_mode=getattr(args, "mm_charge_mode"' in jargs_block


def test_jaxmd_and_ase_suites_accept_lr_solver_and_mm_charge_mode_flags() -> None:
    """The actual downstream parsers (md_pbc_suite/{jaxmd,ase}.py) must
    recognize these flags -- build_command forwarding them is necessary but
    not sufficient if the consuming argparse doesn't define them too."""
    from mmml.cli.run.md_pbc_suite import ase as ase_suite
    from mmml.cli.run.md_pbc_suite import jaxmd as jaxmd_suite

    common_argv = [
        "--lr-solver", "ewald",
        "--ewald-omit-self",
        "--mm-charge-mode", "latent",
        "--composition", "ACO:2",
        "--checkpoint", "/tmp/definitely-not-a-real-checkpoint",
        "--output-dir", "/tmp/out",
    ]
    for suite in (jaxmd_suite, ase_suite):
        with pytest.raises(SystemExit) as exc:
            suite.main(common_argv)
        # argparse rejection raises SystemExit(2) with a stderr usage dump;
        # getting *past* parsing means the failure is the (expected) missing
        # checkpoint, not "unrecognized arguments".
        assert "not found" in str(exc.value) or "Checkpoint" in str(exc.value)


def test_jaxmd_warmup_forwards_include_mm_flag() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    jaxmd_src = (root / "mmml/cli/run/md_pbc_suite/jaxmd.py").read_text(encoding="utf-8")
    warmup_block = jaxmd_src.split("warmup_hybrid_spherical_cutoff(", 1)[1][:400]
    assert "doMM=include_mm" in warmup_block
    assert 'getattr(args, "include_mm", True)' in jaxmd_src


def test_jaxmd_setup_calculator_forwards_ewald_include_self() -> None:
    from pathlib import Path

    src = Path("mmml/cli/run/md_pbc_suite/jaxmd.py").read_text(encoding="utf-8")
    assert "--ewald-omit-self" in src
    assert "ewald_include_self=not bool(getattr(args, \"ewald_omit_self\"" in src
    assert "ewald_include_intra=not bool(getattr(args, \"ewald_omit_self\"" in src


def test_ase_setup_calculator_forwards_ewald_include_intra() -> None:
    """jaxmd/ase must match hybrid_mlpot: --ewald-omit-self drops intra Coulomb."""
    from pathlib import Path

    ase_src = Path("mmml/cli/run/md_pbc_suite/ase.py").read_text(encoding="utf-8")
    assert "ewald_include_intra=not bool(getattr(args, \"ewald_omit_self\"" in ase_src
    assert "ewald_include_intra=bool(ewald_include_intra)" in ase_src
    # Signature must accept the kwarg (not only the call site).
    assert "ewald_include_intra: bool = True" in ase_src

def test_build_command_jaxmd_forwards_ml_gpu_and_profile_flags() -> None:
    from mmml.cli.run.md_system import build_command

    backend, argv = build_command(
        _jaxmd_args(ml_gpu_count=2, ml_batch_size=256, mlpot_profile=True)
    )
    assert backend == "jaxmd"
    assert "--ml-gpu-count" in argv
    assert argv[argv.index("--ml-gpu-count") + 1] == "2"
    assert "--ml-batch-size" in argv
    assert argv[argv.index("--ml-batch-size") + 1] == "256"
    assert "--mlpot-profile" in argv


def test_build_command_jaxmd_forwards_fire_min_steps() -> None:
    from mmml.cli.run.md_system import build_command

    backend, argv = build_command(_jaxmd_args(fire_min_steps=1000, fire_min_maxstep=0.05))
    assert backend == "jaxmd"
    assert "--fire-min-steps" in argv
    assert argv[argv.index("--fire-min-steps") + 1] == "1000"
    assert "--fire-min-maxstep" in argv
    assert argv[argv.index("--fire-min-maxstep") + 1] == "0.05"
