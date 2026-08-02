"""Unit tests for md-system → ASE argv forwarding."""

from __future__ import annotations

from argparse import Namespace

import pytest

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
        temperature=300.0,
        pressure=1.0,
        traj_chunk_frames=0,
        n_molecules=5,
        box_size=25.0,
        checkpoint="/tmp/ckpt.json",
        electrostatics_damping_sigma=None,
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


def test_build_command_ase_forwards_do_ml_flags() -> None:
    from mmml.cli.run.md_system import build_command

    backend, argv = build_command(_ase_args(do_ml=False, do_ml_dimer=False, include_mm=False))
    assert backend == "ase"
    assert "--no-do-ml" in argv
    assert "--no-do-ml-dimer" in argv
    assert "--no-include-mm" in argv


def test_build_command_ase_forwards_fire_min_flags_parse() -> None:
    """md-system always forwards FIRE min flags; ASE backend must accept them."""
    from mmml.cli.run.md_pbc_suite.ase import build_parser
    from mmml.cli.run.md_system import build_command

    backend, argv = build_command(
        _ase_args(fire_min_steps=200, fire_min_maxstep=0.2)
    )
    assert backend == "ase"
    assert "--fire-min-steps" in argv
    assert argv[argv.index("--fire-min-steps") + 1] == "200"
    assert "--fire-min-maxstep" in argv
    assert argv[argv.index("--fire-min-maxstep") + 1] == "0.2"

    parsed = build_parser().parse_args(argv)
    assert parsed.fire_min_steps == 200
    assert parsed.fire_min_maxstep == pytest.approx(0.2)


def test_build_command_forwards_electrostatics_damping_sigma() -> None:
    from mmml.cli.run.md_system import build_command

    backend, argv = build_command(_ase_args(electrostatics_damping_sigma=0.0))
    assert backend == "ase"
    assert "--electrostatics-damping-sigma" in argv
    assert argv[argv.index("--electrostatics-damping-sigma") + 1] == "0.0"

    backend, argv = build_command(_ase_args(backend="jaxmd", electrostatics_damping_sigma=0.0))
    assert backend == "jaxmd"
    assert "--electrostatics-damping-sigma" in argv
    assert argv[argv.index("--electrostatics-damping-sigma") + 1] == "0.0"


# --- every forwarded flag must be parseable by the backend it is sent to -----
#
# `md_system.build_command` builds an argv and hands it to a *subprocess*, so a
# flag the backend parser does not know is not a TypeError anyone sees -- it is
# argparse exit 2 inside the child, after the run has already started. That is
# how `--hybrid-hamiltonian` and `--shared-cutoff` shipped: `run_sim` grew both
# options, md_system forwarded them to every backend unconditionally, and
# neither `md_pbc_suite.ase` nor `md_pbc_suite.jaxmd` declared them.
#
# Checking one flag at a time (as the tests above do) only ever catches the
# flag someone remembered to write a test for. These two parse the whole argv.


def _backend_parser(backend: str):
    if backend == "ase":
        from mmml.cli.run.md_pbc_suite.ase import build_parser
    else:
        from mmml.cli.run.md_pbc_suite.jaxmd import build_parser
    return build_parser()


@pytest.mark.parametrize("backend", ["ase", "jaxmd"])
def test_every_forwarded_flag_is_accepted_by_its_backend(backend: str) -> None:
    from mmml.cli.run.md_system import build_command

    # `extra_args` is user passthrough -- the fixture's are ASE-specific -- so
    # it is excluded here. Everything else in the argv is chosen by md_system.
    resolved, argv = build_command(_ase_args(backend=backend, extra_args=[]))

    assert resolved == backend
    # Raises SystemExit(2) on an unknown flag rather than returning.
    _backend_parser(backend).parse_args(argv)


@pytest.mark.parametrize("backend", ["ase", "jaxmd"])
def test_hybrid_hamiltonian_reaches_the_backend_with_its_value(backend: str) -> None:
    """Accepting the flag is not enough -- it has to arrive intact."""
    from mmml.cli.run.md_system import build_command

    _, argv = build_command(
        _ase_args(
            backend=backend,
            extra_args=[],
            hybrid_hamiltonian="shared_cutoff",
            shared_cutoff=6.5,
        )
    )
    parsed = _backend_parser(backend).parse_args(argv)

    assert parsed.hybrid_hamiltonian == "shared_cutoff"
    assert parsed.shared_cutoff == pytest.approx(6.5)


@pytest.mark.parametrize("backend", ["ase", "jaxmd"])
def test_the_backend_default_matches_run_sim(backend: str) -> None:
    """Three parsers declare these options; a default that drifts between them
    silently changes which Hamiltonian a run uses."""
    from mmml.cli.run.run_sim import build_parser as run_sim_parser

    reference = {
        action.dest: action.default
        for action in run_sim_parser()._actions
        if action.dest in ("hybrid_hamiltonian", "shared_cutoff")
    }
    assert reference == {"hybrid_hamiltonian": "handoff", "shared_cutoff": None}

    parsed = _backend_parser(backend).parse_args([])

    assert parsed.hybrid_hamiltonian == reference["hybrid_hamiltonian"]
    assert parsed.shared_cutoff == reference["shared_cutoff"]
