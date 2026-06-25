"""Shared CLI helpers for CHARMM MLpot workflows (tests and ``mmml md-system``)."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Tuple

import numpy as np

from mmml.paths import _package_dir

# Repository root (parent of the installed ``mmml`` package directory).
REPO_ROOT = _package_dir().parent

DEFAULT_RESIDUE = "ACO"
DEFAULT_N_MOLECULES = 2
DEFAULT_SPACING = 4.0
ACO_ATOMS_PER_MONOMER = 10
NVE_TIMESTEP_PS = 0.00025


def add_charmm_output_args(parser: argparse.ArgumentParser) -> None:
    """CLI flags for CHARMM console verbosity."""
    group = parser.add_argument_group("CHARMM console output")
    group.add_argument(
        "--prnlev",
        type=int,
        default=5,
        help="CHARMM PRNLev (0=quiet, 5=verbose; default: 5)",
    )
    group.add_argument(
        "--warnlev",
        type=int,
        default=5,
        help="CHARMM WRNLev (default: 5)",
    )
    group.add_argument(
        "--bomlev",
        type=int,
        default=-2,
        help="CHARMM BOMBlev (default: -2; use 0 to stop on any warning)",
    )
    group.add_argument(
        "--mlpot-profile",
        action="store_true",
        help="Enable profiling of MLpot callbacks and JAX/XLA compilation timers",
    )
    group.add_argument(
        "--nprint",
        type=int,
        default=50,
        help="SD minimization: print energy every N steps (default: 50)",
    )
    group.add_argument(
        "--dyn-nprint",
        type=int,
        default=500,
        help="Dynamics: print energy every N steps (default: 500)",
    )
    group.add_argument(
        "--dyn-iprfrq",
        type=int,
        default=2000,
        help="Dynamics: detailed status every N steps (default: 2000)",
    )
    group.add_argument(
        "--heat-ihtfrq",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Heating with --heat-thermostat scale: CHARMM ihtfrq (velocity rescaling "
            "every N steps). 0 = use --dyn-nprint (or full stage length when --quiet). "
            "Ignored for --heat-thermostat hoover."
        ),
    )
    group.add_argument(
        "--heat-thermostat",
        choices=("scale", "hoover"),
        default="scale",
        help=(
            "Heat-stage temperature control: scale=IHTFRQ velocity rescaling (default); "
            "hoover=CHARMM Hoover NVT (no ihtfrq; vacuum uses hoover reft/tmass, no CPT)."
        ),
    )
    group.add_argument(
        "--heat-firstt",
        type=float,
        default=None,
        metavar="K",
        help=(
            "Heat start temperature (CHARMM FIRSTT). Default: 0.2×--temperature. "
            "Use 0 for a cold start (zero initial velocities, then IHTFRQ scaling)."
        ),
    )
    group.add_argument(
        "--heat-finalt",
        type=float,
        default=None,
        metavar="K",
        help=(
            "Heat end temperature (CHARMM FINALT / TBATH). "
            "Default: --temperature (e.g. 300). DCM:9 stability often uses 240."
        ),
    )
    group.add_argument(
        "--heat-hoover-tmass",
        type=int,
        default=None,
        metavar="M",
        help=(
            "Hoover CPT heat only: thermostat mass tmass in kcal·mol⁻¹·ps². "
            "Default clamps PSF-derived tmass to 400–1200 (tighter than equi). "
            "Lower = stronger T coupling; try 400–800 if heat overshoots."
        ),
    )
    group.add_argument(
        "--heat-comp-damp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimental: COMP prep on high-|F| hydrogens before heat (does not change "
            "iasvel/iasors). Cleared before equi/NVE/prod. Default off — use reference heat."
        ),
    )
    group.add_argument(
        "--heat-comp-hydrogen-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --heat-comp-damp, damp only hydrogens above --heat-comp-force-min "
            "(default). Use --no-heat-comp-hydrogen-only to damp all atom types."
        ),
    )
    group.add_argument(
        "--heat-comp-force-min",
        type=float,
        default=None,
        metavar="KCAL",
        help=(
            "Min |F| (kcal/mol/Å) for heat COMP damp selection (default: 1.0). "
            "Only used with --heat-comp-damp."
        ),
    )
    group.add_argument(
        "--heat-comp-force-scale",
        type=float,
        default=None,
        help="Scale copied forces into COMP during heat (default: 0.01).",
    )
    group.add_argument(
        "--quiet",
        action="store_true",
        help="Shortcut for --prnlev 0 --warnlev 0; coarse mini/dynamics print",
    )
    group.add_argument(
        "--skip-energy-show",
        action="store_true",
        help="Skip CHARMM energy.show() (avoids segfault on MPI/cluster builds).",
    )
    group.add_argument(
        "--show-energy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Print CHARMM energy tables (off by default; use for debugging). "
            "Honors SKIP_CHARMM_ENERGY_SHOW and SLURM/macOS/pytest/OpenMPI guards."
        ),
    )


def resolve_show_energy(args: argparse.Namespace) -> bool:
    """Whether MLpot workflows should call CHARMM energy.show()."""
    if getattr(args, "skip_energy_show", False):
        return False
    if getattr(args, "quiet", False):
        return False
    show = getattr(args, "show_energy", None)
    if show is not None:
        return bool(show)
    return False


def add_flat_bottom_args(parser: argparse.ArgumentParser) -> None:
    """CHARMM MMFP flat-bottom spherical potential (non-PBC)."""
    group = parser.add_argument_group("Flat-bottom sphere (CHARMM MMFP)")
    group.add_argument(
        "--fb-rad",
        type=float,
        default=None,
        metavar="ANG",
        help="MMFP droff radius (Å); inside = no restraint. Omit to disable.",
    )
    group.add_argument(
        "--fb-forc",
        type=float,
        default=1.0,
        help="MMFP wall force constant (default: 1.0)",
    )
    group.add_argument(
        "--fb-selection",
        type=str,
        default="all",
        help="CHARMM atom selection for MMFP wall (default: all).",
    )
    group.add_argument(
        "--fb-center",
        action="store_true",
        default=True,
        help="Center cluster at origin before MMFP (default: on)",
    )
    group.add_argument(
        "--no-fb-center",
        action="store_false",
        dest="fb_center",
        help="Skip COM translation; use with --fb-xref/--fb-yref/--fb-zref",
    )
    group.add_argument("--fb-xref", type=float, default=0.0, help="Sphere center x (Å)")
    group.add_argument("--fb-yref", type=float, default=0.0, help="Sphere center y (Å)")
    group.add_argument("--fb-zref", type=float, default=0.0, help="Sphere center z (Å)")


def add_dynamics_stability_args(parser: argparse.ArgumentParser) -> None:
    """CHARMM dynamics energy-drift kill switch (``ECHECK``)."""
    group = parser.add_argument_group("Dynamics stability (ECHECK)")
    group.add_argument(
        "--echeck",
        type=float,
        default=100.0,
        metavar="KCAL",
        help="Stop dynamics if total energy change exceeds this (kcal/mol); "
        "default 100. Auto-loosened for large clusters (see --no-scale-echeck). "
        "Use --no-echeck to disable.",
    )
    group.add_argument(
        "--no-echeck",
        action="store_true",
        help="Disable ECHECK (CHARMM -1 = no early stop)",
    )
    group.add_argument(
        "--allow-incomplete-dynamics",
        action="store_true",
        help=(
            "Do not fail staged MD when CHARMM stops early (echeck) or the stage DCD "
            "has too few frames. Default: abort with a clear error."
        ),
    )


def resolve_echeck_from_args(args: argparse.Namespace) -> float:
    if getattr(args, "no_echeck", False):
        return -1.0
    return float(getattr(args, "echeck", 100.0))


def add_force_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Force checkpoint (NPZ)")
    group.add_argument(
        "--save-forces-npz",
        action="store_true",
        help="During dynamics, append CHARMM total forces to <output-dir>/forces.npz",
    )
    group.add_argument(
        "--forces-npz-interval",
        type=int,
        default=1,
        help="Save forces every N integration steps (default: 1; align with --dcd-nsavc)",
    )


def add_dcd_save_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("DCD trajectory output")
    group.add_argument(
        "--dcd-nsavc",
        type=int,
        default=1,
        help="Write a DCD frame every N integration/SD steps (default: 1)",
    )
    group.add_argument(
        "--dcd-interval-ps",
        type=float,
        default=None,
        metavar="PS",
        help="Alternative to --dcd-nsavc: save interval in ps (dynamics only)",
    )
    group.add_argument(
        "--dcd-max-frames",
        type=int,
        default=25,
        metavar="N",
        help=(
            "Cap DCD output to about N frames per stage when --dcd-interval-ps is unset "
            "(default: 25; set 0 to allow every-step output with --dcd-nsavc 1)"
        ),
    )
    group.add_argument(
        "--rescue-old-dcd",
        action="store_true",
        help=(
            "Before each dynamics stage, rename an existing stage DCD to "
            "*.rescued.N.dcd instead of deleting it (default: remove prior DCD)."
        ),
    )


def resolve_dcd_nsavc(
    *,
    dcd_nsavc: int,
    dcd_interval_ps: float | None = None,
    timestep_ps: float | None = None,
    nstep: int | None = None,
    dcd_max_frames: int | None = 25,
) -> int:
    if dcd_interval_ps is not None:
        if timestep_ps is None or timestep_ps <= 0:
            raise ValueError("timestep_ps required when using --dcd-interval-ps")
        nsavc = int(round(float(dcd_interval_ps) / float(timestep_ps)))
    else:
        nsavc = int(dcd_nsavc)
    nsavc = max(1, nsavc)
    if (
        dcd_interval_ps is None
        and dcd_max_frames is not None
        and int(dcd_max_frames) > 0
        and nstep is not None
        and int(nstep) > 1
    ):
        min_nsavc = max(1, (int(nstep) + int(dcd_max_frames) - 1) // int(dcd_max_frames))
        nsavc = max(nsavc, min_nsavc)
    if nstep is not None:
        n = int(nstep)
        if n > 1:
            nsavc = min(nsavc, n - 1)
        else:
            nsavc = min(nsavc, n)
    return nsavc


def resolve_dcd_nsavc_for_args(
    args: argparse.Namespace,
    *,
    nstep: int | None = None,
    timestep_ps: float | None = None,
) -> int:
    """Resolve DCD ``nsavc`` from a parsed ``md-system`` / PyCHARMM namespace."""
    return resolve_dcd_nsavc(
        dcd_nsavc=int(getattr(args, "dcd_nsavc", 1)),
        dcd_interval_ps=getattr(args, "dcd_interval_ps", None),
        timestep_ps=timestep_ps,
        nstep=nstep,
        dcd_max_frames=getattr(args, "dcd_max_frames", 25),
    )


def apply_charmm_output_from_args(args: argparse.Namespace) -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import apply_charmm_verbosity

    bomlev = int(getattr(args, "bomlev", -2))
    if getattr(args, "quiet", False):
        apply_charmm_verbosity(prnlev=0, warnlev=0, bomlev=bomlev)
    else:
        apply_charmm_verbosity(
            prnlev=int(getattr(args, "prnlev", 5)),
            warnlev=int(getattr(args, "warnlev", 5)),
            bomlev=bomlev,
        )
    if getattr(args, "quiet", False):
        mini_nstep = getattr(args, "mini_nstep", getattr(args, "nstep", 100))
        return max(1, int(mini_nstep))
    return max(1, int(getattr(args, "nprint", 50)))


def resolve_nve_boltzmann_temp(
    args: argparse.Namespace,
    *,
    default_temp: float,
) -> float:
    """Kelvin for the one-shot velocity draw before NVE (``iasvel=0``).

    Defaults to ``0.2 × --temperature`` (same as heat FIRSTT) when
    ``--nve-boltzmann-temp`` is unset. Use a low value (e.g. 50 K) after mini
    to avoid large initial forces on stiff ML clusters.
    """
    explicit = getattr(args, "nve_boltzmann_temp", None)
    if explicit is not None:
        return float(explicit)
    return float(default_temp) * 0.2


def resolve_heat_firstt_finalt(
    args: argparse.Namespace,
    *,
    default_temp: float,
) -> tuple[float, float]:
    """Return ``(firstt, finalt)`` for the heat stage (Kelvin)."""
    finalt = getattr(args, "heat_finalt", None)
    firstt = getattr(args, "heat_firstt", None)
    t_end = float(finalt if finalt is not None else default_temp)
    t_start = float(firstt if firstt is not None else t_end * 0.2)
    return t_start, t_end


def opt_attr(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    """Return ``getattr(args, name)`` unless the value is ``None`` (unset CLI default)."""
    val = getattr(args, name, None)
    return default if val is None else val


def resolve_stage_ps(args: argparse.Namespace, stage: str) -> float:
    """Per-stage dynamics length in ps for PyCHARMM staged workflows."""
    if stage == "heat":
        return float(opt_attr(args, "ps_heat", 10.0))
    if stage == "equi":
        return float(opt_attr(args, "ps_equi", 50.0))
    if stage == "prod":
        return float(opt_attr(args, "ps_prod", opt_attr(args, "ps", 100.0)))
    if stage == "nve":
        return float(opt_attr(args, "ps_nve", opt_attr(args, "ps", 50.0)))
    return float(opt_attr(args, "ps", 1.0))


def resolve_stage_pressure_atm(args: argparse.Namespace, stage: str) -> float | None:
    """Target pressure (atm) for NPT equilibration/production stages."""
    setup = str(opt_attr(args, "setup", "") or "")
    if stage not in {"equi", "prod"} or not setup.endswith("npt"):
        return None
    raw = opt_attr(args, "npt_pressure", opt_attr(args, "pressure", 1.0))
    return float(raw) if raw is not None else None


def resolve_stage_temperature_K(args: argparse.Namespace, *, default: float = 300.0) -> float:
    return float(opt_attr(args, "temperature", default))


def resolve_dt_fs(args: argparse.Namespace, *, default: float = 0.25) -> float:
    return float(opt_attr(args, "dt_fs", default))


def resolve_heat_comp_damp(args: argparse.Namespace) -> bool:
    """Whether heat runs experimental COMP force-damp prep (default off)."""
    return bool(getattr(args, "heat_comp_damp", False))


def _requested_heat_thermostat(args: argparse.Namespace) -> str:
    raw = str(getattr(args, "heat_thermostat", "scale") or "scale").strip().lower()
    if raw not in ("scale", "hoover"):
        raise ValueError(f"unknown heat_thermostat: {raw!r}")
    return raw


def resolve_charmm_mm_pretreat_for_staged(
    args: argparse.Namespace,
    *,
    handoff_coords_in_memory: bool,
) -> bool:
    """Whether to run CHARMM MM pretreat before MLpot in :func:`run_staged_workflow`.

    Pretreat relaxes Packmol clashes on cold starts.  When continuing from a jaxmd
    or PyCHARMM handoff, coordinates are already in CHARMM memory — pretreat is
    skipped unless ``--charmm-mm-pretreat-on-handoff`` is set.
    """
    if not bool(getattr(args, "charmm_mm_pretreat", False)):
        return False
    if handoff_coords_in_memory and not bool(
        getattr(args, "charmm_mm_pretreat_on_handoff", False)
    ):
        return False
    return True


def heat_thermostat_requires_hoover_after_pretreat(args: argparse.Namespace) -> bool:
    """True when MLpot heat must use Hoover to avoid two-nose CHARMM conflicts."""
    if not resolve_charmm_mm_pretreat_for_staged(
        args, handoff_coords_in_memory=False
    ):
        return False
    setup = str(getattr(args, "setup", "") or "").strip().lower()
    return setup.startswith("pbc_")


def resolve_heat_thermostat(args: argparse.Namespace) -> str:
    """Heat-stage thermostat: ``scale`` (IHTFRQ) or ``hoover`` (CHARMM Hoover NVT).

    After CHARMM MM pretreat (Hoover NVT equi/prod in-session), ``scale`` heat
    triggers CHARMM ``Calling two different nose methods`` on overlap chunks.
    """
    requested = _requested_heat_thermostat(args)
    if requested == "scale" and heat_thermostat_requires_hoover_after_pretreat(args):
        if not bool(getattr(args, "quiet", False)):
            print(
                "HEAT: forcing heat_thermostat=hoover (CHARMM MM pretreat + PBC heat; "
                "scale/ihtfrq conflicts with Hoover state on overlap restarts)",
                flush=True,
            )
        return "hoover"
    return requested


def resolve_heat_hoover_tmass(
    args: argparse.Namespace,
    *,
    psf_tmass: int | None = None,
) -> int:
    """Hoover CPT ``tmass`` (kcal·mol⁻¹·ps²) for heat — tighter than equi/prod default.

    The old 2000 floor damped too slowly on small ML clusters and allowed kinetic
    T overshoot.  Default clamps PSF-derived ``tmass`` to [400, 1200].
    """
    explicit = getattr(args, "heat_hoover_tmass", None)
    if explicit is not None:
        return max(1, int(explicit))
    if psf_tmass is None:
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
            compute_cpt_piston_masses,
        )

        _, psf_tmass = compute_cpt_piston_masses()
    return max(400, min(int(psf_tmass), 1200))


def resolve_heat_comp_damp_kwargs(args: argparse.Namespace) -> dict[str, float | bool]:
    from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
        DEFAULT_COMP_FORCE_MIN_KCALMOL_A,
        DEFAULT_COMP_FORCE_SCALE,
    )

    min_f = getattr(args, "heat_comp_force_min", None)
    scale = getattr(args, "heat_comp_force_scale", None)
    kw: dict[str, float | bool] = {
        "hydrogen_only": bool(getattr(args, "heat_comp_hydrogen_only", True)),
    }
    if min_f is not None:
        kw["min_force_kcalmol_A"] = float(min_f)
    else:
        kw["min_force_kcalmol_A"] = DEFAULT_COMP_FORCE_MIN_KCALMOL_A
    if scale is not None:
        kw["force_scale"] = float(scale)
    else:
        kw["force_scale"] = DEFAULT_COMP_FORCE_SCALE
    return kw


def resolve_heat_ihtfrq(args: argparse.Namespace, *, nstep: int) -> int:
    """Heating velocity-rescale cadence (CHARMM ``ihtfrq``).

    CHARMM prints GAUSSIAN / COM / velocity-assignment blocks on each rescale,
    which dominates console output when ``ihtfrq`` is small (legacy default 10).
    """
    nstep = max(1, int(nstep))
    explicit = int(getattr(args, "heat_ihtfrq", 0) or 0)
    if explicit > 0:
        return min(explicit, nstep)
    if getattr(args, "quiet", False):
        return nstep
    return min(max(1, int(getattr(args, "dyn_nprint", 500))), nstep)


def resolve_dynamics_print_kwargs(
    args: argparse.Namespace,
    *,
    nstep: int,
    nsavc: int | None = None,
) -> dict[str, int]:
    nstep = max(1, int(nstep))
    if nsavc is not None:
        ns = max(1, int(nsavc))
        return {"nprint": ns, "iprfrq": ns, "isvfrq": ns}
    if getattr(args, "quiet", False):
        return {"nprint": nstep, "iprfrq": nstep, "isvfrq": nstep}
    nprint = max(1, int(getattr(args, "dyn_nprint", 500)))
    iprfrq = max(nprint, int(getattr(args, "dyn_iprfrq", 2000)))
    isvfrq = max(iprfrq, int(getattr(args, "dyn_iprfrq", 2000)))
    nprint = min(nprint, nstep)
    iprfrq = min(iprfrq, nstep)
    isvfrq = min(isvfrq, nstep)
    return {"nprint": nprint, "iprfrq": iprfrq, "isvfrq": isvfrq}



def add_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--residue",
        default=DEFAULT_RESIDUE,
        help="CHARMM residue name when --composition is not set (default: ACO)",
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=DEFAULT_N_MOLECULES,
        help="Number of identical monomers when --composition is not set",
    )
    parser.add_argument(
        "--composition",
        type=str,
        default=None,
        help="Comma-separated RES:N entries (e.g. ACO:4,MEOH:2)",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=DEFAULT_SPACING,
        help="Spacing (Å) when placing multiple residues",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint (.json or Orbax root). Default: MMML_CKPT or repo ckpts.",
    )


def add_packmol_cache_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Packmol cluster cache")
    group.add_argument(
        "--reuse-packmol-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse disk cache for Packmol sphere builds (default: on)",
    )
    group.add_argument(
        "--rebuild-packmol",
        action="store_true",
        help="Ignore Packmol cache and rebuild monomer/Packmol/MM placement",
    )
    group.add_argument(
        "--packmol-cache-dir",
        type=Path,
        default=None,
        help=(
            "Packmol cache root (default: <output-dir>/.packmol_cache, "
            "or MMML_PACKMOL_CACHE)"
        ),
    )


def add_run_state_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Run state checkpoint (Orbax or NPZ)")
    group.add_argument(
        "--save-run-state",
        action="store_true",
        help=(
            "After staged workflow, save positions/velocities and metadata "
            "(PhysNet weights remain in --checkpoint)"
        ),
    )
    group.add_argument(
        "--run-state-dir",
        type=Path,
        default=None,
        help="Run-state output directory (default: <output-dir>/run_state)",
    )
    group.add_argument(
        "--overlap-run-state-dir",
        type=Path,
        default=None,
        help="Overlap-chunk geometry sidecar directory (default: <output-dir>/run_state/overlap)",
    )
    group.add_argument(
        "--overlap-run-state-every-chunks",
        type=int,
        default=0,
        help="Save overlap run-state sidecar every N successful chunks (0=off)",
    )


def overlap_run_state_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    every = int(getattr(args, "overlap_run_state_every_chunks", 0) or 0)
    if every <= 0:
        return {}
    out_dir = getattr(args, "output_dir", None)
    default_dir = (
        Path(out_dir) / "run_state" / "overlap" if out_dir is not None else Path("run_state/overlap")
    )
    rs_dir = Path(getattr(args, "overlap_run_state_dir", None) or default_dir)
    return {
        "overlap_run_state_dir": rs_dir,
        "overlap_run_state_every_chunks": every,
    }


def add_monomer_constraint_args(
    parser: argparse.ArgumentParser,
    *,
    for_dynamics: bool = False,
) -> None:
    group = parser.add_argument_group("Monomer constraints (cons_fix)")
    group.add_argument(
        "--fix-resids",
        type=str,
        default="",
        metavar="IDS",
        help="Monomers fixed in SD pass 2; comma-separated resids (default: none)",
    )
    group.add_argument(
        "--fix-resid",
        type=int,
        default=None,
        help="Deprecated: single resid; use --fix-resids",
    )
    group.add_argument(
        "--no-fix",
        action="store_true",
        help="Skip constrained SD pass 2 (only free minimization)",
    )
    if for_dynamics:
        group.add_argument(
            "--constrain-resids",
            type=str,
            default="",
            metavar="IDS",
            help="Freeze these resids during MD (default: none)",
        )
        group.add_argument(
            "--mini-nstep",
            type=int,
            default=20,
            help="SD steps per minimization pass before dynamics (default: 20)",
        )
        group.add_argument(
            "--no-pre-minimize",
            action="store_true",
            help="Skip pre-dynamics SD minimization",
        )


def add_test_first_args(parser: argparse.ArgumentParser) -> None:
    """Optional derivative tests after MLpot SD minimization."""
    group = parser.add_argument_group("Post-minimize force tests")
    group.add_argument(
        "--test-first",
        action="store_true",
        help="After MLpot SD: Python ML energy vs finite-difference force check",
    )
    group.add_argument(
        "--test-first-charmm",
        action="store_true",
        help="Also run CHARMM TEST FIRSt (ANALYTIC omits USER/MLpot; usually misleading)",
    )
    group.add_argument(
        "--test-first-no-charmm",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    group.add_argument(
        "--test-first-update-nbonds",
        action="store_true",
        help="With --test-first: run UPDATE before CHARMM TEST (SD uses inbfrq=0 by default)",
    )
    group.add_argument(
        "--test-first-tol",
        type=float,
        default=0.005,
        help="TEST FIRSt TOL: report terms differing by more than this (default: 0.005)",
    )
    group.add_argument(
        "--test-first-step",
        type=float,
        default=1.0e-4,
        help="TEST FIRSt finite-difference step in Å (default: 1e-4)",
    )
    group.add_argument(
        "--test-first-resids",
        type=str,
        default="",
        metavar="IDS",
        help="Limit TEST FIRSt to these resids (default: all atoms; use for large clusters)",
    )


def resolve_test_first_config(
    args: argparse.Namespace,
) -> Optional["TestFirstConfig"]:
    if not getattr(args, "test_first", False):
        return None
    from mmml.interfaces.pycharmmInterface.mlpot.derivative_test import TestFirstConfig

    resids = tuple(parse_resid_list(getattr(args, "test_first_resids", "") or ""))
    return TestFirstConfig(
        tol=float(getattr(args, "test_first_tol", 0.005)),
        step=float(getattr(args, "test_first_step", 1.0e-4)),
        resids=resids,
        verbose=not getattr(args, "quiet", False),
        mlpot_python=True,
        charmm_lingo=bool(getattr(args, "test_first_charmm", False))
        and not getattr(args, "test_first_no_charmm", False),
        update_nonbonds=bool(getattr(args, "test_first_update_nbonds", False)),
    )


def parse_resid_list(text: str) -> list[int]:
    if not text or not str(text).strip():
        return []
    parts = str(text).replace(",", " ").split()
    resids: list[int] = []
    for p in parts:
        rid = int(p.strip())
        if rid < 1:
            raise ValueError(f"residue IDs must be >= 1, got {rid}")
        if rid not in resids:
            resids.append(rid)
    return resids


def resolve_fix_resids(args: argparse.Namespace) -> list[int]:
    if getattr(args, "no_fix", False):
        return []
    if getattr(args, "fix_resid", None) is not None:
        return [int(args.fix_resid)]
    return parse_resid_list(getattr(args, "fix_resids", "") or "")


def resolve_constrain_resids(args: argparse.Namespace) -> list[int]:
    return parse_resid_list(getattr(args, "constrain_resids", "") or "")


def build_acetone_cluster(
    n_molecules: int,
    spacing: float = DEFAULT_SPACING,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_molecules < 1:
        raise ValueError(f"n_molecules must be >= 1, got {n_molecules}")
    z, r = build_ase_cluster(DEFAULT_RESIDUE, n_molecules, spacing)
    expected = ACO_ATOMS_PER_MONOMER * n_molecules
    if len(z) != expected:
        raise RuntimeError(
            f"Expected {expected} atoms for ACO×{n_molecules}, got {len(z)}"
        )
    return z, r


def resolve_checkpoint(explicit: Path | None = None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    ckpt_env = os.environ.get("MMML_CKPT")
    candidates: list[Path] = []
    if ckpt_env:
        candidates.append(Path(ckpt_env))
    try:
        from mmml.cli.base import BUNDLED_PORTABLE_SMALL_MOLECULE_PATH

        candidates.append(BUNDLED_PORTABLE_SMALL_MOLECULE_PATH)
    except Exception:
        pass
    candidates.extend(
        [
            REPO_ROOT / "examples/ckpts_json/DESdimers_params.json",
            REPO_ROOT / "examples/ckpts_json",
            REPO_ROOT / "ckpts_json/DESdimers_params.json",
            REPO_ROOT / "mmml/models/physnetjax/ckpts/DESdimers",
            REPO_ROOT / "mmml/models/physnetjax/ckpts",
        ]
    )
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt.resolve()
    raise FileNotFoundError(
        "No checkpoint found. Set MMML_CKPT or pass --checkpoint."
    )


def validate_cluster_geometry(
    positions: np.ndarray,
    *,
    min_axis_span: float = 0.3,
    min_monomer_extent: float = 1.5,
    n_molecules: int | None = None,
) -> dict[str, float]:
    r = np.asarray(positions, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {r.shape}")
    span = r.max(axis=0) - r.min(axis=0)
    if float(span[1]) < min_axis_span or float(span[2]) < min_axis_span:
        raise ValueError(
            f"Cluster not 3D (spans Å x={span[0]:.3f} y={span[1]:.3f} z={span[2]:.3f})"
        )
    if n_molecules is not None and n_molecules > 0 and r.shape[0] % n_molecules == 0:
        n_per = r.shape[0] // n_molecules
        coms: list[np.ndarray] = []
        com_dists: list[float] = []
        for i in range(n_molecules):
            chunk = r[i * n_per : (i + 1) * n_per]
            extent = float(np.linalg.norm(chunk.max(axis=0) - chunk.min(axis=0)))
            if extent < min_monomer_extent:
                raise ValueError(
                    f"Monomer {i + 1} extent {extent:.3f} Å < {min_monomer_extent} Å"
                )
            coms.append(chunk.mean(axis=0))
        if len(coms) > 1:
            for i in range(len(coms)):
                for j in range(i + 1, len(coms)):
                    com_dists.append(float(np.linalg.norm(coms[j] - coms[i])))
        com_sep = com_dists[0] if com_dists else 0.0
    else:
        com_sep = float("nan")
        com_dists = []
    return {
        "span_x": float(span[0]),
        "span_y": float(span[1]),
        "span_z": float(span[2]),
        "com_sep_01": com_sep,
        "com_dist_min": min(com_dists) if com_dists else float("nan"),
        "com_dist_max": max(com_dists) if com_dists else float("nan"),
        "n_molecules": float(n_molecules or 0),
    }


def build_ase_cluster(
    residue: str,
    n_molecules: int,
    spacing: float,
    *,
    reference_npz: str | Path | None = None,
    frame: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a homogeneous CHARMM PSF-ordered cluster for ASE / hybrid calculators.

    Residues with bundled 3D templates (ACO, MEOH) use those templates; all others
    use IC + CHARMM SD/ABNR monomer minimization plus composition placement.

    When ``reference_npz`` is set, build the PSF and load coordinates from that file
    (must be PSF-order and match ``residue`` × ``n_molecules``).

    In Jupyter, call :func:`ensure_charmm_session_ready` once per kernel first (this
    function calls it automatically).
    """
    from mmml.interfaces.pycharmmInterface.cluster_geometry import ensure_charmm_session_ready

    ensure_charmm_session_ready()
    from mmml.cli.run.md_pbc_suite.cluster import (
        _build_psf_ordered_cluster,
        build_cluster_from_reference_npz,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    if reference_npz is not None:
        z, r = build_cluster_from_reference_npz(
            residue.upper(),
            n_molecules,
            reference_npz,
            frame=frame,
        )
    else:
        z, r = _build_psf_ordered_cluster(residue.upper(), n_molecules, spacing)
        sync_charmm_positions(r)
    validate_cluster_geometry(r, n_molecules=n_molecules)
    from mmml.interfaces.pycharmmInterface.mlpot.setup import report_charmm_topology_summary

    report_charmm_topology_summary()
    return z, r


def reference_frame_geometry(
    path: str | Path,
    *,
    frame: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``(Z, R)`` from a reference or handoff NPZ (see ``cluster_geometry``)."""
    from mmml.interfaces.pycharmmInterface.cluster_geometry import (
        reference_frame_geometry as _reference_frame_geometry,
    )

    return _reference_frame_geometry(path, frame=frame)


def atoms_from_reference_npz(path: str | Path, *, frame: int = 0) -> Any:
    """ASE ``Atoms`` from a reference or handoff NPZ frame."""
    from mmml.interfaces.pycharmmInterface.cluster_geometry import (
        atoms_from_reference_npz as _atoms_from_reference_npz,
    )

    return _atoms_from_reference_npz(path, frame=frame)


def prepare_vacuum_nbonds_for_mm() -> None:
    """Vacuum nbonds preset before first hybrid/MM energy (see ``cluster_geometry``)."""
    from mmml.interfaces.pycharmmInterface.cluster_geometry import (
        prepare_vacuum_nbonds_for_mm as _prepare_vacuum_nbonds_for_mm,
    )

    _prepare_vacuum_nbonds_for_mm()


def ensure_charmm_session_ready(**kwargs: Any) -> None:
    """Initialize CHARMM for notebooks (``bomlev=-2``, vacuum, MM BLOCK)."""
    from mmml.interfaces.pycharmmInterface.cluster_geometry import (
        ensure_charmm_session_ready as _ensure_charmm_session_ready,
    )

    _ensure_charmm_session_ready(**kwargs)


def prepare_charmm_notebook(**kwargs: Any) -> None:
    """Alias for :func:`ensure_charmm_session_ready`."""
    ensure_charmm_session_ready(**kwargs)


def prepare_jax_gpu_notebook(*, required: bool = True) -> bool:
    """Prep JAX GPU env for notebooks — **first cell**, before ``import jax``."""
    from mmml.interfaces.pycharmmInterface.cluster_geometry import (
        prepare_jax_gpu_notebook as _prepare_jax_gpu_notebook,
    )

    return _prepare_jax_gpu_notebook(required=required)


def prepare_notebook_kernel(*, jax_required: bool = True) -> None:
    """Bootstrap JAX GPU + CHARMM for Jupyter (JAX prep must run before any ``import jax``)."""
    from mmml.interfaces.pycharmmInterface.cluster_geometry import (
        prepare_notebook_kernel as _prepare_notebook_kernel,
    )

    _prepare_notebook_kernel(jax_required=jax_required)


def composition_tag(composition: list[tuple[str, int]] | None, residue: str, n_molecules: int) -> str:
    if composition:
        parts = [f"{res.lower()}_{count}" for res, count in composition]
        return "_".join(parts)
    return f"{residue.lower()}_{n_molecules}mer"


def use_packmol_placement(args: argparse.Namespace) -> bool:
    from mmml.interfaces.pycharmmInterface.packmol_placement import resolve_packmol_use

    return resolve_packmol_use(
        composition=getattr(args, "composition", None),
        packmol=getattr(args, "packmol", None),
    )


def use_packmol_sphere_placement(args: argparse.Namespace) -> bool:
    """Backward-compatible alias."""
    return use_packmol_placement(args)


def build_cluster_from_args_with_tag(
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """Build cluster; returns ``(Z, positions, n_monomers, tag)``."""
    from mmml.cli.run.md_pbc_suite.ase import (
        _build_cluster_from_composition,
        _build_cluster_from_composition_packmol,
        _parse_composition,
        packmol_sphere_center_from_args,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        resolve_packmol_cube_side,
        resolve_packmol_placement_mode,
        resolve_packmol_sphere_radius,
    )

    spacing = float(args.spacing)
    if getattr(args, "composition", None):
        composition = _parse_composition(args.composition)
        if use_packmol_placement(args):
            import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
            placement = resolve_packmol_placement_mode(
                packmol_placement=getattr(args, "packmol_placement", None),
                packmol_sphere=getattr(args, "packmol_sphere", None),
            )
            center = packmol_sphere_center_from_args(args)
            tolerance = float(getattr(args, "packmol_tolerance", 2.0))
            cube_side: float | None = None
            radius: float | None = None
            if placement == "sphere":
                radius = resolve_packmol_sphere_radius(
                    getattr(args, "packmol_radius", None),
                    getattr(args, "flat_bottom_radius", None),
                )
            else:
                cube_side = resolve_packmol_cube_side(
                    box_size=getattr(args, "box_size", None),
                    packmol_box_size=getattr(args, "packmol_box_size", None),
                    packmol_radius=getattr(args, "packmol_radius", None),
                    flat_bottom_radius=getattr(args, "flat_bottom_radius", None),
                )
            z, r, _atoms_per, _names = _build_cluster_from_composition_packmol(
                composition=composition,
                placement=placement,
                center=center,
                cube_side=cube_side,
                radius=radius,
                tolerance=tolerance,
                seed=int(getattr(args, "seed", 123)),
                charmm_sd_steps=int(getattr(args, "charmm_sd_steps", 50)),
                charmm_abnr_steps=int(getattr(args, "charmm_abnr_steps", 100)),
                charmm_tolenr=float(getattr(args, "charmm_tolenr", 1e-3)),
                charmm_tolgrd=float(getattr(args, "charmm_tolgrd", 1e-3)),
                scratch_dir=(
                    Path(args.output_dir) / "packmol_cluster"
                    if getattr(args, "output_dir", None) is not None
                    else None
                ),
                verbose=not getattr(args, "quiet", False),
                reuse_packmol_cache=bool(getattr(args, "reuse_packmol_cache", True)),
                packmol_cache_dir=getattr(args, "packmol_cache_dir", None),
                force_rebuild_packmol_cache=bool(
                    getattr(args, "rebuild_packmol", False)
                ),
            )
            if placement == "sphere":
                print(
                    f"Packmol sphere: center={center} R={radius:.1f} Å tol={tolerance:.1f} Å"
                )
            else:
                print(
                    f"Packmol cube: center={center} side={cube_side:.1f} Å tol={tolerance:.1f} Å"
                )
        else:
            z, r, _atoms_per, _names = _build_cluster_from_composition(
                composition=composition,
                spacing=spacing,
            )
        n_mol = sum(count for _, count in composition)
        tag = composition_tag(composition, args.residue.upper(), n_mol)
    else:
        residue = args.residue.upper()
        n_mol = int(args.n_molecules)
        if residue == "ACO":
            z, r = build_acetone_cluster(n_mol, spacing)
        else:
            z, r = build_ase_cluster(residue, n_mol, spacing)
        tag = composition_tag(None, residue, n_mol)
    sync_charmm_positions(r)
    validate_cluster_geometry(r, n_molecules=n_mol)
    return z, r, n_mol, tag


def build_cluster_from_args(
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build cluster; returns ``(Z, positions, n_atoms)`` (test-script compatible)."""
    z, r, _n_mol, _tag = build_cluster_from_args_with_tag(args)
    return z, r, len(z)


def print_cluster_geometry_summary(positions: np.ndarray, n_molecules: int) -> None:
    stats = validate_cluster_geometry(positions, n_molecules=n_molecules)
    n_atoms = len(positions)
    msg = (
        f"Cluster geometry OK: {n_molecules} monomer(s), {n_atoms} atoms |"
        f" spans (Å) x={stats['span_x']:.2f} y={stats['span_y']:.2f} z={stats['span_z']:.2f}"
    )
    if n_molecules > 1 and not np.isnan(stats["com_dist_min"]):
        msg += (
            f" | COM distances (Å) min={stats['com_dist_min']:.2f}"
            f" max={stats['com_dist_max']:.2f}"
        )
    print(msg)


def validate_resids_for_cluster(resids: list[int], n_molecules: int) -> None:
    bad = [r for r in resids if r < 1 or r > n_molecules]
    if bad:
        raise ValueError(
            f"residue ID(s) {bad} out of range for {n_molecules} monomer(s)"
        )


def setup_cons_fix_for_resids(resids: list[int]) -> Any:
    if not resids:
        return None
    import pycharmm.cons_fix as cons_fix

    from mmml.interfaces.pycharmmInterface.mlpot.setup import select_by_resids

    sel = select_by_resids(resids)
    if len(sel.get_atom_indexes()) == 0:
        raise RuntimeError(f"cons_fix: no atoms for resid(s) {resids}")
    cons_fix.setup(sel)
    return sel


def turn_off_cons_fix() -> None:
    import pycharmm.cons_fix as cons_fix

    cons_fix.turn_off()


def apply_flat_bottom_from_args(args: argparse.Namespace) -> None:
    fb_rad = getattr(args, "fb_rad", None)
    if fb_rad is None or float(fb_rad) <= 0:
        return
    from mmml.interfaces.pycharmmInterface.mlpot.restraints import apply_flat_bottom_workflow

    cfg = apply_flat_bottom_workflow(
        radius=float(fb_rad),
        force=float(getattr(args, "fb_forc", 1.0)),
        center_at_origin=bool(getattr(args, "fb_center", True)),
        xref=float(getattr(args, "fb_xref", 0.0)),
        yref=float(getattr(args, "fb_yref", 0.0)),
        zref=float(getattr(args, "fb_zref", 0.0)),
        selection=resolve_flat_bottom_selection(args),
    )
    if cfg is not None:
        print(
            "MMFP flat-bottom sphere: "
            f"droff={cfg.radius:.2f} Å force={cfg.force:.2f} "
            f"center=({cfg.xref:.2f}, {cfg.yref:.2f}, {cfg.zref:.2f}) "
            f"selection='{cfg.selection}'"
        )


def resolve_flat_bottom_selection(args: argparse.Namespace) -> str:
    """Resolve the CHARMM selection used for MMFP wall constraints."""
    raw = str(getattr(args, "fb_selection", "all") or "all").strip()
    return raw or "all"


def format_resid_constraint_message(resids: list[int], *, context: str) -> str:
    if not resids:
        return f"{context}: no monomers constrained"
    ids = ", ".join(str(r) for r in resids)
    return f"{context}: cons_fix on resid(s) [{ids}] ({len(resids)} monomer(s))"


def write_vmd_load_script(
    *,
    out_dir: Path,
    tag: str,
    topology_psf: Path,
    trajectory: Path | Sequence[Path] | None = None,
    n_atoms: int,
) -> Path:
    out_dir = out_dir.resolve()
    topology_psf = topology_psf.resolve()
    lines = [
        "# VMD: topology with full PSF connectivity (MLpot uses BLOCK, not bond deletion).",
        f"# Atoms: {n_atoms} — must match trajectory frame count.",
        f"mol new {{{topology_psf}}}",
    ]
    trajectories = _normalize_trajectory_paths(trajectory)
    for traj in trajectories:
        lines.append(f"mol addfile {{{traj}}} waitfor all")
    if trajectories:
        lines.append("animate goto 0")
    lines.append("display update")
    tcl_path = out_dir / f"load_{tag}_in_vmd.tcl"
    tcl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tcl_path


def print_vmd_load_help(
    *,
    out_dir: Path,
    tag: str,
    topology_psf: Path,
    trajectory: Path | Sequence[Path] | None,
    n_atoms: int,
    bondless_psf: Path | None = None,
) -> None:
    topo = topology_psf.resolve()
    print("\n=== VMD ===")
    print(f"  Atoms in this run: {n_atoms}")
    print(f"  Topology (bonds):  {topo}")
    trajectories = _normalize_trajectory_paths(trajectory)
    if trajectories:
        if len(trajectories) == 1:
            print(f"  Trajectory:        {trajectories[0]}")
            print(f"\n  vmd {topo} {trajectories[0]}")
        else:
            print("  Trajectories:")
            for traj in trajectories:
                print(f"    {traj}")
            print(f"\n  vmd {topo}")
        tcl = write_vmd_load_script(
            out_dir=out_dir,
            tag=tag,
            topology_psf=topo,
            trajectory=trajectories,
            n_atoms=n_atoms,
        )
        print(f"  # or: vmd -e {tcl}")
    else:
        print(f"\n  vmd {topo}")
    if bondless_psf is not None:
        print(
            f"\n  Prefer {topology_psf.name} in VMD (full connectivity). "
            f"{bondless_psf.name} is the in-memory PSF after MLpot registration."
        )


def _normalize_trajectory_paths(
    trajectory: Path | Sequence[Path] | None,
) -> list[Path]:
    if trajectory is None:
        return []
    if isinstance(trajectory, Path):
        paths = [trajectory]
    else:
        paths = [Path(p) for p in trajectory]
    return [path.resolve() for path in paths if path.is_file() and path.stat().st_size > 0]


def timestep_ps_from_dt_fs(dt_fs: float) -> float:
    return float(dt_fs) * 1e-3


def dynamics_nstep_from_ps(ps: float, dt_fs: float) -> int:
    timestep_ps = timestep_ps_from_dt_fs(dt_fs)
    if timestep_ps <= 0:
        raise ValueError(f"dt-fs must be > 0, got {dt_fs}")
    return max(1, int(round(float(ps) / timestep_ps)))


def build_acetone_dimer_cluster(spacing: float = DEFAULT_SPACING) -> Tuple[np.ndarray, np.ndarray]:
    return build_acetone_cluster(2, spacing)


def load_physnet_for_cluster(
    checkpoint: Path,
    n_atoms: int,
) -> Tuple[Any, Any]:
    from mmml.cli.base import load_physnet_params_and_ef_model, resolve_checkpoint_paths

    if checkpoint.is_file() and checkpoint.suffix == ".json":
        return load_physnet_params_and_ef_model(checkpoint, natoms=n_atoms)

    _, epoch_dir = resolve_checkpoint_paths(checkpoint)
    from mmml.models.physnetjax.physnetjax.restart.restart import get_params_model

    params, model = get_params_model(str(epoch_dir), natoms=n_atoms)
    return params, model


def all_atom_selection():
    from mmml.interfaces.pycharmmInterface.mlpot.setup import select_all_atoms

    return select_all_atoms()


def setup_charmm_nbonds() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds

    setup_default_nbonds()


def check_mlpot_symbols() -> list[str]:
    import pycharmm.lib as lib

    required = ("mlpot_set_func", "mlpot_set_properties", "mlpot_unset")
    missing = []
    for name in required:
        if not hasattr(lib.charmm, name):
            missing.append(name)
    return missing


def charmm_energy_row() -> dict[str, float]:
    import pycharmm.energy as energy

    df = energy.get_energy()
    row = df.iloc[0].to_dict()
    out: dict[str, float] = {}
    for key, value in row.items():
        if isinstance(value, (int, float, np.floating)):
            out[str(key)] = float(value)
    return out


def charmm_grms() -> float:
    """Current CHARMM gradient RMS (kcal/mol/Å) from the active energy/force state."""
    import pycharmm.energy as energy

    return float(energy.get_grms())


def charmm_grms_after_ener_force(*, silent: bool = True) -> float:
    """Run ``ENER FORCE`` then return GRMS (kcal/mol/Å).

    ``ENER`` alone can leave a stale GRMS from a prior MLpot evaluation; always
    use this before GRMS gates and bonded-MM recovery metrics.
    """
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    if silent:
        from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_silent_command

        with charmm_silent_command():
            pycharmm.lingo.charmm_script("ENER FORCE")
    else:
        pycharmm.lingo.charmm_script("ENER FORCE")
    return charmm_grms()


def charmm_total_forces_kcalmol_A() -> np.ndarray:
    """Per-atom CHARMM forces from the last ``ENER FORCE`` (kcal/mol/Å).

    PyCHARMM ``coor.get_forces()`` exposes ``dx/dy/dz`` as energy gradients
    (``dE/dx``). Physical forces are the negative gradient (see
    ``tests/functionality/mlpot/01_callback_vs_ase_no_charmm.py``).
    """
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.coor as coor

    fdf = coor.get_forces()
    grad = np.column_stack(
        [
            fdf["dx"].to_numpy(dtype=float),
            fdf["dy"].to_numpy(dtype=float),
            fdf["dz"].to_numpy(dtype=float),
        ]
    )
    return -grad


def charmm_total_forces_ev_angstrom() -> np.ndarray:
    """Per-atom CHARMM total forces in eV/Å (canonical evaluate/compare units)."""
    from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol

    return np.asarray(charmm_total_forces_kcalmol_A(), dtype=np.float64) / float(ev2kcalmol)


def mlpot_last_hybrid_forces_kcalmol_A(pyCModel: Any) -> np.ndarray | None:
    """Hybrid ML/MM forces from the last MLpot callback (kcal/mol/Å).

    ``ENER FORCE`` can leave bonded or other CHARMM terms in ``coor.get_forces()``
    even when the BLOCK zeros their energy. For evaluate/compare, prefer these
    callback forces — they match the ASE ``spherical_fn`` path.
    """
    if pyCModel is None:
        return None
    forces = getattr(pyCModel, "_last_ml_forces", None)
    if forces is None:
        calc = getattr(pyCModel, "_registered_calculator", None)
        forces = getattr(calc, "last_ml_forces", None)
    if forces is None:
        return None
    arr = np.asarray(forces, dtype=np.float64).reshape(-1, 3)
    return arr if arr.size else None


def mlpot_hybrid_forces_ev_angstrom(
    pyCModel: Any,
    *,
    natom: int | None = None,
) -> np.ndarray | None:
    """Hybrid MLpot forces in eV/Å from the last callback evaluation."""
    from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol

    forces_kcal = mlpot_last_hybrid_forces_kcalmol_A(pyCModel)
    if forces_kcal is None:
        return None
    if natom is not None:
        forces_kcal = forces_kcal[: int(natom)]
    return np.asarray(forces_kcal, dtype=np.float64) / float(ev2kcalmol)


def mlpot_spherical_forces_ev_angstrom(
    pyCModel: Any,
    *,
    positions: np.ndarray,
    use_pbc: bool,
    box_A: float | None,
) -> np.ndarray | None:
    """Evaluate hybrid ML/MM forces via ``spherical_fn`` (matches ASE evaluate-npz)."""
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        DecomposedMlpotCalculator,
        DecomposedMlpotModel,
    )

    if not isinstance(pyCModel, DecomposedMlpotModel):
        return None
    calc = pyCModel.get_pycharmm_calculator()
    if not isinstance(calc, DecomposedMlpotCalculator) or calc.spherical_fn is None:
        return None

    import jax
    import jax.numpy as jnp

    pos_np = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = int(pos_np.shape[0])
    pos_j = jnp.asarray(pos_np, dtype=jnp.float32)
    z_j = jnp.asarray(calc.atomic_numbers[:n], dtype=jnp.int32)
    kwargs: dict[str, Any] = dict(
        positions=pos_j,
        atomic_numbers=z_j,
        n_monomers=int(calc.n_monomers),
        cutoff_params=calc.cutoff_params,
        doML=True,
        doMM=bool(calc.do_mm),
        doML_dimer=True,
    )
    if use_pbc and box_A is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import cubic_box_matrix_from_side

        box_j = jnp.asarray(cubic_box_matrix_from_side(float(box_A)), dtype=jnp.float32)
        kwargs["box"] = box_j
        mm_pair_idx, mm_pair_mask, use_mm_pairs = calc._resolve_mm_pairs(pos_np, box_j)
        if use_mm_pairs:
            kwargs["mm_pair_idx"] = mm_pair_idx
            kwargs["mm_pair_mask"] = mm_pair_mask
    out = calc.spherical_fn(**kwargs)
    forces = np.asarray(jax.device_get(out.forces), dtype=np.float64).reshape(n, 3)
    return forces


def force_error_metrics_ev_angstrom(
    predicted_forces_ev: np.ndarray,
    reference_forces_ev: np.ndarray,
) -> dict[str, float]:
    """RMSE / MAE / max |ΔF| between two (N, 3) force arrays in eV/Å."""
    pred = np.asarray(predicted_forces_ev, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(reference_forces_ev, dtype=np.float64).reshape(-1, 3)
    if pred.shape != ref.shape:
        raise ValueError(f"force shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    delta = pred - ref
    return {
        "force_rmse_eV_A": float(np.sqrt(np.mean(delta**2))),
        "force_mae_eV_A": float(np.mean(np.abs(delta))),
        "force_max_abs_eV_A": float(np.max(np.abs(delta))),
    }


def collect_evaluate_force_sources_ev_angstrom(
    pyCModel: Any,
    *,
    natom: int,
    positions: np.ndarray,
    use_pbc: bool = False,
    box_A: float | None = None,
) -> dict[str, np.ndarray]:
    """All evaluate force lanes at one geometry (eV/Å)."""
    n = int(natom)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    sources: dict[str, np.ndarray] = {}
    spherical = mlpot_spherical_forces_ev_angstrom(
        pyCModel,
        positions=pos,
        use_pbc=use_pbc,
        box_A=box_A,
    )
    if spherical is not None and int(spherical.shape[0]) == n:
        sources["spherical_fn"] = np.asarray(spherical, dtype=np.float64)
    hybrid = mlpot_hybrid_forces_ev_angstrom(pyCModel, natom=n)
    if hybrid is not None and int(hybrid.shape[0]) == n:
        sources["mlpot_callback"] = np.asarray(hybrid, dtype=np.float64)
    sources["charmm_total"] = np.asarray(
        charmm_total_forces_ev_angstrom()[:n],
        dtype=np.float64,
    )
    return sources


def compare_force_sources_to_reference(
    force_sources: dict[str, np.ndarray],
    reference_forces_ev: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Per-lane force error metrics vs a reference (eV/Å)."""
    ref = np.asarray(reference_forces_ev, dtype=np.float64).reshape(-1, 3)
    out: dict[str, dict[str, float]] = {}
    for name, forces in force_sources.items():
        out[name] = force_error_metrics_ev_angstrom(forces, ref)
    return out


def cross_lane_force_rmse_ev_angstrom(
    force_sources: dict[str, np.ndarray],
    *,
    baseline: str = "spherical_fn",
) -> dict[str, float]:
    """RMSE of each lane vs ``baseline`` (eV/Å)."""
    base = force_sources.get(baseline)
    if base is None:
        return {}
    base_arr = np.asarray(base, dtype=np.float64).reshape(-1, 3)
    out: dict[str, float] = {}
    for name, forces in force_sources.items():
        if name == baseline:
            continue
        pred = np.asarray(forces, dtype=np.float64).reshape(-1, 3)
        if pred.shape != base_arr.shape:
            continue
        delta = pred - base_arr
        out[f"force_rmse_vs_{baseline}_eV_A"] = float(np.sqrt(np.mean(delta**2)))
    return out


def resolve_evaluate_forces_ev_angstrom(
    pyCModel: Any,
    *,
    natom: int,
    positions: np.ndarray | None = None,
    use_pbc: bool = False,
    box_A: float | None = None,
) -> tuple[np.ndarray, str]:
    """Forces for evaluate/compare: spherical_fn, MLpot callback, CHARMM fallback."""
    if positions is not None:
        spherical = mlpot_spherical_forces_ev_angstrom(
            pyCModel,
            positions=positions,
            use_pbc=use_pbc,
            box_A=box_A,
        )
        if spherical is not None and int(spherical.shape[0]) == int(natom):
            return np.asarray(spherical, dtype=np.float64), "spherical_fn"
    hybrid = mlpot_hybrid_forces_ev_angstrom(pyCModel, natom=natom)
    if hybrid is not None and int(hybrid.shape[0]) == int(natom):
        return np.asarray(hybrid, dtype=np.float64), "mlpot_callback"
    return charmm_total_forces_ev_angstrom()[: int(natom)], "charmm_total"


def charmm_positions_angstrom() -> np.ndarray:
    """Current CHARMM coordinates (Å) from the active PSF."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.coor as coor

    pos_df = coor.get_positions()
    return pos_df[["x", "y", "z"]].to_numpy(dtype=np.float64).reshape(-1, 3)


def forces_grms_kcalmol_A(forces: np.ndarray) -> float:
    """RMS of force components (kcal/mol/Å), matching CHARMM ``get_grms`` convention."""
    f = np.asarray(forces, dtype=np.float64).reshape(-1, 3)
    if f.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(f * f)))


GrmsMismatchKind = Literal[
    "ok", "desync_suspected", "geometry_stress", "both_high", "unknown"
]


@dataclass(frozen=True)
class HybridCharmmGrmsDiag:
    """Hybrid vs CHARMM GRMS comparison for MLpot readiness gates."""

    hybrid: float
    charmm: float
    ratio: float
    kind: GrmsMismatchKind


def classify_hybrid_charmm_grms_mismatch(
    hybrid: float,
    charmm: float,
    *,
    warn_ratio: float = 2.0,
    desync_max_ratio: float = 10.0,
    charmm_bonded_ok_max: float = 5.0,
) -> GrmsMismatchKind:
    """Classify hybrid/CHARMM GRMS disagreement for resync vs geometry recovery.

    Hybrid GRMS from the JAX calculator is authoritative for MLpot.  CHARMM
    ``get_grms()`` often stays ~1 kcal/mol/Å with ELEC/VDW blocked on ML atoms,
    so a high CHARMM/hybrid ratio with low hybrid is healthy, not desync.
    """
    if not (np.isfinite(hybrid) and np.isfinite(charmm)):
        return "unknown"
    if charmm <= 1.0e-8:
        return "ok" if hybrid <= charmm_bonded_ok_max else "geometry_stress"
    if hybrid <= charmm_bonded_ok_max:
        return "ok"
    if charmm <= charmm_bonded_ok_max:
        return "geometry_stress"
    ratio = float(max(hybrid / charmm, charmm / hybrid))
    if ratio <= warn_ratio:
        return "ok"
    if ratio <= desync_max_ratio:
        return "desync_suspected"
    return "both_high"


def measure_hybrid_charmm_grms(
    mlpot_ctx: Any | None,
    *,
    prefer_calculator: bool = True,
    warn_ratio: float = 2.0,
    desync_max_ratio: float = 10.0,
    charmm_bonded_ok_max: float = 5.0,
) -> HybridCharmmGrmsDiag:
    """Measure hybrid and CHARMM GRMS and classify their mismatch."""
    hybrid = float("nan")
    if mlpot_ctx is not None and prefer_calculator:
        calc = mlpot_hybrid_grms_from_calculator(mlpot_ctx)
        if calc is not None and np.isfinite(calc):
            hybrid = float(calc)
    if not np.isfinite(hybrid):
        hybrid = float(charmm_grms())
    charmm = float(charmm_grms())
    ratio = (
        float(max(hybrid / charmm, charmm / hybrid))
        if charmm > 1.0e-8 and np.isfinite(hybrid)
        else float("inf")
    )
    kind = classify_hybrid_charmm_grms_mismatch(
        hybrid,
        charmm,
        warn_ratio=warn_ratio,
        desync_max_ratio=desync_max_ratio,
        charmm_bonded_ok_max=charmm_bonded_ok_max,
    )
    return HybridCharmmGrmsDiag(hybrid=hybrid, charmm=charmm, ratio=ratio, kind=kind)


def _print_hybrid_charmm_grms_diag(context: str, diag: HybridCharmmGrmsDiag) -> None:
    if not context:
        return
    if diag.kind == "ok":
        print(
            f"{context}: hybrid GRMS={diag.hybrid:.4f} kcal/mol/Å (calculator)",
            flush=True,
        )
        return
    notes = {
        "desync_suspected": (
            f"possible desync (ratio={diag.ratio:.1f}); light resync recommended"
        ),
        "geometry_stress": (
            "ML geometry stress (CHARMM bonded/MM GRMS low; ELEC/VDW blocked on ML atoms)"
        ),
        "both_high": "high hybrid and CHARMM GRMS",
        "unknown": "non-finite GRMS",
    }
    note = notes.get(diag.kind, diag.kind)
    print(
        f"{context}: hybrid GRMS={diag.hybrid:.4f} kcal/mol/Å (calculator); "
        f"CHARMM GRMS={diag.charmm:.4f} ({note})",
        flush=True,
    )


def light_resync_mlpot_state(
    mlpot_ctx: Any,
    *,
    context: str = "MLpot light resync",
    silent_charmm: bool = True,
    verbose: bool = False,
    restart_path: Path | str | None = None,
) -> float:
    """Reregister MLpot, ``ENER FORCE``, ``UPDATE``, and sync PBC; return hybrid GRMS.

    Does **not** call ``upinb`` / ``update_bnbnd`` (unsafe with MLpot registered).
    """
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    mlpot_ctx.reregister_mlpot(verbose=verbose)
    if getattr(mlpot_ctx, "use_pbc", False):
        pyCModel = getattr(mlpot_ctx, "pyCModel", None)
        if pyCModel is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
                sync_mlpot_pbc_cell_from_charmm,
            )

            side = sync_mlpot_pbc_cell_from_charmm(
                pyCModel,
                fallback_side_A=getattr(mlpot_ctx, "cubic_box_side_A", None),
                restart_path=restart_path,
                verbose=verbose,
            )
            mlpot_ctx.cubic_box_side_A = float(side)
            mlpot_ctx.charmm_cubic_box_side_A = float(side)
    if silent_charmm:
        from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_silent_command

        with charmm_silent_command():
            pycharmm.lingo.charmm_script("ENER FORCE")
            pycharmm.lingo.charmm_script("UPDATE")
    else:
        pycharmm.lingo.charmm_script("ENER FORCE")
        pycharmm.lingo.charmm_script("UPDATE")
    diag = measure_hybrid_charmm_grms(mlpot_ctx)
    if context:
        print(
            f"{context}: hybrid GRMS {diag.hybrid:.4f} kcal/mol/Å after light resync "
            f"(CHARMM {diag.charmm:.4f})",
            flush=True,
        )
    return float(diag.hybrid)


def probe_and_light_resync_if_desync(
    mlpot_ctx: Any,
    *,
    context: str = "",
    silent_charmm: bool = True,
    verbose: bool = False,
    restart_path: Path | str | None = None,
) -> float:
    """Run light resync when hybrid/CHARMM GRMS look desynced; else refresh in place."""
    charmm_grms_after_ener_force(silent=silent_charmm)
    diag = measure_hybrid_charmm_grms(mlpot_ctx)
    _print_hybrid_charmm_grms_diag(context, diag)
    if diag.kind == "desync_suspected":
        return light_resync_mlpot_state(
            mlpot_ctx,
            context=context or "MLpot light resync",
            silent_charmm=silent_charmm,
            verbose=verbose,
            restart_path=restart_path,
        )
    return float(diag.hybrid)


def prepare_mlpot_hybrid_state_for_sd(
    mlpot_ctx: Any,
    *,
    grms_limit: float | None,
    energy_limit: float | None,
    bonded_recovery_nstep: int,
    bonded_recovery_verbose: bool = False,
    bonded_recovery_show_energy: bool = False,
    bonded_recovery_nprint: int = 10,
    bonded_recovery_tolenr: float = 1e-5,
    bonded_recovery_tolgrd: float = 1e-5,
    context_prefix: str = "Pre-SD",
    verbose: bool = True,
    restart_path: Path | str | None = None,
    allow_high_grms: bool | None = None,
) -> tuple[float, float]:
    """Resync, optional bonded recovery, and abort if hybrid GRMS remains too high.

    Returns ``(hybrid_grms, user_kcal)`` when MLpot SD may proceed.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.setup import assert_mlpot_user_active

    if allow_high_grms is None:
        allow_high_grms = bool(os.environ.get("MMML_MLPOT_ALLOW_HIGH_GRMS"))

    user = assert_mlpot_user_active(
        mlpot_ctx,
        context=f"{context_prefix} MLpot SD minimize",
        quiet=not verbose,
    )
    charmm_grms_after_ener_force()
    diag = measure_hybrid_charmm_grms(mlpot_ctx)
    _print_hybrid_charmm_grms_diag(
        f"{context_prefix} hybrid GRMS check" if verbose else "",
        diag,
    )
    hybrid_grms = float(diag.hybrid)

    if diag.kind == "desync_suspected":
        hybrid_grms = light_resync_mlpot_state(
            mlpot_ctx,
            context=f"{context_prefix} light resync" if verbose else "",
            verbose=verbose,
            restart_path=restart_path,
        )
        diag = measure_hybrid_charmm_grms(mlpot_ctx)

    grms_hot = grms_limit is not None and hybrid_grms > float(grms_limit)
    user_hot = energy_limit is not None and user > float(energy_limit)
    if grms_hot or user_hot:
        reasons: list[str] = []
        if user_hot:
            reasons.append(f"USER={user:.1f} kcal/mol > {float(energy_limit):.1f}")
        if grms_hot:
            reasons.append(
                f"GRMS={hybrid_grms:.1f} kcal/mol/Å > {float(grms_limit):.1f}"
            )
        if diag.kind == "geometry_stress":
            recovery_note = "bonded-MM SD (MLpot detached; may not lower hybrid GRMS)"
        else:
            recovery_note = "bonded-MM SD (MLpot detached)"
        print(
            f"{context_prefix} bonded recovery: {', '.join(reasons)}; "
            f"running {recovery_note} ({bonded_recovery_nstep} steps)",
            flush=True,
        )
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
            BondedMmMiniConfig,
            minimize_bonded_mm_recovery,
        )

        minimize_bonded_mm_recovery(
            mlpot_ctx,
            BondedMmMiniConfig(
                nstep_sd=int(bonded_recovery_nstep),
                nprint=max(1, int(bonded_recovery_nprint)),
                tolenr=float(bonded_recovery_tolenr),
                tolgrd=float(bonded_recovery_tolgrd),
                verbose=bonded_recovery_verbose,
                show_energy=bonded_recovery_show_energy,
            ),
        )
        user = assert_mlpot_user_active(
            mlpot_ctx,
            context=f"{context_prefix} MLpot SD (post bonded recovery)",
            quiet=not verbose,
        )
        charmm_grms_after_ener_force()
        diag = measure_hybrid_charmm_grms(mlpot_ctx)
        _print_hybrid_charmm_grms_diag(
            f"{context_prefix} post-recovery hybrid GRMS" if verbose else "",
            diag,
        )
        hybrid_grms = float(diag.hybrid)
        if diag.kind == "desync_suspected":
            hybrid_grms = light_resync_mlpot_state(
                mlpot_ctx,
                context=f"{context_prefix} post-recovery light resync" if verbose else "",
                verbose=verbose,
                restart_path=restart_path,
            )

    if grms_limit is not None and hybrid_grms > float(grms_limit):
        msg = (
            f"{context_prefix}: hybrid GRMS {hybrid_grms:.2f} kcal/mol/Å > "
            f"{float(grms_limit):.1f} after resync/recovery; refusing MLpot SD. "
            "Try more pretreat/--mini-nstep, a composition checkpoint, or "
            "MMML_MLPOT_ALLOW_HIGH_GRMS (not recommended)."
        )
        if not allow_high_grms:
            raise RuntimeError(msg)
        print(f"WARN: {msg}", flush=True)

    return float(hybrid_grms), float(user)


def mlpot_hybrid_grms_from_calculator(
    mlpot_ctx: Any,
    *,
    positions: np.ndarray | None = None,
    natom: int | None = None,
) -> float | None:
    """Hybrid ML/MM GRMS from JAX ``spherical_fn`` at CHARMM (or given) positions.

    Prefer this over :func:`charmm_grms` for MLpot gates: CHARMM SD and MM-only
    blocks can leave ``get_grms()`` reflecting bonded/MM terms while the hybrid
    potential still has large ML forces.
    """
    pyCModel = getattr(mlpot_ctx, "pyCModel", None)
    if pyCModel is None:
        return None

    if natom is None:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        import pycharmm.coor as coor

        natom = int(coor.get_natom())
    n = int(natom)

    if positions is None:
        pos = charmm_positions_angstrom()[:n]
    else:
        pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)[:n]

    use_pbc = bool(getattr(mlpot_ctx, "use_pbc", False))
    box_A = getattr(mlpot_ctx, "cubic_box_side_A", None)
    if box_A is None:
        box_A = getattr(mlpot_ctx, "charmm_cubic_box_side_A", None)

    forces_ev = mlpot_spherical_forces_ev_angstrom(
        pyCModel,
        positions=pos,
        use_pbc=use_pbc,
        box_A=float(box_A) if box_A is not None else None,
    )
    if forces_ev is not None and int(forces_ev.shape[0]) >= n:
        from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol

        forces_kcal = np.asarray(forces_ev[:n], dtype=np.float64) * float(ev2kcalmol)
        return forces_grms_kcalmol_A(forces_kcal)

    forces_kcal = mlpot_last_hybrid_forces_kcalmol_A(pyCModel)
    if forces_kcal is not None and int(forces_kcal.shape[0]) >= n:
        return forces_grms_kcalmol_A(forces_kcal[:n])
    return None


def resolve_mlpot_grms_kcalmol_A(
    mlpot_ctx: Any | None = None,
    *,
    context: str = "",
    prefer_calculator: bool = True,
    charmm_fallback: bool = True,
    stale_warn_ratio: float = 2.0,
) -> float:
    """GRMS for MLpot readiness gates (calculator hybrid forces when available)."""
    if mlpot_ctx is not None and prefer_calculator:
        calc_grms = mlpot_hybrid_grms_from_calculator(mlpot_ctx)
        if calc_grms is not None and np.isfinite(calc_grms):
            if charmm_fallback and context:
                charmm_val = float(charmm_grms())
                kind = classify_hybrid_charmm_grms_mismatch(
                    float(calc_grms),
                    charmm_val,
                    warn_ratio=stale_warn_ratio,
                )
                ratio = (
                    float(max(calc_grms / charmm_val, charmm_val / calc_grms))
                    if charmm_val > 1.0e-8
                    else float("inf")
                )
                diag = HybridCharmmGrmsDiag(
                    hybrid=float(calc_grms),
                    charmm=charmm_val,
                    ratio=ratio,
                    kind=kind,
                )
                if diag.kind != "ok":
                    _print_hybrid_charmm_grms_diag(context, diag)
                else:
                    print(
                        f"{context}: hybrid GRMS={calc_grms:.4f} kcal/mol/Å "
                        "(calculator)",
                        flush=True,
                    )
            elif context:
                print(
                    f"{context}: hybrid GRMS={calc_grms:.4f} kcal/mol/Å "
                    "(calculator)",
                    flush=True,
                )
            return float(calc_grms)

    if not charmm_fallback:
        return float("nan")
    grms = float(charmm_grms())
    if context:
        print(f"{context}: GRMS={grms:.4f} kcal/mol/Å (CHARMM)", flush=True)
    return grms


def refresh_mlpot_energy_and_grms(
    mlpot_ctx: Any | None = None,
    *,
    context: str = "MLpot energy refresh",
    silent_charmm: bool = True,
    reregister: bool = True,
    verbose: bool = False,
) -> float:
    """Re-apply MLpot BLOCK, run ``ENER FORCE``, return hybrid GRMS (kcal/mol/Å).

    When ``mlpot_ctx`` is set, GRMS comes from JAX ``spherical_fn`` at the current
    CHARMM coordinates (not CHARMM ``get_grms()``, which can be stale after SD).

    CHARMM SD can leave a stale GRMS from the minimizer while MLpot USER forces
    are not fully synchronized. Call before pre-dynamics gates and after MLpot mini.

    ``silent_charmm`` (default True) runs ``ENER FORCE`` at PRNLev/WRNLev 0 so gate
    checks do not spam nonbond list banners; Python ``context`` lines still print.

    Set ``reregister=False`` for evaluate-npz frame loops where MLpot BLOCK is
    already active (avoids repeated Rich BLOCK banners and redundant ``upinb``).
    """
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    if mlpot_ctx is not None and reregister:
        mlpot_ctx.reregister_mlpot(verbose=verbose)
    if silent_charmm:
        from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_silent_command

        with charmm_silent_command():
            pycharmm.lingo.charmm_script("ENER FORCE")
    else:
        pycharmm.lingo.charmm_script("ENER FORCE")
    return resolve_mlpot_grms_kcalmol_A(
        mlpot_ctx,
        context=context if context else "MLpot energy refresh",
        prefer_calculator=mlpot_ctx is not None,
    )


def resolve_mini_nstep(
    args: argparse.Namespace,
    n_monomers: int,
    *,
    n_atoms: int | None = None,
    pbc: bool = False,
) -> int:
    """SD steps per pass; scale up for large Packmol clusters unless disabled."""
    base = int(getattr(args, "mini_nstep", 20))
    if getattr(args, "no_scale_mini_nstep", False):
        return max(1, base)
    # Heuristic: large clusters need more than 20 SD steps per pass.
    scaled = max(base, min(300, 8 * max(1, n_monomers)))
    if pbc:
        n_at = int(n_atoms) if n_atoms is not None else 5 * int(n_monomers)
        pbc_floor = max(400, min(800, n_at // 2))
        scaled = max(scaled, min(800, pbc_floor))
    if scaled != base:
        print(
            f"mini-nstep scaled {base} -> {scaled} for {n_monomers} monomer(s)"
            + (" (PBC)" if pbc else ""),
            flush=True,
        )
    return scaled


def recommend_echeck_kcal(n_monomers: int, n_atoms: int) -> float:
    """Size-aware ECHECK floor for MLpot clusters (kcal/mol).

    Single-monomer smoke tests keep 100 kcal/mol. Multi-monomer clusters (e.g.
    DCM:9) scale with size so ML heat/nonbond updates do not trip ECHECK.
    """
    n_mol = max(1, int(n_monomers))
    n_at = max(1, int(n_atoms))
    if n_mol == 1 and n_at < 100:
        return 100.0
    from_mol = float(n_mol) * 50.0
    from_atoms = float(n_at) * 10.0
    return max(500.0, from_mol, from_atoms)


def resolve_echeck_for_cluster(
    args: argparse.Namespace,
    *,
    n_atoms: int,
    n_monomers: int,
) -> float:
    """ECHECK tolerance; large ML clusters use a looser default (see production dyna.inp CPT)."""
    if getattr(args, "no_echeck", False):
        return -1.0
    if getattr(args, "no_scale_echeck", False):
        return float(getattr(args, "echeck", 100.0))
    base = float(getattr(args, "echeck", 100.0))
    recommended = recommend_echeck_kcal(n_monomers, n_atoms)
    scaled = max(base, recommended)
    if scaled != base:
        print(
            f"echeck loosened {base} -> {scaled:.0f} kcal/mol for "
            f"{n_monomers} monomer(s) / {n_atoms} atoms "
            f"(recommended floor {recommended:.0f}; --no-scale-echeck to keep {base})",
            flush=True,
        )
    return scaled


def resolve_max_grms_before_dyn(
    args: argparse.Namespace,
    n_monomers: int,
    n_atoms: int,
    *,
    pbc: bool = False,
) -> float:
    """Size-aware GRMS ceiling before MLpot dynamics (kcal/mol/Å).

    CHARMM SD can report a stale low GRMS while MLpot USER forces are still large
    on dense PBC clusters.  Scale the default 50 kcal/mol/Å floor with system size.
    """
    base = float(getattr(args, "max_grms_before_dyn", 50.0))
    if getattr(args, "allow_high_grms", False) or getattr(args, "no_scale_max_grms", False):
        return base
    n_mol = max(1, int(n_monomers))
    n_at = max(1, int(n_atoms))
    if n_mol == 1 and n_at < 100:
        return base
    from_mol = float(n_mol) * 0.75
    from_atoms = float(n_at) * 0.2
    scaled = max(base, from_mol, from_atoms)
    if pbc:
        scaled = max(scaled, min(250.0, float(n_mol) * 0.85))
    if scaled != base:
        print(
            f"max_grms_before_dyn scaled {base:.0f} -> {scaled:.0f} kcal/mol/Å for "
            f"{n_mol} monomer(s) / {n_at} atoms"
            + (" (PBC)" if pbc else ""),
            flush=True,
        )
    return scaled


def assert_dynamics_ready(
    *,
    max_grms: float = 50.0,
    abort: bool = True,
    require_mlpot_user: bool = False,
    user_zero_tol_kcalmol: float = 1.0e-6,
    mlpot_ctx: Any | None = None,
    silent_charmm: bool = True,
) -> float:
    """Warn or abort if gradients are still huge before starting dynamics."""
    import math

    import pycharmm
    import pycharmm.energy as energy

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_silent_command

    def _ener_force() -> None:
        if silent_charmm:
            with charmm_silent_command():
                pycharmm.lingo.charmm_script("ENER FORCE")
        else:
            pycharmm.lingo.charmm_script("ENER FORCE")

    user_kcal: float | None = None
    if require_mlpot_user and mlpot_ctx is not None:
        grms = refresh_mlpot_energy_and_grms(
            mlpot_ctx, context="", silent_charmm=silent_charmm
        )
    elif require_mlpot_user:
        _ener_force()
        grms = charmm_grms()
    else:
        if silent_charmm:
            with charmm_silent_command():
                pycharmm.lingo.charmm_script("ENER")
        else:
            pycharmm.lingo.charmm_script("ENER")
        grms = charmm_grms()
    if require_mlpot_user:
        try:
            user_kcal = float(energy.get_term_by_name("USER"))
        except Exception:
            user_kcal = None
        user_missing = (
            user_kcal is None
            or not math.isfinite(user_kcal)
            or abs(user_kcal) <= user_zero_tol_kcalmol
        )
        if user_missing:
            msg = (
                "Pre-dynamics check failed: MLpot USER energy inactive "
                f"(USER={user_kcal}, GRMS={grms:.4f}). "
                "Dynamics would integrate a free gas / zeroed BLOCK state."
            )
            if abort:
                raise RuntimeError(msg)
            print(f"WARN: {msg}", flush=True)
            return grms
        if grms > max_grms and mlpot_ctx is not None:
            stale_grms = float(grms)
            grms = refresh_mlpot_energy_and_grms(
                mlpot_ctx,
                context="Pre-dynamics GRMS retry (hybrid force re-eval)",
                silent_charmm=silent_charmm,
            )
            if grms <= max_grms:
                print(
                    f"Pre-dynamics GRMS recovered: {stale_grms:.2f} -> {grms:.4f} "
                    f"kcal/mol/Å after hybrid re-eval (limit {max_grms})",
                    flush=True,
                )
    if grms <= max_grms:
        extra = f", USER={user_kcal:.2f} kcal/mol" if user_kcal is not None else ""
        print(
            f"Pre-dynamics GRMS OK: {grms:.4f} kcal/mol/Å (limit {max_grms}){extra}",
            flush=True,
        )
        return grms
    msg = (
        f"Pre-dynamics GRMS {grms:.2f} kcal/mol/Å > {max_grms} — "
        "MLpot minimization did not converge; dynamics skipped. "
        "Try more --mini-nstep, a composition-specific checkpoint, or "
        "--allow-high-grms (not recommended)."
    )
    if abort and not os.environ.get("MMML_MLPOT_ALLOW_HIGH_GRMS"):
        raise RuntimeError(msg)
    print(f"WARN: {msg}", flush=True)
    return grms


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def add_staged_md_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Staged MLpot MD (mini / heat / NVE / equi / prod)")
    group.add_argument(
        "--md-stages",
        type=str,
        default=None,
        help=(
            "Comma-separated stages: mini,heat,nve,equi,prod. "
            "Default depends on --setup / --phase (pycharmm_full runs all)."
        ),
    )
    group.add_argument(
        "--md-stage",
        type=str,
        default=None,
        choices=["mini", "heat", "nve", "equi", "prod"],
        help="Run a single dynamics stage (implies --skip-cluster-build when artifacts exist).",
    )
    group.add_argument(
        "--ps-heat",
        type=float,
        default=10.0,
        help="Heating segment length in ps (default: 10)",
    )
    group.add_argument(
        "--ps-nve",
        type=float,
        default=None,
        help="NVE segment length in ps (default: --ps or 50)",
    )
    group.add_argument(
        "--nve-boltzmann-temp",
        type=float,
        default=None,
        metavar="K",
        help=(
            "pycharmm: Kelvin for Boltzmann velocities before NVE (memory handoff after mini). "
            "Default 0.2×--temperature; use 50–100 K for gentler start than full --temperature."
        ),
    )
    group.add_argument(
        "--ps-equi",
        type=float,
        default=50.0,
        help="NPT equilibration length in ps (default: 50)",
    )
    group.add_argument(
        "--ps-prod",
        type=float,
        default=None,
        help="Production length in ps (default: --ps or 100)",
    )
    group.add_argument(
        "--npt-thermostat",
        type=str,
        choices=["hoover", "berendsen"],
        default="hoover",
        help=(
            "NPT temperature control for equi/prod stages (default: hoover). "
            "Hoover uses CPT extended-system thermostat with pmass/tmass from PSF mass."
        ),
    )
    group.add_argument(
        "--npt-pressure",
        type=float,
        default=1.0,
        help="NPT reference pressure in atm for equi/prod (default: 1.0)",
    )
    group.add_argument(
        "--npt-pgamma",
        type=float,
        default=5.0,
        help=(
            "CPT barostat Langevin collision frequency in 1/ps (default: 5). "
            "Set to 0 to disable barostat coupling."
        ),
    )
    group.add_argument(
        "--n-heat-segments",
        type=int,
        default=1,
        help=(
            "Split heating into short chained restart segments (default: 1). "
            "Use >1 so overlap rescue can run between segments during the ramp."
        ),
    )
    group.add_argument(
        "--n-equi-segments",
        type=int,
        default=1,
        help=(
            "Split NPT equilibration into chained restart segments (default: 1). "
            "Use multiple segments to equilibrate pressure before production."
        ),
    )
    group.add_argument(
        "--n-prod-segments",
        type=int,
        default=1,
        help="Split production into chained restart segments (default: 1)",
    )
    pretreat = parser.add_argument_group(
        "CHARMM MM pretreat (CGENFF before MLpot registration)"
    )
    pretreat.add_argument(
        "--charmm-mm-pretreat",
        action="store_true",
        help=(
            "Before MLpot: CGENFF SD/ABNR minimize + CHARMM heat/equi/prod (no USER/ML). "
            "Coordinates feed MLpot mini/heat/NVE."
        ),
    )
    pretreat.add_argument(
        "--charmm-mm-pretreat-ps-heat",
        type=float,
        default=None,
        metavar="PS",
        help=(
            "Pretreat CHARMM heat length in ps (overrides --charmm-mm-pretreat-heat-nstep "
            "when set)"
        ),
    )
    pretreat.add_argument(
        "--charmm-mm-pretreat-heat-nstep",
        type=int,
        default=2000,
        metavar="N",
        help="Integration steps for pretreat CHARMM heat (default: 2000)",
    )
    pretreat.add_argument(
        "--charmm-mm-pretreat-ps-equi",
        type=float,
        default=0.0,
        metavar="PS",
        help="Pretreat CHARMM NPT equilibration in ps (0 skips; default: 0)",
    )
    pretreat.add_argument(
        "--charmm-mm-pretreat-ps-prod",
        type=float,
        default=0.0,
        metavar="PS",
        help="Pretreat CHARMM NPT production in ps (0 skips; default: 0)",
    )
    pretreat.add_argument(
        "--charmm-mm-pretreat-mini-sd",
        type=int,
        default=None,
        metavar="N",
        help="Pretreat CHARMM SD steps (default: --charmm-sd-steps)",
    )
    pretreat.add_argument(
        "--charmm-mm-pretreat-mini-abnr",
        type=int,
        default=None,
        metavar="N",
        help="Pretreat CHARMM ABNR steps (default: --charmm-abnr-steps)",
    )
    add_bonded_mm_mini_args(parser)
    group.add_argument(
        "--mlpot-mm-internal-scale",
        type=float,
        default=0.0,
        metavar="W",
        help=(
            "During MLpot, scale CGENFF BOND/ANGL/DIHE on ML atoms via BLOCK "
            "(0=full ML only, 0.1=10%% MM internal). ELEC/VDW remain off."
        ),
    )
    group.add_argument(
        "--min-com-restraint-distance",
        type=float,
        default=None,
        metavar="Å",
        help=(
            "Pairwise inter-monomer COM lower wall for ML/MM dynamics. Adds "
            "0.5*k*(r_min-r)^2 when COM distance r < r_min (default: disabled)."
        ),
    )
    group.add_argument(
        "--min-com-restraint-k",
        type=float,
        default=1.0,
        metavar="eV/Å²",
        help="Force constant for --min-com-restraint-distance (default: 1.0).",
    )
    group.add_argument(
        "--restart-from",
        type=Path,
        default=None,
        help="CHARMM restart (.res) for the first dynamics stage",
    )
    group.add_argument(
        "--from-psf",
        type=Path,
        default=None,
        help="Load topology from PSF instead of rebuilding the cluster",
    )
    group.add_argument(
        "--from-crd",
        type=Path,
        default=None,
        help="Load coordinates from CRD (with --from-psf)",
    )
    group.add_argument(
        "--skip-cluster-build",
        action="store_true",
        help="Do not run Packmol/IC build; requires --from-psf/--from-crd or prior mini artifacts",
    )
    group.add_argument(
        "--skip-if-crd-exists",
        action="store_true",
        help="Skip MLpot SD when mini CRD already exists in --output-dir",
    )
    group.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Artifact tag for staged outputs (default: from composition/residue)",
    )
    group.add_argument(
        "--free-space",
        action="store_true",
        help="Vacuum cluster (no CHARMM crystal); default unless --setup pbc_* or --box-size",
    )
    group.add_argument(
        "--mlpot-pbc",
        action="store_true",
        help=(
            "Enable ML MIC / periodic dimer lists (default for --setup pbc_* only). "
            "With --setup free_* and --box-size, CHARMM uses a large crystal for CPT "
            "but ML stays open-boundary unless this flag is set."
        ),
    )
    group.add_argument(
        "--dyn-inbfrq",
        type=int,
        default=None,
        help=(
            "CHARMM nonbond list rebuild cadence for dynamics (inbfrq). "
            "Vacuum default in builders is 50; -1 rebuilds when the cluster moves. "
            "MLpot mini uses inbfrq=0; use --pre-nve-charmm-update before NVE."
        ),
    )
    group.add_argument(
        "--dyn-imgfrq",
        type=int,
        default=None,
        metavar="N",
        help=(
            "PBC dynamics: image/HB/extended list rebuild every N steps "
            "(sets imgfrq, ihbfrq, ilbfrq; default 50). Larger N reduces "
            "UPDECI/mlpot_update CPU cost; use only when lists stay valid."
        ),
    )
    group.add_argument(
        "--pre-nve-charmm-update",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before vacuum NVE: CHARMM ENER+UPDATE after mini to sync lists (default: on). "
            "Does not call update_bnbnd/upinb."
        ),
    )


def resolve_dyn_inbfrq(args: argparse.Namespace) -> int | None:
    """Explicit ``--dyn-inbfrq`` for dynamics stages, if set."""
    raw = getattr(args, "dyn_inbfrq", None)
    if raw is None:
        return None
    return int(raw)


def resolve_dyn_imgfrq(args: argparse.Namespace) -> int | None:
    """Explicit ``--dyn-imgfrq`` for PBC dynamics (image/HB list cadence)."""
    raw = getattr(args, "dyn_imgfrq", None)
    if raw is None:
        return None
    return int(raw)


def _default_stages_for_setup(setup: str | None) -> list[str]:
    s = (setup or "").strip().lower()
    if s == "pycharmm_minimize":
        return ["mini"]
    if s == "free_nvt":
        return ["mini", "heat"]
    if s == "free_nve":
        return ["mini", "nve"]
    if s in ("pycharmm_full", "pbc_nve", "pbc_npt"):
        return ["mini", "heat", "nve", "equi", "prod"]
    if s == "pbc_nvt":
        return ["mini", "heat", "equi"]
    return ["mini", "heat", "nve", "equi", "prod"]


def add_bonded_mm_mini_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "Bonded MM recovery (fix bad conformations after heat)"
    )
    group.add_argument(
        "--bonded-mm-mini",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compare MM bonded GRMS to post-MM-pre-min baseline after selected stages; "
            "run bonded-only SD (BLOCK toggle, MLpot stays on) if higher "
            "(default: on; heat is always checked when enabled)"
        ),
    )
    group.add_argument(
        "--bonded-mm-mini-after",
        type=str,
        default="mini,heat",
        help="Comma-separated dynamics stages to check (default: mini,heat; heat always)",
    )
    group.add_argument(
        "--bonded-mm-mini-steps",
        type=int,
        default=50,
        help="SD steps for bonded-only recovery mini (default: 50)",
    )
    group.add_argument(
        "--bonded-mm-mini-always",
        action="store_true",
        help=(
            "Run bonded SD after every stage in --bonded-mm-mini-after, even when "
            "GRMS/ANGL/internal strain is below baseline (default: strain-gated only). "
            "All-ML clusters on MPI-linked CHARMM use MLpot SD instead of PSF reload."
        ),
    )
    group.add_argument(
        "--bonded-mm-internal-margin",
        type=float,
        default=0.0,
        help="Legacy alias for --bonded-mm-grms-margin when grms-margin unset (default: 0)",
    )
    group.add_argument(
        "--bonded-mm-grms-margin",
        type=float,
        default=None,
        help="kcal/mol/Å above baseline GRMS before recovery SD (default: bonded-mm-internal-margin)",
    )
    group.add_argument(
        "--bonded-mm-internal-energy-margin",
        type=float,
        default=0.0,
        help="kcal/mol above baseline bonded internal energy before recovery (0=off)",
    )
    group.add_argument(
        "--bonded-mm-angl-margin",
        type=float,
        default=0.0,
        help="kcal/mol above baseline ANGL term before recovery (0=off)",
    )
    group.add_argument(
        "--bonded-mm-max-angl-kcal",
        type=float,
        default=None,
        help="Abort after MM pre-min if ANGL exceeds this (e.g. 15 for tight clusters)",
    )
    group.add_argument(
        "--bonded-mm-max-internal-kcal",
        type=float,
        default=None,
        help="Abort after MM pre-min if bonded internal energy exceeds this",
    )
    group.add_argument(
        "--allow-high-bonded-strain",
        action="store_true",
        help="Continue when --bonded-mm-max-angl-kcal / max-internal limits are exceeded",
    )


def resolve_md_stages(args: argparse.Namespace) -> list[str]:
    if getattr(args, "md_stage", None):
        return [str(args.md_stage)]
    raw = getattr(args, "md_stages", None)
    if raw:
        stages = [p.strip().lower() for p in str(raw).split(",") if p.strip()]
    else:
        setup = getattr(args, "setup", None)
        phase = getattr(args, "phase", None)
        if phase == "minimize":
            return ["mini"]
        if phase == "dynamics" and setup in (None, ""):
            return ["nve"]
        stages = _default_stages_for_setup(setup)
    valid = {"mini", "heat", "nve", "equi", "prod"}
    bad = [s for s in stages if s not in valid]
    if bad:
        raise ValueError(f"unknown md stage(s): {bad}; allowed: {sorted(valid)}")
    return stages


def resolve_charmm_use_pbc(args: argparse.Namespace) -> bool:
    """CHARMM crystal / IMAGE / CPT (independent of ML MIC)."""
    if getattr(args, "free_space", False):
        return False
    setup = (getattr(args, "setup", None) or "").strip().lower()
    if setup.startswith("pbc_"):
        return True
    if getattr(args, "box_size", None) is not None:
        return True
    return False


def resolve_mlpot_use_pbc(args: argparse.Namespace) -> bool:
    """ML decomposed PhysNet MIC and periodic dimer neighbor lists."""
    if getattr(args, "free_space", False):
        return False
    if getattr(args, "mlpot_pbc", False):
        return True
    setup = (getattr(args, "setup", None) or "").strip().lower()
    return setup.startswith("pbc_")


def resolve_loose_pbc(charmm_pbc: bool, mlpot_pbc: bool) -> bool:
    """CHARMM crystal on for CPT/Hoover, but ML uses open boundary (no MIC).

    Typical: ``--box-size`` without ``--mlpot-pbc``. Image/extended list rebuilds
    are unnecessary for a cluster restrained near the box center.
    """
    return bool(charmm_pbc and not mlpot_pbc)


def resolve_use_pbc(args: argparse.Namespace) -> bool:
    """Backward-compatible alias for :func:`resolve_charmm_use_pbc`."""
    return resolve_charmm_use_pbc(args)


def resolve_pbc_box_side(args: argparse.Namespace, positions: np.ndarray) -> float:
    if getattr(args, "box_size", None) is not None:
        return float(args.box_size)
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        cubic_box_length_from_geometry,
    )

    ml_cutoff = float(getattr(args, "ml_cutoff", 12.0))
    return cubic_box_length_from_geometry(positions, ml_cutoff=ml_cutoff)
