"""Shared CLI helpers for CHARMM MLpot workflows (tests and ``mmml md-system``)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

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
) -> int:
    if dcd_interval_ps is not None:
        if timestep_ps is None or timestep_ps <= 0:
            raise ValueError("timestep_ps required when using --dcd-interval-ps")
        nsavc = int(round(float(dcd_interval_ps) / float(timestep_ps)))
    else:
        nsavc = int(dcd_nsavc)
    nsavc = max(1, nsavc)
    if nstep is not None:
        n = int(nstep)
        if n > 1:
            nsavc = min(nsavc, n - 1)
        else:
            nsavc = min(nsavc, n)
    return nsavc


def apply_charmm_output_from_args(args: argparse.Namespace) -> int:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import apply_charmm_verbosity

    if getattr(args, "quiet", False):
        apply_charmm_verbosity(prnlev=0, warnlev=0, bomlev=args.bomlev)
    else:
        apply_charmm_verbosity(
            prnlev=args.prnlev,
            warnlev=args.warnlev,
            bomlev=args.bomlev,
        )
    if getattr(args, "quiet", False):
        mini_nstep = getattr(args, "mini_nstep", getattr(args, "nstep", 100))
        return max(1, int(mini_nstep))
    return max(1, int(args.nprint))


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


def resolve_heat_comp_damp(args: argparse.Namespace) -> bool:
    """Whether heat runs experimental COMP force-damp prep (default off)."""
    return bool(getattr(args, "heat_comp_damp", False))


def resolve_heat_thermostat(args: argparse.Namespace) -> str:
    """Heat-stage thermostat: ``scale`` (IHTFRQ) or ``hoover`` (CHARMM Hoover NVT)."""
    raw = str(getattr(args, "heat_thermostat", "scale") or "scale").strip().lower()
    if raw not in ("scale", "hoover"):
        raise ValueError(f"unknown heat_thermostat: {raw!r}")
    return raw


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
) -> dict[str, int]:
    nstep = max(1, int(nstep))
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
) -> Tuple[np.ndarray, np.ndarray]:
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    from mmml.cli.run.md_pbc_suite.cluster import _build_psf_ordered_cluster
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    z, r = _build_psf_ordered_cluster(residue.upper(), n_molecules, spacing)
    sync_charmm_positions(r)
    validate_cluster_geometry(r, n_molecules=n_molecules)
    return z, r


def composition_tag(composition: list[tuple[str, int]] | None, residue: str, n_molecules: int) -> str:
    if composition:
        parts = [f"{res.lower()}_{count}" for res, count in composition]
        return "_".join(parts)
    return f"{residue.lower()}_{n_molecules}mer"


def use_packmol_sphere_placement(args: argparse.Namespace) -> bool:
    from mmml.interfaces.pycharmmInterface.packmol_placement import resolve_packmol_sphere_use

    return resolve_packmol_sphere_use(
        composition=getattr(args, "composition", None),
        packmol_radius=getattr(args, "packmol_radius", None),
        flat_bottom_radius=getattr(args, "flat_bottom_radius", None),
        packmol_sphere=getattr(args, "packmol_sphere", None),
    )


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
    from mmml.interfaces.pycharmmInterface.packmol_placement import resolve_packmol_sphere_radius

    spacing = float(args.spacing)
    if getattr(args, "composition", None):
        composition = _parse_composition(args.composition)
        if use_packmol_sphere_placement(args):
            import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
            radius = resolve_packmol_sphere_radius(
                getattr(args, "packmol_radius", None),
                getattr(args, "flat_bottom_radius", None),
            )
            center = packmol_sphere_center_from_args(args)
            tolerance = float(getattr(args, "packmol_tolerance", 2.0))
            z, r, _atoms_per, _names = _build_cluster_from_composition_packmol(
                composition=composition,
                center=center,
                radius=radius,
                tolerance=tolerance,
                seed=int(getattr(args, "seed", 123)),
                charmm_sd_steps=int(getattr(args, "charmm_sd_steps", 50)),
                charmm_abnr_steps=int(getattr(args, "charmm_abnr_steps", 100)),
                charmm_tolenr=float(getattr(args, "charmm_tolenr", 1e-3)),
                charmm_tolgrd=float(getattr(args, "charmm_tolgrd", 1e-3)),
                scratch_dir=(
                    Path(args.output_dir) / "packmol_sphere"
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
            fb_r = getattr(args, "flat_bottom_radius", None)
            print(
                f"Packmol sphere: center={center} R={radius:.1f} Å tol={tolerance:.1f} Å"
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


def resolve_mini_nstep(args: argparse.Namespace, n_monomers: int) -> int:
    """SD steps per pass; scale up for large Packmol clusters unless disabled."""
    base = int(getattr(args, "mini_nstep", 20))
    if getattr(args, "no_scale_mini_nstep", False):
        return max(1, base)
    # Heuristic: large clusters need more than 20 SD steps per pass.
    scaled = max(base, min(300, 8 * max(1, n_monomers)))
    if scaled != base:
        print(f"mini-nstep scaled {base} -> {scaled} for {n_monomers} monomer(s)")
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


def assert_dynamics_ready(
    *,
    max_grms: float = 50.0,
    abort: bool = True,
    require_mlpot_user: bool = False,
    user_zero_tol_kcalmol: float = 1.0e-6,
) -> float:
    """Warn or abort if gradients are still huge before starting dynamics."""
    import math

    import pycharmm
    import pycharmm.energy as energy

    pycharmm.lingo.charmm_script("ENER")
    grms = charmm_grms()
    user_kcal: float | None = None
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
        if user_missing or grms < 1.0e-4:
            msg = (
                "Pre-dynamics check failed: MLpot USER energy inactive or GRMS≈0 "
                f"(USER={user_kcal}, GRMS={grms:.4f}). "
                "Dynamics would integrate a free gas / zeroed BLOCK state."
            )
            if abort:
                raise RuntimeError(msg)
            print(f"WARN: {msg}", flush=True)
            return grms
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
            "Before MLpot: CGENFF SD/ABNR minimize + CHARMM heating (no USER/ML). "
            "Coordinates feed MLpot mini/heat/NVE."
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
        action="store_true",
        help=(
            "Compare MM bonded GRMS to post-MM-pre-min baseline after selected stages; "
            "run bonded-only SD (BLOCK toggle, MLpot stays on) if higher"
        ),
    )
    group.add_argument(
        "--bonded-mm-mini-after",
        type=str,
        default="heat",
        help="Comma-separated dynamics stages to check (default: heat)",
    )
    group.add_argument(
        "--bonded-mm-mini-steps",
        type=int,
        default=50,
        help="SD steps for bonded-only recovery mini (default: 50)",
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


def resolve_use_pbc(args: argparse.Namespace) -> bool:
    if getattr(args, "free_space", False):
        return False
    setup = (getattr(args, "setup", None) or "").strip().lower()
    if setup.startswith("pbc_"):
        return True
    if getattr(args, "box_size", None) is not None:
        return True
    return False


def resolve_pbc_box_side(args: argparse.Namespace, positions: np.ndarray) -> float:
    if getattr(args, "box_size", None) is not None:
        return float(args.box_size)
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        cubic_box_length_from_geometry,
    )

    ml_cutoff = float(getattr(args, "ml_cutoff", 12.0))
    return cubic_box_length_from_geometry(positions, ml_cutoff=ml_cutoff)
