"""Shared helpers for MLpot exploration scripts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np

# Repo root: tests/functionality/mlpot -> parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Defaults aligned with DESdimers / acetone cluster tests
DEFAULT_RESIDUE = "ACO"
DEFAULT_N_MOLECULES = 2
DEFAULT_SPACING = 4.0
ACO_ATOMS_PER_MONOMER = 10


def add_charmm_output_args(parser: argparse.ArgumentParser) -> None:
    """CLI flags for CHARMM console verbosity (scripts 04–05)."""
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
        default=0,
        help="CHARMM BOMBlev (default: 0 = stop on errors)",
    )
    group.add_argument(
        "--nprint",
        type=int,
        default=1,
        help="Print energy/status every N minimization or dynamics steps (default: 1)",
    )
    group.add_argument(
        "--quiet",
        action="store_true",
        help="Shortcut for --prnlev 0 --warnlev 0 and coarse nprint",
    )


def add_dcd_save_args(parser: argparse.ArgumentParser) -> None:
    """CLI flags for DCD trajectory frame spacing (scripts 04–05)."""
    group = parser.add_argument_group("DCD trajectory output")
    group.add_argument(
        "--dcd-nsavc",
        type=int,
        default=1,
        help="Write a DCD frame every N integration/SD steps (CHARMM nsavc; default: 1)",
    )
    group.add_argument(
        "--dcd-interval-ps",
        type=float,
        default=None,
        metavar="PS",
        help="Alternative to --dcd-nsavc: save interval in ps (dynamics only; uses timestep)",
    )


def resolve_dcd_nsavc(
    *,
    dcd_nsavc: int,
    dcd_interval_ps: float | None = None,
    timestep_ps: float | None = None,
    nstep: int | None = None,
) -> int:
    """Resolve CHARMM ``nsavc`` from step count or a target time interval."""
    if dcd_interval_ps is not None:
        if timestep_ps is None or timestep_ps <= 0:
            raise ValueError("timestep_ps required when using --dcd-interval-ps")
        nsavc = int(round(float(dcd_interval_ps) / float(timestep_ps)))
    else:
        nsavc = int(dcd_nsavc)
    nsavc = max(1, nsavc)
    if nstep is not None:
        nsavc = min(nsavc, max(1, int(nstep)))
    return nsavc


def apply_charmm_output_from_args(args: argparse.Namespace) -> int:
    """Apply PRNLev/WRNLev from argparse; return effective ``nprint``."""
    # Import setup submodule directly (avoid pulling full mlpot via package __init__).
    from mmml.interfaces.pycharmmInterface.mlpot.setup import apply_charmm_verbosity

    if getattr(args, "quiet", False):
        apply_charmm_verbosity(prnlev=0, warnlev=0, bomlev=args.bomlev)
        nstep = getattr(args, "nstep", 100)
        return max(1, nstep)
    apply_charmm_verbosity(
        prnlev=args.prnlev,
        warnlev=args.warnlev,
        bomlev=args.bomlev,
    )
    return max(1, int(args.nprint))


def add_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--residue",
        default=DEFAULT_RESIDUE,
        help="CHARMM residue name for CGenFF cluster (default: ACO)",
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=DEFAULT_N_MOLECULES,
        help="Number of identical monomers (CGenFF residues) in the cluster (default: 2)",
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


def add_monomer_constraint_args(
    parser: argparse.ArgumentParser,
    *,
    for_dynamics: bool = False,
) -> None:
    """CLI flags to fix/constrain specific monomers (CHARMM ``resid`` = monomer index)."""
    group = parser.add_argument_group("Monomer constraints (cons_fix)")
    if for_dynamics:
        group.add_argument(
            "--constrain-resids",
            type=str,
            default="",
            metavar="IDS",
            help="Comma-separated residue IDs frozen for the whole dynamics run (e.g. 1,2)",
        )
    else:
        group.add_argument(
            "--fix-resids",
            type=str,
            default="1",
            metavar="IDS",
            help="Monomers fixed during SD pass 1 only (comma-separated resids; default: 1)",
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
            help="Skip the constrained SD pass (only run free minimization)",
        )


def parse_resid_list(text: str) -> list[int]:
    """Parse ``1,2,3`` or ``1 2 3`` into unique positive residue IDs."""
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
    """Resids to hold fixed in minimization pass 1 (empty if --no-fix)."""
    if getattr(args, "no_fix", False):
        return []
    if getattr(args, "fix_resid", None) is not None:
        return [int(args.fix_resid)]
    return parse_resid_list(getattr(args, "fix_resids", "") or "")


def resolve_constrain_resids(args: argparse.Namespace) -> list[int]:
    """Resids frozen for an entire dynamics run."""
    return parse_resid_list(getattr(args, "constrain_resids", "") or "")


def build_acetone_cluster(
    n_molecules: int,
    spacing: float = DEFAULT_SPACING,
) -> Tuple[np.ndarray, np.ndarray]:
    """Acetone cluster (ACO × n) with bundled 3D monomer template; 10 atoms per monomer."""
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
    candidates.extend(
        [
            PROJECT_ROOT / "examples/ckpts_json/DESdimers_params.json",
            PROJECT_ROOT / "examples/ckpts_json",
            PROJECT_ROOT / "ckpts_json/DESdimers_params.json",
            PROJECT_ROOT / "mmml/models/physnetjax/ckpts/DESdimers",
            PROJECT_ROOT / "mmml/models/physnetjax/ckpts",
        ]
    )
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt.resolve()
    raise FileNotFoundError(
        "No checkpoint found. Set MMML_CKPT or pass --checkpoint."
    )


def load_physnet_for_cluster(
    checkpoint: Path,
    n_atoms: int,
) -> Tuple[Any, Any]:
    """Return (params, model) for ``n_atoms`` atoms."""
    from mmml.cli.base import load_physnet_params_and_ef_model, resolve_checkpoint_paths

    if checkpoint.is_file() and checkpoint.suffix == ".json":
        return load_physnet_params_and_ef_model(checkpoint, natoms=n_atoms)

    _, epoch_dir = resolve_checkpoint_paths(checkpoint)
    from mmml.models.physnetjax.physnetjax.restart.restart import get_params_model

    params, model = get_params_model(str(epoch_dir), natoms=n_atoms)
    return params, model


def validate_cluster_geometry(
    positions: np.ndarray,
    *,
    min_axis_span: float = 0.3,
    min_monomer_extent: float = 1.5,
    n_molecules: int | None = None,
) -> dict[str, float]:
    """Raise if coordinates look collapsed or non-physical; else return summary stats."""
    r = np.asarray(positions, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {r.shape}")
    span = r.max(axis=0) - r.min(axis=0)
    if float(span[1]) < min_axis_span or float(span[2]) < min_axis_span:
        raise ValueError(
            f"Cluster is nearly planar/collinear (spans Å x={span[0]:.3f} y={span[1]:.3f} z={span[2]:.3f})"
        )
    if n_molecules is not None and n_molecules > 0 and r.shape[0] % n_molecules == 0:
        n_per = r.shape[0] // n_molecules
        coms = []
        for i in range(n_molecules):
            chunk = r[i * n_per : (i + 1) * n_per]
            extent = float(np.linalg.norm(chunk.max(axis=0) - chunk.min(axis=0)))
            if extent < min_monomer_extent:
                raise ValueError(
                    f"Monomer {i + 1} extent {extent:.3f} Å < {min_monomer_extent} Å (likely bad template/ic.build)"
                )
            coms.append(chunk.mean(axis=0))
        com_dists: list[float] = []
        if len(coms) > 1:
            for i in range(len(coms)):
                for j in range(i + 1, len(coms)):
                    com_dists.append(float(np.linalg.norm(coms[j] - coms[i])))
            com_sep = com_dists[0]
        else:
            com_sep = 0.0
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
    """Build CGenFF cluster in CHARMM and return (Z, positions) without keeping CHARMM state."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401 — sets CHARMM env
    from mmml.cli.run.md_pbc_suite.cluster import _build_psf_ordered_cluster

    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    z, r = _build_psf_ordered_cluster(residue.upper(), n_molecules, spacing)
    sync_charmm_positions(r)
    validate_cluster_geometry(r, n_molecules=n_molecules)
    return z, r


def build_acetone_dimer_cluster(spacing: float = DEFAULT_SPACING) -> Tuple[np.ndarray, np.ndarray]:
    """ACO × 2 acetone dimer (20 atoms); alias for :func:`build_acetone_cluster`."""
    return build_acetone_cluster(2, spacing)


def build_cluster_from_args(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build cluster from CLI args; returns ``(Z, positions, n_atoms)``."""
    residue = args.residue.upper()
    n_mol = int(args.n_molecules)
    spacing = float(args.spacing)
    if residue == "ACO":
        z, r = build_acetone_cluster(n_mol, spacing)
    else:
        z, r = build_ase_cluster(residue, n_mol, spacing)
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
    """Ensure each resid maps to a built monomer (1 … n_molecules)."""
    bad = [r for r in resids if r < 1 or r > n_molecules]
    if bad:
        raise ValueError(
            f"residue ID(s) {bad} out of range for {n_molecules} monomer(s) "
            "(resids are 1-based monomer indices from the PSF)"
        )


def setup_cons_fix_for_resids(resids: list[int]) -> Any:
    """Apply ``cons_fix`` to all atoms in the given residue IDs."""
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
    trajectory: Path | None = None,
    n_atoms: int,
) -> Path:
    """Write a small Tcl script that loads topology (with bonds) + optional trajectory."""
    out_dir = out_dir.resolve()
    topology_psf = topology_psf.resolve()
    lines = [
        "# VMD: topology written BEFORE MLpot (bonds intact).",
        f"# Atoms: {n_atoms} — must match trajectory frame count.",
        f"mol new {{{topology_psf}}}",
    ]
    if trajectory is not None:
        traj = trajectory.resolve()
        lines.append(f"mol addfile {{{traj}}} waitfor all")
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
    trajectory: Path | None,
    n_atoms: int,
    bondless_psf: Path | None = None,
) -> None:
    """Print how to load the system in VMD (avoid bondless / wrong-atom-count PSF)."""
    topo = topology_psf.resolve()
    print("\n=== VMD ===")
    print(f"  Atoms in this run: {n_atoms}")
    print(f"  Topology (bonds):  {topo}")
    if trajectory is not None:
        traj = trajectory.resolve()
        print(f"  Trajectory:        {traj}")
        print(f"\n  vmd {topo} {traj}")
        tcl = write_vmd_load_script(
            out_dir=out_dir,
            tag=tag,
            topology_psf=topo,
            trajectory=traj,
            n_atoms=n_atoms,
        )
        print(f"  # or: vmd -e {tcl}")
    else:
        print(f"\n  vmd {topo}")
    if bondless_psf is not None:
        print(
            f"\n  Do NOT use {bondless_psf.name} in VMD — written after MLpot "
            "(no bonds; for CHARMM restart only)."
        )
    print(
        "  Do not mix PSF/DCD from different --n-molecules runs "
        "(atom counts must match)."
    )


def all_atom_selection():
    """PyCHARMM selection for all atoms (for MLpot on full cluster)."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import select_all_atoms

    return select_all_atoms()


def setup_charmm_nbonds() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds

    setup_default_nbonds()


def check_mlpot_symbols() -> list[str]:
    """Return missing MLpot symbols on libcharmm, if any."""
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


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")
