"""PyCHARMM vs JAX CGenFF energy comparison along a coordinate trajectory.

User-facing demo: ``scripts/demo_charmm_jax_trajectory_energy.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mmml.interfaces.pycharmmInterface.charmm_jax_energy_benchmark import (
    ForceDelta,
    TermDelta,
    _force_delta,
)

MM_TERM_ORDER: tuple[str, ...] = (
    "bond",
    "angle",
    "urey",
    "torsion",
    "improper",
    "cmap",
    "bonded_total",
    "vdw",
    "elec",
    "nb_total",
    "total",
)


@dataclass(frozen=True, slots=True)
class FrameEnergyComparison:
    """Per-frame PyCHARMM vs JAX MM energy deltas."""

    frame: int
    terms: tuple[TermDelta, ...]
    forces: ForceDelta


@dataclass(frozen=True, slots=True)
class TermTrajectoryStats:
    """Aggregate error statistics for one energy term over a trajectory."""

    term: str
    max_abs_diff: float
    mean_abs_diff: float
    rms_abs_diff: float
    max_rel_diff: float
    mean_rel_diff: float


@dataclass(frozen=True, slots=True)
class TrajectoryEnergyComparison:
    """Full trajectory cross-check report."""

    name: str
    description: str
    n_atoms: int
    n_frames: int
    frames: tuple[FrameEnergyComparison, ...]
    term_stats: tuple[TermTrajectoryStats, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrajectoryMmContext:
    """Preloaded JAX MM system data (topology does not change along the traj)."""

    psf_path: Path
    prm_path: Path
    cell: np.ndarray
    nb_settings: Any
    extra_prm_files: tuple[Path, ...]
    bonded_system: Any
    nbond_data: Any


def _charmm_mm_energy_components_kcalmol() -> dict[str, float]:
    import pycharmm.energy as energy

    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        charmm_bonded_energy_components_kcalmol,
        charmm_nonbonded_energy_components_kcalmol,
    )

    bonded = charmm_bonded_energy_components_kcalmol()
    nb = charmm_nonbonded_energy_components_kcalmol()
    urey = float(bonded.get("urey", 0.0)) + float(bonded.get("ub", 0.0))
    return {
        "bond": float(bonded.get("bond", 0.0)),
        "angle": float(bonded.get("angl", 0.0)),
        "urey": urey,
        "torsion": float(bonded.get("dihe", 0.0)),
        "improper": float(bonded.get("impr", 0.0)),
        "cmap": float(bonded.get("cmap", 0.0)),
        "bonded_total": float(bonded.get("total", 0.0)),
        "vdw": float(nb["vdw"]),
        "elec": float(nb["elec"]),
        "nb_total": float(nb["total"]),
        "total": float(energy.get_total()),
    }


def _jax_mm_energy_components_kcalmol(result: Any) -> dict[str, float]:
    bonded = result.bonded
    nb = result.nonbonded
    return {
        "bond": float(bonded.get("bond", 0.0)),
        "angle": float(bonded.get("angle", 0.0)),
        "urey": float(bonded.get("urey", 0.0)),
        "torsion": float(bonded.get("torsion", 0.0)),
        "improper": float(bonded.get("improper", 0.0)),
        "cmap": float(bonded.get("cmap", 0.0)),
        "bonded_total": float(bonded["total"]),
        "vdw": float(nb["vdw"]),
        "elec": float(nb["elec"]),
        "nb_total": float(nb["total"]),
        "total": float(result.total_energy),
    }


def term_deltas_from_component_maps(
    jax_terms: dict[str, float],
    charmm_terms: dict[str, float],
    *,
    term_order: Sequence[str] = MM_TERM_ORDER,
) -> tuple[TermDelta, ...]:
    """Build per-term deltas for the shared MM component keys."""
    out: list[TermDelta] = []
    for term in term_order:
        if term not in jax_terms or term not in charmm_terms:
            continue
        out.append(
            TermDelta.from_pair(term, charmm_terms[term], jax_terms[term])
        )
    return tuple(out)


def load_trajectory_mm_context(
    *,
    psf_path: Path | str,
    prm_path: Path | str,
    cell: np.ndarray,
    nb_settings: Any,
    positions0: np.ndarray,
    extra_prm_files: Sequence[Path | str] = (),
) -> TrajectoryMmContext:
    """Load bonded/nonbonded JAX data once for a trajectory comparison."""
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        load_bonded_system_from_psf,
        load_nonbonded_system_from_charmm,
    )

    pos0 = np.asarray(positions0, dtype=np.float64)
    bonded = load_bonded_system_from_psf(
        psf_path,
        pos0,
        prm_file=prm_path,
        extra_prm_files=extra_prm_files,
    )
    nbond_data = load_nonbonded_system_from_charmm(psf_path, prm_path)
    return TrajectoryMmContext(
        psf_path=Path(psf_path),
        prm_path=Path(prm_path),
        cell=np.asarray(cell, dtype=np.float64),
        nb_settings=nb_settings,
        extra_prm_files=tuple(Path(p) for p in extra_prm_files),
        bonded_system=bonded,
        nbond_data=nbond_data,
    )


def _box_side_from_cell(cell: np.ndarray) -> float | None:
    arr = np.asarray(cell, dtype=np.float64)
    if arr.shape != (3, 3):
        return None
    return float(arr[0, 0])


def ensure_full_cgenff_mm_session(
    ctx: TrajectoryMmContext,
    *,
    reregister_cgenff: bool = False,
    verbose: bool = False,
    nbxmod: int = 5,
) -> None:
    """Optional one-time CGENFF restore before a multi-frame loop (no BLOCK).

    Fresh box builds already have full CGENFF + PBC loaded; the default is a
    no-op.  Set ``reregister_cgenff=True`` only after MLpot zeroed bonded params
  (``READ PARAM APPEND`` via :func:`apply_full_cgenff_params`, then rebuild PBC).
    """
    if not reregister_cgenff:
        return

    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import (
        apply_full_cgenff_params,
    )

    apply_full_cgenff_params(verbose=verbose)
    box_side = _box_side_from_cell(ctx.cell)
    if box_side is not None and box_side > 0.0:
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
            restore_charmm_cubic_crystal_lattice,
        )

        restore_charmm_cubic_crystal_lattice(box_side, nbxmod=int(nbxmod))


def compare_frame_mm_energy(
    positions: np.ndarray,
    ctx: TrajectoryMmContext,
    *,
    frame_index: int = 0,
) -> FrameEnergyComparison:
    """Compare live PyCHARMM ``ENER FORCE`` vs JAX CGenFF clone at one geometry."""
    import pycharmm.energy as energy

    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        run_charmm_bonded_ener_force,
        set_charmm_positions,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        charmm_total_forces_kcalmol_A,
    )
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        mm_system_energy_and_forces,
    )

    pos = np.asarray(positions, dtype=np.float64)
    set_charmm_positions(pos)
    import pycharmm

    pycharmm.lingo.charmm_script("UPDATE")
    run_charmm_bonded_ener_force(silent=True)

    charmm_terms = _charmm_mm_energy_components_kcalmol()
    charmm_forces = charmm_total_forces_kcalmol_A()

    result = mm_system_energy_and_forces(
        pos,
        ctx.bonded_system,
        ctx.nbond_data,
        ctx.cell,
        ctx.nb_settings,
    )
    jax_terms = _jax_mm_energy_components_kcalmol(result)
    terms = term_deltas_from_component_maps(jax_terms, charmm_terms)
    forces = _force_delta(charmm_forces, np.asarray(result.forces, dtype=np.float64))

    # Sanity: CHARMM total should match energy module after ENER FORCE.
    _ = float(energy.get_total())

    return FrameEnergyComparison(frame=frame_index, terms=terms, forces=forces)


def summarize_trajectory_term_errors(
    frames: Sequence[FrameEnergyComparison],
    *,
    term_order: Sequence[str] = MM_TERM_ORDER,
) -> tuple[TermTrajectoryStats, ...]:
    """Aggregate absolute/relative deltas per energy term across frames."""
    if not frames:
        return ()

    by_term: dict[str, list[TermDelta]] = {term: [] for term in term_order}
    for frame in frames:
        for delta in frame.terms:
            by_term.setdefault(delta.term, []).append(delta)

    stats: list[TermTrajectoryStats] = []
    for term in term_order:
        deltas = by_term.get(term, [])
        if not deltas:
            continue
        abs_vals = np.asarray([d.abs_diff for d in deltas], dtype=np.float64)
        rel_vals = np.asarray([d.rel_diff for d in deltas], dtype=np.float64)
        stats.append(
            TermTrajectoryStats(
                term=term,
                max_abs_diff=float(np.max(np.abs(abs_vals))),
                mean_abs_diff=float(np.mean(np.abs(abs_vals))),
                rms_abs_diff=float(np.sqrt(np.mean(abs_vals * abs_vals))),
                max_rel_diff=float(np.max(np.abs(rel_vals))),
                mean_rel_diff=float(np.mean(np.abs(rel_vals))),
            )
        )
    return tuple(stats)


def compare_trajectory_mm_energy(
    positions: np.ndarray,
    ctx: TrajectoryMmContext,
    *,
    name: str = "trajectory",
    description: str = "",
    metadata: dict[str, Any] | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    reregister_cgenff: bool = False,
    progress: bool = True,
) -> TrajectoryEnergyComparison:
    """Compare PyCHARMM vs JAX MM energies for every frame in ``positions``."""
    traj = np.asarray(positions, dtype=np.float64)
    if traj.ndim != 3 or traj.shape[2] != 3:
        raise ValueError(f"positions must be (n_frames, n_atoms, 3), got {traj.shape}")

    stride = max(1, int(frame_stride))
    indices = list(range(0, traj.shape[0], stride))
    if max_frames is not None:
        indices = indices[: int(max_frames)]

    if reregister_cgenff:
        if progress:
            print("Restoring full CGENFF params (no BLOCK)...", flush=True)
        ensure_full_cgenff_mm_session(ctx, reregister_cgenff=True)

    frame_reports: list[FrameEnergyComparison] = []
    for i, frame_idx in enumerate(indices, start=1):
        if progress:
            print(f"  frame {frame_idx} ({i}/{len(indices)})", flush=True)
        frame_reports.append(compare_frame_mm_energy(traj[frame_idx], ctx, frame_index=frame_idx))

    term_stats = summarize_trajectory_term_errors(frame_reports)
    return TrajectoryEnergyComparison(
        name=name,
        description=description,
        n_atoms=int(traj.shape[1]),
        n_frames=len(frame_reports),
        frames=tuple(frame_reports),
        term_stats=term_stats,
        metadata=dict(metadata or {}),
    )


def run_short_nvt_dynamics_dcd(
    *,
    dcd_path: Path | str,
    n_frames: int = 10,
    timestep_ps: float = 0.0002,
    temp: float = 300.0,
    box_side_A: float | None = None,
    minimize_sd_steps: int = 20,
) -> Path:
    """Run a short Hoover NVT segment and write coordinates to a DCD file."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmMmMinimizeConfig,
        CharmmTrajectoryFiles,
        build_hoover_heat_dynamics,
        minimize_charmm_mm_only,
        run_dynamics,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import ensure_charmm_crystal_for_cpt

    if int(n_frames) < 1:
        raise ValueError("n_frames must be >= 1")

    dcd = Path(dcd_path)
    dcd.parent.mkdir(parents=True, exist_ok=True)
    if dcd.is_file():
        dcd.unlink()

    if box_side_A is not None:
        ensure_charmm_crystal_for_cpt(float(box_side_A), quiet=True)

    if minimize_sd_steps > 0:
        minimize_charmm_mm_only(
            CharmmMmMinimizeConfig(
                nstep_sd=int(minimize_sd_steps),
                use_pbc=box_side_A is not None,
                verbose=False,
            )
        )

    nsavc = 1
    nstep = int(n_frames) * nsavc
    duration_ps = float(timestep_ps) * float(nstep)
    save_interval_ps = float(timestep_ps) * float(nsavc)

    io = CharmmTrajectoryFiles(trajectory=dcd)
    open_files, io_kw, _aliases = io.open_for_run()

    kw = build_hoover_heat_dynamics(
        temp=float(temp),
        firstt=float(temp),
        finalt=float(temp),
        use_pbc=box_side_A is not None,
        duration_ps=duration_ps,
        save_interval_ps=save_interval_ps,
        timestep_ps=float(timestep_ps),
        echeck=500.0,
        tmass=100,
    )
    kw.update(io_kw)
    kw["nstep"] = nstep
    kw["nsavc"] = nsavc
    kw["start"] = True
    kw["iasvel"] = 1
    kw["restart"] = False
    kw["iunrea"] = -1

    try:
        run_dynamics(kw)
    finally:
        for handle in open_files:
            try:
                handle.close()
            except Exception:
                pass

    if not dcd.is_file():
        raise RuntimeError(f"Dynamics did not produce trajectory file: {dcd}")
    return dcd.resolve()


def read_trajectory_positions(
    dcd_path: Path | str,
    *,
    max_frames: int | None = None,
    frame_stride: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read coordinates from a CHARMM DCD file."""
    from mmml.utils.dcd_reader import read_dcd_trajectory

    positions, header = read_dcd_trajectory(
        dcd_path,
        max_frames=max_frames,
        frame_stride=frame_stride,
    )
    return positions, dict(header)


def render_trajectory_markdown(report: TrajectoryEnergyComparison) -> str:
    """Human-readable per-frame and aggregate error report."""
    lines = [
        "# PyCHARMM vs JAX CGenFF trajectory energy comparison",
        "",
        report.description,
        f"System: **{report.name}**  |  Atoms: {report.n_atoms}  |  Frames: {report.n_frames}",
        "",
        "CHARMM nonbonded components aggregate PBC image terms "
        "(``vdw`` = VDW+IMNB, ``elec`` = ELEC+IMEL[+EXTE]) to match JAX MIC pair totals.",
        "",
        "## Aggregate term errors (kcal/mol)",
        "",
        "| Term | max |Δ| | mean |Δ| | RMS |Δ| | max |rel Δ| | mean |rel Δ| |",
        "|------|--------|---------|---------|-----------|------------|",
    ]
    for stat in report.term_stats:
        lines.append(
            f"| {stat.term} | {stat.max_abs_diff:.3e} | {stat.mean_abs_diff:.3e} | "
            f"{stat.rms_abs_diff:.3e} | {stat.max_rel_diff:.3e} | {stat.mean_rel_diff:.3e} |"
        )
    lines.extend(["", "## Per-frame component errors", ""])
    for frame in report.frames:
        lines.append(f"### Frame {frame.frame}")
        lines.append("")
        lines.append("| Term | CHARMM | JAX | Δ | rel Δ |")
        lines.append("|------|--------|-----|---|-------|")
        for term in frame.terms:
            lines.append(
                f"| {term.term} | {term.charmm_kcal:.6f} | {term.jax_kcal:.6f} | "
                f"{term.abs_diff:+.2e} | {term.rel_diff:+.2e} |"
            )
        lines.append("")
        lines.append(
            f"Force RMS Δ: {frame.forces.force_rms:.4e}  "
            f"max |ΔF|: {frame.forces.force_max:.4e}"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def render_trajectory_json(report: TrajectoryEnergyComparison) -> str:
    """Machine-readable trajectory comparison report."""
    payload = {
        **asdict(report),
        "frames": [asdict(frame) for frame in report.frames],
        "term_stats": [asdict(stat) for stat in report.term_stats],
    }
    return json.dumps(payload, indent=2)


def synthetic_trajectory_from_seed(
    positions0: np.ndarray,
    *,
    n_frames: int = 10,
    seed: int = 17,
    scale: float = 0.02,
) -> np.ndarray:
    """Build a deterministic coordinate series by cumulative Gaussian noise."""
    rng = np.random.default_rng(seed)
    base = np.asarray(positions0, dtype=np.float64)
    frames = [base.copy()]
    pos = base.copy()
    for _ in range(int(n_frames) - 1):
        pos = pos + rng.normal(scale=scale, size=pos.shape)
        frames.append(pos.copy())
    return np.stack(frames, axis=0)


__all__ = [
    "FrameEnergyComparison",
    "MM_TERM_ORDER",
    "TermTrajectoryStats",
    "TrajectoryEnergyComparison",
    "TrajectoryMmContext",
    "compare_frame_mm_energy",
    "compare_trajectory_mm_energy",
    "ensure_full_cgenff_mm_session",
    "load_trajectory_mm_context",
    "read_trajectory_positions",
    "render_trajectory_json",
    "render_trajectory_markdown",
    "run_short_nvt_dynamics_dcd",
    "summarize_trajectory_term_errors",
    "synthetic_trajectory_from_seed",
    "term_deltas_from_component_maps",
]
