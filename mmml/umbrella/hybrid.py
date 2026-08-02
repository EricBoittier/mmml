"""Solvated mechanical-embedding umbrella sampling (ML solute + MM solvent)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from mmml.md.ml_region import (
    compact_mol_id,
    merge_ml_region_mol_id,
    resolve_ml_region_indices,
)
from mmml.md.system import MolecularSystem, SystemSpec
from mmml.umbrella.config import UmbrellaConfig
from mmml.umbrella.io import (
    BIN_MINIMA_TRAJ,
    SNAPSHOTS_NPZ,
    SUMMARY_JSON,
    save_snapshots,
    write_summary,
)
from mmml.umbrella.sample import (
    UmbrellaResult,
    center_com_positions,
    select_lowest_energy_frames,
)

__all__ = [
    "bind_hybrid_atom_names",
    "find_atom_index_by_name",
    "merge_ml_region_mol_id",
    "mic_distance",
    "relax_around_frozen_seed",
    "resolve_ml_region_indices",
    "run_umbrella_hybrid_nvt",
    "save_failure_trace",
    "seed_force_maxima",
    "stretch_antisymmetric_seed_mic",
    "stretch_distance_seed_mic",
]


def find_atom_index_by_name(
    atom_names: Sequence[str],
    resnames: Sequence[str],
    *,
    atom_name: str,
    ml_resnames: Sequence[str] | None = None,
) -> int:
    """Find the first atom with ``atom_name`` (optionally restricted to ML residues)."""
    target = str(atom_name).strip().upper()
    want_res = (
        None
        if ml_resnames is None
        else {str(r).strip().upper() for r in ml_resnames}
    )
    hits: list[int] = []
    for i, (aname, rname) in enumerate(zip(atom_names, resnames, strict=True)):
        if str(aname).strip().upper() != target:
            continue
        if want_res is not None and str(rname).strip().upper() not in want_res:
            continue
        hits.append(i)
    if not hits:
        raise ValueError(
            f"atom name {atom_name!r} not found"
            + (f" in residues {sorted(want_res)}" if want_res else "")
        )
    if len(hits) > 1:
        raise ValueError(
            f"atom name {atom_name!r} is ambiguous ({len(hits)} hits); "
            "restrict with ml_resnames or use explicit atom indices"
        )
    return int(hits[0])


def mic_distance(
    positions: np.ndarray,
    atom_i: int,
    atom_j: int,
    box: np.ndarray | None,
) -> float:
    """Scalar distance (Å), minimum-image when ``box`` is set."""
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
        mic_displacement_numpy,
    )

    r = np.asarray(positions, dtype=np.float64)
    disp = mic_displacement_numpy(r[atom_i], r[atom_j], box)
    return float(np.linalg.norm(disp))


def stretch_distance_seed_mic(
    positions: np.ndarray,
    atom_i: int,
    atom_j: int,
    target_A: float,
    box: np.ndarray | None,
    move_with: Sequence[int] | None = None,
) -> np.ndarray:
    """MIC-aware rigid stretch of ``atom_j`` (+ ``move_with``); ``atom_i`` fixed."""
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
        mic_displacement_numpy,
    )

    r = np.asarray(positions, dtype=np.float64).copy()
    if target_A <= 0:
        raise ValueError(f"target_A must be > 0 (got {target_A})")
    disp = mic_displacement_numpy(r[atom_i], r[atom_j], box)
    dist = float(np.linalg.norm(disp))
    if dist < 1e-8:
        raise ValueError(
            f"cannot stretch atoms ({atom_i}, {atom_j}): current distance ~0"
        )
    u = disp / dist
    shift = (float(target_A) - dist) * u
    group = {int(atom_j)}
    if move_with:
        group.update(int(a) for a in move_with)
    for a in sorted(group):
        r[a] = r[a] + shift
    return r


def _load_box_json(path: Path) -> float | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "box_side_A",
        "final_cubic_side_A",
        "box_size",
        "side_length_A",
        "L",
    ):
        if key in data and data[key] is not None:
            return float(data[key])
    return None


def _cubic_box(side: float) -> np.ndarray:
    L = float(side)
    return np.diag([L, L, L]).astype(np.float64)


def _psf_atom_tables(psf_path: Path) -> tuple[list[str], list[str]]:
    from mmml.utils.domdec_psf_order import read_psf_atoms_and_bonds

    atoms, _bonds = read_psf_atoms_and_bonds(psf_path)
    names = [a.atom_name for a in atoms]
    resnames = [a.resname for a in atoms]
    return names, resnames


def _load_coords(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from ase.io import read

    atoms = read(str(path))
    if isinstance(atoms, list):
        atoms = atoms[0]
    r = np.asarray(atoms.get_positions(), dtype=np.float64)
    z = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
    return r, z


def _resolve_atom_ref(
    ref,
    atom_names: Sequence[str],
    resnames: Sequence[str],
    *,
    ml_resnames: Sequence[str] | None,
) -> int:
    """Map a YAML atom reference (index or name like ``C1``) to a 0-based index."""
    if isinstance(ref, str):
        s = ref.strip()
        try:
            return int(s)
        except ValueError:
            return find_atom_index_by_name(
                atom_names,
                resnames,
                atom_name=s,
                ml_resnames=ml_resnames,
            )
    return int(ref)


def bind_hybrid_atom_names(
    cfg: UmbrellaConfig,
    atom_names: Sequence[str],
    resnames: Sequence[str],
) -> UmbrellaConfig:
    """Resolve PSF atom *names* in ``cv_x`` / walls to integer indices.

    Enables YAML such as::

        cv_x:
          pairs: [[C1, CL1], [C1, N1]]
          coefficients: [1.0, -1.0]   # ξ = r(C–Cl) − r(C–N)
    """
    from mmml.md.restraints import LinearDistanceCV
    from mmml.umbrella.config import _resolve_wall, _spec_needs_name_bind

    ml = cfg.ml_resnames

    def _bind_pair(pair):
        return (
            _resolve_atom_ref(pair[0], atom_names, resnames, ml_resnames=ml),
            _resolve_atom_ref(pair[1], atom_names, resnames, ml_resnames=ml),
        )

    def _bind_cv(spec):
        if spec is None:
            return None
        if isinstance(spec, LinearDistanceCV):
            return spec
        data = dict(spec)
        data["pairs"] = [_bind_pair(p) for p in data["pairs"]]
        return LinearDistanceCV.from_spec(data)

    def _bind_wall(spec):
        if not isinstance(spec, dict) or not _spec_needs_name_bind(spec):
            return _resolve_wall(spec)
        data = dict(spec)
        if "pairs" in data:
            data["pairs"] = [_bind_pair(p) for p in data["pairs"]]
        if "atoms" in data:
            data["atoms"] = [
                _resolve_atom_ref(a, atom_names, resnames, ml_resnames=ml)
                for a in data["atoms"]
            ]
        if "cv" in data:
            data["cv"] = _bind_cv(data["cv"])
        return _resolve_wall(data)

    cv_x = _bind_cv(cfg.cv_x)
    cv_y = _bind_cv(cfg.cv_y)
    walls = tuple(_bind_wall(w) for w in cfg.walls)

    atom_i, atom_j = cfg.atom_i, cfg.atom_j
    if cv_x is None:
        atom_i = int(cfg.atom_i)
        atom_j = int(cfg.atom_j)
        if cfg.atom_name_i is not None:
            atom_i = find_atom_index_by_name(
                atom_names, resnames, atom_name=cfg.atom_name_i, ml_resnames=ml
            )
        if cfg.atom_name_j is not None:
            atom_j = find_atom_index_by_name(
                atom_names, resnames, atom_name=cfg.atom_name_j, ml_resnames=ml
            )
        cv_x = LinearDistanceCV.distance(atom_i, atom_j)
    else:
        atom_i, atom_j = int(cv_x.pairs[0][0]), int(cv_x.pairs[0][1])

    return replace(
        cfg,
        cv_x=cv_x,
        cv_y=cv_y,
        walls=walls,
        atom_i=atom_i,
        atom_j=atom_j,
        atom_name_i=None,
        atom_name_j=None,
    )


def stretch_antisymmetric_seed_mic(
    positions: np.ndarray,
    pair_plus: tuple[int, int],
    pair_minus: tuple[int, int],
    target_xi: float,
    box: np.ndarray | None,
    *,
    move_with_plus: Sequence[int] | None = None,
    move_with_minus: Sequence[int] | None = None,
    min_distance_A: float = 1.4,
) -> np.ndarray:
    """MIC-aware seed for ``ξ = r_plus − r_minus`` (hold sum, split difference)."""
    r_ref = np.asarray(positions, dtype=np.float64)
    d_plus = mic_distance(r_ref, pair_plus[0], pair_plus[1], box)
    d_minus = mic_distance(r_ref, pair_minus[0], pair_minus[1], box)
    total = d_plus + d_minus
    xi = float(target_xi)
    new_plus = 0.5 * (total + xi)
    new_minus = 0.5 * (total - xi)
    if new_plus < min_distance_A or new_minus < min_distance_A:
        needed = abs(xi) + 2.0 * min_distance_A
        new_plus = 0.5 * (needed + xi)
        new_minus = 0.5 * (needed - xi)
    r = stretch_distance_seed_mic(
        r_ref, pair_plus[0], pair_plus[1], new_plus, box, move_with=move_with_plus
    )
    return stretch_distance_seed_mic(
        r, pair_minus[0], pair_minus[1], new_minus, box, move_with=move_with_minus
    )


def build_hybrid_umbrella_system(cfg: UmbrellaConfig) -> tuple[MolecularSystem, np.ndarray, list[str], list[str]]:
    """Build a PBC ``MolecularSystem`` with ML-region ``mol_id`` merge.

    Returns ``(system, ml_indices, atom_names, resnames)``.
    """
    from mmml.md.builders import PsfSystemBuilder
    from mmml.md.builders._topology import monomer_indices_from_mol_id
    from mmml.interfaces.pycharmmInterface.charmm_paths import resolve_cgenff_toppar_paths

    if cfg.composition is not None and cfg.from_psf is None:
        from mmml.cli.run.md_system_unified import build_packmol_system_with_ffparams
        from mmml.md.system import SystemSpec as Spec

        if cfg.box_size is None:
            raise ValueError("composition hybrid path requires box_size")
        system = build_packmol_system_with_ffparams(
            Spec(
                builder="packmol",
                composition=cfg.composition,
                box_size=float(cfg.box_size),
                seed=int(cfg.seed),
                params={"cube_side": float(cfg.box_size)},
            )
        )
        # Packmol metadata stores one residue name per molecule, not per atom.
        per_mol = list(system.metadata.get("residue_names") or [])
        if not per_mol or not system.monomer_indices:
            raise ValueError(
                "packmol hybrid system missing residue_names / monomer_indices metadata"
            )
        if len(per_mol) != len(system.monomer_indices):
            raise ValueError(
                f"residue_names length {len(per_mol)} != "
                f"n_molecules {len(system.monomer_indices)}"
            )
        resnames = [""] * system.n_atoms
        for mol_ix, group in enumerate(system.monomer_indices):
            name = str(per_mol[mol_ix])
            for a in np.asarray(group, dtype=int):
                resnames[int(a)] = name
        atom_names = [f"X{i}" for i in range(system.n_atoms)]
        ml_indices = resolve_ml_region_indices(resnames, cfg.ml_resnames)
        mol_id = compact_mol_id(merge_ml_region_mol_id(system.mol_id, ml_indices))
        monomers = monomer_indices_from_mol_id(mol_id)
        system = MolecularSystem(
            R=system.R,
            Z=system.Z,
            box=system.box if system.box is not None else _cubic_box(float(cfg.box_size)),
            mol_id=mol_id,
            monomer_indices=monomers,
            water_indices=system.water_indices,
            psf_path=system.psf_path,
            ff_params=system.ff_params,
            metadata={
                **dict(system.metadata),
                "ml_atom_indices": ml_indices.tolist(),
                "engine": "hybrid_jaxmd",
            },
        )
        return system, ml_indices, atom_names, resnames

    psf_path = Path(cfg.from_psf).expanduser().resolve()  # type: ignore[arg-type]
    coord_path = (
        Path(cfg.from_pdb).expanduser().resolve()
        if cfg.from_pdb is not None
        else Path(cfg.from_crd).expanduser().resolve()
        if cfg.from_crd is not None
        else Path(cfg.structure).expanduser().resolve()  # type: ignore[arg-type]
    )
    positions, atomic_numbers = _load_coords(coord_path)
    atom_names, resnames = _psf_atom_tables(psf_path)
    if len(atom_names) != positions.shape[0]:
        raise ValueError(
            f"PSF atom count {len(atom_names)} != coordinate atoms {positions.shape[0]}"
        )

    box_side = cfg.box_size
    if box_side is None:
        box_side = _load_box_json(coord_path.parent / "box.json")
    if box_side is None:
        raise ValueError(
            "hybrid_jaxmd requires box_size or a sibling box.json next to the coordinate file"
        )
    box = _cubic_box(float(box_side))

    toppar = resolve_cgenff_toppar_paths()
    prm_paths = [toppar.prm]
    extra_prm = Path(__file__).resolve().parents[2] / "examples" / "m" / "par_ch3cl.prm"
    if extra_prm.is_file():
        prm_paths.append(extra_prm)

    system = PsfSystemBuilder().build(
        SystemSpec(
            builder="psf",
            params={
                "psf_path": psf_path,
                "prm_paths": prm_paths,
                "positions": positions,
                "atomic_numbers": atomic_numbers,
                "box": box,
            },
        )
    )
    ml_indices = resolve_ml_region_indices(resnames, cfg.ml_resnames)
    mol_id = compact_mol_id(merge_ml_region_mol_id(system.mol_id, ml_indices))
    monomers = monomer_indices_from_mol_id(mol_id)
    system = MolecularSystem(
        R=system.R,
        Z=system.Z,
        box=box,
        mol_id=mol_id,
        monomer_indices=monomers,
        water_indices=system.water_indices,
        psf_path=system.psf_path,
        ff_params=system.ff_params,
        metadata={
            **dict(system.metadata),
            "ml_atom_indices": ml_indices.tolist(),
            "residue_names": list(resnames),
            "atom_names": list(atom_names),
            "engine": "hybrid_jaxmd",
        },
    )
    return system, ml_indices, atom_names, resnames


def _numpy_bias_cv(
    positions: np.ndarray,
    cv,
    target: float,
    k_ev_A2: float,
    box: np.ndarray | None,
) -> float:
    xi = float(cv.value_numpy(positions, cell=box))
    return 0.5 * float(k_ev_A2) * (xi - float(target)) ** 2


def save_failure_trace(
    output_dir: Path,
    wid: int,
    exc,
    *,
    keep_frames: int = 10,
) -> Path | None:
    """Dump the frames preceding a non-finite abort to ``windows/wXXX.trace.npz``.

    Without this a blown-up window leaves only the step number: the driver logs
    ``step N/80000`` and the checkpoint it writes is all NaN. The energy and
    kinetic series say whether the run was heating steadily or snapped in one
    frame, and the last geometries say which atoms were involved.

    Only the tail of the trajectory is kept, so a window that dies at 52000
    steps does not write hundreds of megabytes.
    """
    from mmml.umbrella.hybrid_windows import windows_dir

    positions = list(getattr(exc, "positions", []) or [])
    energies = list(getattr(exc, "energies", []) or [])
    kinetic = list(getattr(exc, "kinetic_energies", []) or [])
    if not positions and not energies:
        return None
    tail = positions[-int(keep_frames) :] if keep_frames > 0 else []
    path = windows_dir(output_dir) / f"w{int(wid):03d}.trace.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        # Full series: cheap, and the shape of the run-up is the whole point.
        energies=np.asarray(energies, dtype=np.float64),
        kinetic_energies=np.asarray(kinetic, dtype=np.float64),
        positions_tail=np.asarray(tail, dtype=np.float64),
        n_frames_recorded=np.int64(len(positions)),
        step=np.int64(getattr(exc, "step", -1)),
        n_steps=np.int64(getattr(exc, "n_steps", -1)),
        message=np.asarray(str(exc)),
    )
    return path


def seed_force_maxima(
    forces: np.ndarray,
    ml_indices: Sequence[int],
) -> tuple[float, float]:
    """``(max |F| over the ML region, max |F| over every atom)`` in eV/Å.

    The two differ by orders of importance. Only the first responds to the seed:
    solvent coordinates are shared by every window, so the whole-system maximum
    is a constant offset set by the packing, not by how far this window had to
    displace the solute.
    """
    f = np.abs(np.asarray(forces, dtype=np.float64))
    if f.size == 0:
        return float("nan"), float("nan")
    idx = np.asarray(list(ml_indices), dtype=np.int64).reshape(-1)
    fmax_all = float(f.max())
    fmax_ml = float(f[idx].max()) if idx.size else float("nan")
    return fmax_ml, fmax_all


def relax_around_frozen_seed(
    atoms,
    *,
    frozen_indices: Sequence[int],
    fmax: float,
    steps: int,
) -> tuple[np.ndarray, int]:
    """FIRE-relax everything except ``frozen_indices``; returns ``(R, n_steps)``.

    Holding the solute keeps the window at exactly the ξ it was seeded at while
    the surroundings open up around it. Constraints are cleared before returning
    so a subsequent ``get_forces()`` reports true forces on the frozen atoms
    rather than the zeros a constrained call would give.
    """
    from ase.constraints import FixAtoms
    from ase.optimize import FIRE

    idx = [int(i) for i in np.asarray(list(frozen_indices), dtype=np.int64).reshape(-1)]
    atoms.set_constraint(FixAtoms(indices=idx))
    try:
        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=float(fmax), steps=int(steps))
        n_steps = int(opt.get_number_of_steps())
    finally:
        atoms.set_constraint()
    return np.asarray(atoms.get_positions(), dtype=np.float64), n_steps


def _seed_window_geometry(
    r0: np.ndarray,
    cv,
    xi0: float,
    box: np.ndarray | None,
    move_with: Sequence[int],
) -> np.ndarray:
    """Stretch plain distance or antisymmetric-difference CV to ``xi0``."""
    if len(cv.pairs) == 1 and abs(float(cv.coefficients[0]) - 1.0) < 1e-12:
        i, j = cv.pairs[0]
        return stretch_distance_seed_mic(r0, i, j, xi0, box, move_with=move_with)
    if len(cv.pairs) == 2:
        (pair_a, pair_b), (coef_a, coef_b) = cv.pairs, cv.coefficients
        if coef_a * coef_b >= 0:
            raise ValueError(
                f"hybrid stretch seed expects opposite-sign coefficients; got {cv.label()}"
            )
        plus, minus = (pair_a, pair_b) if coef_a > 0 else (pair_b, pair_a)
        return stretch_antisymmetric_seed_mic(
            r0,
            plus,
            minus,
            xi0,
            box,
            move_with_plus=(),
            move_with_minus=move_with,
        )
    raise ValueError(
        f"hybrid stretch seed does not support CV {cv.label()}; use seed_mode=frames"
    )


def _pre_equilibrate(
    *, cfg, r0, cv, sched, box, masses, dt, build_leg, output_dir, n_atoms
):
    """Relax the packed liquid once, before any window runs.

    A Packmol box has the right density from the first step but no liquid
    structure at all, and the first solvation shell around a charged solute
    takes tens to hundreds of picoseconds to form. Windows started from it
    spend their whole trajectory in a solvent that is still relaxing, which
    under-solvates the ion pair and biases the barrier high. Turan et al. run
    500 ps NpT + 2 ns NVT first; this is the same idea at a cost we can pay.

    Heating is staged from ``heat_start_fraction * T`` because the packed box
    carries no kinetic energy and assigning full-target velocities in one step
    is a thermal shock.

    Returns the relaxed coordinates, cached per (n_atoms, seed, ps) so the cost
    is paid once per box rather than once per campaign.
    """
    import time

    from mmml.md.config import EnsembleSpec
    from mmml.md.drivers import JaxmdDriver

    ps = float(getattr(cfg, "pre_equilibrate_ps", 0.0) or 0.0)
    if ps <= 0.0:
        return r0

    cache = (
        output_dir.parent
        / f"equilibrated_{n_atoms}atoms_seed{int(cfg.seed)}_{ps:g}ps.npz"
    )
    if cache.is_file():
        cached = np.load(cache)
        print(f"  using cached equilibrated box {cache.name} ({ps:g} ps)")
        return np.asarray(cached["R"], dtype=np.float64)

    # Restrain at the schedule point nearest the base geometry so relaxing the
    # solvent does not drag the solute off the coordinate.
    r0_cv = float(cv.value_numpy(r0, cell=box))
    start = int(np.argmin(np.abs(np.asarray(sched.xi0) - r0_cv)))
    xi0, k_w = float(sched.xi0[start]), float(sched.k_x[start])
    T = float(cfg.temperature_K)
    stages = int(getattr(cfg, "heat_stages", 0) or 0)
    frac = float(getattr(cfg, "heat_start_fraction", 0.2) or 0.2)

    print(
        f"\n  pre-equilibrating the packed box for {ps:g} ps at ξ₀={xi0:.3f} "
        f"({stages} heat stages)  → {cache.name}"
    )
    pos = np.asarray(r0, dtype=np.float64)
    temps = (
        [T] if stages <= 0
        else list(np.linspace(frac * T, T, stages, dtype=float))
    )
    # Heating shares the budget; the remainder runs at the target temperature.
    n_total = int(round(ps * 1000.0 / dt))
    n_heat = int(n_total * 0.3) // max(1, len(temps)) if stages > 0 else 0
    n_hold = max(1, n_total - n_heat * len(temps))

    for i, (T_i, n_i) in enumerate(
        [*[(t, n_heat) for t in temps], (T, n_hold)]
    ):
        if n_i <= 0:
            continue
        leg_system, leg_energy, leg_nbr = build_leg(pos, xi0, k_w)
        driver = JaxmdDriver(
            record_every=n_i,
            block_size=min(n_i, 500),
            neighbor_fn=leg_nbr,
            output_path=None,
            name=f"preequil_{i}",
            progress_every=0,
        )
        t0 = time.time()
        traj = driver.run(
            leg_system,
            leg_energy,
            EnsembleSpec(
                ensemble="nvt",
                space="pbc" if box is not None else "free",
                temperature_K=float(T_i),
                dt_fs=dt,
                n_steps=int(n_i),
                thermostat="langevin",
                params={
                    "seed": int(cfg.seed),
                    "masses": masses,
                    "float64": True,
                    "langevin_gamma": float(getattr(cfg, "langevin_gamma", 0.1) or 0.1),
                    "center_velocity": False,
                },
            ),
        )
        frames = np.asarray(traj.metadata["positions"], dtype=np.float64)
        e = np.asarray(traj.metadata["energies"], dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(frames[-1])):
            raise RuntimeError(
                f"pre-equilibration went non-finite at {T_i:.0f} K. The packed "
                f"box is too strained to heat directly; lower "
                f"--timestep-fs or raise heat_stages."
            )
        pos = frames[-1]
        label = "hold" if i == len(temps) else f"heat {T_i:6.0f} K"
        print(
            f"    {label}  E {e[0]:12.3f} -> {e[-1]:12.3f} eV   "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, R=pos, ps=ps)
    print(f"  cached → {cache}")
    return pos


def run_umbrella_hybrid_nvt(cfg: UmbrellaConfig) -> UmbrellaResult:
    """Per-window hybrid (ML solute + MM solvent) NVT umbrella sampling."""
    import jax
    from ase import Atoms
    from ase.data import atomic_masses
    from ase.io import write

    from mmml.md.assemble import build_hybrid_energy
    from mmml.md.config import EnsembleSpec
    from mmml.md.drivers import JaxmdDriver
    from mmml.md.energy.registry import EnergyContext
    from mmml.md.neighbors import make_intermolecular_neighbor_fn
    from mmml.md.static_pairs import make_static_pair_fn
    from mmml.md.nl_cadence import resolve_block_steps
    from mmml.md.restraints import LinearDistanceCV
    from mmml.cli.run.md_system_unified import _load_model

    jax.config.update("jax_enable_x64", True)

    if cfg.engine != "hybrid_jaxmd":
        raise ValueError(f"run_umbrella_hybrid_nvt requires engine=hybrid_jaxmd (got {cfg.engine})")

    from mmml.umbrella.hybrid_windows import (
        bootstrap_windows_from_snapshots,
        load_all_window_arrays,
        save_window_checkpoint,
        select_windows_to_run,
        should_bootstrap_windows,
        windows_dir,
    )

    output_dir = Path(cfg.output_dir).expanduser().resolve()
    resume = bool(getattr(cfg, "resume", False))
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not cfg.overwrite
        and not resume
    ):
        raise FileExistsError(
            f"output_dir is not empty: {output_dir} "
            "(pass --overwrite to wipe, or --resume to fill missing windows)"
        )
    if cfg.overwrite and output_dir.exists() and not resume:
        # Fresh campaign: drop per-window checkpoints so stale wXXX.npz cannot
        # be mistaken for finished work.
        import shutil

        wdir = windows_dir(output_dir)
        if wdir.is_dir():
            shutil.rmtree(wdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_system, ml_indices, atom_names, resnames = build_hybrid_umbrella_system(cfg)
    cfg = bind_hybrid_atom_names(cfg, atom_names, resnames)
    sched = cfg.resolve_schedule()
    if sched.ndim != 1:
        raise ValueError("hybrid_jaxmd supports 1D only")
    cv = LinearDistanceCV.from_spec(sched.cvs[0])
    atom_i, atom_j = int(cv.pairs[0][0]), int(cv.pairs[0][1])
    k_windows = sched.n_windows
    box = None if base_system.box is None else np.asarray(base_system.box, dtype=np.float64)
    wall_specs = [w.to_spec() for w in sched.walls]

    model, params = _load_model(Path(cfg.checkpoint).expanduser().resolve())
    ctx = EnergyContext(model=model, params=params, options={})

    masses = np.array(
        [atomic_masses[int(zi)] for zi in base_system.Z], dtype=np.float64
    )
    r0 = np.asarray(base_system.R, dtype=np.float64)
    r0_cv = float(cv.value_numpy(r0, cell=box))

    savefreq = cfg.effective_savefreq()
    dt = float(cfg.timestep_fs)
    n_atoms = base_system.n_atoms
    nsteps = int(cfg.nsteps)
    equil = int(cfg.equilibration_steps)
    printfreq = int(cfg.printfreq) if int(cfg.printfreq) > 0 else savefreq
    ps_per_window = nsteps * dt / 1000.0
    # Driver records frame 0 plus every savefreq steps → ~nsteps/savefreq + 1.
    n_frames_est = 1 + (nsteps // savefreq if savefreq > 0 else 0)

    print(
        f"=== Hybrid umbrella NVT ({k_windows} windows, 1D, "
        f"T={cfg.temperature_K} K, dt={dt} fs, engine=hybrid_jaxmd) ==="
    )
    print(
        f"  ML region: {len(ml_indices)} atoms resnames={list(cfg.ml_resnames)}  "
        f"CV={cv.label()}  r0={r0_cv:.4f} Å"
    )
    print(f"  n_atoms={n_atoms}  box_diag={None if box is None else np.diag(box).tolist()}")
    print(
        f"  nsteps={nsteps}  ({ps_per_window:g} ps/window)  "
        f"equil={equil}  savefreq={savefreq}  printfreq={printfreq}  "
        f"~{n_frames_est} frames/window"
    )
    print(
        f"  max_seed_force={float(cfg.max_seed_force):g} eV/Å over the ML region  "
        f"(failed windows are skipped, not fatal)"
    )
    relax_steps = int(getattr(cfg, "relax_seed_steps", 0))
    print(
        f"  seed relaxation: {relax_steps} FIRE steps around the frozen solute"
        + (
            f" to fmax={float(getattr(cfg, 'relax_seed_fmax', 1.0)):g} eV/Å"
            if relax_steps > 0
            else " (disabled)"
        )
    )
    print(f"  output_dir={output_dir}")
    if wall_specs:
        print(f"  walls={len(wall_specs)}  {[w.label() for w in sched.walls]}")

    only = tuple(getattr(cfg, "only_windows", ()) or ())
    if should_bootstrap_windows(resume=resume, only_windows=only):
        boot = bootstrap_windows_from_snapshots(output_dir, n_windows=k_windows)
        if boot:
            print(
                f"  resume: imported {len(boot)} window(s) from existing "
                f"{SNAPSHOTS_NPZ} → {windows_dir(output_dir).name}/",
                flush=True,
            )

    to_run, already_ok = select_windows_to_run(
        k_windows,
        output_dir,
        resume=resume,
        resume_failed=bool(getattr(cfg, "resume_failed", True)),
        only_windows=only if only else None,
    )
    if resume or only:
        print(
            f"  resume={resume}: run {len(to_run)} window(s) {to_run}; "
            f"keep {len(already_ok)} finished",
            flush=True,
        )
    if not to_run and resume:
        print("  resume: nothing to run — reassembling snapshots from windows/", flush=True)

    def _nan_window() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos = np.full((n_frames_est, n_atoms, 3), np.nan, dtype=np.float64)
        cv_nan = np.full(n_frames_est, np.nan, dtype=np.float64)
        e_nan = np.full(n_frames_est, np.nan, dtype=np.float64)
        return pos, cv_nan, e_nan, e_nan.copy()

    # Structure each window is stretched from. With chaining this advances to
    # the previous window's final frame, so the relaxed solvation shell is
    # carried along the ladder instead of every window re-forming it from the
    # same packed configuration.
    seed_source = r0
    chain = bool(getattr(cfg, "seed_from_previous_window", False))

    def _build_leg(positions, xi0, k_w, verbose_pairs=False):
        """System + biased energy + pair list for one leg at centre ``xi0``.

        Shared by the pre-equilibration leg and every window so the two cannot
        drift apart in force field, solver or pair-list treatment.
        """
        leg_system = MolecularSystem(
            R=positions,
    for wid in to_run:
        xi0 = float(sched.xi0[wid])
        k_w = float(sched.k_x[wid])
        seeded = _seed_window_geometry(r0, cv, xi0, box, cfg.move_with)
        win_system = MolecularSystem(
            R=seeded,
            Z=base_system.Z,
            box=base_system.box,
            mol_id=base_system.mol_id,
            monomer_indices=base_system.monomer_indices,
            water_indices=base_system.water_indices,
            psf_path=base_system.psf_path,
            ff_params=base_system.ff_params,
            metadata=base_system.metadata,
        )
        extra_prm: list = []
        ch3cl_prm = Path(__file__).resolve().parents[2] / "examples" / "m" / "par_ch3cl.prm"
        if ch3cl_prm.is_file():
            extra_prm.append(ch3cl_prm)
        leg_energy = build_hybrid_energy(
            leg_system,
            ("ml_intra", "mm_bonded", "mm_nonbonded", "rxncoor"),
            ctx,
            term_kwargs={
                "ml_intra": {"monomer_indices": [ml_indices]},
                "mm_bonded": {
                    "ml_atom_indices": ml_indices,
                    "extra_prm_files": extra_prm,
                },
                "mm_nonbonded": {"lr_solver": str(cfg.lr_solver)},
                "rxncoor": {
                    "cv": cv,
                    "k_ev_per_A2": k_w,
                    "target": xi0,
                    "walls": list(sched.walls),
                },
            },
        )
        if cfg.static_pairs:
            # Complete intermolecular list, uploaded once. The switching
            # functions cull by distance on the GPU, so no host rebuild is
            # needed and none of the per-block transfer cost is paid. Measured
            # 8.3 -> 24 steps/s on this 2625-atom box.
            leg_nbr = make_static_pair_fn(leg_system, verbose=verbose_pairs)
        else:
            leg_nbr = make_intermolecular_neighbor_fn(leg_system, cutoff_A=12.0)
        return leg_system, leg_energy, leg_nbr

    seed_source = _pre_equilibrate(
        cfg=cfg,
        r0=r0,
        cv=cv,
        sched=sched,
        box=box,
        masses=masses,
        dt=dt,
        build_leg=_build_leg,
        output_dir=output_dir,
        n_atoms=n_atoms,
    )

    for wid in range(k_windows):
        xi0 = float(sched.xi0[wid])
        k_w = float(sched.k_x[wid])
        seeded = _seed_window_geometry(seed_source, cv, xi0, box, cfg.move_with)
        win_system, energy, neighbor_fn = _build_leg(
            seeded, xi0, k_w, verbose_pairs=(wid == 0)
        neighbor_fn = make_intermolecular_neighbor_fn(
            win_system,
            cutoff_A=12.0,
            skin_A=float(getattr(cfg, "nl_skin_A", 0.0) or 0.0),
        )
        # Relax the surroundings around the frozen seed, then preflight, both via
        # the ASE face when available; otherwise skip.
        fmax = float("nan")
        fmax_all = float("nan")
        n_relax = 0
        seed_error: str | None = None
        try:
            atoms = Atoms(
                numbers=win_system.Z,
                positions=seeded,
                cell=box,
                pbc=box is not None,
            )
            atoms.calc = energy.as_ase_calculator()
            if int(getattr(cfg, "relax_seed_steps", 0)) > 0:
                seeded, n_relax = relax_around_frozen_seed(
                    atoms,
                    frozen_indices=ml_indices,
                    fmax=float(getattr(cfg, "relax_seed_fmax", 1.0)),
                    steps=int(cfg.relax_seed_steps),
                )
                win_system = replace(win_system, R=seeded)
            fmax, fmax_all = seed_force_maxima(atoms.get_forces(), ml_indices)
        except (ValueError, NotImplementedError):
            # No usable ASE face here (some lr_solver choices have none): the
            # relaxation and the gate are both skipped, as before.
            fmax = float("nan")
            fmax_all = float("nan")
        except Exception as exc:  # noqa: BLE001
            # A blown-up minimisation must cost one window, not the campaign:
            # the serial path runs all 30 in a single process.
            seed_error = f"seed relaxation failed: {type(exc).__name__}: {exc}"

        # Default Langevin: hybrid NHC was hard-coded and blew up solvent
        # windows (high-T / stiff ξ), then Snakemake still scheduled MBAR.
        thermo = str(getattr(cfg, "thermostat", None) or "langevin").strip().lower()
        if thermo not in {"langevin", "lgv", "nhc", "nose_hoover", "nose-hoover"}:
            thermo = "langevin"
        if thermo in {"nose_hoover", "nose-hoover"}:
            thermo = "nhc"
        ensemble = EnsembleSpec(
            ensemble="nvt",
            space="pbc" if box is not None else "free",
            temperature_K=float(cfg.temperature_K),
            dt_fs=dt,
            n_steps=int(cfg.nsteps),
            thermostat=thermo,
            params={
                "seed": int(cfg.seed) + wid,
                "masses": masses,
                "float64": True,
                "langevin_gamma": float(getattr(cfg, "langevin_gamma", 0.1) or 0.1),
                "center_velocity": False,
            },
        )
        driver = JaxmdDriver(
            record_every=int(savefreq),
            # Block size is the MM pair refresh cadence, not a recording
            # concern; resolve_block_steps keeps it a divisor of savefreq so a
            # frame still lands on a refresh boundary.
            block_size=resolve_block_steps(
                steps_per_recording=int(savefreq),
                use_pbc=box is not None,
                has_update_fn=True,
                update_interval=getattr(cfg, "nl_update_interval", None) or int(savefreq),
                ensemble="nvt",
            ),
            neighbor_fn=neighbor_fn,
            output_path=None,
            name=f"umbrella_hybrid_w{wid:03d}",
            progress_every=int(printfreq),
            abort_nonfinite=True,
        )
        print(
            f"  window {wid + 1}/{k_windows}  ξ₀={xi0:.3f}  k={k_w:.3f}  "
            f"nsteps={nsteps}  relax_steps={n_relax}  "
            f"seed_max|F|_ML={fmax:.2f}  seed_max|F|_all={fmax_all:.2f}",
            flush=True,
        )

        fail_msg: str | None = seed_error
        if fail_msg is None and fmax == fmax and fmax > float(cfg.max_seed_force):
            fail_msg = (
                f"seed max|F| over the ML region={fmax:.2f} eV/Å exceeds "
                f"max_seed_force={cfg.max_seed_force:g} (ξ₀={xi0:.3f}; "
                f"whole-system max {fmax_all:.2f})"
            )
        if fail_msg is None:
            try:
                traj = driver.run(win_system, energy, ensemble)
            except RuntimeError as exc:
                fail_msg = str(exc)
                traj = None
                trace_path = save_failure_trace(output_dir, wid, exc)
                if trace_path is not None:
                    print(f"    trace → {trace_path.name}", flush=True)
            if fail_msg is None:
                frames = traj.metadata.get("positions") if traj is not None else None
                energies = traj.metadata.get("energies") if traj is not None else None
                if frames is None:
                    fail_msg = "driver returned no positions in metadata"
                else:
                    pos_w = np.asarray(frames, dtype=np.float64)
                    if pos_w.ndim == 2:
                        pos_w = pos_w[None, ...]
                    e_tot = np.asarray(energies, dtype=np.float64).reshape(-1)
                    if e_tot.shape[0] != pos_w.shape[0]:
                        e_tot = np.resize(e_tot, pos_w.shape[0])
                    if not (
                        np.all(np.isfinite(pos_w)) and np.all(np.isfinite(e_tot))
                    ):
                        fail_msg = "non-finite positions/energies in recorded frames"
                    else:
                        cv_w = np.array(
                            [
                                float(cv.value_numpy(pos_w[t], cell=box))
                                for t in range(pos_w.shape[0])
                            ],
                            dtype=np.float64,
                        )
                        w_w = np.array(
                            [
                                _numpy_bias_cv(pos_w[t], cv, xi0, k_w, box)
                                for t in range(pos_w.shape[0])
                            ],
                            dtype=np.float64,
                        )
                        e_unb = e_tot - w_w
                        if not np.all(np.isfinite(cv_w)):
                            fail_msg = "non-finite CV values in recorded frames"
                        else:
                            save_window_checkpoint(
                                output_dir,
                                wid,
                                status="ok",
                                positions=pos_w,
                                cv=cv_w,
                                energies=e_tot,
                                energies_unbiased=e_unb,
                                xi0=xi0,
                                k_ev_A2=k_w,
                            )
                            print(
                                f"    done: {pos_w.shape[0]} frames  "
                                f"⟨ξ⟩={float(cv_w.mean()):.3f}  "
                                f"⟨E_unb⟩={float(e_unb.mean()):.4f} eV  "
                                f"→ windows/w{wid:03d}.npz",
                                flush=True,
                            )

        if fail_msg is not None:
            pos_w, cv_w, e_tot, e_unb = _nan_window()
            save_window_checkpoint(
                output_dir,
                wid,
                status="failed",
                positions=pos_w,
                cv=cv_w,
                energies=e_tot,
                energies_unbiased=e_unb,
                xi0=xi0,
                k_ev_A2=k_w,
                fail_reason=fail_msg,
            )
            print(
                f"    FAILED window {wid + 1}/{k_windows}: {fail_msg}  "
                f"→ windows/w{wid:03d}.npz",
                flush=True,
            )

    if only:
        # A per-window Slurm job must not touch the shared aggregate. Eight run
        # concurrently, each would assemble whatever happened to be on disk at
        # the time and write that partial view to the same two paths, and the
        # last writer would win. The assemble step rebuilds both once every
        # window exists.
        print(
            f"  --windows set: wrote windows/ only; {SNAPSHOTS_NPZ} and "
            f"{SUMMARY_JSON} are left to the assemble step",
            flush=True,
        )
        return UmbrellaResult(
            output_dir=output_dir,
            snapshots_path=output_dir / SNAPSHOTS_NPZ,
            summary_path=output_dir / SUMMARY_JSON,
            n_windows=k_windows,
            n_frames=int(n_frames_est),
            paths={"windows": windows_dir(output_dir)},
        )
        if chain:
            # Hand the relaxed solvent to the next window. Refuse to propagate
            # a diverged frame: a single NaN would otherwise seed every
            # remaining window and the run would report finite-looking garbage.
            last = pos_w[-1]
            if np.all(np.isfinite(last)):
                seed_source = np.asarray(last, dtype=np.float64)
            else:
                print(
                    f"    window {wid} final frame is non-finite; keeping the "
                    f"previous seed rather than propagating it"
                )

    # Assemble from per-window checkpoints (resume-safe source of truth).
    positions, cv_2d, energies, energies_unbiased, failed_windows, fail_reasons = (
        load_all_window_arrays(
            output_dir,
            k_windows,
            n_frames=n_frames_est,
            n_atoms=n_atoms,
        )
    )
    n_frames = int(positions.shape[1])
    cv_traj = cv_2d[..., None]

    minima_pos, minima_idx, minima_e = select_lowest_energy_frames(
        positions, energies
    )
    z = np.asarray(base_system.Z, dtype=np.int32)
    snapshots_path = output_dir / SNAPSHOTS_NPZ
    extra = {
        "ndim": np.int32(1),
        "grid_shape": np.asarray(sched.grid_shape, dtype=np.int32),
        "energies_ev": np.asarray(energies, dtype=np.float64),
        "energies_unbiased_ev": np.asarray(energies_unbiased, dtype=np.float64),
        "bin_minima_frame_idx": np.asarray(minima_idx, dtype=np.int64),
        "bin_minima_energy_ev": np.asarray(minima_e, dtype=np.float64),
        "engine": np.asarray("hybrid_jaxmd"),
        "ml_atom_indices": np.asarray(ml_indices, dtype=np.int32),
        # Authoritative CV (difference / combination); atom_i/j are legacy only.
        "cv_spec": np.asarray(json.dumps(sched.cv_specs())),
        "wall_spec": np.asarray(json.dumps(wall_specs)),
        "failed_windows": np.asarray(failed_windows, dtype=np.int32),
        "fail_reasons": np.asarray(json.dumps({str(k): v for k, v in fail_reasons.items()})),
    }
    if box is not None:
        extra["box"] = np.asarray(box, dtype=np.float64)

    save_snapshots(
        snapshots_path,
        positions=positions,
        Z=z,
        atom_i=atom_i,
        atom_j=atom_j,
        xi0=np.asarray(sched.xi0, dtype=np.float64),
        k_ev_A2=np.asarray(sched.k_x, dtype=np.float64),
        temperature_K=float(cfg.temperature_K),
        dt_fs=dt,
        cv_traj=cv_traj,
        checkpoint=str(Path(cfg.checkpoint).expanduser().resolve()),
        extra=extra,
    )

    minima_centered = center_com_positions(minima_pos, masses)
    minima_path = output_dir / BIN_MINIMA_TRAJ
    minima_frames = [
        Atoms(
            numbers=z,
            positions=minima_centered[wid],
            masses=masses,
            cell=box,
            pbc=box is not None,
            info={
                "window": wid,
                "frame_idx": int(minima_idx[wid]),
                "energy_ev": float(minima_e[wid]),
            },
        )
        for wid in range(k_windows)
    ]
    write(minima_path, minima_frames)

    summary = {
        "args": cfg.to_dict(),
        "engine": "hybrid_jaxmd",
        "ndim": 1,
        "n_windows": k_windows,
        "n_frames": int(n_frames),
        "n_atoms": int(n_atoms),
        "atom_pairs": [list(p) for p in cv.pairs],
        "cv_spec": sched.cv_specs(),
        "cv_label": [cv.label()],
        "wall_spec": wall_specs,
        "wall_label": [w.label() for w in sched.walls],
        "xi0": list(sched.xi0),
        "yi0": None,
        "k_ev_A2": list(sched.k_x),
        "k_y_ev_A2": None,
        "grid_shape": list(sched.grid_shape),
        "r0_cv_A": [r0_cv],
        "seed_mode": cfg.seed_mode,
        "ml_atom_indices": ml_indices.tolist(),
        "ml_resnames": list(cfg.ml_resnames),
        "replica_exchange": False,
        "cv_mean": np.nanmean(cv_traj, axis=1).reshape(k_windows, -1).tolist(),
        "cv_std": np.nanstd(cv_traj, axis=1).reshape(k_windows, -1).tolist(),
        "snapshots": str(snapshots_path),
        "bin_minima": str(minima_path),
        "bin_minima_frame_idx": minima_idx.tolist(),
        "bin_minima_energy_ev": minima_e.tolist(),
        "has_energies_unbiased_ev": True,
        "failed_windows": failed_windows,
        "fail_reasons": {str(k): v for k, v in fail_reasons.items()},
        "n_failed_windows": len(failed_windows),
    }
    if failed_windows:
        print(
            f"WARNING: {len(failed_windows)}/{k_windows} windows failed and will be "
            f"dropped by MBAR: {failed_windows}",
            flush=True,
        )
    summary_path = write_summary(output_dir / SUMMARY_JSON, summary)
    return UmbrellaResult(
        output_dir=output_dir,
        snapshots_path=snapshots_path,
        summary_path=summary_path,
        n_windows=k_windows,
        n_frames=int(n_frames),
        paths={"snapshots": snapshots_path, "summary": summary_path, "bin_minima": minima_path},
    )
