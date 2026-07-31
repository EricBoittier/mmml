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
    "resolve_ml_region_indices",
    "run_umbrella_hybrid_nvt",
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
        f"  max_seed_force={float(cfg.max_seed_force):g} eV/Å  "
        f"(failed windows are skipped, not fatal)"
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
        energy = build_hybrid_energy(
            win_system,
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
        neighbor_fn = make_intermolecular_neighbor_fn(
            win_system,
            cutoff_A=12.0,
        )
        # Preflight seed force via ASE face when available; otherwise skip.
        fmax = float("nan")
        try:
            calc = energy.as_ase_calculator()
            atoms = Atoms(
                numbers=win_system.Z,
                positions=seeded,
                cell=box,
                pbc=box is not None,
            )
            atoms.calc = calc
            f0 = atoms.get_forces()
            fmax = float(np.max(np.abs(f0)))
        except (ValueError, NotImplementedError):
            fmax = float("nan")

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
            block_size=int(savefreq),
            neighbor_fn=neighbor_fn,
            output_path=None,
            name=f"umbrella_hybrid_w{wid:03d}",
            progress_every=int(printfreq),
            abort_nonfinite=True,
        )
        print(
            f"  window {wid + 1}/{k_windows}  ξ₀={xi0:.3f}  k={k_w:.3f}  "
            f"nsteps={nsteps}  "
            f"seed_max|F|={fmax if fmax == fmax else float('nan'):.2f}",
            flush=True,
        )

        fail_msg: str | None = None
        if fmax == fmax and fmax > float(cfg.max_seed_force):
            fail_msg = (
                f"seed max|F|={fmax:.2f} eV/Å exceeds max_seed_force="
                f"{cfg.max_seed_force:g} (ξ₀={xi0:.3f})"
            )
        else:
            try:
                traj = driver.run(win_system, energy, ensemble)
            except RuntimeError as exc:
                fail_msg = str(exc)
                traj = None
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
