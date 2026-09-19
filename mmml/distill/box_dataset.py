"""Many-seed periodic PET dataset: random packing → FIRE (kept) → NVT.

Each seed builds its own box (random lattice sites, COM jitter, random
rotations), FIRE-minimizes it while saving intermediate frames, then runs
Langevin NVT from seed-specific velocities and saves frames. Every frame is
labelled with the driving model's cached E/F (eV, eV/Å) and tagged in
``info``: ``seed``, ``phase`` (``fire`` / ``md``), ``step``, ``T_target_K``.

FIRE intermediates cover strained, compressed and clashing-but-finite
geometries that thermal MD rarely visits; MD covers the thermal ensemble.
Frames whose max |F| exceeds ``max_force_eVA`` are dropped (the model is not
trustworthy there and they would dominate a force loss).

Output per seed: ``<out>/seed_<k>/traj.extxyz`` + ``summary.json``.
Feed the extxyz files to metatrain, or to
``mmml pet-physnet-distill --from-box-extxyz`` for ML/MM cluster labels.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

PHASE_FIRE = "fire"
PHASE_MD = "md"


@dataclass(frozen=True)
class BoxDatasetConfig:
    monomer_xyz: Path
    n_molecules: int
    box_side_A: float
    temperature_K: float = 300.0
    dt_fs: float = 0.5
    friction_per_fs: float = 0.01
    fire_steps: int = 200
    fire_every: int = 5
    fire_fmax: float = 0.05
    md_steps: int = 2000
    md_every: int = 20
    com_jitter_frac: float = 0.25
    max_force_eVA: float = 15.0


def random_packed_box(
    monomer: Atoms, n_molecules: int, box_side_A: float, *, seed: int, jitter_frac: float
) -> Atoms:
    """Random subset of simple-cubic sites, COM jitter (fraction of spacing), random rotations."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(int(seed))
    n_mol = int(n_molecules)
    n_side = int(np.ceil(n_mol ** (1.0 / 3.0)))
    if n_side**3 == n_mol:
        n_side += 1  # keep vacancies so site choice varies between seeds
    step = float(box_side_A) / n_side
    grid = (np.stack(np.meshgrid(*[np.arange(n_side)] * 3, indexing="ij"), -1).reshape(-1, 3) + 0.5) * step
    sites = grid[rng.choice(len(grid), size=n_mol, replace=False)]
    sites += rng.uniform(-0.5, 0.5, size=sites.shape) * float(jitter_frac) * step
    ref = np.asarray(monomer.get_positions(), dtype=float)
    ref = ref - ref.mean(axis=0)
    rots = Rotation.random(n_mol, random_state=rng.integers(2**31)).as_matrix()
    pos = np.einsum("mij,aj->mai", rots, ref) + sites[:, None, :]
    atoms = Atoms(
        numbers=np.tile(monomer.get_atomic_numbers(), n_mol),
        positions=pos.reshape(-1, 3),
        cell=[float(box_side_A)] * 3,
        pbc=True,
    )
    atoms.wrap()
    return atoms


def run_seed(calc, cfg: BoxDatasetConfig, *, seed: int, out_dir: Path) -> dict:
    """Generate one seed's frames. ``calc`` is a loaded ASE calculator (reused across seeds)."""
    from ase import units
    from ase.constraints import FixCom
    from ase.io import read as ase_read
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import Stationary, thermalize_momenta
    from ase.optimize import FIRE

    from mmml.md.metatomic_pbc import append_training_frame

    out = Path(out_dir) / f"seed_{int(seed):04d}"
    out.mkdir(parents=True, exist_ok=True)
    traj = out / "traj.extxyz"
    traj.unlink(missing_ok=True)
    counts = {"fire_kept": 0, "fire_dropped": 0, "md_kept": 0, "md_dropped": 0}

    atoms = random_packed_box(
        ase_read(str(cfg.monomer_xyz)),
        cfg.n_molecules,
        cfg.box_side_A,
        seed=seed,
        jitter_frac=cfg.com_jitter_frac,
    )
    atoms.calc = calc

    def _save(phase: str, step: int) -> None:
        fmax = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
        ok = np.isfinite(fmax) and fmax <= cfg.max_force_eVA
        counts[f"{phase}_{'kept' if ok else 'dropped'}"] += 1
        if not ok:
            return
        append_training_frame(
            traj,
            atoms,
            step=step,
            dt_fs=cfg.dt_fs if phase == PHASE_MD else 0.0,
            info={"seed": int(seed), "phase": phase, "T_target_K": float(cfg.temperature_K)},
        )

    t0 = time.perf_counter()
    e_start = float(atoms.get_potential_energy())
    opt = FIRE(atoms, logfile=None, maxstep=0.1)
    opt.attach(lambda: _save(PHASE_FIRE, int(opt.nsteps)), interval=max(int(cfg.fire_every), 1))
    opt.run(fmax=float(cfg.fire_fmax), steps=int(cfg.fire_steps))
    e_min = float(atoms.get_potential_energy())
    t_fire = time.perf_counter() - t0

    rng = np.random.default_rng(int(seed) + 10_000)
    thermalize_momenta(atoms, temperature_K=float(cfg.temperature_K), rng=rng)
    Stationary(atoms)
    atoms.set_constraint(FixCom())
    dyn = Langevin(
        atoms,
        timestep=float(cfg.dt_fs) * units.fs,
        temperature_K=float(cfg.temperature_K),
        friction=float(cfg.friction_per_fs),
        fixcm=False,
        rng=rng,
    )
    temps: list[float] = []
    dyn.attach(lambda: temps.append(float(atoms.get_temperature())), interval=max(int(cfg.md_every), 1))
    dyn.attach(
        lambda: _save(PHASE_MD, int(dyn.get_number_of_steps())),
        interval=max(int(cfg.md_every), 1),
    )
    t1 = time.perf_counter()
    dyn.run(int(cfg.md_steps))
    summary = {
        "seed": int(seed),
        "traj_extxyz": str(traj),
        "n_atoms": len(atoms),
        **counts,
        "E_start_eV": e_start,
        "E_fire_end_eV": e_min,
        "fire_steps_run": int(opt.nsteps),
        "fire_s": t_fire,
        "md_s": time.perf_counter() - t1,
        "T_md_mean_K": float(np.mean(temps[len(temps) // 2 :])) if temps else float("nan"),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
