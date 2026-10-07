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

Collapse checks. A random packing can FIRE-relax into a fused, reacted box
(seen with PET-MAD xs for ACO/DCM: hundreds of eV below the other seeds, then
NVT at 10^3-10^8 K). Two guards, neither relying on the force filter:

* damaged molecules: a molecule is damaged when a covalent bond (graph from the
  monomer xyz) is more than ``max_bond_stretch_A`` off its reference length or
  when it is covalently bonded to another molecule. A seed whose FIRE end state
  has more than ``max_damaged_fraction`` damaged molecules, or two molecules'
  atoms closer than ``min_intermolecular_A``, is rejected and its NVT skipped.
  Frames with such overlaps are dropped; a few reacted molecules are left to
  ``box_cluster_pool``, which removes them per cluster (PET-MAD xs acetone
  boxes carry 3-7% reacted molecules: proton transfers and new C-O bonds between
  molecules). Each frame records ``n_damaged`` in ``info``;
* energy outliers: :func:`flag_energy_outliers` rejects seeds whose FIRE-end
  energy per molecule lies more than ``energy_outlier_eV_per_mol`` below the
  median of the other accepted seeds.

A rejected seed keeps its frames in ``traj.rejected.extxyz`` (not matched by
``seed_*/traj.extxyz``) and records ``rejected`` in ``summary.json``.

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
    max_bond_stretch_A: float | None = 0.4
    max_damaged_fraction: float = 0.10
    min_intermolecular_A: float = 0.5
    energy_outlier_eV_per_mol: float | None = 0.25


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


TRAJ_NAME = "traj.extxyz"
REJECTED_TRAJ_NAME = "traj.rejected.extxyz"


def molecule_damage_check(monomer: Atoms, tol_A: float | None):
    """Return ``check(atoms) -> (n_damaged, closest intermolecular Å)`` for boxes of
    ``monomer`` copies, or None when disabled."""
    if tol_A is None:
        return None
    from mmml.distill.box_clusters import bond_graph, damaged_molecules

    ref = np.asarray(monomer.get_positions(), dtype=np.float64)
    bonds = bond_graph(monomer.get_atomic_numbers(), ref)
    ref_len = np.linalg.norm(ref[bonds[:, 0]] - ref[bonds[:, 1]], axis=-1)
    apm = len(monomer)

    def check(atoms: Atoms) -> tuple[int, float]:
        damaged, d_min = damaged_molecules(atoms, apm, bonds, ref_len, float(tol_A))
        return int(damaged.sum()), d_min

    return check


def reject_seed(seed_dir: Path, summary: dict, reason: str) -> dict:
    """Move the seed's frames out of the ``seed_*/traj.extxyz`` glob and record why."""
    seed_dir = Path(seed_dir)
    traj = seed_dir / TRAJ_NAME
    if traj.exists():
        traj.replace(seed_dir / REJECTED_TRAJ_NAME)
    summary = {**summary, "rejected": reason, "traj_extxyz": str(seed_dir / REJECTED_TRAJ_NAME)}
    (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def flag_energy_outliers(out_dir: Path, threshold_eV_per_mol: float | None) -> list[dict]:
    """Reject seeds in ``out_dir`` whose FIRE-end E/molecule is far below the others' median.

    Each seed is compared with the median of the *other* accepted seeds, so one
    collapse cannot drag the reference. Needs at least three accepted seeds;
    summaries from earlier runs in ``out_dir`` count too. Returns the newly
    rejected summaries.
    """
    if threshold_eV_per_mol is None:
        return []
    seeds = []
    for f in sorted(Path(out_dir).glob("seed_*/summary.json")):
        s = json.loads(f.read_text())
        if not s.get("rejected") and s.get("n_molecules"):
            seeds.append((f.parent, s))
    if len(seeds) < 3:
        return []
    e = np.array([s["E_fire_end_eV"] / s["n_molecules"] for _, s in seeds])
    rejected = []
    for k, (seed_dir, s) in enumerate(seeds):
        ref = float(np.median(np.delete(e, k)))
        if e[k] < ref - float(threshold_eV_per_mol):
            reason = (f"FIRE-end energy {e[k]:.3f} eV/molecule is {ref - e[k]:.3f} below the other seeds' "
                      f"median {ref:.3f} (threshold {float(threshold_eV_per_mol):g})")
            rejected.append(reject_seed(seed_dir, s, reason))
    return rejected


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
    traj = out / TRAJ_NAME
    traj.unlink(missing_ok=True)
    (out / REJECTED_TRAJ_NAME).unlink(missing_ok=True)
    counts = {"fire_kept": 0, "fire_dropped": 0, "md_kept": 0, "md_dropped": 0, "fire_overlap": 0, "md_overlap": 0}

    monomer = ase_read(str(cfg.monomer_xyz))
    damage = molecule_damage_check(monomer, cfg.max_bond_stretch_A)
    atoms = random_packed_box(
        monomer,
        cfg.n_molecules,
        cfg.box_side_A,
        seed=seed,
        jitter_frac=cfg.com_jitter_frac,
    )
    atoms.calc = calc

    def _save(phase: str, step: int) -> None:
        fmax = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
        ok = np.isfinite(fmax) and fmax <= cfg.max_force_eVA
        n_damaged = -1
        if ok and damage is not None:
            n_damaged, d_min = damage(atoms)
            if d_min < cfg.min_intermolecular_A:
                counts[f"{phase}_overlap"] += 1
                ok = False
        counts[f"{phase}_{'kept' if ok else 'dropped'}"] += 1
        if not ok:
            return
        append_training_frame(
            traj,
            atoms,
            step=step,
            dt_fs=cfg.dt_fs if phase == PHASE_MD else 0.0,
            info={"seed": int(seed), "phase": phase, "T_target_K": float(cfg.temperature_K), "n_damaged": n_damaged},
        )

    t0 = time.perf_counter()
    e_start = float(atoms.get_potential_energy())
    opt = FIRE(atoms, logfile=None, maxstep=0.1)
    opt.attach(lambda: _save(PHASE_FIRE, int(opt.nsteps)), interval=max(int(cfg.fire_every), 1))
    opt.run(fmax=float(cfg.fire_fmax), steps=int(cfg.fire_steps))
    e_min = float(atoms.get_potential_energy())
    t_fire = time.perf_counter() - t0
    n_damaged_fire_end, d_min_fire_end = (0, float("inf")) if damage is None else damage(atoms)
    collapse = []
    if n_damaged_fire_end > cfg.max_damaged_fraction * cfg.n_molecules:
        collapse.append(
            f"{n_damaged_fire_end}/{cfg.n_molecules} molecules damaged after FIRE "
            f"(limit {cfg.max_damaged_fraction:.0%})"
        )
    if d_min_fire_end < cfg.min_intermolecular_A:
        collapse.append(f"intermolecular atoms {d_min_fire_end:.2f} Å apart after FIRE")

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
    md_steps = 0 if collapse else int(cfg.md_steps)  # collapsed box: skip NVT
    dyn.attach(lambda: temps.append(float(atoms.get_temperature())), interval=max(int(cfg.md_every), 1))
    dyn.attach(
        lambda: _save(PHASE_MD, int(dyn.get_number_of_steps())),
        interval=max(int(cfg.md_every), 1),
    )
    t1 = time.perf_counter()
    dyn.run(md_steps)
    summary = {
        "seed": int(seed),
        "traj_extxyz": str(traj),
        "n_atoms": len(atoms),
        "n_molecules": int(cfg.n_molecules),
        "rejected": None,
        "n_damaged_fire_end": int(n_damaged_fire_end),
        "min_intermolecular_A_fire_end": d_min_fire_end,
        **counts,
        "E_start_eV": e_start,
        "E_fire_end_eV": e_min,
        "fire_steps_run": int(opt.nsteps),
        "fire_s": t_fire,
        "md_s": time.perf_counter() - t1,
        "T_md_mean_K": float(np.mean(temps[len(temps) // 2 :])) if temps and md_steps else float("nan"),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if collapse:
        summary = reject_seed(out, summary, "; ".join(collapse) + " (NVT skipped)")
    return summary
