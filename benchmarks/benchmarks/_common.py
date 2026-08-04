"""Fixtures and helpers shared by the mmml asv benchmarks.

Three rules the benchmark modules follow, all enforced or supported from here:

1. **No heavy imports at module scope.** asv imports every benchmark module just
   to discover benchmarks; a module-level ``import jax`` would pay the JAX import
   on discovery and turn any import error into a whole-file failure. Import
   inside ``setup()`` and let :func:`skip` turn a missing dependency into an asv
   *skip* rather than an error.
2. **Always block on JAX.** Dispatch is asynchronous, so a timing that does not
   call :func:`block` measures Python overhead, not the kernel.
3. **Precision is a process-global.** ``jax_enable_x64`` cannot be flipped per
   benchmark without leaking into whatever runs next in the same worker, so it is
   fixed here from the environment (``MMML_BENCH_X64``, default on to match the
   production MD path in ``examples/md_cpu/_env.sh``). Numbers are only
   comparable between runs that used the same setting — :func:`precision_tag`
   reports it and ``bench_meta`` records it alongside every result set.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

__all__ = [
    "X64",
    "REPO_ROOT",
    "block",
    "require_jax",
    "require_jax_md",
    "skip",
    "precision_tag",
    "default_checkpoint",
    "water_box",
    "aco_cluster",
    "synthetic_ff_params",
    "synthetic_system",
    "padded_pair_list",
]


def _truthy(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


#: float64 unless ``MMML_BENCH_X64`` says otherwise. Set *before* JAX is imported
#: anywhere, so JAX picks it up from its own environment variable.
X64 = _truthy(os.environ.get("MMML_BENCH_X64"), default=True)
os.environ.setdefault("JAX_ENABLE_X64", "1" if X64 else "0")

REPO_ROOT = Path(__file__).resolve().parents[2]


class BenchSkipped(NotImplementedError):
    """asv treats ``NotImplementedError`` from ``setup()`` as 'skip this benchmark'."""


def skip(reason: str) -> "BenchSkipped":
    """Return the exception that asks asv to skip — ``raise skip("no GPU")``."""
    return BenchSkipped(reason)


def require_jax():
    """Import JAX, or ask asv to skip the benchmark."""
    try:
        import jax
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise skip(f"jax unavailable: {exc}") from exc
    return jax


def require_jax_md():
    """Import jax-md, or ask asv to skip the benchmark."""
    try:
        import jax_md
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise skip(f"jax-md unavailable: {exc}") from exc
    return jax_md


def block(x):
    """Wait for every array in ``x`` to be materialised on device.

    Without this the timed region ends at dispatch and the benchmark measures
    Python, not the kernel.
    """
    import jax

    return jax.block_until_ready(x)


def precision_tag() -> str:
    return "x64" if X64 else "x32"


def device_tag() -> str:
    """Platform of JAX's default device (``cpu`` / ``gpu``), or ``unknown``."""
    try:
        import jax

        return str(jax.devices()[0].platform)
    except Exception:  # pragma: no cover - environment-dependent
        return "unknown"


def default_checkpoint() -> Path:
    """The bundled ACO/DESdimers PhysNet JSON checkpoint (``MMML_CKPT`` overrides)."""
    env = os.environ.get("MMML_BENCH_CKPT") or os.environ.get("MMML_CKPT")
    path = Path(env) if env else REPO_ROOT / "examples" / "ckpts_json" / "DESdimers_params.json"
    if not path.exists():
        raise skip(f"checkpoint not found: {path}")
    return path


# --------------------------------------------------------------------------
# Geometry fixtures
#
# Plain NumPy on purpose: these build the *input* to the code under test, so
# they must be fast, deterministic, and free of anything that could itself show
# up in a timing. Sizes are chosen to bracket the systems mmml actually runs —
# an ACO dimer (20 atoms) through a ~7000-atom solvent box.
# --------------------------------------------------------------------------

_TIP3_R_OH = 0.9572
_TIP3_THETA_DEG = 104.52
_WATER_MOLAR_MASS = 18.01528  # g/mol
_AVOGADRO = 6.02214076e23


def _tip3_template() -> np.ndarray:
    theta = np.deg2rad(_TIP3_THETA_DEG)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [_TIP3_R_OH, 0.0, 0.0],
            [_TIP3_R_OH * np.cos(theta), _TIP3_R_OH * np.sin(theta), 0.0],
        ]
    )


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def water_box(
    n_molecules: int,
    *,
    density_g_cm3: float = 0.997,
    seed: int = 0,
) -> dict:
    """Rigid TIP3 waters on a jittered lattice in a cubic box at liquid density.

    Returns a dict with ``R`` (N, 3), ``Z`` (N,), ``box`` (3, 3), ``mol_id`` (N,),
    ``box_L``, and ``masses`` — geometries are ideal, so the SHAKE/RATTLE
    benchmarks start on the constraint manifold rather than projecting onto it.
    """
    n_molecules = int(n_molecules)
    rng = np.random.default_rng(seed)

    volume_A3 = n_molecules * _WATER_MOLAR_MASS / _AVOGADRO / float(density_g_cm3) * 1.0e24
    box_L = float(volume_A3 ** (1.0 / 3.0))

    n_side = int(np.ceil(n_molecules ** (1.0 / 3.0)))
    pitch = box_L / n_side
    template = _tip3_template()

    coords = np.empty((n_molecules, 3, 3), dtype=np.float64)
    for m in range(n_molecules):
        cell_idx = np.array([m % n_side, (m // n_side) % n_side, m // (n_side * n_side)])
        # Jitter stays under a quarter pitch so molecules never overlap; the MM
        # kernels would otherwise spend their time on a 1/r singularity.
        centre = (cell_idx + 0.5) * pitch + rng.uniform(-0.2 * pitch, 0.2 * pitch, size=3)
        coords[m] = template @ _random_rotation(rng).T + centre

    return {
        "R": coords.reshape(-1, 3),
        "Z": np.tile(np.array([8, 1, 1], dtype=np.int32), n_molecules),
        "box": np.diag([box_L, box_L, box_L]),
        "box_L": box_L,
        "mol_id": np.repeat(np.arange(n_molecules, dtype=np.int32), 3),
        "masses": np.tile(np.array([15.9994, 1.008, 1.008]), n_molecules),
        "n_molecules": n_molecules,
    }


def aco_cluster(n_monomers: int = 2, *, spacing: float = 5.0) -> dict:
    """ACO (acetone) monomers from the bundled template PDB, laid out on a grid.

    Matches ``examples/md_cpu/_geometry.py`` so the ML benchmarks and the CPU MD
    smokes speak about the same system.
    """
    try:
        from ase.io import read as ase_read

        from mmml.paths import default_aco_template_pdb
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise skip(f"ASE / mmml.paths unavailable: {exc}") from exc

    monomer = ase_read(str(default_aco_template_pdb()))
    z_mono = np.asarray(monomer.get_atomic_numbers(), dtype=np.int32)
    r_mono = np.asarray(monomer.get_positions(), dtype=np.float64)
    r_mono = r_mono - r_mono.mean(axis=0)
    atoms_per = int(z_mono.shape[0])

    n_side = int(np.ceil(np.sqrt(n_monomers)))
    chunks = []
    for mi in range(int(n_monomers)):
        shift = np.array([(mi % n_side) * spacing, (mi // n_side) * spacing, 0.0])
        chunks.append(r_mono + shift)

    return {
        "R": np.vstack(chunks),
        "Z": np.tile(z_mono, int(n_monomers)),
        "atoms_per_monomer": atoms_per,
        "n_monomers": int(n_monomers),
        "mol_id": np.repeat(np.arange(int(n_monomers), dtype=np.int32), atoms_per),
    }


def synthetic_ff_params(z: np.ndarray, mol_id: np.ndarray, *, seed: int = 1):
    """A neutral :class:`~mmml.md.system.FFParams` with plausible CHARMM-scale values.

    Real CGenFF parameters need a PSF and a CHARMM build, which is not available
    on every machine that can run these benchmarks. The MM kernels are
    element-agnostic — cost depends on pair count and cutoffs, not on the exact
    epsilon — so synthetic values give an honest timing with no native
    dependency. Charges are neutralised per molecule so the Ewald benchmarks are
    not measuring a charged-cell artefact.
    """
    from mmml.md.system import FFParams

    rng = np.random.default_rng(seed)
    n = int(z.shape[0])
    charges = rng.uniform(-0.8, 0.8, size=n)
    for mol in np.unique(mol_id):
        sel = mol_id == mol
        charges[sel] -= charges[sel].mean()

    return FFParams(
        charges=charges,
        epsilon=rng.uniform(0.04, 0.20, size=n),
        rmin_half=rng.uniform(1.2, 2.0, size=n),
        at_codes=np.asarray(z, dtype=np.int32),
        exclusions=np.empty((0, 2), dtype=np.int32),
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )


def synthetic_system(n_molecules: int, *, seed: int = 0):
    """A periodic :class:`~mmml.md.system.MolecularSystem` of TIP3-shaped waters."""
    from mmml.md.system import MolecularSystem

    box = water_box(n_molecules, seed=seed)
    ff = synthetic_ff_params(box["Z"], box["mol_id"], seed=seed + 1)
    system = MolecularSystem(
        R=box["R"],
        Z=box["Z"],
        box=box["box"],
        mol_id=box["mol_id"],
        ff_params=ff,
    )
    return system, box


def padded_pair_list(system, cutoff_A: float, *, headroom: float = 1.15):
    """Build the intermolecular pair list once and pad it to a fixed capacity.

    The jitted MM energy takes fixed-shape pair arrays, so this mirrors what
    ``mmml.md.neighbors.make_intermolecular_neighbor_fn`` hands the driver —
    without paying the rebuild inside the timed region.
    """
    from mmml.interfaces.jaxmdInterface.hybrid_energy import get_intermolecular_pairs

    excluded = frozenset()
    if system.ff_params is not None:
        excluded = frozenset(map(tuple, system.ff_params.exclusions.tolist()))

    pi, pj = get_intermolecular_pairs(
        np.asarray(system.R, dtype=np.float64),
        np.asarray(system.box, dtype=np.float64),
        excluded,
        float(cutoff_A),
        np.asarray(system.mol_id, dtype=np.int32),
    )
    n_pairs = int(len(pi))
    capacity = max(int(n_pairs * float(headroom)), 16)

    pair_i = np.zeros(capacity, dtype=np.int32)
    pair_j = np.zeros(capacity, dtype=np.int32)
    pair_mask = np.zeros(capacity, dtype=np.float64 if X64 else np.float32)
    pair_i[:n_pairs] = pi
    pair_j[:n_pairs] = pj
    pair_mask[:n_pairs] = 1.0
    return {
        "pair_i": pair_i,
        "pair_j": pair_j,
        "pair_mask": pair_mask,
        "n_pairs": n_pairs,
        "capacity": capacity,
    }
