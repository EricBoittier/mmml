"""Frame store for trajectory reweighting: load frames, decompose, re-evaluate.

Pipeline::

    frames = load_frames("run_npt.h5", atoms_per_molecule=10)   # or .extxyz / .traj
    handles = hybrid_term_fns(sph, get_update_fn, z, cutoff_params, n_monomers, x0, box0)
    cache = decompose(frames, handles.terms, handles.pairs, handles.update_fn,
                      cache_path="aco_T200.npz")
    du = delta_u(theta, lam, cache, handles.update_fn.energy_with_lj, cache.type_map())
    w = reweight_weights(du, jnp.zeros_like(du), cache.frames.temperature_K)

The sampled potential is U_0 = E_ml_mono + E_ml_dimer + E_mm(theta_0); the
reweighting target is ``U(theta, lam) = E_ml_mono + lam E_ml_dimer +
E_mm(theta)``. Only ``E_mm`` depends on theta, so the ML terms are evaluated
once and cached. All energies are total-box values in kcal/mol, lengths in A.

JAX-MD NPT HDF5 (``<prefix>_npt.h5`` written by ``mmml md-system --backend
jaxmd``; ``make_jaxmd_reporter(include_positions=True)`` is always on, there is
no flag to set) stores per record: ``positions`` (F, N, 3) real-space A (per
atom or per monomer wrapped), ``potential_energy`` (eV), ``density_g_cm3``,
``temperature``, ``time_ps``, ``velocities``, ``charges``; attrs
``atomic_numbers``, ``temperature_target``, ``ensemble``. The box is NOT
stored; the isotropic NPT box edge is recovered from the recorded density and
the ASE masses of ``atomic_numbers`` (the runner's ``Si_mass``):
``L = (sum(m) * 1.66053906660 / rho)^(1/3)``. The companion ``pbc_npt.traj``
(ASE) carries the cell per frame and can be loaded directly instead.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from mmml.data.units import EV_TO_KCAL_MOL
from mmml.fit.lj_theta import LjTypeMap, Theta, per_atom_lj

# g/cm^3 per (amu / A^3).
AMU_PER_A3_TO_G_CM3 = 1.66053906660

# Loaders drop records whose potential energy deviates from the reference
# cluster by more than this (kcal/mol per atom): a blown-up integrator, not a
# fluctuation (thermal spread is ~kT sqrt(N) / N << 0.1 kcal/mol/atom for liquids).
MAX_U_DEV_KCAL_MOL_PER_ATOM = 1.0
# Absolute sanity bound: |U| may not exceed this multiple of the Thomas-Fermi
# total energy of the free atoms, sum_i 0.7687 Z_i^(7/3) Hartree (an upper bound
# on |E_atom| for every element; valid for total-energy and atomization-energy
# models alike). Catches a blown-up record even when it is the only one.
MAX_ABS_U_THOMAS_FERMI_FACTOR = 2.0
HARTREE_TO_KCAL_MOL = 627.509474
# Fewer usable records than this: the relative blow-up filter has no majority to
# compare against, so the loaders warn.
MIN_FRAMES_FOR_OUTLIER_CHECK = 3

TermsFn = Callable[[np.ndarray, np.ndarray], Mapping[str, float]]
PairsFn = Callable[[np.ndarray, np.ndarray], tuple[Any, Any]]
MmEnergyWithLj = Callable[..., jnp.ndarray]


def density_g_cm3(total_mass_amu: float, box_A: np.ndarray) -> np.ndarray:
    """Mass density (g/cm^3) of orthorhombic boxes ``(..., 3)`` in A."""
    vol = np.prod(np.asarray(box_A, dtype=np.float64), axis=-1)
    return float(total_mass_amu) * AMU_PER_A3_TO_G_CM3 / vol


def cubic_box_from_density(total_mass_amu: float, rho_g_cm3: np.ndarray) -> np.ndarray:
    """Cubic box ``(F, 3)`` in A reproducing densities ``(F,)`` in g/cm^3."""
    rho = np.asarray(rho_g_cm3, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(rho) & (rho > 0.0)):
        raise ValueError(f"densities must be finite and > 0, got {rho[~(np.isfinite(rho) & (rho > 0.0))]}")
    edge = np.cbrt(float(total_mass_amu) * AMU_PER_A3_TO_G_CM3 / rho)
    return np.repeat(edge[:, None], 3, axis=1)


def make_whole(positions: np.ndarray, atoms_per_molecule: int, box_A: np.ndarray) -> np.ndarray:
    """Unwrap each molecule about its first atom, then wrap its centroid into [0, L).

    Orthorhombic box ``(3,)`` in A; ``positions`` is ``(N, 3)`` with molecules
    stored contiguously. PhysNet monomer energies need intact molecules.
    """
    box = np.asarray(box_A, dtype=np.float64).reshape(3)
    p = np.asarray(positions, dtype=np.float64).reshape(-1, int(atoms_per_molecule), 3)
    d = p - p[:, :1]
    d -= box * np.round(d / box)
    p = p[:, :1] + d
    cog = p.mean(axis=1, keepdims=True)
    return (p - box * np.floor(cog / box)).reshape(-1, 3)


@dataclass(frozen=True)
class FrameSet:
    """Stored configurations of one state point (A, kcal/mol, K, g/cm^3)."""

    positions: np.ndarray  # (F, N, 3) float64, A
    box: np.ndarray  # (F, 3) float64, orthorhombic edges, A
    atomic_numbers: np.ndarray  # (N,) int32
    masses: np.ndarray  # (N,) float64, amu
    temperature_K: float
    n_molecules: int
    density_g_cm3: np.ndarray  # (F,)
    u_ref: np.ndarray  # (F,) sampling-run potential energy, kcal/mol (NaN if absent)
    pressure_atm: float | None = None

    @property
    def n_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[1])

    @property
    def atoms_per_molecule(self) -> int:
        return self.n_atoms // self.n_molecules

    def select(self, index: slice | np.ndarray) -> FrameSet:
        """Subset of frames (slice or integer index array)."""
        return replace(
            self,
            positions=self.positions[index],
            box=self.box[index],
            density_g_cm3=self.density_g_cm3[index],
            u_ref=self.u_ref[index],
        )


def _masses(atomic_numbers: np.ndarray) -> np.ndarray:
    from ase.data import atomic_masses

    return np.asarray(atomic_masses[np.asarray(atomic_numbers, dtype=int)], dtype=np.float64)


def _n_molecules(n_atoms: int, atoms_per_molecule: int) -> int:
    if n_atoms % int(atoms_per_molecule):
        raise ValueError(f"{n_atoms} atoms is not a multiple of {atoms_per_molecule} per molecule")
    return n_atoms // int(atoms_per_molecule)


def max_abs_potential_kcal_mol(atomic_numbers: np.ndarray, factor: float = MAX_ABS_U_THOMAS_FERMI_FACTOR) -> float:
    """Largest physical ``|U|`` (kcal/mol) of the atoms: ``factor`` x Thomas-Fermi total energy."""
    z = np.asarray(atomic_numbers, dtype=np.float64)
    return float(factor) * 0.7687 * float(np.sum(z ** (7.0 / 3.0))) * HARTREE_TO_KCAL_MOL


def _reference_energy(u: np.ndarray, half_width: float) -> float:
    """Median of the largest cluster of energies (window ``+-half_width``).

    Unlike the plain median this is not dragged along when most records are
    blown up to scattered values. Ties go to the cluster holding the earliest
    record (a blow-up is absorbing, so early records are the physical ones).
    """
    order = np.argsort(u, kind="stable")
    us = u[order]
    lo = np.searchsorted(us, us - half_width, side="left")
    hi = np.searchsorted(us, us + half_width, side="right")
    counts = hi - lo
    best = np.flatnonzero(counts == counts.max())
    first = [int(order[lo[b] : hi[b]].min()) for b in best]
    b = int(best[int(np.argmin(first))])
    return float(np.median(us[lo[b] : hi[b]]))


def _good_frames(
    path: str | Path,
    positions: np.ndarray,
    u_kcal_mol: np.ndarray,
    extra: Mapping[str, np.ndarray],
    max_u_dev_kcal_mol_per_atom: float | None,
    atomic_numbers: np.ndarray,
) -> np.ndarray:
    """Mask ``(F,)`` of usable records; warns about every dropped frame.

    Drops non-finite positions / ``extra`` arrays (per-frame, e.g. density),
    non-finite energies when the energy is recorded at all, energies beyond the
    absolute bound :func:`max_abs_potential_kcal_mol`, and energy outliers
    beyond ``max_u_dev_kcal_mol_per_atom`` from the reference cluster
    (:func:`_reference_energy`). Warns when fewer than
    ``MIN_FRAMES_FOR_OUTLIER_CHECK`` records remain.
    """
    n_f, n_at = positions.shape[0], positions.shape[1]
    ok = np.isfinite(positions).reshape(n_f, -1).all(axis=1)
    reasons = {"non-finite positions": ~ok}
    for name, v in extra.items():
        bad = ~np.isfinite(np.asarray(v).reshape(n_f, -1)).all(axis=1)
        reasons[f"non-finite {name}"] = bad
        ok &= ~bad
    has_u = not np.all(np.isnan(u_kcal_mol))
    if has_u:
        bad = ~np.isfinite(u_kcal_mol)
        reasons["non-finite potential energy"] = bad
        ok &= ~bad
        u_max = max_abs_potential_kcal_mol(atomic_numbers)
        with np.errstate(invalid="ignore"):
            bad = ok & (np.abs(u_kcal_mol) > u_max)
        reasons[f"|U| > {u_max:.4g} kcal/mol (unphysical, {MAX_ABS_U_THOMAS_FERMI_FACTOR:g} x Thomas-Fermi)"] = bad
        ok &= ~bad
        if max_u_dev_kcal_mol_per_atom is not None and ok.any():
            tol = float(max_u_dev_kcal_mol_per_atom) * n_at
            ref = _reference_energy(u_kcal_mol[ok], tol)
            with np.errstate(invalid="ignore"):
                bad = ok & (np.abs(u_kcal_mol - ref) > tol)
            reasons[f"|U - U_ref| > {max_u_dev_kcal_mol_per_atom} kcal/mol/atom (U_ref {ref:.6g})"] = bad
            ok &= ~bad
    for why, bad in reasons.items():
        if bad.any():
            warnings.warn(
                f"{path}: dropping {int(bad.sum())}/{n_f} frames with {why}: {np.flatnonzero(bad).tolist()}",
                stacklevel=3,
            )
    if not ok.any():
        raise ValueError(f"{path}: no usable frames")
    if has_u and max_u_dev_kcal_mol_per_atom is not None and ok.sum() < MIN_FRAMES_FOR_OUTLIER_CHECK:
        warnings.warn(
            f"{path}: only {int(ok.sum())} usable frame(s); the relative blow-up filter needs "
            f">= {MIN_FRAMES_FOR_OUTLIER_CHECK} to compare against (only the absolute bound applied)",
            stacklevel=3,
        )
    return ok


def load_jaxmd_h5(
    path: str | Path,
    atoms_per_molecule: int,
    *,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    temperature_K: float | None = None,
    box_A: np.ndarray | None = None,
    max_u_dev_kcal_mol_per_atom: float | None = MAX_U_DEV_KCAL_MOL_PER_ATOM,
) -> FrameSet:
    """Load a JAX-MD HDF5 record file (see module docstring for the layout).

    ``box_A`` ``(3,)`` or ``(F_all, 3)`` overrides the box; otherwise NPT boxes
    come from ``density_g_cm3`` and an NVT/NVE run needs ``box_A``. Records
    with non-finite positions / energy / density or blown-up energies are
    dropped with a warning (see :func:`_good_frames`).
    """
    import h5py

    sl = slice(start, stop, stride)
    with h5py.File(path, "r") as f:
        z = np.asarray(f.attrs["atomic_numbers"], dtype=np.int32)
        pos = np.asarray(f["positions"][sl], dtype=np.float64)
        n_all = int(f["positions"].shape[0])
        u = (
            np.asarray(f["potential_energy"][sl], dtype=np.float64) * EV_TO_KCAL_MOL
            if "potential_energy" in f
            else np.full(pos.shape[0], np.nan)
        )
        rho = np.asarray(f["density_g_cm3"][sl], dtype=np.float64) if "density_g_cm3" in f else None
        t_attr = f.attrs.get("temperature_target")
    masses = _masses(z)
    if box_A is None and rho is None:
        raise ValueError(f"{path}: no density_g_cm3 (not NPT); pass box_A")
    extra = {} if rho is None else {"density_g_cm3": rho}
    ok = _good_frames(path, pos, u, extra, max_u_dev_kcal_mol_per_atom, z)
    if box_A is not None:
        box = np.asarray(box_A, dtype=np.float64)
        box = np.broadcast_to(box, (n_all, 3))[sl] if box.ndim == 2 else np.tile(box, (pos.shape[0], 1))
        box = np.array(box, dtype=np.float64)[ok]
    pos, u = pos[ok], u[ok]
    if rho is not None:
        rho = rho[ok]
    if box_A is None:
        box = cubic_box_from_density(masses.sum(), rho)
    if rho is None:
        rho = density_g_cm3(masses.sum(), box)
    if temperature_K is None:
        if t_attr is None:
            raise ValueError(f"{path}: no temperature_target attr; pass temperature_K")
        temperature_K = float(t_attr)
    return FrameSet(
        positions=pos,
        box=box,
        atomic_numbers=z,
        masses=masses,
        temperature_K=float(temperature_K),
        n_molecules=_n_molecules(pos.shape[1], atoms_per_molecule),
        density_g_cm3=rho,
        u_ref=u,
    )


def load_ase_frames(
    path: str | Path,
    atoms_per_molecule: int,
    *,
    index: str = ":",
    temperature_K: float | None = None,
    stages: Sequence[str] | None = ("equi", "prod"),
    max_u_dev_kcal_mol_per_atom: float | None = MAX_U_DEV_KCAL_MOL_PER_ATOM,
) -> FrameSet:
    """Load extxyz / ASE ``.traj`` frames with an orthorhombic cell per frame.

    ``u_ref`` is the stored potential energy (eV -> kcal/mol) when present.
    Frames carrying ``info['stage']`` are kept only if the stage is in
    ``stages`` (default drops non-equilibrium ``heat`` frames; ``None`` keeps
    all). ``temperature_K`` falls back to ``info['T_target_K']`` /
    ``info['temperature_K']``, which must agree across the kept frames.
    """
    from ase.io import read

    images = read(str(path), index=index)
    if not isinstance(images, list):
        images = [images]
    if not images:
        raise ValueError(f"{path}: no frames")
    if stages is not None:
        keep = {str(s) for s in stages}
        dropped = [str(a.info["stage"]) for a in images if "stage" in a.info and str(a.info["stage"]) not in keep]
        if dropped:
            counts = {s: dropped.count(s) for s in sorted(set(dropped))}
            warnings.warn(
                f"{path}: dropping {len(dropped)}/{len(images)} frames from stages {counts} (keeping {sorted(keep)})",
                stacklevel=2,
            )
        images = [a for a in images if "stage" not in a.info or str(a.info["stage"]) in keep]
        if not images:
            raise ValueError(f"{path}: no frames in stages {sorted(keep)}")
    t_info = {float(t) for a in images if (t := a.info.get("T_target_K", a.info.get("temperature_K"))) is not None}
    if len(t_info) > 1:
        raise ValueError(f"{path}: frames have different target temperatures {sorted(t_info)}")
    z = np.asarray(images[0].get_atomic_numbers(), dtype=np.int32)
    pos = np.stack([np.asarray(a.positions, dtype=np.float64) for a in images])
    cells = np.stack([np.asarray(a.cell[:], dtype=np.float64) for a in images])
    off_diag = cells - np.einsum("fii->fi", cells)[:, :, None] * np.eye(3)
    if not np.allclose(off_diag, 0.0, atol=1e-8):
        raise ValueError(f"{path}: only orthorhombic cells are supported")
    box = np.einsum("fii->fi", cells).copy()
    if np.any(box <= 0.0):
        raise ValueError(f"{path}: frames need a periodic cell")
    u = np.full(len(images), np.nan)
    for i, a in enumerate(images):
        try:
            u[i] = float(a.get_potential_energy()) * EV_TO_KCAL_MOL
        except Exception:  # noqa: BLE001 - no calculator / energy stored
            e = a.info.get("energy")
            if e is not None:
                u[i] = float(e) * EV_TO_KCAL_MOL
    if temperature_K is None:
        if not t_info:
            raise ValueError(f"{path}: no temperature in info; pass temperature_K")
        temperature_K = t_info.pop()
    elif t_info and not np.isclose(float(temperature_K), next(iter(t_info))):
        warnings.warn(
            f"{path}: temperature_K={temperature_K} overrides info T_target_K={next(iter(t_info))}",
            stacklevel=2,
        )
    ok = _good_frames(path, pos, u, {"cell": box}, max_u_dev_kcal_mol_per_atom, z)
    pos, box, u = pos[ok], box[ok], u[ok]
    masses = _masses(z)
    return FrameSet(
        positions=pos,
        box=box,
        atomic_numbers=z,
        masses=masses,
        temperature_K=float(temperature_K),
        n_molecules=_n_molecules(pos.shape[1], atoms_per_molecule),
        density_g_cm3=density_g_cm3(masses.sum(), box),
        u_ref=u,
    )


def load_frames(path: str | Path, atoms_per_molecule: int, **kwargs: Any) -> FrameSet:
    """Dispatch on suffix: ``.h5``/``.hdf5`` -> :func:`load_jaxmd_h5`, else ASE."""
    if Path(path).suffix.lower() in {".h5", ".hdf5"}:
        return load_jaxmd_h5(path, atoms_per_molecule, **kwargs)
    return load_ase_frames(path, atoms_per_molecule, **kwargs)


# ---------------------------------------------------------------------------
# Decomposition cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameCache:
    """Per-frame energy terms (kcal/mol) plus what ``E_mm(theta)`` needs later.

    ``positions`` are the molecule-whole coordinates actually evaluated.
    Pair lists keep only unmasked pairs, padded to a common length (padding:
    index (0, 0), mask 0); ``pair_capacity`` is the live list length that
    ``energy_with_lj`` expects (its switching closure is sized to it).
    ``E_mm0`` is ``mm_energy_with_lj`` at the base LJ (the theta_0 reference
    used by :func:`delta_u`); ``E_mm_calc`` is the calculator's own MM term.
    ``fp_frames`` / ``fp_lj`` / ``fp_model`` fingerprint the input frames, the
    base MM parameters and the rest of the Hamiltonian (caller's ``model_tag``
    plus the ``whole`` flag; :func:`frames_fingerprint`, :func:`lj_fingerprint`,
    :func:`model_fingerprint`) so :func:`decompose` can refuse a stale cache.
    """

    frames: FrameSet
    E_ml_mono: np.ndarray  # (F,)
    E_ml_dimer: np.ndarray  # (F,) switched PhysNet dimer term
    E_mm0: np.ndarray  # (F,)
    E_mm_calc: np.ndarray  # (F,)
    pair_idx: np.ndarray  # (F, P, 2) int32
    pair_mask: np.ndarray  # (F, P)
    base_rmins: np.ndarray  # (N,) CHARMM Rmin/2, A
    base_epsilons: np.ndarray  # (N,) CHARMM epsilon (<= 0), kcal/mol
    at_codes: np.ndarray  # (N,) int, index into atc_names
    atc_names: tuple[str, ...]
    pair_capacity: int = 0  # 0 = use the stored length
    fp_frames: str = ""
    fp_lj: str = ""
    fp_model: str = ""

    @property
    def u_ref(self) -> np.ndarray:
        """Decomposed sampling potential at (theta_0, lam=1), kcal/mol."""
        return self.E_ml_mono + self.E_ml_dimer + self.E_mm0

    def type_map(self, fit_types: list[str] | None = None) -> LjTypeMap:
        return LjTypeMap.from_atc(self.at_codes, self.atc_names, fit_types)

    def save(self, path: str | Path) -> None:
        arrays: dict[str, Any] = {}
        for fld in fields(FrameSet):
            v = getattr(self.frames, fld.name)
            arrays[f"frames__{fld.name}"] = np.asarray(np.nan if v is None else v)
        for fld in fields(self):
            if fld.name == "frames":
                continue
            arrays[fld.name] = np.asarray(getattr(self, fld.name))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> FrameCache:
        with np.load(path, allow_pickle=False) as d:
            fs: dict[str, Any] = {}
            for fld in fields(FrameSet):
                v = d[f"frames__{fld.name}"]
                fs[fld.name] = v if v.ndim else v.item()
            fs["temperature_K"] = float(fs["temperature_K"])
            fs["n_molecules"] = int(fs["n_molecules"])
            p = fs["pressure_atm"]
            fs["pressure_atm"] = None if p is None or np.isnan(p) else float(p)
            rest = {fld.name: d[fld.name] for fld in fields(cls) if fld.name != "frames" and fld.name in d}
        rest["atc_names"] = tuple(str(s) for s in rest["atc_names"])
        rest["pair_capacity"] = int(rest.get("pair_capacity", 0))
        for key in ("fp_frames", "fp_lj", "fp_model"):
            if key in rest:
                rest[key] = str(rest[key])
        return cls(frames=FrameSet(**fs), **rest)


def _pad_pairs(idx_list: list[np.ndarray], mask_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Drop masked rows, then pad all frames to the largest valid-pair count.

    The live neighbour list over-allocates (~18x for the acetone box) and
    masked pairs contribute exactly 0, so storage is compact;
    :func:`mm_energies` re-pads each frame to ``pair_capacity`` on the fly.
    """
    kept = [(i[m > 0], m[m > 0]) for i, m in zip(idx_list, mask_list, strict=True)]
    cap = max(1, max(int(i.shape[0]) for i, _ in kept))
    idx = np.zeros((len(kept), cap, 2), dtype=np.int32)
    mask = np.zeros((len(kept), cap), dtype=np.float32)
    for f, (i, m) in enumerate(kept):
        idx[f, : i.shape[0]] = i
        mask[f, : m.shape[0]] = m
    return idx, mask


def _sha256(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        a = np.ascontiguousarray(np.asarray(p))
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def frames_fingerprint(frames: FrameSet) -> str:
    """Hash of the input positions, boxes, atomic numbers and temperature."""
    return _sha256(
        np.asarray(frames.positions, dtype=np.float64),
        np.asarray(frames.box, dtype=np.float64),
        np.asarray(frames.atomic_numbers, dtype=np.int64),
        np.float64(frames.temperature_K),
    )


def lj_fingerprint(mm_update_fn: Any) -> str:
    """Hash of the MM object's base LJ (Rmin/2, epsilon), ``at_codes`` and type names."""
    return _sha256(
        np.asarray(mm_update_fn.lj_rmins, dtype=np.float64),
        np.asarray(mm_update_fn.lj_epsilons, dtype=np.float64),
        np.asarray(mm_update_fn.at_codes, dtype=np.int64),
        np.asarray([str(s) for s in mm_update_fn.atc_names], dtype=str),
    )


def hamiltonian_tag(checkpoint: str | Path | None = None, cutoff_params: Any = None, **extra: Any) -> str:
    """Hash identifying the hybrid Hamiltonian for :func:`decompose`'s ``model_tag``.

    Covers the checkpoint file contents (every file, if a directory), the
    cutoff / switching parameters (``to_dict()`` or ``repr``) and any ``extra``
    key/values (e.g. charges source, LJ sidecar path).
    """
    h = hashlib.sha256()
    if checkpoint is not None:
        ck = Path(checkpoint)
        files = sorted(q for q in ck.rglob("*") if q.is_file()) if ck.is_dir() else [ck]
        for q in files:
            h.update(q.name.encode())
            h.update(q.read_bytes())
    if cutoff_params is not None:
        d = cutoff_params.to_dict() if hasattr(cutoff_params, "to_dict") else cutoff_params
        h.update(repr(sorted(d.items()) if isinstance(d, Mapping) else d).encode())
    h.update(repr(sorted(extra.items())).encode())
    return h.hexdigest()


def model_fingerprint(model_tag: str, whole: bool) -> str:
    """Hash of the caller's ``model_tag`` and the molecule-``whole`` flag."""
    return _sha256(np.asarray(str(model_tag)), np.asarray(bool(whole)))


# Stored E_mm0 / ML terms of frame 0 vs a fresh evaluation on cache reuse:
# a mismatch beyond atol + rtol |E| means the Hamiltonian changed (kcal/mol).
VERIFY_ATOL_KCAL_MOL = 0.01
VERIFY_RTOL = 1e-4


def _verify_frame0(cache: FrameCache, terms_fn: TermsFn | None, mm_update_fn: Any) -> list[str]:
    """Re-evaluate frame 0 of a cache with the requested functions; mismatch messages."""
    bad = []

    def off(stored: float, fresh: float) -> bool:
        return not abs(fresh - stored) <= VERIFY_ATOL_KCAL_MOL + VERIFY_RTOL * abs(stored)

    if mm_update_fn is not None:
        c0 = replace(
            cache,
            frames=cache.frames.select(slice(0, 1)),
            pair_idx=cache.pair_idx[:1],
            pair_mask=cache.pair_mask[:1],
        )
        e = float(np.asarray(mm_energies(None, c0, mm_update_fn.energy_with_lj))[0])
        if off(float(cache.E_mm0[0]), e):
            bad.append(f"frame 0 E_mm(theta_0) is {e:.6g}, cached {float(cache.E_mm0[0]):.6g} kcal/mol")
    if terms_fn is not None:
        out = terms_fn(cache.frames.positions[0], cache.frames.box[0])
        for key, name in (("internal_E", "E_ml_mono"), ("ml_2b_E", "E_ml_dimer"), ("mm_E", "E_mm_calc")):
            stored, fresh = float(getattr(cache, name)[0]), float(out[key]) * EV_TO_KCAL_MOL
            if off(stored, fresh):
                bad.append(f"frame 0 {name} is {fresh:.6g}, cached {stored:.6g} kcal/mol")
    return bad


def _cache_mismatches(
    cache: FrameCache,
    frames: FrameSet,
    pairs_fn: PairsFn | None,
    mm_update_fn: Any,
    whole: bool,
    model_tag: str | None,
) -> list[str]:
    """Reasons the stored cache does not correspond to the requested inputs."""
    bad = []
    if not cache.fp_frames or not cache.fp_lj or not cache.fp_model:
        bad.append("cache has no (complete) fingerprint (written by an older version)")
    if model_tag is not None and cache.fp_model and cache.fp_model != model_fingerprint(model_tag, whole):
        bad.append("model_tag (checkpoint / cutoffs) or the whole flag differ")
    if cache.fp_frames and cache.fp_frames != frames_fingerprint(frames):
        bad.append(
            f"frames differ (cached {cache.frames.n_frames} frames at T={cache.frames.temperature_K} K, "
            f"requested {frames.n_frames} at T={frames.temperature_K} K)"
        )
    if mm_update_fn is not None and cache.fp_lj and cache.fp_lj != lj_fingerprint(mm_update_fn):
        bad.append("base LJ parameters / at_codes / type names differ from mm_update_fn")
    if pairs_fn is not None and cache.pair_capacity:
        x = frames.positions[0]
        if whole:
            x = make_whole(x, frames.atoms_per_molecule, frames.box[0])
        cap = int(np.asarray(pairs_fn(x, frames.box[0])[0]).shape[0])
        if cap != cache.pair_capacity:
            bad.append(f"pair capacity differs (cached {cache.pair_capacity}, requested {cap})")
    return bad


def check_decomposition(
    cache: FrameCache,
    *,
    mm_atol_kcal_mol: float = 1.0,
    mm_rtol: float = 1e-3,
    u_ref_tol_kcal_mol_per_molecule: float = 0.01,
) -> dict[str, Any]:
    """Compare the decomposition with the calculator and the sampler (kcal/mol).

    * ``mm``: ``E_mm0`` (re-evaluated MM) vs ``E_mm_calc`` (the calculator's own
      MM term); fails when ``|diff| > mm_atol + mm_rtol |E_mm_calc|``.
    * ``u_ref``: when the sampling run recorded its potential (``frames.u_ref``
      finite), ``frames.u_ref - cache.u_ref`` must be constant across frames
      (terms outside the decomposition, e.g. a wall, add an offset); fails when
      its standard deviation per molecule exceeds ``u_ref_tol``. A wrong LJ
      type assignment or a different Hamiltonian shows up here.

    Returns the statistics plus ``problems`` (list of messages, empty if OK).
    """
    out: dict[str, Any] = {"problems": []}
    d_mm = np.asarray(cache.E_mm0) - np.asarray(cache.E_mm_calc)
    fin = np.isfinite(cache.E_mm_calc)
    if fin.any():
        tol = mm_atol_kcal_mol + mm_rtol * np.abs(cache.E_mm_calc[fin])
        out["mm_max_abs_diff_kcal_mol"] = float(np.max(np.abs(d_mm[fin])))
        if np.any(~np.isfinite(d_mm[fin]) | (np.abs(d_mm[fin]) > tol)):
            out["problems"].append(
                f"E_mm0 (energy_with_lj at base LJ) disagrees with the calculator's mm_E: "
                f"max |diff| {out['mm_max_abs_diff_kcal_mol']:.4g} kcal/mol"
            )
    u_s = np.asarray(cache.frames.u_ref, dtype=np.float64)
    ok = np.isfinite(u_s)
    if ok.sum() >= 2:
        diff = u_s[ok] - cache.u_ref[ok]
        n_mol = cache.frames.n_molecules
        out["u_ref_offset_kcal_mol"] = float(np.mean(diff))
        out["u_ref_spread_kcal_mol_per_molecule"] = float(np.std(diff) / n_mol)
        out["u_ref_range_kcal_mol"] = float(np.ptp(diff))
        if not out["u_ref_spread_kcal_mol_per_molecule"] <= u_ref_tol_kcal_mol_per_molecule:
            out["problems"].append(
                f"sampler u_ref - decomposed U_0 is not constant across frames: std "
                f"{out['u_ref_spread_kcal_mol_per_molecule']:.4g} kcal/mol/molecule "
                f"(tol {u_ref_tol_kcal_mol_per_molecule}); the decomposition does not reproduce "
                "the sampled Hamiltonian (LJ types? cutoffs? different model?)"
            )
    return out


def decompose(
    frames: FrameSet,
    terms_fn: TermsFn | None,
    pairs_fn: PairsFn | None,
    mm_update_fn: Any,
    *,
    cache_path: str | Path | None = None,
    overwrite: bool = False,
    whole: bool = True,
    check: Literal["warn", "raise", "off"] = "warn",
    check_kwargs: Mapping[str, float] | None = None,
    model_tag: str | None = None,
    verify_on_load: bool = True,
    verbose: bool = False,
) -> FrameCache:
    """Evaluate the hybrid terms per frame and cache them (kcal/mol).

    ``terms_fn(positions, box) -> {"internal_E", "ml_2b_E", "mm_E"}`` in eV
    (the spherical-cutoff calculator's ModelOutput fields); ``pairs_fn(positions,
    box) -> (pair_idx, pair_mask)``; ``mm_update_fn`` is the MM pair-update
    object carrying ``energy_with_lj``, ``lj_rmins``, ``lj_epsilons``,
    ``at_codes`` and ``atc_names``.

    ``model_tag`` identifies the rest of the Hamiltonian (checkpoint, cutoffs;
    see :func:`hamiltonian_tag`); ``None`` takes ``terms_fn.model_tag`` (set by
    :func:`hybrid_term_fns`) or ``""``.

    An existing ``cache_path`` is reused only if its fingerprint matches
    ``frames``, ``model_tag`` / ``whole`` and (when given) ``mm_update_fn`` /
    ``pairs_fn``, and (``verify_on_load``) frame 0 re-evaluated with
    ``mm_update_fn`` / ``terms_fn`` reproduces the stored terms; otherwise
    ``ValueError`` is raised, or the cache is recomputed with ``overwrite``.
    The result is checked with :func:`check_decomposition` (``check`` selects
    warn / raise / off; ``check_kwargs`` sets its tolerances).
    """
    if check not in ("warn", "raise", "off"):
        raise ValueError(f"check must be 'warn', 'raise' or 'off', got {check!r}")
    if model_tag is None and terms_fn is not None:
        model_tag = str(getattr(terms_fn, "model_tag", ""))
    if cache_path is not None and Path(cache_path).exists() and not overwrite:
        cached = FrameCache.load(cache_path)
        bad = _cache_mismatches(cached, frames, pairs_fn, mm_update_fn, whole, model_tag)
        if not bad and verify_on_load:
            bad = _verify_frame0(cached, terms_fn, mm_update_fn)
        if bad:
            raise ValueError(f"stale cache {cache_path}: " + "; ".join(bad) + " (delete it or pass overwrite=True)")
        return cached
    if terms_fn is None or pairs_fn is None or mm_update_fn is None:
        raise ValueError("terms_fn, pairs_fn and mm_update_fn are required to compute a cache")
    apm = frames.atoms_per_molecule
    pos_out = np.empty_like(frames.positions)
    mono, dimer, mm_calc = (np.empty(frames.n_frames) for _ in range(3))
    idx_list, mask_list = [], []
    for f in range(frames.n_frames):
        box = frames.box[f]
        x = make_whole(frames.positions[f], apm, box) if whole else frames.positions[f]
        pos_out[f] = x
        out = terms_fn(x, box)
        mono[f] = float(out["internal_E"]) * EV_TO_KCAL_MOL
        dimer[f] = float(out["ml_2b_E"]) * EV_TO_KCAL_MOL
        mm_calc[f] = float(out["mm_E"]) * EV_TO_KCAL_MOL
        pi, pm = pairs_fn(x, box)
        pi, pm = np.asarray(pi, dtype=np.int32), np.asarray(pm, dtype=np.float32)
        idx_list.append(pi)
        mask_list.append(pm)
        if verbose:
            print(
                f"[decompose] frame {f}: mono {mono[f]:.3f} dimer {dimer[f]:.3f} "
                f"mm(calc) {mm_calc[f]:.3f} kcal/mol, {int(pm.sum())} MM pairs",
                flush=True,
            )
    capacity = {int(i.shape[0]) for i in idx_list}
    if len(capacity) != 1:
        raise RuntimeError(f"MM pair capacity changed between frames: {sorted(capacity)}")
    pair_idx, pair_mask = _pad_pairs(idx_list, mask_list)
    cache = FrameCache(
        frames=replace(frames, positions=pos_out),
        E_ml_mono=mono,
        E_ml_dimer=dimer,
        E_mm0=np.zeros(frames.n_frames),
        E_mm_calc=mm_calc,
        pair_idx=pair_idx,
        pair_mask=pair_mask,
        base_rmins=np.asarray(mm_update_fn.lj_rmins, dtype=np.float64),
        base_epsilons=np.asarray(mm_update_fn.lj_epsilons, dtype=np.float64),
        at_codes=np.asarray(mm_update_fn.at_codes, dtype=np.int32),
        atc_names=tuple(str(s) for s in mm_update_fn.atc_names),
        pair_capacity=capacity.pop(),
        fp_frames=frames_fingerprint(frames),
        fp_lj=lj_fingerprint(mm_update_fn),
        fp_model=model_fingerprint(model_tag or "", whole),
    )
    # Same code path (padded pairs, lax.map) as delta_u, so delta_u(theta_0, 1) == 0.
    e_mm0 = mm_energies(None, cache, mm_update_fn.energy_with_lj)
    cache = replace(cache, E_mm0=np.asarray(e_mm0, dtype=np.float64))
    if verbose:
        print(f"[decompose] E_mm(theta_0) kcal/mol: {np.round(cache.E_mm0, 3)}", flush=True)
    if check != "off":
        report = check_decomposition(cache, **dict(check_kwargs or {}))
        if verbose:
            print(f"[decompose] consistency: {report}", flush=True)
        if report["problems"]:
            msg = "decomposition check failed: " + " | ".join(report["problems"])
            if check == "raise":
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        cache.save(cache_path)
    return cache


# ---------------------------------------------------------------------------
# Differentiable re-evaluation
# ---------------------------------------------------------------------------


def mm_energies(
    theta: Theta | None,
    cache: FrameCache,
    mm_energy_with_lj: MmEnergyWithLj,
    type_map: LjTypeMap | None = None,
) -> jnp.ndarray:
    """``E_mm(theta)`` per frame (F,), kcal/mol; ``theta=None`` uses the base LJ.

    Frames are evaluated sequentially (``lax.map``) under ``jax.checkpoint``,
    so ``jax.grad`` through it needs O(one frame) of temporaries.
    """
    rm = jnp.asarray(cache.base_rmins)
    ep = jnp.asarray(cache.base_epsilons)
    if theta is not None:
        if type_map is None:
            raise ValueError("type_map is required with theta")
        rm, ep = per_atom_lj(theta, type_map, rm, ep)
    pos = jnp.asarray(cache.frames.positions)
    cells = jax.vmap(jnp.diag)(jnp.asarray(cache.frames.box))
    idx = jnp.asarray(cache.pair_idx)
    mask = jnp.asarray(cache.pair_mask)
    pad = max(0, int(cache.pair_capacity) - int(idx.shape[1]))

    # Rematerialised per frame: reverse mode keeps only the (compact) inputs of
    # each frame and recomputes its capacity-sized pair intermediates in the
    # backward sweep, so gradient memory is ~one frame's worth, not F x capacity.
    @jax.checkpoint
    def one(args: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        x, pi, pm, cell = args
        if pad:
            pi = jnp.pad(pi, ((0, pad), (0, 0)))
            pm = jnp.pad(pm, (0, pad))
        return mm_energy_with_lj(x, pi, pm, cell, lj_rmins=rm, lj_epsilons=ep)

    return jax.lax.map(one, (pos, idx, mask, cells))


def delta_u(
    theta: Theta | None,
    lam: float | jnp.ndarray,
    cache: FrameCache,
    mm_energy_with_lj: MmEnergyWithLj,
    type_map: LjTypeMap | None = None,
) -> jnp.ndarray:
    """``U(theta, lam) - U_0`` per frame (F,), kcal/mol.

    ``(lam - 1) E_ml_dimer + E_mm(theta) - E_mm0``: free of the large
    monomer energy, so it is usable in float32. It is exactly 0 at
    (theta_0, 1) when evaluated at the precision ``decompose`` ran with;
    across precisions it is 0 up to float32 rounding of ``E_mm``. Feed it to
    ``reweight_weights(du, 0.0 * du, T)``.
    """
    e_mm = mm_energies(theta, cache, mm_energy_with_lj, type_map)
    dimer = jnp.asarray(cache.E_ml_dimer)
    return (lam - 1.0) * dimer + (e_mm - jnp.asarray(cache.E_mm0))


def u_theta(
    theta: Theta | None,
    lam: float | jnp.ndarray,
    cache: FrameCache,
    mm_energy_with_lj: MmEnergyWithLj,
    type_map: LjTypeMap | None = None,
) -> jnp.ndarray:
    """``E_ml_mono + lam E_ml_dimer + E_mm(theta)`` per frame (F,), kcal/mol.

    Needs ``jax_enable_x64`` to resolve differences against the ~1e5 kcal/mol
    monomer sum; prefer :func:`delta_u` for reweighting.
    """
    e_mm = mm_energies(theta, cache, mm_energy_with_lj, type_map)
    return jnp.asarray(cache.E_ml_mono) + lam * jnp.asarray(cache.E_ml_dimer) + e_mm


# ---------------------------------------------------------------------------
# Adapter for the real hybrid calculator
# ---------------------------------------------------------------------------


@dataclass
class HybridTermFns:
    """``terms`` / ``pairs`` callables for :func:`decompose` plus the MM update object.

    ``model_tag`` (:func:`hamiltonian_tag`) is also set as ``terms.model_tag``.
    """

    terms: TermsFn
    pairs: PairsFn
    update_fn: Any
    model_tag: str = ""


def hybrid_term_fns(
    spherical_fn: Callable[..., Any],
    get_update_fn: Callable[..., Any],
    atomic_numbers: np.ndarray,
    cutoff_params: Any,
    n_monomers: int,
    reference_positions: np.ndarray,
    reference_box: np.ndarray,
    *,
    checkpoint: str | Path | None = None,
) -> HybridTermFns:
    """Wrap ``setup_calculator`` outputs (spherical fn, update-fn factory).

    Builds the MM function once at ``reference_positions`` / ``reference_box``
    (A); per-frame boxes are passed to both the pair update and the energy.
    The model tag hashes ``checkpoint`` (when given) and ``cutoff_params``.
    """
    update_fn = get_update_fn(np.asarray(reference_positions), cutoff_params, box=np.asarray(reference_box))
    if update_fn is None or not hasattr(update_fn, "energy_with_lj"):
        raise RuntimeError("MM pair-update function with energy_with_lj is required (dynamic MM path)")
    z = jnp.asarray(atomic_numbers, dtype=jnp.int32)

    def pairs(x: np.ndarray, box: np.ndarray) -> tuple[Any, Any]:
        return update_fn(np.asarray(x), box=np.asarray(box, dtype=np.float64))

    def terms(x: np.ndarray, box: np.ndarray) -> dict[str, float]:
        idx, msk = pairs(x, box)
        out = spherical_fn(
            atomic_numbers=z,
            positions=jnp.asarray(x, dtype=jnp.float32),
            n_monomers=int(n_monomers),
            cutoff_params=cutoff_params,
            doML=True,
            doMM=True,
            doML_dimer=True,
            mm_pair_idx=idx,
            mm_pair_mask=msk,
            box=jnp.asarray(box, dtype=jnp.float32),
        )
        return {k: float(np.asarray(getattr(out, k)).reshape(-1)[0]) for k in ("internal_E", "ml_2b_E", "mm_E")}

    tag = hamiltonian_tag(checkpoint, cutoff_params)
    terms.model_tag = tag  # type: ignore[attr-defined]
    return HybridTermFns(terms=terms, pairs=pairs, update_fn=update_fn, model_tag=tag)
