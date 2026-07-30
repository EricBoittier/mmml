"""MM intramolecular bonded term (bonds, angles, Urey-Bradley, torsions, impropers).

``mm_nonbonded`` is intermolecular-only and ``ml_intra`` covers just the atoms
the ML model owns, so without this term every MM molecule in a unified
``mmml.md`` run has *no* intramolecular energy at all and comes apart within a
few hundred femtoseconds. That is fine for the rigid-water-free peptide setups
the stack grew up on, but not for an explicit-solvent reactive run.

Thin wrapper over the existing JAX CGenFF implementation in
:mod:`mmml.interfaces.pycharmmInterface.cgenff_bonded`; the topology and
parameters come from the PSF the builder already wrote.

``ml_atoms``
------------
Bonded terms touching an ML-owned atom must be dropped. CGenFF has a harmonic
``C1-CL1`` bond in ``CH3CL`` with k ~ 220 kcal/mol/A^2; leaving it in place while
PhysNet independently describes the same C-Cl coordinate both double-counts the
interaction and pins the leaving group, making the SN2 reaction impossible. The
same argument applies to every bond, angle, torsion, and improper wholly or
partly inside the ML region, so any row referencing an ``ml_atoms`` index is
removed. This is the JAX-side equivalent of the ``delete bond``/``delete angle``
lingo the PyCHARMM ADUMB path issues before dynamics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mmml.data.units import KCAL_MOL_TO_EV
from mmml.md.energy.registry import EnergyContext, TermFns, register_term
from mmml.md.energy.terms._common import ase_contribution_from_jax, resolve_displacement_fn
from mmml.md.system import MolecularSystem

__all__ = ["MMBondedTerm"]

# (topology field, index-array width, aligned BondedParameters fields)
_ROW_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bonds", ("bond_k", "bond_r0")),
    ("angles", ("angle_k", "angle_theta0")),
    ("torsions", ("torsion_k", "torsion_n", "torsion_gamma")),
    ("impropers", ("improper_k", "improper_n", "improper_gamma")),
)


def _replace(obj: Any, **updates: Any) -> Any:
    """Rebuild ``obj`` with ``updates`` applied.

    jax-md's ``Topology`` / ``BondedParameters`` are registered as JAX pytree
    nodes rather than stdlib dataclasses, and both ``dataclasses.replace`` and
    ``jax_md.dataclasses.replace`` reject them, so reconstruct from the declared
    field names instead.
    """
    fields = getattr(obj, "_fields", None) or tuple(type(obj).__annotations__)
    kwargs = {name: updates.get(name, getattr(obj, name)) for name in fields}
    return type(obj)(**kwargs)


def _rows_touching(rows: np.ndarray, atoms: frozenset[int]) -> np.ndarray:
    """Boolean mask of ``rows`` (M, k) that reference any index in ``atoms``.

    Tolerates the empty/degenerate shapes CHARMM emits for absent term classes
    (a system with no CMAP can give a 0-d or 1-d array rather than ``(0, k)``).
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.size == 0:
        return np.zeros(rows.shape[0] if rows.ndim >= 1 else 0, dtype=bool)
    return np.isin(rows, list(atoms)).any(axis=1)


def _drop_ml_rows(topology: Any, bonded: Any, ml_atoms: frozenset[int]) -> tuple[Any, Any, dict]:
    """Remove every bonded row referencing an ML-owned atom.

    Urey-Bradley terms ride on the angle rows, so they are filtered implicitly by
    the angle mask; the caller filters ``urey_k`` / ``urey_r0`` with the same
    mask.
    """

    topo_updates: dict[str, Any] = {}
    param_updates: dict[str, Any] = {}
    masks: dict[str, np.ndarray] = {}
    dropped: dict[str, int] = {}

    for field, param_fields in _ROW_GROUPS:
        rows = np.asarray(getattr(topology, field))
        if rows.size == 0:
            masks[field] = np.zeros(0, dtype=bool)
            continue
        touching = _rows_touching(rows, ml_atoms)
        keep = ~touching
        masks[field] = keep
        dropped[field] = int(touching.sum())
        topo_updates[field] = rows[keep]
        for pf in param_fields:
            values = getattr(bonded, pf, None)
            if values is None:
                continue
            arr = np.asarray(values)
            if arr.shape[0] == rows.shape[0]:
                param_updates[pf] = arr[keep]

    # CMAP is protein-only and never spans an ML solute here, but drop any grid
    # whose atoms overlap the ML region rather than silently keeping it.
    cmap_atoms = np.asarray(getattr(topology, "cmap_atoms", np.empty((0, 0))))
    if cmap_atoms.ndim == 2 and cmap_atoms.size:
        keep_cmap = ~_rows_touching(cmap_atoms, ml_atoms)
        topo_updates["cmap_atoms"] = cmap_atoms[keep_cmap]
        cmap_idx = np.asarray(getattr(topology, "cmap_map_idx", np.empty(0)))
        if cmap_idx.shape[0] == cmap_atoms.shape[0]:
            topo_updates["cmap_map_idx"] = cmap_idx[keep_cmap]
        dropped["cmap"] = int((~keep_cmap).sum())

    topology = _replace(topology, **topo_updates)
    bonded = _replace(bonded, **param_updates)
    return topology, bonded, {"dropped": dropped, "angle_mask": masks.get("angles")}


@register_term("mm_bonded")
class MMBondedTerm:
    """CGenFF bonded energy on MM atoms (ML-region bonded interactions dropped)."""

    name = "mm_bonded"

    def __init__(
        self,
        psf_path: str | Path | None = None,
        prm_paths: Sequence[str | Path] = (),
        ml_atoms: Sequence[int] | None = None,
        include_cmap: bool = True,
        bonds: Any = None,
        bond_k: Any = None,
        bond_r0: Any = None,
        angles: Any = None,
        angle_k: Any = None,
        angle_theta0: Any = None,
        *,
        ml_atom_indices: Sequence[int] | None = None,
        extra_prm_files: Sequence[str | Path] = (),
    ):
        # Callers (ml_region mechanical embedding, md-system unified) pass
        # ``ml_atom_indices`` / ``extra_prm_files``; keep ``ml_atoms`` /
        # ``prm_paths`` as the canonical names.
        if ml_atoms is None and ml_atom_indices is not None:
            ml_atoms = ml_atom_indices
        if not prm_paths and extra_prm_files:
            prm_paths = extra_prm_files
        self.psf_path = Path(psf_path) if psf_path is not None else None
        self.prm_paths = tuple(Path(p) for p in prm_paths)
        self.ml_atoms = None if ml_atoms is None else frozenset(int(a) for a in ml_atoms)
        self.include_cmap = bool(include_cmap)
        # Explicit harmonic terms, for systems built without CHARMM. When given,
        # the PSF is not consulted at all: no live CHARMM state, no PSF round
        # trip, and the caller owns every number that enters the energy.
        self.bonds = None if bonds is None else np.asarray(bonds, dtype=np.int32)
        self.bond_k = None if bond_k is None else np.asarray(bond_k, dtype=np.float64)
        self.bond_r0 = None if bond_r0 is None else np.asarray(bond_r0, dtype=np.float64)
        self.angles = None if angles is None else np.asarray(angles, dtype=np.int32)
        self.angle_k = None if angle_k is None else np.asarray(angle_k, dtype=np.float64)
        self.angle_theta0 = (
            None if angle_theta0 is None else np.asarray(angle_theta0, dtype=np.float64)
        )
        if self.bonds is not None and (self.bond_k is None or self.bond_r0 is None):
            raise ValueError("explicit bonds require bond_k and bond_r0")
        if self.angles is not None and (self.angle_k is None or self.angle_theta0 is None):
            raise ValueError("explicit angles require angle_k and angle_theta0")

    @property
    def _explicit(self) -> bool:
        return self.bonds is not None or self.angles is not None

    def neighbor_request(self, system: MolecularSystem):
        return None  # bonded rows are a fixed index list, not a neighbor list

    def _resolve_psf(self, system: MolecularSystem) -> Path:
        candidates = [
            self.psf_path,
            system.psf_path,
            getattr(system.ff_params, "psf_path", None) if system.ff_params else None,
        ]
        for candidate in candidates:
            if candidate is not None and Path(candidate).is_file():
                return Path(candidate)
        raise ValueError(
            "mm_bonded needs a PSF: pass psf_path=..., or build the system with "
            "one (MolecularSystem.psf_path / FFParams.psf_path)."
        )

    def _resolve_prm(self, ctx: EnergyContext) -> tuple[Path, tuple[Path, ...]]:
        from mmml.interfaces.pycharmmInterface.cgenff_topology import default_cgenff_paths

        options = dict(getattr(ctx, "options", {}) or {})
        prm = options.get("cgenff_prm")
        if prm is None:
            _, prm = default_cgenff_paths()
        extra = tuple(self.prm_paths) or tuple(
            Path(p) for p in options.get("cgenff_extra_prm", ())
        )
        return Path(prm), extra

    def _resolve_ml_atoms(self, system: MolecularSystem, ctx: EnergyContext) -> frozenset[int]:
        if self.ml_atoms is not None:
            return self.ml_atoms
        options = dict(getattr(ctx, "options", {}) or {})
        from_ctx = options.get("ml_atoms")
        if from_ctx is None:
            from_ctx = options.get("ml_atom_indices")
        if from_ctx is not None:
            return frozenset(int(a) for a in from_ctx)
        return frozenset()

    def _make_explicit(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        """Harmonic bonds + angles from caller-supplied arrays (no CHARMM).

        Same functional form as the CHARMM terms -- E = k(r-r0)^2 and
        k(theta-theta0)^2, i.e. CHARMM's convention where k already absorbs the
        factor of 1/2 -- so parameters can be copied straight out of a CHARMM
        parameter file. Rows touching an ML atom are dropped exactly as in the
        PSF path.
        """
        import jax
        import jax.numpy as jnp

        ml_atoms = self._resolve_ml_atoms(system, ctx)
        bonds = self.bonds if self.bonds is not None else np.zeros((0, 2), np.int32)
        angles = self.angles if self.angles is not None else np.zeros((0, 3), np.int32)
        bond_k = self.bond_k if self.bond_k is not None else np.zeros(0)
        bond_r0 = self.bond_r0 if self.bond_r0 is not None else np.zeros(0)
        angle_k = self.angle_k if self.angle_k is not None else np.zeros(0)
        angle_t0 = self.angle_theta0 if self.angle_theta0 is not None else np.zeros(0)

        dropped = {}
        if ml_atoms:
            keep_b = ~_rows_touching(bonds, ml_atoms)
            keep_a = ~_rows_touching(angles, ml_atoms)
            dropped = {"bonds": int((~keep_b).sum()), "angles": int((~keep_a).sum())}
            bonds, bond_k, bond_r0 = bonds[keep_b], bond_k[keep_b], bond_r0[keep_b]
            angles, angle_k, angle_t0 = angles[keep_a], angle_k[keep_a], angle_t0[keep_a]

        for name, rows in (("bonds", bonds), ("angles", angles)):
            if rows.size and int(rows.max()) >= system.n_atoms:
                raise ValueError(
                    f"mm_bonded {name} reference atom {int(rows.max())} but the "
                    f"system has {system.n_atoms} atoms"
                )

        displacement_fn = resolve_displacement_fn(system, ctx)
        bi, bj = jnp.asarray(bonds[:, 0]), jnp.asarray(bonds[:, 1])
        ai, aj, ak = (jnp.asarray(angles[:, c]) for c in range(3))
        bk, br0 = jnp.asarray(bond_k), jnp.asarray(bond_r0)
        ak_, at0 = jnp.asarray(angle_k), jnp.asarray(np.deg2rad(angle_t0))
        has_b, has_a = bonds.shape[0] > 0, angles.shape[0] > 0

        def energy_fn(R, **kwargs):
            del kwargs
            total = jnp.asarray(0.0, dtype=R.dtype)
            if has_b:
                d = jax.vmap(displacement_fn)(R[bi], R[bj])
                r = jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-12)
                total = total + jnp.sum(bk * (r - br0) ** 2)
            if has_a:
                d1 = jax.vmap(displacement_fn)(R[ai], R[aj])
                d2 = jax.vmap(displacement_fn)(R[ak], R[aj])
                n1 = d1 / jnp.linalg.norm(d1, axis=-1, keepdims=True)
                n2 = d2 / jnp.linalg.norm(d2, axis=-1, keepdims=True)
                cos = jnp.clip(jnp.sum(n1 * n2, axis=-1), -1.0, 1.0)
                total = total + jnp.sum(ak_ * (jnp.arccos(cos) - at0) ** 2)
            return total * KCAL_MOL_TO_EV

        energy_fn.bonded_report = {  # type: ignore[attr-defined]
            "source": "explicit",
            "n_bonds": int(bonds.shape[0]),
            "n_angles": int(angles.shape[0]),
            "n_ml_atoms": len(ml_atoms),
            "dropped": dropped,
        }
        return TermFns(
            jax_energy_fn=energy_fn,
            ase_contribution=ase_contribution_from_jax(energy_fn),
            neighbor_request=None,
        )

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        if self._explicit:
            return self._make_explicit(system, ctx)

        import jax.numpy as jnp

        from mmml.interfaces.pycharmmInterface.cgenff_bonded import (
            bonded_energy_components,
        )
        from mmml.interfaces.pycharmmInterface.mm_system_energy import (
            load_bonded_system_from_psf,
        )

        psf_path = self._resolve_psf(system)
        prm, extra_prm = self._resolve_prm(ctx)
        bonded_system = load_bonded_system_from_psf(
            psf_path,
            np.asarray(system.R, dtype=np.float64),
            prm_file=prm,
            extra_prm_files=extra_prm,
        )

        topology = bonded_system.topology
        bonded = bonded_system.bonded
        urey_k = bonded_system.urey_k
        urey_r0 = bonded_system.urey_r0

        ml_atoms = self._resolve_ml_atoms(system, ctx)
        report: dict[str, Any] = {}
        if ml_atoms:
            n_angles_before = int(np.asarray(topology.angles).shape[0])
            topology, bonded, report = _drop_ml_rows(topology, bonded, ml_atoms)
            angle_mask = report.get("angle_mask")
            if angle_mask is not None and angle_mask.size == n_angles_before:
                if urey_k is not None:
                    urey_k = np.asarray(urey_k)[angle_mask]
                if urey_r0 is not None:
                    urey_r0 = np.asarray(urey_r0)[angle_mask]

        displacement_fn = resolve_displacement_fn(system, ctx)
        include_cmap = self.include_cmap and int(
            np.asarray(getattr(topology, "cmap_atoms", np.empty((0, 0)))).size
        ) > 0

        def energy_fn(R, **kwargs) -> Any:
            del kwargs  # bonded rows are static; no per-step neighbor arrays
            components = bonded_energy_components(
                jnp.asarray(R),
                topology,
                bonded,
                displacement_fn,
                urey_k=urey_k,
                urey_r0=urey_r0,
                include_cmap=include_cmap,
            )
            # cgenff_bonded works in kcal/mol; the rest of the stack is in eV.
            return components["total"] * KCAL_MOL_TO_EV

        energy_fn.bonded_report = {  # type: ignore[attr-defined]
            "psf": str(psf_path),
            "n_bonds": int(np.asarray(topology.bonds).shape[0]),
            "n_angles": int(np.asarray(topology.angles).shape[0]),
            "n_torsions": int(np.asarray(topology.torsions).shape[0]),
            "n_impropers": int(np.asarray(topology.impropers).shape[0]),
            "n_ml_atoms": len(ml_atoms),
            "dropped": report.get("dropped", {}),
        }

        return TermFns(
            jax_energy_fn=energy_fn,
            ase_contribution=ase_contribution_from_jax(energy_fn),
            neighbor_request=None,
        )
