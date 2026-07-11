"""Builders wrapping the legacy Packmol, PyXtal, and peptide-water backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from mmml.md.system import MolecularSystem, SystemSpec

__all__ = ["PackmolSystemBuilder", "PyxtalSystemBuilder", "PeptideWaterSystemBuilder"]


def _composition(spec: SystemSpec) -> list[tuple[str, int]]:
    if not spec.composition:
        raise ValueError(f"{spec.builder} builder requires SystemSpec.composition")
    # Lazy import keeps the shared MD package free of PyCHARMM at import time.
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import parse_composition

    return parse_composition(spec.composition)


def _box(spec: SystemSpec, params: dict[str, Any]) -> np.ndarray | None:
    raw = params.pop("box", None)
    if raw is not None:
        box = np.asarray(raw, dtype=np.float64)
    else:
        side = params.pop("box_side_A", spec.box_size)
        box = None if side is None else np.eye(3, dtype=np.float64) * float(side)
    if box is not None and box.shape != (3, 3):
        raise ValueError(f"box must have shape (3, 3), got {box.shape}")
    return box


def _groups(atoms_per_molecule: list[int], n_atoms: int) -> tuple[np.ndarray, list[np.ndarray]]:
    sizes = np.asarray(atoms_per_molecule, dtype=np.int64)
    if sizes.ndim != 1 or np.any(sizes <= 0) or int(sizes.sum()) != n_atoms:
        raise ValueError(
            f"atoms_per_molecule must be positive and sum to {n_atoms}; got {sizes.tolist()}"
        )
    mol_id = np.repeat(np.arange(len(sizes), dtype=np.int32), sizes)
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    groups = [np.arange(offsets[i], offsets[i + 1], dtype=np.int32) for i in range(len(sizes))]
    return mol_id, groups


def _placement_system(
    *,
    name: str,
    spec: SystemSpec,
    z: Any,
    positions: Any,
    atoms_per_molecule: list[int],
    residue_names: list[str],
    box: np.ndarray | None,
) -> MolecularSystem:
    R = np.asarray(positions, dtype=np.float64)
    Z = np.asarray(z, dtype=np.int32)
    if R.ndim != 2 or R.shape[1] != 3 or Z.shape != (R.shape[0],):
        raise ValueError(f"backend returned incompatible positions {R.shape} and Z {Z.shape}")
    mol_id, groups = _groups(atoms_per_molecule, len(R))
    if len(residue_names) != len(groups):
        raise ValueError("residue_names and atoms_per_molecule must have equal length")
    waters = [g for g, residue in zip(groups, residue_names, strict=True) if residue.upper() in {"TIP3", "HOH", "WAT"}]
    return MolecularSystem(
        R=R,
        Z=Z,
        box=box,
        mol_id=mol_id,
        monomer_indices=groups,
        water_indices=waters,
        metadata={"builder": name, "residue_names": tuple(residue_names)},
    )


def _lower_optional_psf(
    system: MolecularSystem,
    *,
    psf_path: Any,
    prm_paths: Any,
    psf_builder: Any = None,
) -> MolecularSystem:
    if psf_path is None:
        return system
    if psf_builder is None:
        from mmml.md.builders.psf import PsfSystemBuilder

        psf_builder = PsfSystemBuilder()
    lowered = psf_builder.build(
        SystemSpec(
            builder="psf",
            params={
                "psf_path": psf_path,
                "prm_paths": prm_paths or (),
                "positions": system.R,
                "atomic_numbers": system.Z,
                "box": system.box,
            },
        )
    )
    return MolecularSystem(
        R=lowered.R,
        Z=lowered.Z,
        box=lowered.box,
        mol_id=lowered.mol_id,
        monomer_indices=lowered.monomer_indices,
        water_indices=system.water_indices,
        psf_path=lowered.psf_path,
        ff_params=lowered.ff_params,
        metadata={**system.metadata, "ff_source": "psf"},
    )


@dataclass(frozen=True)
class PackmolSystemBuilder:
    """Wrap ``build_packmol_composition_cluster`` behind ``SystemBuilder``."""

    build_fn: Callable[..., Any] | None = None
    name: str = "packmol"

    def build(self, spec: SystemSpec) -> MolecularSystem:
        params = dict(spec.params)
        psf_path = params.pop("psf_path", None)
        prm_paths = params.pop("prm_paths", ())
        box = _box(spec, params)
        composition = _composition(spec)
        if "center" not in params:
            if box is None:
                raise ValueError("packmol builder requires center or a box size")
            params["center"] = tuple(np.diag(box) / 2.0)
        params.setdefault("cube_side", spec.box_size)
        fn = self.build_fn
        if fn is None:
            from mmml.cli.run.md_pbc_suite.cluster import build_packmol_composition_cluster

            fn = build_packmol_composition_cluster
        z, positions, sizes, residues = fn(composition=composition, seed=spec.seed, **params)
        system = _placement_system(
            name=self.name,
            spec=spec,
            z=z,
            positions=positions,
            atoms_per_molecule=list(sizes),
            residue_names=list(residues),
            box=box,
        )
        return _lower_optional_psf(system, psf_path=psf_path, prm_paths=prm_paths)


@dataclass(frozen=True)
class PyxtalSystemBuilder:
    """Wrap ``build_pyxtal_composition_cluster`` behind ``SystemBuilder``."""

    build_fn: Callable[..., Any] | None = None
    name: str = "pyxtal"

    def build(self, spec: SystemSpec) -> MolecularSystem:
        params = dict(spec.params)
        psf_path = params.pop("psf_path", None)
        prm_paths = params.pop("prm_paths", ())
        box = _box(spec, params)
        composition = _composition(spec)
        fn = self.build_fn
        if fn is None:
            from mmml.cli.run.md_pbc_suite.cluster import build_pyxtal_composition_cluster

            fn = build_pyxtal_composition_cluster
        z, positions, sizes, residues = fn(composition=composition, seed=spec.seed, **params)
        system = _placement_system(
            name=self.name,
            spec=spec,
            z=z,
            positions=positions,
            atoms_per_molecule=list(sizes),
            residue_names=list(residues),
            box=box,
        )
        return _lower_optional_psf(system, psf_path=psf_path, prm_paths=prm_paths)


@dataclass(frozen=True)
class PeptideWaterSystemBuilder:
    """Wrap the trialanine-water builder and lower its emitted PSF to FFParams."""

    build_fn: Callable[..., Any] | None = None
    atomic_numbers_fn: Callable[[], Any] | None = None
    psf_builder: Any = None
    name: str = "peptide_water"

    def build(self, spec: SystemSpec) -> MolecularSystem:
        params = dict(spec.params)
        peptide_n = int(params.pop("peptide_n_atoms", 42))
        if spec.box_size is not None:
            params.setdefault("box_side_A", spec.box_size)
        if spec.n_molecules is not None:
            params.setdefault("n_waters", spec.n_molecules)
        fn = self.build_fn
        if fn is None:
            from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
                build_trialanine_water_box_in_charmm,
            )

            fn = build_trialanine_water_box_in_charmm
        result = fn(seed=spec.seed, **params)
        get_z = self.atomic_numbers_fn
        if get_z is None:
            from mmml.interfaces.pycharmmInterface.utils import get_Z_from_psf

            get_z = get_Z_from_psf
        z = np.asarray(get_z(), dtype=np.int32)
        builder = self.psf_builder
        if builder is None:
            from mmml.md.builders.psf import PsfSystemBuilder

            builder = PsfSystemBuilder()
        prm_paths = [result.cgenff_prm, *result.cmap_extra_prm_files]
        system = builder.build(
            SystemSpec(
                builder="psf",
                params={
                    "psf_path": result.psf_path,
                    "prm_paths": prm_paths,
                    "positions": result.positions,
                    "atomic_numbers": z,
                    "box": np.eye(3) * float(result.box_side_A),
                },
            )
        )
        waters = [g for g in system.monomer_indices if len(g) == 3 and int(g[0]) >= peptide_n]
        return MolecularSystem(
            R=system.R,
            Z=system.Z,
            box=system.box,
            mol_id=system.mol_id,
            monomer_indices=system.monomer_indices,
            water_indices=waters,
            psf_path=system.psf_path,
            ff_params=system.ff_params,
            metadata={**system.metadata, "builder": self.name, "n_waters": int(result.n_waters)},
        )
