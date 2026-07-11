"""Builder: assemble a ``MolecularSystem`` (with ``FFParams``) from a CHARMM PSF.

Given a PSF (+ CHARMM parameter files) and caller-supplied coordinates + atomic
numbers, this resolves the force-field state once via
``load_nonbonded_system_from_charmm`` and partitions atoms into molecules from
the PSF bond graph. It is the first concrete :class:`SystemBuilder` and the place
``FFParams`` is populated field-for-field from ``NonbondedSystemData``
(decision A; see ``docs/hybrid-mlmm-decomposition.md`` §2).

Coordinate/element acquisition is intentionally *not* here — the caller passes
``positions`` and ``atomic_numbers`` (e.g. via ``ase.io.read``), keeping the
builder deterministic and free of live-CHARMM coordinate state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mmml.md.builders._topology import (
    molecule_ids_from_bonds,
    monomer_indices_from_mol_id,
)
from mmml.md.system import FFParams, MolecularSystem, SystemSpec

__all__ = ["PsfSystemBuilder"]


class PsfSystemBuilder:
    """Build a :class:`MolecularSystem` from a PSF + params + coordinates.

    Expected ``SystemSpec.params`` keys:

    - ``psf_path`` (str/Path) — CHARMM PSF EXT file.
    - ``prm_paths`` (sequence) — CHARMM parameter files for the LJ tables
      (used when no matching PSF is loaded live in CHARMM).
    - ``positions`` — ``(N, 3)`` coordinates in Å.
    - ``atomic_numbers`` — ``(N,)`` atomic numbers Z.
    - ``box`` (optional) — ``(3, 3)`` cell, or None for free space.
    """

    name = "psf"

    def build(self, spec: SystemSpec) -> MolecularSystem:
        params = dict(spec.params)
        psf_path = Path(params["psf_path"])
        prm_paths = [Path(p) for p in params.get("prm_paths", ())]
        positions = np.asarray(params["positions"], dtype=np.float64)
        atomic_numbers = np.asarray(params["atomic_numbers"], dtype=np.int32)
        box = params.get("box")
        box = None if box is None else np.asarray(box, dtype=np.float64)

        n_atoms = int(positions.shape[0])
        if atomic_numbers.shape[0] != n_atoms:
            raise ValueError(
                f"positions has {n_atoms} atoms but atomic_numbers has "
                f"{atomic_numbers.shape[0]}"
            )

        # Lazy import: keep CHARMM/jax out of module import; only load when building.
        from mmml.interfaces.pycharmmInterface.mm_system_energy import (
            load_nonbonded_system_from_charmm,
        )

        nbdata = load_nonbonded_system_from_charmm(psf_path, *prm_paths)
        if nbdata.charges.shape[0] != n_atoms:
            raise ValueError(
                f"PSF {psf_path.name} has {nbdata.charges.shape[0]} atoms but "
                f"coordinates have {n_atoms}"
            )

        ff_params = FFParams.from_nonbonded_system_data(nbdata)

        bonds = np.asarray(nbdata.psf_bonds, dtype=np.int64) if nbdata.psf_bonds is not None \
            else np.zeros((0, 2), dtype=np.int64)
        mol_id = molecule_ids_from_bonds(n_atoms, bonds)
        monomer_indices = monomer_indices_from_mol_id(mol_id)

        return MolecularSystem(
            R=positions,
            Z=atomic_numbers,
            box=box,
            mol_id=mol_id,
            monomer_indices=monomer_indices,
            water_indices=[],
            psf_path=psf_path,
            ff_params=ff_params,
            metadata={"builder": self.name, "n_molecules": len(monomer_indices)},
        )
