"""CGenFF metadata for the fixed five-molecule dimer validation set."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from ase import Atoms


# Atom order matches ``mmml.analysis.dimer_molecules.MOLECULES`` exactly.
CGENFF_ATOM_TYPES: dict[str, tuple[str, ...]] = {
    "DCM": ("CG321", "CLGA1", "CLGA1", "HGA2", "HGA2"),
    "ACE": ("OG2D3", "CG2O5", "CG331", "CG331") + ("HGA3",) * 6,
    "BENZ": ("CG2R61",) * 6 + ("HGR61",) * 6,
    "TIP3": ("OT", "HT", "HT"),
    "MEOH": ("CG331", "OG311", "HGP1", "HGA3", "HGA3", "HGA3"),
}


@lru_cache(maxsize=4)
def load_cgenff_sigma_epsilon(
    prm_path: str,
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Return type indices, conventional sigma (A), and |epsilon| (kcal/mol).

    CHARMM stores ``Rmin/2`` in the fourth NONBONDED field.  The SpookyNet
    fixed-LJ implementation uses the conventional ``4*epsilon`` LJ form and
    Lorentz arithmetic sigma combination, so convert each atom type with::

        sigma = 2 * (Rmin/2) / 2**(1/6)
    """
    type_to_idx: dict[str, int] = {}
    sigmas: list[float] = []
    epsilons: list[float] = []
    in_nonbonded = False
    for raw in Path(prm_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        upper = line.upper()
        if upper.startswith("NONBONDED"):
            in_nonbonded = True
            continue
        if not in_nonbonded:
            continue
        if upper.startswith("NBFIX"):
            break
        parts = line.split("!", 1)[0].split()
        if len(parts) < 4:
            continue
        try:
            epsilon = abs(float(parts[2]))
            rmin_half = float(parts[3])
        except ValueError:
            continue
        atom_type = parts[0]
        if atom_type in type_to_idx:
            continue
        type_to_idx[atom_type] = len(sigmas)
        sigmas.append(2.0 * rmin_half / (2.0 ** (1.0 / 6.0)))
        epsilons.append(epsilon)
    if not type_to_idx:
        raise ValueError(f"No CHARMM NONBONDED records parsed from {prm_path}")
    return (
        type_to_idx,
        np.asarray(sigmas, dtype=np.float64),
        np.asarray(epsilons, dtype=np.float64),
    )


def attach_cgenff_dimer_metadata(
    atoms: Atoms,
    pair: tuple[str, str],
    fragments: tuple[np.ndarray, np.ndarray],
    *,
    prm_path: str | Path,
) -> None:
    """Attach the dynamic fixed-LJ inputs consumed by ``SpookyPhysNet``."""
    type_to_idx, sigmas, epsilons = load_cgenff_sigma_epsilon(str(Path(prm_path).resolve()))
    atom_type_idx = np.zeros(len(atoms), dtype=np.int32)
    mol_id = np.zeros(len(atoms), dtype=np.int32)
    for molecule_index, (label, indices) in enumerate(zip(pair, fragments, strict=True)):
        names = CGENFF_ATOM_TYPES[label]
        if len(names) != len(indices):
            raise ValueError(
                f"{label}: {len(names)} CGenFF types for {len(indices)} geometry atoms"
            )
        missing = sorted({name for name in names if name not in type_to_idx})
        if missing:
            raise KeyError(f"{label}: CGenFF types missing from parameter table: {missing}")
        atom_type_idx[np.asarray(indices, dtype=np.int32)] = [type_to_idx[name] for name in names]
        mol_id[np.asarray(indices, dtype=np.int32)] = molecule_index
    atoms.set_array("mol_id", mol_id)
    atoms.set_array("cgenff_type_idx", atom_type_idx)
    atoms.info["cgenff_master_sigmas"] = sigmas
    atoms.info["cgenff_master_epsilons"] = epsilons
    atoms.info["cgenff_radius_source"] = "CHARMM Rmin/2 converted to conventional sigma"

