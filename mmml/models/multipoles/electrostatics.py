"""Prototype electrostatics from learned QCML molecular multipoles.

The current implementation uses the learned molecular monopole and dipole only.
Inputs are ASE structures in Angstrom. Internally, fragment origins and
interaction vectors are converted to Bohr, and electrostatic energies are
computed in atomic units:

    charge: e
    dipole: e * bohr
    distance: bohr
    energy: hartree

ASE-facing energies are returned in eV. Electric fields are returned in atomic
units by default, with helpers for V/Angstrom.
"""

from __future__ import annotations

import json
from dataclasses import fields
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any, Iterable, Sequence

import e3x
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

try:
    from ase import Atom, Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.units import Bohr, Hartree
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without ASE.
    raise ModuleNotFoundError(
        "QCML multipole electrostatics prototype requires ASE."
    ) from exc

from mmml.models.multipoles.model import E3xMultipoleModel
from mmml.models.multipoles.config import TrainConfig
from mmml.models.multipoles.representations import irrep_blocks_to_traceless

ANGSTROM_TO_BOHR = 1.0 / Bohr
BOHR_TO_ANGSTROM = Bohr
HARTREE_TO_EV = Hartree
AU_POTENTIAL_TO_V = Hartree
AU_FIELD_TO_V_PER_ANGSTROM = 51.4220674763259


def _load_checkpoint_payload(checkpoint: str | Path) -> dict[str, Any]:
    return ocp.PyTreeCheckpointer().restore(Path(checkpoint).expanduser())


def _multipole_config_from_params(params: Any) -> dict[str, int]:
    """Infer architecture kwargs from a multipole checkpoint's weight shapes.

    Portable exports sometimes ship params only. The atom embedding fixes
    ``(max_atomic_number + 1, ..., features)``, the number of ``MessagePass_*``
    submodules fixes ``num_iterations``, the filter kernel's first axis fixes
    ``num_basis_functions``, and the highest irrep degree present fixes
    ``max_degree`` -- enough to rebuild the backbone without a config.
    """
    backbone = params.get("_E3xMultipoleBackbone_0", params) if isinstance(params, dict) else {}
    cfg: dict[str, int] = {}
    embed = backbone.get("Embed_0", {}).get("embedding")
    if embed is not None:
        emb = np.asarray(embed)
        cfg["max_atomic_number"] = int(emb.shape[0]) - 1
        cfg["features"] = int(emb.shape[-1])
    n_mp = sum(1 for key in backbone if key.startswith("MessagePass_"))
    if n_mp:
        cfg["num_iterations"] = n_mp
    mp0 = backbone.get("MessagePass_0", {}).get("filter", {})
    degrees = [k for k in mp0 if k[:1].isdigit()]
    if degrees:
        cfg["max_degree"] = max(int(k[:-1]) for k in degrees)
        any_kernel = np.asarray(mp0[degrees[0]]["kernel"])
        cfg["num_basis_functions"] = int(any_kernel.shape[0])
    return cfg


def load_multipole_model(checkpoint: str | Path) -> tuple[E3xMultipoleModel, Any]:
    """Load a trained unified QCML multipole model checkpoint.

    Accepts either an Orbax checkpoint directory (with a sibling
    ``model_config.json``) or a portable JSON file produced by
    :func:`mmml.utils.model_checkpoint.orbax_to_json` (params + config bundled
    together, no sibling file needed). A JSON export that shipped params only
    (no ``config``) is rebuilt from the weight shapes with a warning.
    """
    checkpoint = Path(checkpoint).expanduser()
    valid = {field.name for field in fields(TrainConfig)}
    if checkpoint.is_file() and checkpoint.suffix == ".json":
        from mmml.utils.model_checkpoint import json_to_params

        restored = json_to_params(checkpoint)
        raw_config = restored.get("config")
        if not raw_config:
            import warnings

            inferred = _multipole_config_from_params(restored.get("params"))
            if "features" not in inferred:
                raise ValueError(
                    f"JSON checkpoint {checkpoint} has no 'config' and its architecture "
                    "could not be inferred from the weight shapes; re-export with config."
                )
            warnings.warn(
                f"Multipole checkpoint {checkpoint} has no 'config'; inferred "
                f"architecture {inferred} from the saved weight shapes (cutoff and "
                "output flags fall back to E3xMultipoleModel defaults). Re-export "
                "with config to remove the guess.",
                stacklevel=2,
            )
            raw_config = inferred
        model_config = {key: value for key, value in raw_config.items() if key in valid}
        TrainConfig(**model_config)
        model_config.pop("target_degree", None)
        if "params" not in restored:
            raise KeyError(f"Checkpoint {checkpoint} does not contain params")
        return E3xMultipoleModel(**model_config), restored["params"]

    config_path = checkpoint / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model_config.json: {config_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = {key: value for key, value in raw_config.items() if key in valid}
    TrainConfig(**model_config)
    model_config.pop("target_degree", None)
    payload = _load_checkpoint_payload(checkpoint)
    return E3xMultipoleModel(**model_config), payload["params"]


def _indices_from_atoms_subset(parent: Atoms, subset: Atoms) -> np.ndarray:
    """Map an ASE Atoms subset back to parent indices by Z and coordinates."""
    parent_numbers = np.asarray(parent.get_atomic_numbers())
    parent_positions = np.asarray(parent.get_positions(), dtype=np.float64)
    subset_numbers = np.asarray(subset.get_atomic_numbers())
    subset_positions = np.asarray(subset.get_positions(), dtype=np.float64)
    used: set[int] = set()
    indices = []
    for number, position in zip(subset_numbers, subset_positions, strict=True):
        candidates = np.flatnonzero(
            (parent_numbers == number)
            & np.all(np.isclose(parent_positions, position[None, :], atol=1e-8), axis=1)
        )
        match = next((int(candidate) for candidate in candidates if int(candidate) not in used), None)
        if match is None:
            raise ValueError(
                "Could not map an ASE Atoms fragment back to parent atom indices. "
                "Pass explicit integer index lists when duplicate atoms/geometries are ambiguous."
            )
        used.add(match)
        indices.append(match)
    return np.asarray(indices, dtype=np.int64)


def _coerce_fragment_indices(parent: Atoms, fragment: Any) -> np.ndarray:
    if isinstance(fragment, Atoms):
        return _indices_from_atoms_subset(parent, fragment)
    if isinstance(fragment, np.ndarray) and fragment.dtype == bool:
        if fragment.shape[0] != len(parent):
            raise ValueError("Boolean fragment masks must have length len(atoms)")
        return np.flatnonzero(fragment).astype(np.int64)
    if isinstance(fragment, Atom):
        return np.asarray([fragment.index], dtype=np.int64)
    if isinstance(fragment, SequenceABC) and not isinstance(fragment, (str, bytes)):
        items = list(fragment)
        if items and all(isinstance(item, Atom) for item in items):
            return np.asarray([atom.index for atom in items], dtype=np.int64)
    try:
        indices = np.asarray(fragment, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Fragments must be integer index sequences, boolean masks, ASE Atoms "
            "slices, or sequences of ASE Atom objects."
        ) from exc
    if indices.ndim != 1:
        raise ValueError("Each fragment must be one-dimensional")
    return indices


def fragment_indices_from_atoms(
    atoms: Atoms,
    fragments: Sequence[Sequence[int]] | None = None,
    *,
    mol_id_array: str = "mol_id",
) -> list[np.ndarray]:
    """Return fragment atom indices.

    If explicit fragments are not provided, ``atoms.arrays[mol_id_array]`` is
    used when present. Otherwise the full structure is treated as one fragment.
    """
    if fragments is not None:
        return [_coerce_fragment_indices(atoms, fragment) for fragment in fragments]
    if mol_id_array in atoms.arrays:
        mol_ids = np.asarray(atoms.arrays[mol_id_array])
        return [
            np.flatnonzero(mol_ids == mol_id)
            for mol_id in np.unique(mol_ids)
        ]
    return [np.arange(len(atoms), dtype=np.int64)]


def fragment_origin_bohr(
    atoms: Atoms,
    indices: np.ndarray,
    *,
    origin: str = "nuclear_charge_centroid",
) -> np.ndarray:
    """Compute a fragment origin in Bohr from ASE Angstrom coordinates."""
    positions = np.asarray(atoms.get_positions()[indices], dtype=np.float64)
    if origin == "geometric_centroid":
        origin_angstrom = np.mean(positions, axis=0)
    elif origin == "center_of_mass":
        masses = np.asarray(atoms.get_masses()[indices], dtype=np.float64)
        origin_angstrom = np.average(positions, axis=0, weights=masses)
    elif origin == "nuclear_charge_centroid":
        weights = np.asarray(atoms.get_atomic_numbers()[indices], dtype=np.float64)
        origin_angstrom = np.average(positions, axis=0, weights=weights)
    else:
        raise ValueError(
            "origin must be one of: geometric_centroid, center_of_mass, "
            "nuclear_charge_centroid"
        )
    return origin_angstrom * ANGSTROM_TO_BOHR


def atoms_fragment_to_model_batch(
    atoms: Atoms,
    indices: np.ndarray,
    origin_bohr: np.ndarray,
    *,
    charge: float = 0.0,
    multiplicity: float = 1.0,
) -> dict[str, jax.Array]:
    """Build a batch_size=1 E3x batch for a fragment, centered at origin."""
    positions_bohr = (
        np.asarray(atoms.get_positions()[indices], dtype=np.float32)
        * ANGSTROM_TO_BOHR
        - np.asarray(origin_bohr, dtype=np.float32)[None, :]
    )
    atomic_numbers = np.asarray(atoms.get_atomic_numbers()[indices], dtype=np.int32)
    num_atoms = len(atomic_numbers)
    dst_idx, src_idx = map(np.asarray, e3x.ops.sparse_pairwise_indices(num_atoms))
    return {
        "positions": jnp.asarray(positions_bohr.reshape(-1, 3)),
        "atomic_numbers": jnp.asarray(atomic_numbers.reshape(-1)),
        "charge": jnp.asarray([charge], dtype=jnp.float32),
        "spin": jnp.asarray([multiplicity], dtype=jnp.float32),
        "dst_idx": jnp.asarray(dst_idx, dtype=jnp.int32),
        "src_idx": jnp.asarray(src_idx, dtype=jnp.int32),
        "batch_segments": jnp.zeros(num_atoms, dtype=jnp.int32),
        "batch_size": 1,
        "atom_mask": jnp.ones(num_atoms, dtype=jnp.float32),
        "edge_mask": jnp.ones(len(dst_idx), dtype=jnp.float32),
    }


def _point_multipole_potential_field_au(
    points_bohr: np.ndarray,
    origins_bohr: np.ndarray,
    charges: np.ndarray,
    dipoles_bohr: np.ndarray,
    quadrupoles_bohr: np.ndarray | None = None,
    octupoles_bohr: np.ndarray | None = None,
    *,
    exclude_index: int | None = None,
    softening_bohr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Potential and electric field from point multipole sources (up to rank 3).

    Potential is in Hartree/e. Electric field is in atomic units,
    Hartree/(e*bohr).
    """
    points = np.asarray(points_bohr, dtype=np.float64)
    origins = np.asarray(origins_bohr, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64).reshape(-1)
    dipoles = np.asarray(dipoles_bohr, dtype=np.float64)
    n_sources = len(origins)

    if quadrupoles_bohr is None:
        quadrupoles = np.zeros((n_sources, 3, 3), dtype=np.float64)
    else:
        quadrupoles = np.asarray(quadrupoles_bohr, dtype=np.float64)

    if octupoles_bohr is None:
        octupoles = np.zeros((n_sources, 3, 3, 3), dtype=np.float64)
    else:
        octupoles = np.asarray(octupoles_bohr, dtype=np.float64)

    potential = np.zeros(points.shape[0], dtype=np.float64)
    field = np.zeros((points.shape[0], 3), dtype=np.float64)
    softening2 = float(softening_bohr) ** 2
    for source_index in range(n_sources):
        if exclude_index is not None and source_index == exclude_index:
            continue
        origin = origins[source_index]
        charge = charges[source_index]
        dipole = dipoles[source_index]
        quadrupole = quadrupoles[source_index]
        octupole = octupoles[source_index]

        displacement = points - origin[None, :]
        r2 = np.sum(displacement * displacement, axis=1) + softening2
        r = np.sqrt(r2)
        inv_r = 1.0 / np.maximum(r, 1e-12)
        inv_r3 = inv_r**3
        inv_r5 = inv_r**5
        inv_r7 = inv_r**7
        inv_r9 = inv_r**9

        # Dot products and contractions with displacement
        mu_dot_r = displacement @ dipole
        Q_r = displacement @ quadrupole
        r_Q_r = np.sum(displacement * Q_r, axis=1)
        O_r = np.tensordot(displacement, octupole, axes=(1, 2))
        O_rr = np.sum(O_r * displacement[:, :, None], axis=1)
        O_rrr = np.sum(O_rr * displacement, axis=1)

        # Potential terms
        potential += charge * inv_r + mu_dot_r * inv_r3 + 1.5 * r_Q_r * inv_r5 + 2.5 * O_rrr * inv_r7

        # Field terms (E = -grad V)
        field += charge * displacement * inv_r3[:, None]
        field += (
            3.0 * displacement * mu_dot_r[:, None] * inv_r5[:, None]
            - dipole[None, :] * inv_r3[:, None]
        )
        field += (
            7.5 * displacement * r_Q_r[:, None] * inv_r7[:, None]
            - 3.0 * Q_r * inv_r5[:, None]
        )
        field += (
            17.5 * displacement * O_rrr[:, None] * inv_r9[:, None]
            - 7.5 * O_rr * inv_r7[:, None]
        )

    return potential, field


def _point_charge_dipole_potential_field_au(
    points_bohr: np.ndarray,
    origins_bohr: np.ndarray,
    charges: np.ndarray,
    dipoles_bohr: np.ndarray,
    *,
    exclude_index: int | None = None,
    softening_bohr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Potential and electric field from point charge+dipole sources."""
    return _point_multipole_potential_field_au(
        points_bohr,
        origins_bohr,
        charges,
        dipoles_bohr,
        None,
        None,
        exclude_index=exclude_index,
        softening_bohr=softening_bohr,
    )


def pair_energy_multipole_au(
    origin_a_bohr: np.ndarray,
    charge_a: float,
    dipole_a_bohr: np.ndarray,
    quadrupole_a_bohr: np.ndarray,
    octupole_a_bohr: np.ndarray,
    origin_b_bohr: np.ndarray,
    charge_b: float,
    dipole_b_bohr: np.ndarray,
    quadrupole_b_bohr: np.ndarray,
    octupole_b_bohr: np.ndarray,
    *,
    softening_bohr: float = 0.0,
    return_components: bool = False,
) -> float | dict[str, float]:
    """Symmetric multipole-multipole pair interaction in Hartree (up to rank 3).

    ``R`` points from A to B. The formula is the atomic-unit interaction of two
    point multipole objects (charge + dipole + quadrupole + octupole).
    """
    r_vec = np.asarray(origin_b_bohr, dtype=np.float64) - np.asarray(origin_a_bohr, dtype=np.float64)
    r2 = float(np.dot(r_vec, r_vec) + softening_bohr**2)
    r = max(np.sqrt(r2), 1e-12)
    inv_r = 1.0 / r
    inv_r3 = inv_r**3
    inv_r5 = inv_r**5
    inv_r7 = inv_r**7
    inv_r9 = inv_r**9
    inv_r11 = inv_r**11
    inv_r13 = inv_r**13

    # Dipole contractions
    p_a_r = float(np.dot(dipole_a_bohr, r_vec))
    p_b_r = float(np.dot(dipole_b_bohr, r_vec))

    # Quadrupole contractions with r
    Q_a_r = np.asarray(quadrupole_a_bohr, dtype=np.float64) @ r_vec
    Q_b_r = np.asarray(quadrupole_b_bohr, dtype=np.float64) @ r_vec
    r_Q_a_r = float(np.dot(r_vec, Q_a_r))
    r_Q_b_r = float(np.dot(r_vec, Q_b_r))

    # Octupole contractions with r
    oct_a = np.asarray(octupole_a_bohr, dtype=np.float64)
    oct_b = np.asarray(octupole_b_bohr, dtype=np.float64)
    O_a_r = np.tensordot(oct_a, r_vec, axes=(2, 0))
    O_b_r = np.tensordot(oct_b, r_vec, axes=(2, 0))
    O_a_rr = O_a_r @ r_vec
    O_b_rr = O_b_r @ r_vec
    O_a_rrr = float(np.dot(O_a_rr, r_vec))
    O_b_rrr = float(np.dot(O_b_rr, r_vec))

    # 1. Monopole-Monopole (0-0)
    e_00 = charge_a * charge_b * inv_r

    # 2. Monopole-Dipole (0-1)
    e_01 = (charge_b * p_a_r - charge_a * p_b_r) * inv_r3

    # 3. Dipole-Dipole (1-1)
    e_11 = (np.dot(dipole_a_bohr, dipole_b_bohr) * inv_r3 - 3.0 * p_a_r * p_b_r * inv_r5)

    # 4. Monopole-Quadrupole (0-2)
    e_02 = 1.5 * (charge_b * r_Q_a_r + charge_a * r_Q_b_r) * inv_r5

    # 5. Dipole-Quadrupole (1-2)
    e_12 = (3.0 * (np.dot(dipole_b_bohr, Q_a_r) - np.dot(dipole_a_bohr, Q_b_r)) * inv_r5
            - 7.5 * (r_Q_a_r * p_b_r - r_Q_b_r * p_a_r) * inv_r7)

    # 6. Quadrupole-Quadrupole (2-2)
    e_22 = (1.5 * np.sum(quadrupole_a_bohr * quadrupole_b_bohr) * inv_r5
            - 15.0 * np.dot(Q_a_r, Q_b_r) * inv_r7
            + 26.25 * r_Q_a_r * r_Q_b_r * inv_r9)

    # 7. Monopole-Octupole (0-3)
    e_03 = 2.5 * (charge_b * O_a_rrr - charge_a * O_b_rrr) * inv_r7

    # 8. Dipole-Octupole (1-3)
    e_13 = (7.5 * (np.dot(dipole_b_bohr, O_a_rr) + np.dot(dipole_a_bohr, O_b_rr)) * inv_r7
            - 17.5 * (O_a_rrr * p_b_r + O_b_rrr * p_a_r) * inv_r9)

    # 9. Quadrupole-Octupole (2-3)
    e_23 = (7.5 * (np.sum(quadrupole_b_bohr * O_a_r) - np.sum(quadrupole_a_bohr * O_b_r)) * inv_r7
            - 52.5 * (np.dot(O_a_rr, Q_b_r) - np.dot(O_b_rr, Q_a_r)) * inv_r9
            + 78.75 * (O_a_rrr * r_Q_b_r - O_b_rrr * r_Q_a_r) * inv_r11)

    # 10. Octupole-Octupole (3-3)
    e_33 = (2.5 * np.sum(oct_a * oct_b) * inv_r7
            - 52.5 * np.sum(O_a_r * O_b_r) * inv_r9
            + 236.25 * np.dot(O_a_rr, O_b_rr) * inv_r11
            - 288.75 * O_a_rrr * O_b_rrr * inv_r13)

    total = float(e_00 + e_01 + e_11 + e_02 + e_12 + e_22 + e_03 + e_13 + e_23 + e_33)
    if return_components:
        return {
            "0-0": float(e_00),
            "0-1": float(e_01),
            "1-1": float(e_11),
            "0-2": float(e_02),
            "1-2": float(e_12),
            "2-2": float(e_22),
            "0-3": float(e_03),
            "1-3": float(e_13),
            "2-3": float(e_23),
            "3-3": float(e_33),
            "total": total,
        }
    return total


def pair_energy_charge_dipole_au(
    origin_a_bohr: np.ndarray,
    charge_a: float,
    dipole_a_bohr: np.ndarray,
    origin_b_bohr: np.ndarray,
    charge_b: float,
    dipole_b_bohr: np.ndarray,
    *,
    softening_bohr: float = 0.0,
) -> float:
    """Symmetric q+dipole pair interaction in Hartree."""
    return pair_energy_multipole_au(
        origin_a_bohr,
        charge_a,
        dipole_a_bohr,
        np.zeros((3, 3)),
        np.zeros((3, 3, 3)),
        origin_b_bohr,
        charge_b,
        dipole_b_bohr,
        np.zeros((3, 3)),
        np.zeros((3, 3, 3)),
        softening_bohr=softening_bohr,
    )


def resolve_multipoles_checkpoint(explicit: str | Path | None = None) -> Path:
    """Resolve default Multipoles checkpoint path."""
    import os

    env_ckpt = (os.environ.get("MULTIPOLES_CKPT") or os.environ.get("MULTIPOLES_CHECKPOINT") or "").strip()
    target = explicit or (env_ckpt if env_ckpt else None)
    if target is not None:
        p = Path(target).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Multipoles checkpoint not found at: {target}")
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "multipoles_20260711-100037_epoch-0100.json",
        repo_root / "examples" / "multipoles_20260711-100037_epoch-0100.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "No default Multipoles checkpoint found. Set MULTIPOLES_CKPT or pass explicit path."
    )


class LearnedMolecularMultipoleElectrostatics(Calculator):
    """ASE calculator for learned molecular multipoles.

    The model predicts one molecular multipole vector per fragment. Current
    energy includes only l=0 and l=1. l=2/l=3 are stored in results for later
    extension.
    """

    implemented_properties = ["energy"]

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        fragments: Sequence[Sequence[int]] | None = None,
        charges: Sequence[float] | None = None,
        multiplicities: Sequence[float] | None = None,
        origin: str = "nuclear_charge_centroid",
        mol_id_array: str = "mol_id",
        softening_bohr: float = 0.0,
        max_ell: int = 3,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        resolved_ckpt = resolve_multipoles_checkpoint(checkpoint)
        self.model, self.params = load_multipole_model(resolved_ckpt)
        self.fragments = fragments
        self.charges = None if charges is None else np.asarray(charges, dtype=np.float64)
        self.multiplicities = (
            None if multiplicities is None else np.asarray(multiplicities, dtype=np.float64)
        )
        self.origin = origin
        self.mol_id_array = mol_id_array
        self.softening_bohr = float(softening_bohr)
        self.max_ell = int(max_ell)
        self._predict = jax.jit(self._predict_impl)

    def _predict_impl(self, batch: dict[str, jax.Array]) -> jax.Array:
        output = self.model.apply(
            {"params": self.params},
            positions=batch["positions"],
            atomic_numbers=batch["atomic_numbers"],
            charge=batch["charge"],
            spin=batch["spin"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=1,
            atom_mask=batch["atom_mask"],
            edge_mask=batch["edge_mask"],
        )
        return output["multipoles"][0]

    def predict_fragment_multipoles(
        self,
        atoms: Atoms,
    ) -> dict[str, np.ndarray | list[np.ndarray]]:
        fragments = fragment_indices_from_atoms(
            atoms,
            self.fragments,
            mol_id_array=self.mol_id_array,
        )
        n_fragments = len(fragments)
        charges = (
            np.zeros(n_fragments, dtype=np.float64)
            if self.charges is None
            else np.asarray(self.charges, dtype=np.float64)
        )
        multiplicities = (
            np.ones(n_fragments, dtype=np.float64)
            if self.multiplicities is None
            else np.asarray(self.multiplicities, dtype=np.float64)
        )
        if len(charges) != n_fragments or len(multiplicities) != n_fragments:
            raise ValueError("charges and multiplicities must match number of fragments")

        origins = []
        multipoles = []
        for index, fragment in enumerate(fragments):
            origin_bohr = fragment_origin_bohr(atoms, fragment, origin=self.origin)
            batch = atoms_fragment_to_model_batch(
                atoms,
                fragment,
                origin_bohr,
                charge=float(charges[index]),
                multiplicity=float(multiplicities[index]),
            )
            origins.append(origin_bohr)
            multipoles.append(np.asarray(self._predict(batch), dtype=np.float64))
        multipoles_array = np.asarray(multipoles, dtype=np.float64)
        tensors = irrep_blocks_to_traceless(multipoles_array)
        return {
            "fragments": fragments,
            "origins_bohr": np.asarray(origins, dtype=np.float64),
            "origins_angstrom": np.asarray(origins, dtype=np.float64) * BOHR_TO_ANGSTROM,
            "multipoles": multipoles_array,
            "charges": multipoles_array[:, 0],
            "dipoles_bohr": multipoles_array[:, 1:4],
            "quadrupoles_bohr": np.asarray(tensors["l2_quadrupole_tensor"], dtype=np.float64),
            "octupoles_bohr": np.asarray(tensors["l3_octupole_tensor"], dtype=np.float64),
        }

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Iterable[str] = ("energy",),
        system_changes: Iterable[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        prediction = self.predict_fragment_multipoles(self.atoms)
        origins = np.asarray(prediction["origins_bohr"], dtype=np.float64)
        charges = np.asarray(prediction["charges"], dtype=np.float64)
        dipoles = np.asarray(prediction["dipoles_bohr"], dtype=np.float64)
        quadrupoles = np.asarray(prediction["quadrupoles_bohr"], dtype=np.float64)
        octupoles = np.asarray(prediction["octupoles_bohr"], dtype=np.float64)

        # Mask moments based on max_ell
        if self.max_ell < 3:
            octupoles = np.zeros_like(octupoles)
        if self.max_ell < 2:
            quadrupoles = np.zeros_like(quadrupoles)
        if self.max_ell < 1:
            dipoles = np.zeros_like(dipoles)
        if self.max_ell < 0:
            charges = np.zeros_like(charges)

        energy_hartree = 0.0
        pair_rows = []
        pair_component_rows_ha = []
        pair_component_rows_ev = []
        for i in range(len(charges)):
            for j in range(i + 1, len(charges)):
                components = pair_energy_multipole_au(
                    origins[i],
                    charges[i],
                    dipoles[i],
                    quadrupoles[i],
                    octupoles[i],
                    origins[j],
                    charges[j],
                    dipoles[j],
                    quadrupoles[j],
                    octupoles[j],
                    softening_bohr=self.softening_bohr,
                    return_components=True,
                )
                pair_energy = components["total"]
                energy_hartree += pair_energy
                pair_rows.append((i, j, pair_energy, pair_energy * HARTREE_TO_EV))
                
                pair_comp_ha = {"pair": (i, j), **components}
                pair_comp_ev = {"pair": (i, j)}
                for key, val in components.items():
                    pair_comp_ev[key] = val * HARTREE_TO_EV
                pair_component_rows_ha.append(pair_comp_ha)
                pair_component_rows_ev.append(pair_comp_ev)

        # Calculate potentials at origins decomposed by component
        n_fragments = len(charges)
        potentials_au = np.zeros((n_fragments, 4), dtype=np.float64)
        for i in range(n_fragments):
            for j in range(n_fragments):
                if i == j:
                    continue
                r_vec = origins[i] - origins[j]
                r2 = np.dot(r_vec, r_vec) + self.softening_bohr**2
                r_val = np.sqrt(r2)
                inv_r = 1.0 / max(r_val, 1e-12)
                
                # l=0
                pot_0 = charges[j] * inv_r
                # l=1
                p_r = np.dot(dipoles[j], r_vec)
                pot_1 = p_r * inv_r**3
                # l=2
                Q_r = quadrupoles[j] @ r_vec
                r_Q_r = np.dot(r_vec, Q_r)
                pot_2 = 1.5 * r_Q_r * inv_r**5
                # l=3
                O_r = np.tensordot(octupoles[j], r_vec, axes=(2, 0))
                O_rr = O_r @ r_vec
                O_rrr = np.dot(O_rr, r_vec)
                pot_3 = 2.5 * O_rrr * inv_r**7
                
                potentials_au[i, 0] += pot_0
                potentials_au[i, 1] += pot_1
                potentials_au[i, 2] += pot_2
                potentials_au[i, 3] += pot_3

        self.results["energy"] = energy_hartree * HARTREE_TO_EV
        self.results["energy_hartree"] = energy_hartree
        self.results["pair_energies"] = pair_rows
        self.results["pair_energies_by_component_hartree"] = pair_component_rows_ha
        self.results["pair_energies_by_component_ev"] = pair_component_rows_ev
        self.results["pair_energies_by_component"] = pair_component_rows_ev
        self.results["potentials_at_origins_au"] = potentials_au
        self.results["potentials_at_origins_v"] = potentials_au * AU_POTENTIAL_TO_V
        
        masked_prediction = {
            **prediction,
            "charges": charges,
            "dipoles_bohr": dipoles,
            "quadrupoles_bohr": quadrupoles,
            "octupoles_bohr": octupoles,
        }
        self.results.update(masked_prediction)


def field_on_slice(
    origins_bohr: np.ndarray,
    charges: np.ndarray,
    dipoles_bohr: np.ndarray,
    quadrupoles_bohr: np.ndarray | None = None,
    octupoles_bohr: np.ndarray | None = None,
    *,
    plane: str = "xy",
    center_bohr: Sequence[float] | None = None,
    extent_angstrom: float = 8.0,
    n_grid: int = 121,
    softening_bohr: float = 0.5,
) -> dict[str, np.ndarray]:
    """Evaluate potential and field on an ``xy``, ``xz``, or ``yz`` slice."""
    if plane not in {"xy", "xz", "yz"}:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    center = (
        np.mean(np.asarray(origins_bohr, dtype=np.float64), axis=0)
        if center_bohr is None
        else np.asarray(center_bohr, dtype=np.float64)
    )
    half_extent_bohr = 0.5 * float(extent_angstrom) * ANGSTROM_TO_BOHR
    axis = np.linspace(-half_extent_bohr, half_extent_bohr, n_grid)
    u, v = np.meshgrid(axis, axis, indexing="xy")
    points = np.tile(center[None, :], (n_grid * n_grid, 1))
    if plane == "xy":
        points[:, 0] += u.reshape(-1)
        points[:, 1] += v.reshape(-1)
        component_indices = (0, 1)
    elif plane == "xz":
        points[:, 0] += u.reshape(-1)
        points[:, 2] += v.reshape(-1)
        component_indices = (0, 2)
    else:
        points[:, 1] += u.reshape(-1)
        points[:, 2] += v.reshape(-1)
        component_indices = (1, 2)
    potential, field = _point_multipole_potential_field_au(
        points,
        origins_bohr,
        charges,
        dipoles_bohr,
        quadrupoles_bohr,
        octupoles_bohr,
        softening_bohr=softening_bohr,
    )
    return {
        "plane": np.asarray(plane),
        "u_angstrom": axis * BOHR_TO_ANGSTROM,
        "v_angstrom": axis * BOHR_TO_ANGSTROM,
        "points_bohr": points.reshape(n_grid, n_grid, 3),
        "potential_au": potential.reshape(n_grid, n_grid),
        "field_au": field.reshape(n_grid, n_grid, 3),
        "field_v_per_angstrom": (
            field.reshape(n_grid, n_grid, 3) * AU_FIELD_TO_V_PER_ANGSTROM
        ),
        "field_in_plane_au": field.reshape(n_grid, n_grid, 3)[..., component_indices],
        "origins_bohr": np.asarray(origins_bohr, dtype=np.float64),
        "origins_angstrom": np.asarray(origins_bohr, dtype=np.float64) * BOHR_TO_ANGSTROM,
    }


def field_on_line(
    origins_bohr: np.ndarray,
    charges: np.ndarray,
    dipoles_bohr: np.ndarray,
    quadrupoles_bohr: np.ndarray | None = None,
    octupoles_bohr: np.ndarray | None = None,
    *,
    axis: str = "x",
    center_bohr: Sequence[float] | None = None,
    offset_bohr: Sequence[float] | None = None,
    extent_angstrom: float = 12.0,
    n_points: int = 401,
    softening_bohr: float = 0.5,
) -> dict[str, np.ndarray]:
    """Evaluate potential and field on a 1D Cartesian line scan.

    ``axis`` is the horizontal scan direction. ``offset_bohr`` shifts the line
    in transverse directions while leaving the scan-axis coordinate unchanged.
    """

    axis_indices = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_indices:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    axis_index = axis_indices[axis]
    center = (
        np.mean(np.asarray(origins_bohr, dtype=np.float64), axis=0)
        if center_bohr is None
        else np.asarray(center_bohr, dtype=np.float64)
    )
    if offset_bohr is not None:
        offset = np.asarray(offset_bohr, dtype=np.float64)
        if offset.shape != (3,):
            raise ValueError("offset_bohr must have shape (3,)")
        offset = offset.copy()
        offset[axis_index] = 0.0
        center = center + offset

    half_extent_bohr = 0.5 * float(extent_angstrom) * ANGSTROM_TO_BOHR
    coordinate_bohr = np.linspace(-half_extent_bohr, half_extent_bohr, n_points)
    points = np.tile(center[None, :], (n_points, 1))
    points[:, axis_index] += coordinate_bohr
    potential, field = _point_multipole_potential_field_au(
        points,
        origins_bohr,
        charges,
        dipoles_bohr,
        quadrupoles_bohr,
        octupoles_bohr,
        softening_bohr=softening_bohr,
    )
    return {
        "axis": np.asarray(axis),
        "coordinate_bohr": coordinate_bohr,
        "coordinate_angstrom": coordinate_bohr * BOHR_TO_ANGSTROM,
        "points_bohr": points,
        "potential_au": potential,
        "potential_v": potential * AU_POTENTIAL_TO_V,
        "field_au": field,
        "field_v_per_angstrom": field * AU_FIELD_TO_V_PER_ANGSTROM,
        "field_horizontal_au": field[:, axis_index],
        "field_horizontal_v_per_angstrom": field[:, axis_index] * AU_FIELD_TO_V_PER_ANGSTROM,
        "origins_bohr": np.asarray(origins_bohr, dtype=np.float64),
        "origins_angstrom": np.asarray(origins_bohr, dtype=np.float64) * BOHR_TO_ANGSTROM,
    }


def plot_field_line_scan(
    origins_bohr: np.ndarray,
    charges: np.ndarray,
    dipoles_bohr: np.ndarray,
    quadrupoles_bohr: np.ndarray | None = None,
    octupoles_bohr: np.ndarray | None = None,
    *,
    axis: str = "x",
    extent_angstrom: float = 12.0,
    n_points: int = 401,
    softening_bohr: float = 0.5,
    output: str | Path | None = None,
):
    """Plot potential and horizontal field from a 1D electrostatic scan."""

    import matplotlib.pyplot as plt

    scan = field_on_line(
        origins_bohr,
        charges,
        dipoles_bohr,
        quadrupoles_bohr,
        octupoles_bohr,
        axis=axis,
        extent_angstrom=extent_angstrom,
        n_points=n_points,
        softening_bohr=softening_bohr,
    )
    coordinate = scan["coordinate_angstrom"]
    potential = scan["potential_v"]
    field_horizontal = scan["field_horizontal_v_per_angstrom"]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 3.8),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(coordinate, potential, color="tab:blue", linewidth=2.0)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Potential along scan")
    axes[0].set_ylabel("Potential [V]")

    axes[1].plot(coordinate, field_horizontal, color="tab:red", linewidth=2.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title(f"{axis.upper()} field along scan")
    axes[1].set_ylabel(f"E{axis} [V/Å]")

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    center = np.mean(np.asarray(origins_bohr, dtype=np.float64), axis=0)
    source_coordinates = (
        np.asarray(origins_bohr, dtype=np.float64)[:, axis_index] - center[axis_index]
    ) * BOHR_TO_ANGSTROM
    for plot_axis in axes:
        for source_coordinate in source_coordinates:
            plot_axis.axvline(source_coordinate, color="0.2", alpha=0.25, linewidth=1.0)
        plot_axis.set_xlabel(f"{axis} scan coordinate [Å]")

    figure.suptitle("Learned molecular multipole electrostatic line scan", fontsize=13)
    if output is not None:
        figure.savefig(Path(output).expanduser(), dpi=180)
    return figure


def plot_field_summary(
    origins_bohr: np.ndarray,
    charges: np.ndarray,
    dipoles_bohr: np.ndarray,
    quadrupoles_bohr: np.ndarray | None = None,
    octupoles_bohr: np.ndarray | None = None,
    *,
    planes: Sequence[str] = ("xy", "xz", "yz"),
    extent_angstrom: float = 8.0,
    n_grid: int = 121,
    softening_bohr: float = 0.5,
    output: str | Path | None = None,
):
    """Create a compact potential/field summary plot for three orthogonal slices."""
    import matplotlib.pyplot as plt

    n_planes = len(planes)
    figure, axes = plt.subplots(
        n_planes,
        2,
        figsize=(10, 3.6 * n_planes),
        squeeze=False,
        constrained_layout=True,
    )
    for row, plane in enumerate(planes):
        grid = field_on_slice(
            origins_bohr,
            charges,
            dipoles_bohr,
            quadrupoles_bohr,
            octupoles_bohr,
            plane=plane,
            extent_angstrom=extent_angstrom,
            n_grid=n_grid,
            softening_bohr=softening_bohr,
        )
        u = grid["u_angstrom"]
        v = grid["v_angstrom"]
        potential = grid["potential_au"]
        field = grid["field_v_per_angstrom"]
        field_norm = np.linalg.norm(field, axis=-1)
        in_plane = grid["field_in_plane_au"] * AU_FIELD_TO_V_PER_ANGSTROM
        potential_limit = np.nanquantile(np.abs(potential), 0.98)
        field_limit = np.nanquantile(field_norm, 0.98)

        image = axes[row, 0].imshow(
            potential,
            origin="lower",
            extent=(u[0], u[-1], v[0], v[-1]),
            cmap="coolwarm",
            vmin=-potential_limit,
            vmax=potential_limit,
            aspect="equal",
        )
        figure.colorbar(image, ax=axes[row, 0], label="Potential [Ha/e]")
        axes[row, 0].set_title(f"{plane.upper()} potential")

        image = axes[row, 1].imshow(
            field_norm,
            origin="lower",
            extent=(u[0], u[-1], v[0], v[-1]),
            cmap="magma",
            vmin=0.0,
            vmax=field_limit,
            aspect="equal",
        )
        stride = max(1, n_grid // 25)
        axes[row, 1].quiver(
            u[::stride],
            v[::stride],
            in_plane[::stride, ::stride, 0],
            in_plane[::stride, ::stride, 1],
            color="white",
            alpha=0.75,
            pivot="mid",
            scale=None,
            width=0.0025,
        )
        figure.colorbar(image, ax=axes[row, 1], label="|E| [V/Å]")
        axes[row, 1].set_title(f"{plane.upper()} field")

        for axis in axes[row]:
            axis.set_xlabel(f"{plane[0]} [Å]")
            axis.set_ylabel(f"{plane[1]} [Å]")
            projected = np.asarray(origins_bohr)[:, {"x": 0, "y": 1, "z": 2}[plane[0:1]]]
            projected_v = np.asarray(origins_bohr)[:, {"x": 0, "y": 1, "z": 2}[plane[1:2]]]
            center = np.mean(np.asarray(origins_bohr, dtype=np.float64), axis=0)
            axis.scatter(
                (projected - center[{"x": 0, "y": 1, "z": 2}[plane[0:1]]]) * BOHR_TO_ANGSTROM,
                (projected_v - center[{"x": 0, "y": 1, "z": 2}[plane[1:2]]]) * BOHR_TO_ANGSTROM,
                s=35,
                c="cyan",
                edgecolors="black",
                linewidths=0.5,
            )

    figure.suptitle("Learned molecular multipole electrostatic field", fontsize=14)
    if output is not None:
        figure.savefig(Path(output).expanduser(), dpi=180)
    return figure
