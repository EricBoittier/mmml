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
from pathlib import Path
from typing import Any, Iterable, Sequence

import e3x
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

try:
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.units import Bohr, Hartree
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without ASE.
    raise ModuleNotFoundError(
        "QCML multipole electrostatics prototype requires ASE."
    ) from exc

from mmml.models.multipoles.model import E3xMultipoleModel
from scripts.train_qcml_multipoles import TrainConfig

ANGSTROM_TO_BOHR = 1.0 / Bohr
BOHR_TO_ANGSTROM = Bohr
HARTREE_TO_EV = Hartree
AU_FIELD_TO_V_PER_ANGSTROM = 51.4220674763259


def _load_checkpoint_payload(checkpoint: str | Path) -> dict[str, Any]:
    return ocp.PyTreeCheckpointer().restore(Path(checkpoint).expanduser())


def load_multipole_model(checkpoint: str | Path) -> tuple[E3xMultipoleModel, Any]:
    """Load a trained unified QCML multipole model checkpoint."""
    checkpoint = Path(checkpoint).expanduser()
    config_path = checkpoint / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model_config.json: {config_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    valid = {field.name for field in fields(TrainConfig)}
    model_config = {key: value for key, value in raw_config.items() if key in valid}
    TrainConfig(**model_config)
    model_config.pop("target_degree", None)
    payload = _load_checkpoint_payload(checkpoint)
    return E3xMultipoleModel(**model_config), payload["params"]


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
        return [np.asarray(fragment, dtype=np.int64) for fragment in fragments]
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


def _point_charge_dipole_potential_field_au(
    points_bohr: np.ndarray,
    origins_bohr: np.ndarray,
    charges: np.ndarray,
    dipoles_bohr: np.ndarray,
    *,
    exclude_index: int | None = None,
    softening_bohr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Potential and electric field from point charge+dipole sources.

    Potential is in Hartree/e. Electric field is in atomic units,
    Hartree/(e*bohr).
    """
    points = np.asarray(points_bohr, dtype=np.float64)
    origins = np.asarray(origins_bohr, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64).reshape(-1)
    dipoles = np.asarray(dipoles_bohr, dtype=np.float64)
    potential = np.zeros(points.shape[0], dtype=np.float64)
    field = np.zeros((points.shape[0], 3), dtype=np.float64)
    softening2 = float(softening_bohr) ** 2
    for source_index, (origin, charge, dipole) in enumerate(zip(origins, charges, dipoles, strict=True)):
        if exclude_index is not None and source_index == exclude_index:
            continue
        displacement = points - origin[None, :]
        r2 = np.sum(displacement * displacement, axis=1) + softening2
        r = np.sqrt(r2)
        inv_r = 1.0 / np.maximum(r, 1e-12)
        inv_r3 = inv_r**3
        inv_r5 = inv_r**5
        mu_dot_r = displacement @ dipole
        potential += charge * inv_r + mu_dot_r * inv_r3
        field += charge * displacement * inv_r3[:, None]
        field += (
            3.0 * displacement * mu_dot_r[:, None] * inv_r5[:, None]
            - dipole[None, :] * inv_r3[:, None]
        )
    return potential, field


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
    """Symmetric q+dipole pair interaction in Hartree.

    ``R`` points from A to B. The formula is the atomic-unit interaction of two
    point monopole+dipole objects:

        qA qB / R
        + qB muA.R / R^3 - qA muB.R / R^3
        + muA.muB / R^3 - 3 (muA.R)(muB.R) / R^5
    """
    r_vec = np.asarray(origin_b_bohr, dtype=np.float64) - np.asarray(origin_a_bohr, dtype=np.float64)
    r2 = float(np.dot(r_vec, r_vec) + softening_bohr**2)
    r = max(np.sqrt(r2), 1e-12)
    inv_r = 1.0 / r
    inv_r3 = inv_r**3
    inv_r5 = inv_r**5
    mu_a = np.asarray(dipole_a_bohr, dtype=np.float64)
    mu_b = np.asarray(dipole_b_bohr, dtype=np.float64)
    mu_a_r = float(np.dot(mu_a, r_vec))
    mu_b_r = float(np.dot(mu_b, r_vec))
    return float(
        charge_a * charge_b * inv_r
        + charge_b * mu_a_r * inv_r3
        - charge_a * mu_b_r * inv_r3
        + float(np.dot(mu_a, mu_b)) * inv_r3
        - 3.0 * mu_a_r * mu_b_r * inv_r5
    )


class LearnedMolecularMultipoleElectrostatics(Calculator):
    """ASE calculator for intermolecular q+dipole electrostatics.

    The model predicts one molecular multipole vector per fragment. Current
    energy includes only l=0 and l=1. l=2/l=3 are stored in results for later
    extension.
    """

    implemented_properties = ["energy"]

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        fragments: Sequence[Sequence[int]] | None = None,
        charges: Sequence[float] | None = None,
        multiplicities: Sequence[float] | None = None,
        origin: str = "nuclear_charge_centroid",
        mol_id_array: str = "mol_id",
        softening_bohr: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model, self.params = load_multipole_model(checkpoint)
        self.fragments = fragments
        self.charges = None if charges is None else np.asarray(charges, dtype=np.float64)
        self.multiplicities = (
            None if multiplicities is None else np.asarray(multiplicities, dtype=np.float64)
        )
        self.origin = origin
        self.mol_id_array = mol_id_array
        self.softening_bohr = float(softening_bohr)
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
        return {
            "fragments": fragments,
            "origins_bohr": np.asarray(origins, dtype=np.float64),
            "origins_angstrom": np.asarray(origins, dtype=np.float64) * BOHR_TO_ANGSTROM,
            "multipoles": multipoles_array,
            "charges": multipoles_array[:, 0],
            "dipoles_bohr": multipoles_array[:, 1:4],
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
        energy_hartree = 0.0
        pair_rows = []
        for i in range(len(charges)):
            for j in range(i + 1, len(charges)):
                pair_energy = pair_energy_charge_dipole_au(
                    origins[i],
                    charges[i],
                    dipoles[i],
                    origins[j],
                    charges[j],
                    dipoles[j],
                    softening_bohr=self.softening_bohr,
                )
                energy_hartree += pair_energy
                pair_rows.append((i, j, pair_energy, pair_energy * HARTREE_TO_EV))

        self.results["energy"] = energy_hartree * HARTREE_TO_EV
        self.results["energy_hartree"] = energy_hartree
        self.results["pair_energies"] = pair_rows
        self.results.update(prediction)


def field_on_slice(
    origins_bohr: np.ndarray,
    charges: np.ndarray,
    dipoles_bohr: np.ndarray,
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
    potential, field = _point_charge_dipole_potential_field_au(
        points,
        origins_bohr,
        charges,
        dipoles_bohr,
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


def plot_field_summary(
    origins_bohr: np.ndarray,
    charges: np.ndarray,
    dipoles_bohr: np.ndarray,
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

    figure.suptitle("Learned molecular q+dipole electrostatic field", fontsize=14)
    if output is not None:
        figure.savefig(Path(output).expanduser(), dpi=180)
    return figure
