"""ASE calculator wrapper for learned QCML MBD models."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

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
    raise ModuleNotFoundError("QCML MBD calculator requires ASE.") from exc

from mmml.models.mbd.model import E3xMBDModel, mbd_energy_and_forces


@dataclass(frozen=True)
class MBDModelConfig:
    features: int = 64
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 16
    cutoff: float = 12.0
    max_atomic_number: int = 118

ANGSTROM_TO_BOHR = 1.0 / Bohr
HARTREE_TO_EV = Hartree
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = Hartree / Bohr


def _validate_mbd_config_against_params(
    config: MBDModelConfig, params: Any, checkpoint: str | Path
) -> None:
    """Fail loudly if assumed ``config`` disagrees with the saved weight shapes.

    The atom embedding fixes ``(max_atomic_number + 1, ..., features)`` and the
    number of ``MessagePass_*`` submodules fixes ``num_iterations``; both are
    cheap to read and the most likely things to differ from the defaults.
    """
    if not isinstance(params, dict):
        return
    embed = params.get("Embed_0", {}).get("embedding")
    if embed is not None:
        emb = np.asarray(embed)
        n_emb, feats = int(emb.shape[0]), int(emb.shape[-1])
        if n_emb != config.max_atomic_number + 1 or feats != config.features:
            raise ValueError(
                f"MBD checkpoint {checkpoint} weight shapes (max_atomic_number="
                f"{n_emb - 1}, features={feats}) disagree with assumed default "
                f"config (max_atomic_number={config.max_atomic_number}, "
                f"features={config.features}); re-export the checkpoint with its config."
            )
    n_mp = sum(1 for key in params if key.startswith("MessagePass_"))
    if n_mp and n_mp != config.num_iterations:
        raise ValueError(
            f"MBD checkpoint {checkpoint} has {n_mp} MessagePass layers but assumed "
            f"num_iterations={config.num_iterations}; re-export the checkpoint with its config."
        )


def load_mbd_model(checkpoint: str | Path) -> tuple[E3xMBDModel, Any]:
    """Load a trained QCML MBD checkpoint.

    Accepts either an Orbax checkpoint directory (with a sibling
    ``model_config.json``) or a portable JSON file produced by
    :func:`mmml.utils.model_checkpoint.orbax_to_json` (params + config bundled
    together, no sibling file needed).
    """
    checkpoint = Path(checkpoint).expanduser()
    if checkpoint.is_file() and checkpoint.suffix == ".json":
        from mmml.utils.model_checkpoint import json_to_params

        restored = json_to_params(checkpoint)
        raw_config = restored.get("config")
        if not raw_config:
            # Portable exports sometimes ship params only. Fall back to the
            # default MBD architecture, but verify it is consistent with the
            # saved weights (embedding shape pins max_atomic_number + features)
            # so we fail loudly on a genuine mismatch rather than mis-loading.
            import warnings

            defaults = MBDModelConfig()
            _validate_mbd_config_against_params(defaults, restored.get("params"), checkpoint)
            warnings.warn(
                f"JSON checkpoint {checkpoint} has no 'config'; assuming default "
                f"MBDModelConfig ({defaults}). This matches the saved weight shapes, "
                "but re-export with config to remove the guess.",
                stacklevel=2,
            )
            raw_config = {field.name: getattr(defaults, field.name) for field in fields(MBDModelConfig)}
        valid = {field.name for field in fields(MBDModelConfig)}
        model_config = {key: value for key, value in raw_config.items() if key in valid}
        MBDModelConfig(**model_config)
        if "params" not in restored:
            raise KeyError(f"Checkpoint {checkpoint} does not contain params")
        return E3xMBDModel(**model_config), restored["params"]

    config_path = checkpoint / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model_config.json: {config_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    valid = {field.name for field in fields(MBDModelConfig)}
    model_config = {key: value for key, value in raw_config.items() if key in valid}
    MBDModelConfig(**model_config)
    restored = ocp.PyTreeCheckpointer().restore(checkpoint)
    if "params" not in restored:
        raise KeyError(f"Checkpoint {checkpoint} does not contain params")
    return E3xMBDModel(**model_config), restored["params"]


def atoms_to_mbd_batch(
    atoms: Atoms,
    *,
    charge: float = 0.0,
    multiplicity: float = 1.0,
) -> dict[str, jax.Array]:
    """Build a batch_size=1 fully connected E3x MBD batch from ASE atoms."""
    positions_bohr = np.asarray(atoms.get_positions(), dtype=np.float32) * ANGSTROM_TO_BOHR
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
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


def predict_mbd_from_atoms(
    model: E3xMBDModel,
    params: Any,
    atoms: Atoms,
    *,
    charge: float = 0.0,
    multiplicity: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Predict learned MBD energy/forces/properties for one ASE structure."""
    batch = atoms_to_mbd_batch(atoms, charge=charge, multiplicity=multiplicity)
    inputs = dict(batch)
    inputs.pop("batch_size")
    output, forces = mbd_energy_and_forces(
        model,
        params,
        **inputs,
        batch_size=1,
    )
    forces_hartree_bohr = np.asarray(forces, dtype=np.float64).reshape(len(atoms), 3)
    energy_hartree = float(np.asarray(output["energy"])[0])
    return {
        "energy_hartree": energy_hartree,
        "energy_ev": energy_hartree * HARTREE_TO_EV,
        "forces_hartree_bohr": forces_hartree_bohr,
        "forces_ev_angstrom": forces_hartree_bohr * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
        "polarizabilities_bohr3": np.asarray(output["polarizabilities"], dtype=np.float64),
        "c6_native": np.asarray(output["c6_coefficients"], dtype=np.float64),
    }


def resolve_mbd_checkpoint(explicit: str | Path | None = None) -> Path:
    """Resolve default MBD checkpoint path."""
    import os

    env_ckpt = (os.environ.get("MBD_CKPT") or os.environ.get("MBD_CHECKPOINT") or "").strip()
    target = explicit or (env_ckpt if env_ckpt else None)
    if target is not None:
        p = Path(target).expanduser().resolve()
        if p.exists():
            return p
        remapped = remap_missing_mbd_checkpoint(p)
        if remapped is not None:
            return remapped
        raise FileNotFoundError(f"MBD checkpoint not found at: {target}")
    for candidate in default_mbd_checkpoint_candidates():
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "No default MBD checkpoint found. Set MBD_CKPT or pass explicit path."
    )


def default_mbd_checkpoint_candidates() -> list[Path]:
    """Portable MBD JSON twins shipped with the repo / examples layout."""
    repo_root = Path(__file__).resolve().parents[3]
    return [
        repo_root / "mbd_20260711-100037_epoch-0100.json",
        repo_root / "examples" / "mbd_20260711-100037_epoch-0100.json",
    ]


def remap_missing_mbd_checkpoint(recorded: str | Path) -> Path | None:
    """Map a missing training-time MBD path to a local portable twin.

    Training configs often record cluster Orbax dirs such as::

        /mmhome/.../mbd_restart_20260711-100037/epoch-0100

    while the portable copy lives at::

        examples/mbd_20260711-100037_epoch-0100.json

    Returns an existing local path, or ``None`` if no twin is found.
    """
    import re

    recorded_path = Path(recorded).expanduser()
    if recorded_path.exists():
        return recorded_path.resolve()

    text = str(recorded)
    repo_root = Path(__file__).resolve().parents[3]
    candidates: list[Path] = []

    match = re.search(r"mbd_restart_(\d{8}-\d{6}).*?epoch-0*(\d+)", text)
    if match:
        stamp = match.group(1)
        epoch = int(match.group(2))
        for name in (
            f"mbd_{stamp}_epoch-{epoch:04d}.json",
            f"mbd_{stamp}_epoch-{epoch}.json",
        ):
            candidates.extend(
                [
                    repo_root / name,
                    repo_root / "examples" / name,
                ]
            )

    # Basename already looks like the portable JSON name.
    name = recorded_path.name
    if name.endswith(".json") and name.startswith("mbd_"):
        candidates.extend([repo_root / name, repo_root / "examples" / name])

    # Deduplicate while preserving order.
    seen: set[Path] = set()
    for candidate in candidates:
        key = candidate.resolve() if candidate.exists() else candidate
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()

    # Known default companion used by the shipped Spooky MBD training runs.
    if match and match.group(1) == "20260711-100037":
        for candidate in default_mbd_checkpoint_candidates():
            if candidate.is_file():
                return candidate.resolve()
    return None


def resolve_companion_mbd(
    mbd_checkpoint: str | Path | bool | None,
    mbd_weight: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[Path | None, float, str | None]:
    """Resolve a Spooky/hybrid companion MBD checkpoint for inference.

    Returns ``(load_path, weight, missing_recorded_path)``.
    Pass ``mbd_checkpoint=False`` to force Spooky/PhysNet-only (no auto-load).
    ``None`` reads ``config["mbd_checkpoint"]`` when present.

    Missing cluster Orbax dirs such as
    ``.../mbd_restart_20260711-100037/epoch-0100`` are remapped to
    ``examples/mbd_20260711-100037_epoch-0100.json`` when present.
    """
    cfg = dict(config or {})
    if mbd_checkpoint is False:
        return None, 0.0, None
    if mbd_checkpoint is not None and mbd_checkpoint is not True:
        candidate: str | Path | None = mbd_checkpoint
    else:
        candidate = cfg.get("mbd_checkpoint")
    weight = float(
        mbd_weight if mbd_weight is not None else cfg.get("mbd_weight", 1.0)
    )
    if not candidate:
        return None, weight, None
    resolved = Path(candidate).expanduser()
    if resolved.exists():
        return resolved, weight, None
    remapped = remap_missing_mbd_checkpoint(resolved)
    if remapped is not None:
        return remapped, weight, None
    return None, weight, str(resolved)


class QCMLMBDCalculator(Calculator):
    """ASE calculator for the learned QCML MBD surrogate.

    ASE-facing `energy` is eV and `forces` are eV/Angstrom. Raw atomic-unit
    values are retained in `results`.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        charge: float = 0.0,
        multiplicity: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        resolved_ckpt = resolve_mbd_checkpoint(checkpoint)
        self.checkpoint = resolved_ckpt
        self.model, self.params = load_mbd_model(resolved_ckpt)
        self.charge = float(charge)
        self.multiplicity = float(multiplicity)
        self._predict = jax.jit(self._predict_impl)

    def _predict_impl(self, batch: dict[str, jax.Array]) -> tuple[dict[str, jax.Array], jax.Array]:
        inputs = dict(batch)
        inputs.pop("batch_size")
        return mbd_energy_and_forces(
            self.model,
            self.params,
            **inputs,
            batch_size=1,
        )

    def predict_mbd(self, atoms: Atoms) -> dict[str, np.ndarray | float]:
        batch = atoms_to_mbd_batch(
            atoms,
            charge=self.charge,
            multiplicity=self.multiplicity,
        )
        output, forces = self._predict(batch)
        forces_hartree_bohr = np.asarray(forces, dtype=np.float64).reshape(len(atoms), 3)
        energy_hartree = float(np.asarray(output["energy"])[0])
        return {
            "energy_hartree": energy_hartree,
            "energy_ev": energy_hartree * HARTREE_TO_EV,
            "forces_hartree_bohr": forces_hartree_bohr,
            "forces_ev_angstrom": forces_hartree_bohr * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            "polarizabilities_bohr3": np.asarray(output["polarizabilities"], dtype=np.float64),
            "c6_native": np.asarray(output["c6_coefficients"], dtype=np.float64),
        }

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Iterable[str] = ("energy", "forces"),
        system_changes: Iterable[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        prediction = self.predict_mbd(self.atoms)
        self.results["energy"] = prediction["energy_ev"]
        self.results["forces"] = prediction["forces_ev_angstrom"]
        self.results.update(prediction)
