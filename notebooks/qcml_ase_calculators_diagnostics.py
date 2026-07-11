# %% [markdown]
# # QCML ASE calculators and diagnostics
#
# Notebook-style script for:
# - loading QCML multipole and MBD checkpoints,
# - evaluating an `ase.Atoms` molecule through lightweight ASE calculators,
# - plotting molecule-level predicted components,
# - plotting test-set spherical multipole `(l, m)` diagnostic pyramids.
#
# Open this file as a notebook with Jupyter/VS Code percent-cell support, or import it
# from a notebook. Positions passed through ASE are converted from Angstrom to Bohr,
# matching the QCML cache convention.

# %%
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import e3x
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mmml-matplotlib"))
import matplotlib.pyplot as plt
from matplotlib import colors

try:
    from cmap import Colormap
except Exception:  # pragma: no cover - optional plotting dependency.
    Colormap = None

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree

from mmml.models.mbd import E3xMBDModel, mbd_energy_and_forces
from mmml.models.multipoles import E3xMultipoleModel
from scripts.train_qcml_mbd import MBDTrainConfig
from scripts.train_qcml_multipoles import TrainConfig
from scripts.plot_qcml_multipole_components import (
    collect_predictions as collect_multipole_predictions,
    component_metrics as multipole_component_metrics,
    load_scale_vector,
)
from scripts.analyze_qcml_mbd import (
    collect_targets as collect_mbd_targets,
    compute_metrics as compute_mbd_metrics,
    predict_shard as predict_mbd_shard,
    resolve_split_paths as resolve_mbd_split_paths,
    single_shard_indices as single_mbd_shard_indices,
)
from scripts.train_qcml_mbd import eligible_indices as eligible_mbd_indices
from scripts.train_qcml_mbd import restore_cache as restore_mbd_cache

ANGSTROM_TO_BOHR = 1.0 / Bohr
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = Hartree / Bohr
L_MAX = 3
M_OFF = L_MAX

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 13,
    }
)


# %% [markdown]
# ## Checkpoint loading and single-molecule batching

# %%
def _load_checkpoint_payload(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint).expanduser()
    payload = ocp.PyTreeCheckpointer().restore(checkpoint)
    if "params" not in payload:
        raise KeyError(f"Checkpoint {checkpoint} does not contain a 'params' tree")
    return payload


def _load_model_config(checkpoint: str | Path, config_type: type) -> dict[str, Any]:
    checkpoint = Path(checkpoint).expanduser()
    config_path = checkpoint / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model_config.json: {config_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    valid = {field.name for field in fields(config_type)}
    model_config = {key: value for key, value in raw_config.items() if key in valid}
    config_type(**model_config)
    return model_config


def load_multipole_model(checkpoint: str | Path) -> tuple[E3xMultipoleModel, Any]:
    model_config = _load_model_config(checkpoint, TrainConfig)
    payload = _load_checkpoint_payload(checkpoint)
    return E3xMultipoleModel(**model_config), payload["params"]


def load_mbd_model(checkpoint: str | Path) -> tuple[E3xMBDModel, Any]:
    model_config = _load_model_config(checkpoint, MBDTrainConfig)
    payload = _load_checkpoint_payload(checkpoint)
    return E3xMBDModel(**model_config), payload["params"]


def atoms_to_model_batch(
    atoms: Atoms,
    *,
    charge: float = 0.0,
    multiplicity: float = 1.0,
) -> dict[str, jax.Array]:
    """Build a batch_size=1 fully connected E3x batch from ASE atoms."""
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
    positions_bohr = np.asarray(atoms.get_positions(), dtype=np.float32) * ANGSTROM_TO_BOHR
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


def multipole_component_metadata(max_degree: int = 3) -> pd.DataFrame:
    rows = []
    offset = 0
    for degree in range(max_degree + 1):
        for component, order in enumerate(range(-degree, degree + 1)):
            rows.append(
                {
                    "index": offset + component,
                    "degree": degree,
                    "order": order,
                    "name": f"l{degree}_m{order:+d}",
                }
            )
        offset += 2 * degree + 1
    return pd.DataFrame(rows)


# %% [markdown]
# ## ASE calculators

# %%
class QCMLMultipoleCalculator(Calculator):
    """ASE-compatible wrapper for the QCML molecular multipole model.

    This model is not an energy model. `energy` is reported as zero so the object
    can be attached to ASE atoms, while the useful quantities are available under
    `results['multipoles']` and `results['multipole_components']`.
    """

    implemented_properties = ["energy", "multipoles"]

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        charge: float = 0.0,
        multiplicity: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model, self.params = load_multipole_model(checkpoint)
        self.charge = charge
        self.multiplicity = multiplicity
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

    def predict_multipoles(self, atoms: Atoms) -> np.ndarray:
        batch = atoms_to_model_batch(
            atoms,
            charge=self.charge,
            multiplicity=self.multiplicity,
        )
        return np.asarray(self._predict(batch), dtype=np.float64)

    def calculate(self, atoms=None, properties=("energy", "multipoles"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        multipoles = self.predict_multipoles(self.atoms)
        components = multipole_component_metadata()
        components["value"] = multipoles
        self.results["energy"] = 0.0
        self.results["multipoles"] = multipoles
        self.results["multipole_components"] = components


class QCMLMBDCalculator(Calculator):
    """ASE calculator for the QCML MBD surrogate.

    ASE-facing `energy` is eV and `forces` are eV/Angstrom. Raw model units are
    also stored in `results['energy_hartree']` and `results['forces_hartree_bohr']`.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        charge: float = 0.0,
        multiplicity: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model, self.params = load_mbd_model(checkpoint)
        self.charge = charge
        self.multiplicity = multiplicity
        self._predict = jax.jit(self._predict_impl)

    def _predict_impl(self, batch: dict[str, jax.Array]) -> tuple[dict[str, jax.Array], jax.Array]:
        inputs = dict(batch)
        inputs.pop("batch_size")
        output, forces = mbd_energy_and_forces(
            self.model,
            self.params,
            **inputs,
            batch_size=1,
        )
        return output, forces

    def predict_mbd(self, atoms: Atoms) -> dict[str, np.ndarray | float]:
        batch = atoms_to_model_batch(
            atoms,
            charge=self.charge,
            multiplicity=self.multiplicity,
        )
        output, forces = self._predict(batch)
        forces_hartree_bohr = np.asarray(forces, dtype=np.float64).reshape(len(atoms), 3)
        energy_hartree = float(np.asarray(output["energy"])[0])
        return {
            "energy_hartree": energy_hartree,
            "energy_ev": energy_hartree * Hartree,
            "forces_hartree_bohr": forces_hartree_bohr,
            "forces_ev_angstrom": forces_hartree_bohr * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            "polarizabilities_bohr3": np.asarray(output["polarizabilities"], dtype=np.float64),
            "c6_native": np.asarray(output["c6_coefficients"], dtype=np.float64),
        }

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        prediction = self.predict_mbd(self.atoms)
        self.results["energy"] = prediction["energy_ev"]
        self.results["forces"] = prediction["forces_ev_angstrom"]
        self.results.update(prediction)


# %% [markdown]
# ## Plotting helpers

# %%
def _resolve_cmap(name: str):
    if Colormap is not None and name.startswith("crameri:"):
        return Colormap(name).to_mpl().copy()
    fallback = {
        "crameri:vik": "coolwarm",
        "crameri:batlow": "viridis",
        "crameri:berlin": "magma",
    }.get(name, name)
    return plt.get_cmap(fallback).copy()


def multipole_pyramid(values: np.ndarray, value_column: str | None = None) -> np.ma.MaskedArray:
    grid = np.full((L_MAX + 1, 2 * L_MAX + 1), np.nan, dtype=np.float64)
    if isinstance(values, pd.DataFrame):
        if value_column is None:
            raise ValueError("value_column is required when values is a DataFrame")
        rows = values.to_dict("records")
    else:
        metadata = multipole_component_metadata()
        rows = [
            {"degree": row.degree, "order": row.order, "value": float(values[int(row.index)])}
            for row in metadata.itertuples()
        ]
        value_column = "value"
    for row in rows:
        grid[int(row["degree"]), int(row["order"]) + M_OFF] = float(row[value_column])
    return np.ma.masked_invalid(grid)


def annotate_pyramid(axis, grid, cmap, norm, fmt: str) -> None:
    for degree in range(L_MAX + 1):
        for order in range(-degree, degree + 1):
            value = grid[degree, order + M_OFF]
            if np.ma.is_masked(value):
                continue
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            axis.text(
                order + M_OFF,
                degree,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=10.5,
                color="white" if luminance < 0.5 else "black",
            )


def format_pyramid_axis(axis, title: str) -> None:
    axis.set_xticks(np.arange(-0.5, 2 * L_MAX + 1), minor=True)
    axis.set_yticks(np.arange(-0.5, L_MAX + 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", length=0)
    axis.set_xticks(range(2 * L_MAX + 1))
    axis.set_xticklabels([f"${m:+d}$" if m else "$0$" for m in range(-L_MAX, L_MAX + 1)])
    axis.set_yticks(range(L_MAX + 1))
    axis.set_yticklabels([rf"$\ell = {degree}$" for degree in range(L_MAX + 1)])
    axis.xaxis.set_ticks_position("bottom")
    axis.set_xlabel("order  $m$")
    axis.set_title(title, pad=12)
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_molecule_multipoles(
    multipoles: np.ndarray,
    *,
    title: str = "Predicted molecular multipoles",
    out: str | Path | None = None,
):
    grid = multipole_pyramid(np.asarray(multipoles, dtype=np.float64))
    finite = np.asarray(grid.compressed())
    vmax = float(np.quantile(np.abs(finite), 0.95)) if finite.size else 1.0
    vmax = max(vmax, 1e-12)
    norm = colors.SymLogNorm(linthresh=vmax / 100.0, vmin=-vmax, vmax=vmax)
    cmap = _resolve_cmap("crameri:vik")
    cmap.set_bad("#f0f0f0")
    fig, ax = plt.subplots(figsize=(6.3, 4.4), constrained_layout=True)
    image = ax.matshow(grid, cmap=cmap, norm=norm)
    annotate_pyramid(ax, grid, cmap, norm, "{:+.2e}")
    format_pyramid_axis(ax, title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="QCML native")
    if out is not None:
        fig.savefig(out, dpi=180, bbox_inches="tight")
    return fig, ax


def plot_mbd_atom_properties(
    atoms: Atoms,
    prediction: dict[str, np.ndarray | float],
    *,
    out: str | Path | None = None,
):
    labels = [f"{symbol}{idx}" for idx, symbol in enumerate(atoms.get_chemical_symbols())]
    c6 = np.asarray(prediction["c6_native"], dtype=np.float64)
    alpha = np.asarray(prediction["polarizabilities_bohr3"], dtype=np.float64)
    force_norm = np.linalg.norm(np.asarray(prediction["forces_ev_angstrom"], dtype=np.float64), axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for axis, values, ylabel in (
        (axes[0], alpha, r"$\alpha$ [bohr$^3$]"),
        (axes[1], c6, "C6 [QCML native]"),
        (axes[2], force_norm, "|force| [eV/Å]"),
    ):
        axis.bar(np.arange(len(values)), values, color="#4c78a8")
        axis.set_xticks(np.arange(len(values)))
        axis.set_xticklabels(labels, rotation=60, ha="right")
        axis.set_ylabel(ylabel)
    fig.suptitle(
        f"MBD prediction: E = {float(prediction['energy_hartree']):.6g} Ha "
        f"({float(prediction['energy_ev']):.6g} eV)",
        fontweight="bold",
    )
    if out is not None:
        fig.savefig(out, dpi=180, bbox_inches="tight")
    return fig, axes




def _flatten_masked(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None):
    if mask is None:
        return np.ravel(target), np.ravel(prediction)
    broadcast = np.broadcast_to(mask, target.shape)
    return target[broadcast], prediction[broadcast]


def plot_mbd_test_diagnostics(
    targets: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    *,
    out: str | Path | None = None,
):
    """Parity plots for MBD energy, force components, C6, and polarizability."""
    atom_mask = targets["atom_mask"].astype(bool)
    panels = [
        (
            "Energy [Ha]",
            np.ravel(targets["energy"]),
            np.ravel(predictions["energy"]),
        ),
        (
            "Force components [Ha/bohr]",
            *_flatten_masked(targets["forces"], predictions["forces"], atom_mask[:, :, None]),
        ),
        (
            "C6 [native]",
            *_flatten_masked(targets["c6"], predictions["c6"], atom_mask),
        ),
        (
            r"$\alpha$ [bohr$^3$]",
            *_flatten_masked(targets["alpha"], predictions["alpha"], atom_mask),
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for axis, (title, target, prediction) in zip(axes.flat, panels, strict=True):
        axis.scatter(target, prediction, s=5, alpha=0.25, rasterized=True)
        low = float(min(np.min(target), np.min(prediction)))
        high = float(max(np.max(target), np.max(prediction)))
        if math.isclose(low, high):
            pad = max(abs(low) * 0.05, 1.0)
        else:
            pad = 0.04 * (high - low)
        limits = (low - pad, high + pad)
        axis.plot(limits, limits, color="black", linewidth=1, linestyle="--")
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_title(title)
        axis.set_xlabel("Reference")
        axis.set_ylabel("Prediction")
    fig.suptitle("QCML MBD test-set diagnostics", fontweight="bold")
    if out is not None:
        fig.savefig(out, dpi=180, bbox_inches="tight")
    return fig, axes


def plot_multipole_metric_pyramids(
    metrics: pd.DataFrame,
    *,
    title: str = r"QCML multipole predictions — per-component $(\ell, m)$ diagnostics",
    out: str | Path | None = None,
):
    panels = [
        ("correlation", "Correlation ($r$, pred vs target)", "crameri:vik", colors.Normalize(vmin=-1, vmax=1), "{:+.2f}"),
        ("normalized_rmse", r"Normalized RMSE", "crameri:batlow", colors.LogNorm(vmin=0.05, vmax=3.0), "{:.2f}"),
        ("normalized_mae", r"Normalized MAE", "crameri:batlow", colors.LogNorm(vmin=0.002, vmax=0.1), "{:.3f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0), constrained_layout=True)
    fig.suptitle(title, fontsize=18, fontweight="bold")
    for axis, (column, panel_title, cmap_name, norm, fmt) in zip(axes, panels, strict=True):
        grid = multipole_pyramid(metrics, column)
        cmap = _resolve_cmap(cmap_name)
        cmap.set_bad("#f0f0f0")
        image = axis.matshow(grid, cmap=cmap, norm=norm)
        annotate_pyramid(axis, grid, cmap, norm, fmt)
        format_pyramid_axis(axis, panel_title)
        fig.colorbar(image, ax=axis, fraction=0.038, pad=0.03)
    if out is not None:
        fig.savefig(out, dpi=180, bbox_inches="tight")
    return fig, axes


# %% [markdown]
# ## Test-set multipole metrics

# %%
def evaluate_multipole_test_metrics(
    *,
    cache: str | Path,
    checkpoint: str | Path,
    split: str = "test",
    scale_json: str | Path | None = None,
    data_split: str | Path | None = None,
    max_structures: int | None = 100_000,
    max_atoms: int | None = 32,
    batch_size: int = 256,
    bucket_width: int = 16,
    validation_shards: int = 2,
    test_shards: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Evaluate the multipole checkpoint and return `(l, m)` component metrics."""
    args = argparse.Namespace(
        cache=Path(cache).expanduser(),
        checkpoint=Path(checkpoint).expanduser(),
        split=split,
        data_split=None if data_split is None else Path(data_split).expanduser(),
        validation_shards=validation_shards,
        test_shards=test_shards,
        validation_fraction=0.1,
        seed=seed,
        max_structures=max_structures,
        max_atoms=max_atoms,
        batch_size=batch_size,
        bucket_width=bucket_width,
    )
    target, prediction, indices, num_atoms = collect_multipole_predictions(args)
    scale_vector = load_scale_vector(None if scale_json is None else Path(scale_json).expanduser())
    metrics = pd.DataFrame(multipole_component_metrics(target, prediction, scale_vector))
    metrics.attrs["num_structures"] = int(len(target))
    metrics.attrs["dataset_indices"] = indices
    metrics.attrs["num_atoms"] = num_atoms
    return metrics




def evaluate_mbd_test_predictions(
    *,
    cache: str | Path,
    checkpoint: str | Path,
    split: str = "test",
    data_split: str | Path | None = None,
    max_structures: int | None = 100_000,
    max_atoms: int | None = 32,
    batch_size: int = 64,
    bucket_width: int = 16,
    validation_shards: int = 2,
    test_shards: int = 2,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Evaluate the MBD checkpoint and return targets, predictions, and metrics."""
    checkpoint = Path(checkpoint).expanduser()
    model, params = load_mbd_model(checkpoint)
    shard_paths = resolve_mbd_split_paths(
        Path(cache).expanduser(),
        checkpoint,
        split,
        validation_shards,
        test_shards,
        None if data_split is None else Path(data_split).expanduser(),
    )
    all_targets: dict[str, list[np.ndarray]] = {key: [] for key in ("energy", "forces", "c6", "alpha", "atom_mask")}
    all_predictions: dict[str, list[np.ndarray]] = {key: [] for key in ("energy", "forces", "c6", "alpha", "num_atoms")}
    remaining = max_structures

    if shard_paths is None:
        cache_data = restore_mbd_cache(Path(cache).expanduser())
        indices = single_mbd_shard_indices(cache_data, split, 0.1, seed, max_atoms)
        if remaining is not None:
            indices = indices[:remaining]
        shard_work = [(Path(cache).expanduser(), cache_data, indices)]
    else:
        shard_work = []
        for shard_path in shard_paths:
            if remaining is not None and remaining <= 0:
                break
            cache_data = restore_mbd_cache(shard_path)
            indices = eligible_mbd_indices(cache_data, max_atoms)
            if remaining is not None:
                indices = indices[:remaining]
                remaining -= len(indices)
            shard_work.append((shard_path, cache_data, indices))

    for shard_path, cache_data, indices in shard_work:
        if not len(indices):
            continue
        prediction = predict_mbd_shard(model, params, cache_data, indices, batch_size, bucket_width)
        width = prediction["forces"].shape[1]
        target = collect_mbd_targets(cache_data, prediction["indices"], width)
        for key in all_targets:
            all_targets[key].append(target[key])
        for key in all_predictions:
            all_predictions[key].append(prediction[key])

    if not all_targets["energy"]:
        raise ValueError(f"The selected {split} MBD split contains no eligible structures")
    targets = {key: np.concatenate(values, axis=0) for key, values in all_targets.items()}
    predictions = {key: np.concatenate(values, axis=0) for key, values in all_predictions.items()}
    metrics = compute_mbd_metrics(
        targets["energy"],
        predictions["energy"],
        targets["forces"],
        predictions["forces"],
        targets["c6"],
        predictions["c6"],
        targets["alpha"],
        predictions["alpha"],
        targets["atom_mask"],
        predictions["num_atoms"],
    )
    return targets, predictions, metrics


# %% [markdown]
# ## Example notebook usage
#
# Edit these paths on the cluster, then run the cells.

# %%
EXAMPLE_MULTIPOLE_CHECKPOINT = Path("~/qcml_runs/multipoles_restart_YYYYMMDD-HHMMSS/epoch-XXXX").expanduser()
EXAMPLE_MBD_CHECKPOINT = Path("~/qcml_runs/mbd_restart_YYYYMMDD-HHMMSS/epoch-XXXX").expanduser()
EXAMPLE_MULTIPOLE_CACHE = Path("~/orbax_cache/qcml_multipoles_traceless").expanduser()
EXAMPLE_TARGET_SCALE = EXAMPLE_MULTIPOLE_CHECKPOINT.parent / "target_scale.json"


# %%
def demo_molecule_from_atoms(
    atoms: Atoms,
    *,
    multipole_checkpoint: str | Path,
    mbd_checkpoint: str | Path,
    charge: float = 0.0,
    multiplicity: float = 1.0,
):
    multipole_calc = QCMLMultipoleCalculator(
        multipole_checkpoint,
        charge=charge,
        multiplicity=multiplicity,
    )
    mbd_calc = QCMLMBDCalculator(
        mbd_checkpoint,
        charge=charge,
        multiplicity=multiplicity,
    )
    multipoles = multipole_calc.predict_multipoles(atoms)
    mbd = mbd_calc.predict_mbd(atoms)
    plot_molecule_multipoles(multipoles)
    plot_mbd_atom_properties(atoms, mbd)
    return multipoles, mbd


# %% [markdown]
# Example:
#
# ```python
# from ase.build import molecule
# atoms = molecule("H2O")
# multipoles, mbd = demo_molecule_from_atoms(
#     atoms,
#     multipole_checkpoint="~/qcml_runs/multipoles_restart_20260710-141113/epoch-0031",
#     mbd_checkpoint="~/qcml_runs/mbd_restart_20260710-141113/epoch-0080",
#     charge=0,
#     multiplicity=1,
# )
# metrics = evaluate_multipole_test_metrics(
#     cache="~/orbax_cache/qcml_multipoles_traceless",
#     checkpoint="~/qcml_runs/multipoles_restart_20260710-141113/epoch-0031",
#     scale_json="~/qcml_runs/multipoles_restart_20260710-141113/target_scale.json",
#     max_structures=100000,
# )
# plot_multipole_metric_pyramids(metrics)
# ```


# %% [markdown]
# ## Optional command-line mode

# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="QCML ASE calculator and multipole diagnostic notebook script.")
    parser.add_argument("--multipole-checkpoint", type=Path, required=True)
    parser.add_argument("--mbd-checkpoint", type=Path)
    parser.add_argument("--multipole-cache", type=Path)
    parser.add_argument("--mbd-cache", type=Path, default=Path("~/orbax_cache/qcml_mbd"))
    parser.add_argument("--scale-json", type=Path)
    parser.add_argument("--xyz", type=Path, help="Optional molecule file readable by ASE.")
    parser.add_argument("--charge", type=float, default=0.0)
    parser.add_argument("--multiplicity", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("qcml_notebook_diagnostics"))
    parser.add_argument("--max-structures", type=int, default=100_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.xyz is not None:
        from ase.io import read

        atoms = read(args.xyz)
        multipole_calc = QCMLMultipoleCalculator(
            args.multipole_checkpoint,
            charge=args.charge,
            multiplicity=args.multiplicity,
        )
        multipoles = multipole_calc.predict_multipoles(atoms)
        plot_molecule_multipoles(
            multipoles,
            out=args.output_dir / "molecule_multipoles.png",
        )
        pd.DataFrame(
            {
                **multipole_component_metadata().to_dict("list"),
                "value": multipoles,
            }
        ).to_csv(args.output_dir / "molecule_multipoles.csv", index=False)

        if args.mbd_checkpoint is not None:
            mbd_calc = QCMLMBDCalculator(
                args.mbd_checkpoint,
                charge=args.charge,
                multiplicity=args.multiplicity,
            )
            mbd = mbd_calc.predict_mbd(atoms)
            plot_mbd_atom_properties(atoms, mbd, out=args.output_dir / "molecule_mbd.png")
            np.savez_compressed(args.output_dir / "molecule_mbd.npz", **mbd)

    if args.multipole_cache is not None:
        metrics = evaluate_multipole_test_metrics(
            cache=args.multipole_cache,
            checkpoint=args.multipole_checkpoint,
            scale_json=args.scale_json,
            max_structures=args.max_structures,
        )
        metrics.to_csv(args.output_dir / "component_metrics.csv", index=False)
        plot_multipole_metric_pyramids(
            metrics,
            out=args.output_dir / "multipole_metric_pyramids.png",
        )
        if args.mbd_checkpoint is not None:
            targets, predictions, mbd_metrics = evaluate_mbd_test_predictions(
                cache=args.mbd_cache,
                checkpoint=args.mbd_checkpoint,
                max_structures=args.max_structures,
            )
            (args.output_dir / "mbd_metrics.json").write_text(
                json.dumps(mbd_metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            plot_mbd_test_diagnostics(
                targets,
                predictions,
                out=args.output_dir / "mbd_test_diagnostics.png",
            )

    print(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
