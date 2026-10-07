"""Run ASE NEB with an MMML PhysNet (or compatible) checkpoint calculator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from mmml.data.units import EV_TO_KCAL_MOL
from mmml.neb.config import NebConfig


def _import_neb():
    try:
        from ase.mep import NEB
    except ImportError:  # ASE < 3.23
        from ase.neb import NEB
    return NEB


def _optimizer_cls(name: str):
    from ase.optimize import BFGS, FIRE, MDMin

    mapping = {"BFGS": BFGS, "FIRE": FIRE, "MDMin": MDMin}
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unknown optimizer {name!r}; choose from {sorted(mapping)}") from exc


def path_length_coordinate(images: Sequence[Any]) -> np.ndarray:
    """Cumulative RMS displacement between consecutive images (Å)."""
    distances = [0.0]
    for i in range(len(images) - 1):
        dR = images[i + 1].get_positions() - images[i].get_positions()
        distances.append(float(np.linalg.norm(dR)))
    return np.cumsum(np.asarray(distances, dtype=np.float64))


def pair_distances(images: Sequence[Any], pairs: Sequence[tuple[int, int]]) -> dict[str, np.ndarray]:
    """Per-image distances for each (i, j) atom pair."""
    out: dict[str, np.ndarray] = {}
    for i, j in pairs:
        key = f"d_{i}_{j}"
        vals = []
        for image in images:
            pos = image.get_positions()
            vals.append(float(np.linalg.norm(pos[j] - pos[i])))
        out[key] = np.asarray(vals, dtype=np.float64)
    return out


def relative_energies_kcal(
    images: Sequence[Any],
    *,
    reference_index: int = 0,
) -> np.ndarray:
    """Potential energies relative to ``images[reference_index]``, in kcal/mol."""
    e0 = float(images[reference_index].get_potential_energy())
    energies_ev = np.asarray(
        [float(image.get_potential_energy()) for image in images],
        dtype=np.float64,
    )
    return (energies_ev - e0) * float(EV_TO_KCAL_MOL)


def _attach_calculators(
    images: list[Any],
    make_calc: Callable[[], Any],
    *,
    shared: bool,
) -> None:
    if shared:
        calc = make_calc()
        for image in images:
            image.calc = calc
        return
    for image in images:
        image.calc = make_calc()


def _default_calculator_factory(
    checkpoint: Path,
    *,
    calculator: str | None = None,
) -> Callable[[], Any]:
    from mmml.models.kernnn import KerNNCalculator, is_kernnn_checkpoint

    calc_name = (calculator or "").strip().lower()
    if calc_name == "kernnn" or (not calc_name and is_kernnn_checkpoint(checkpoint)):
        path = Path(checkpoint)

        def make_kernnn() -> Any:
            return KerNNCalculator(path)

        return make_kernnn

    from mmml.interfaces.calculators.checkpoint_loading import (
        create_calculator_from_checkpoint,
        load_checkpoint_bundle,
        _build_physnet_ef_calculator,
    )

    bundle = load_checkpoint_bundle(Path(checkpoint))
    cfg = bundle.config
    is_joint = "physnet_config" in cfg and (
        "dcmnet_config" in cfg or "noneq_config" in cfg
    )
    if is_joint:
        path = str(Path(checkpoint))

        def make() -> Any:
            return create_calculator_from_checkpoint(path)

        return make

    def make() -> Any:
        return _build_physnet_ef_calculator(cfg, bundle.params, cutoff=None)

    return make


@dataclass
class NebResult:
    """Outputs from a finished NEB run."""

    images: list[Any]
    reaction_coordinate: np.ndarray
    energy_kcal_mol: np.ndarray
    pair_distance_angstrom: dict[str, np.ndarray]
    output_dir: Path
    summary: dict[str, Any]
    paths: dict[str, Path]

    @property
    def barrier_kcal_mol(self) -> float:
        return float(np.max(self.energy_kcal_mol))


def _write_profile(
    path: Path,
    reaction_coordinate: np.ndarray,
    energy_kcal_mol: np.ndarray,
    pair_distance_angstrom: dict[str, np.ndarray],
) -> None:
    pair_keys = sorted(pair_distance_angstrom)
    header = ["reaction_coordinate_ang", "energy_kcal_mol", *pair_keys]
    cols = [reaction_coordinate, energy_kcal_mol, *[pair_distance_angstrom[k] for k in pair_keys]]
    data = np.column_stack(cols)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# " + " ".join(header) + "\n")
        for row in data:
            fh.write(" ".join(f"{x:.10g}" for x in row) + "\n")


def _write_plot(
    path: Path,
    reaction_coordinate: np.ndarray,
    energy_kcal_mol: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(reaction_coordinate, energy_kcal_mol, marker="o", ms=3.0, lw=1.5)
    ax.set_xlabel("Reaction coordinate (Å)")
    ax.set_ylabel("Energy (kcal/mol)")
    ax.set_title("NEB energy profile")
    ax.axhline(0.0, color="0.6", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run_neb(
    config: NebConfig,
    *,
    calculator_factory: Callable[[], Any] | None = None,
) -> NebResult:
    """Interpolate, relax an NEB band, and write profile artifacts."""
    from ase.io import read, write

    initial_path = Path(config.initial).expanduser().resolve()
    final_path = Path(config.final).expanduser().resolve()
    ckpt_path = Path(config.checkpoint).expanduser().resolve()
    out_dir = Path(config.output_dir).expanduser().resolve()

    if not initial_path.is_file():
        raise FileNotFoundError(f"initial geometry not found: {initial_path}")
    if not final_path.is_file():
        raise FileNotFoundError(f"final geometry not found: {final_path}")
    if not ckpt_path.exists() and calculator_factory is None:
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "neb.traj"
    xyz_path = out_dir / "neb.xyz"
    profile_path = out_dir / "neb_profile.dat"
    plot_path = out_dir / "neb_plot.png"
    summary_path = out_dir / "neb_summary.json"

    for path in (traj_path, xyz_path, profile_path, plot_path, summary_path):
        if path.exists() and not config.overwrite:
            raise FileExistsError(
                f"refusing to overwrite {path}; pass overwrite=True / --overwrite"
            )

    initial = read(str(initial_path))
    final = read(str(final_path))
    if len(initial) != len(final):
        raise ValueError(
            f"initial/final atom counts differ: {len(initial)} vs {len(final)}"
        )
    if not np.array_equal(initial.get_atomic_numbers(), final.get_atomic_numbers()):
        raise ValueError("initial/final atomic numbers differ (check atom ordering)")

    n_atoms = len(initial)
    for i, j in config.pair_indices:
        if not (0 <= i < n_atoms and 0 <= j < n_atoms):
            raise ValueError(
                f"pair index ({i}, {j}) out of range for {n_atoms}-atom system"
            )

    images = [initial.copy()]
    images.extend(initial.copy() for _ in range(config.n_images - 2))
    images.append(final.copy())

    make_calc = calculator_factory or _default_calculator_factory(
        ckpt_path, calculator=getattr(config, "calculator", None)
    )
    _attach_calculators(images, make_calc, shared=config.shared_calculator)

    NEB = _import_neb()
    neb = NEB(
        images,
        k=float(config.spring_k),
        climb=bool(config.climb),
        method=str(config.neb_method),
        allow_shared_calculator=bool(config.shared_calculator),
    )
    if config.interpolate == "idpp":
        neb.interpolate(method="idpp")
    else:
        neb.interpolate()

    opt_cls = _optimizer_cls(config.optimizer)
    dyn = opt_cls(neb, trajectory=str(traj_path))
    if config.max_steps is None:
        dyn.run(fmax=float(config.fmax))
    else:
        dyn.run(fmax=float(config.fmax), steps=int(config.max_steps))

    write(str(xyz_path), images)
    rc = path_length_coordinate(images)
    e_kcal = relative_energies_kcal(images)
    pairs = pair_distances(images, config.pair_indices)
    _write_profile(profile_path, rc, e_kcal, pairs)

    paths: dict[str, Path] = {
        "traj": traj_path,
        "xyz": xyz_path,
        "profile": profile_path,
        "summary": summary_path,
    }
    if config.plot:
        _write_plot(plot_path, rc, e_kcal)
        paths["plot"] = plot_path

    barrier = float(np.max(e_kcal))
    barrier_idx = int(np.argmax(e_kcal))
    summary = {
        "checkpoint": str(ckpt_path),
        "initial": str(initial_path),
        "final": str(final_path),
        "n_images": int(config.n_images),
        "fmax": float(config.fmax),
        "climb": bool(config.climb),
        "interpolate": config.interpolate,
        "optimizer": config.optimizer,
        "neb_method": config.neb_method,
        "spring_k": float(config.spring_k),
        "shared_calculator": bool(config.shared_calculator),
        "max_steps": config.max_steps,
        "barrier_kcal_mol": barrier,
        "barrier_image_index": barrier_idx,
        "delta_e_product_kcal_mol": float(e_kcal[-1]),
        "reaction_coordinate_ang": rc.tolist(),
        "energy_kcal_mol": e_kcal.tolist(),
        "pair_distance_angstrom": {k: v.tolist() for k, v in pairs.items()},
        "pair_indices": [list(p) for p in config.pair_indices],
        "artifacts": {k: str(v) for k, v in paths.items()},
        "config": config.to_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return NebResult(
        images=images,
        reaction_coordinate=rc,
        energy_kcal_mol=e_kcal,
        pair_distance_angstrom=pairs,
        output_dir=out_dir,
        summary=summary,
        paths=paths,
    )
