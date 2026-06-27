"""
Compare reference (PySCF / QM) and model NPZ arrays with metrics and plots.

Supports energies, forces, dipoles, and per-atom / per-element force breakdowns.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from ase.data import chemical_symbols
except ImportError:  # pragma: no cover
    chemical_symbols = [""] + [f"Z{z}" for z in range(1, 120)]


# Display conversions (canonical NPZ energies/forces are usually eV and eV/Å).
EV_TO_KCAL_MOL = 23.060549
HARTREE_TO_KCAL_MOL = 627.509474


@dataclass(frozen=True)
class ScalarMetrics:
    mae: float
    rmse: float
    r2: float
    bias: float
    max_abs_error: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _as_2d_energy(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    return a


def _as_3d_forces(arr: np.ndarray) -> np.ndarray:
    f = np.asarray(arr, dtype=np.float64)
    if f.ndim == 2:
        return f[np.newaxis, ...]
    if f.ndim == 4 and f.shape[1] == 1:
        return f[:, 0, ...]
    if f.ndim != 3:
        raise ValueError(f"Expected forces shape (n, natom, 3), got {f.shape}")
    return f


def _as_2d_dipole(arr: np.ndarray) -> np.ndarray:
    d = np.asarray(arr, dtype=np.float64)
    if d.ndim == 3 and d.shape[1] == 1:
        d = d[:, 0, :]
    if d.ndim != 2 or d.shape[-1] != 3:
        raise ValueError(f"Expected dipole shape (n, 3), got {d.shape}")
    return d


def _atomic_numbers_batch(z: np.ndarray, n_frames: int, n_atoms: int) -> np.ndarray:
    z = np.asarray(z)
    if z.ndim == 1:
        return np.broadcast_to(z, (n_frames, n_atoms)).copy()
    if z.ndim == 2:
        return z
    raise ValueError(f"Unexpected Z shape {z.shape}")


def compute_scalar_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> ScalarMetrics:
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targ = np.asarray(targets, dtype=np.float64).reshape(-1)
    if pred.shape != targ.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs targ {targ.shape}")
    mask = np.isfinite(pred) & np.isfinite(targ)
    pred = pred[mask]
    targ = targ[mask]
    n = int(pred.size)
    if n == 0:
        return ScalarMetrics(
            mae=float("nan"),
            rmse=float("nan"),
            r2=float("nan"),
            bias=float("nan"),
            max_abs_error=float("nan"),
            n=0,
        )
    err = pred - targ
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((targ - np.mean(targ)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return ScalarMetrics(
        mae=mae,
        rmse=rmse,
        r2=r2,
        bias=float(np.mean(err)),
        max_abs_error=float(np.max(np.abs(err))),
        n=n,
    )


def compute_force_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float | int]:
    pred = _as_3d_forces(predictions)
    targ = _as_3d_forces(targets)
    if pred.shape != targ.shape:
        raise ValueError(f"Force shape mismatch: {pred.shape} vs {targ.shape}")

    err = pred - targ
    mask = np.isfinite(err).all(axis=-1)
    err_flat = err[mask].reshape(-1)
    pred_flat = pred[mask].reshape(-1)
    targ_flat = targ[mask].reshape(-1)

    overall = compute_scalar_metrics(pred_flat, targ_flat)
    comp = {}
    for i, axis in enumerate("xyz"):
        m = compute_scalar_metrics(pred[..., i][mask], targ[..., i][mask])
        comp[f"mae_{axis}"] = m.mae
        comp[f"rmse_{axis}"] = m.rmse
        comp[f"r2_{axis}"] = m.r2

    pred_mag = np.linalg.norm(pred, axis=-1)[mask]
    targ_mag = np.linalg.norm(targ, axis=-1)[mask]
    mag = compute_scalar_metrics(pred_mag, targ_mag)

    return {
        "n_components": int(err_flat.size),
        "mae": overall.mae,
        "rmse": overall.rmse,
        "r2": overall.r2,
        "bias": overall.bias,
        "max_abs_error": overall.max_abs_error,
        "mae_magnitude": mag.mae,
        "rmse_magnitude": mag.rmse,
        "r2_magnitude": mag.r2,
        **comp,
    }


def compute_per_atom_force_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    atomic_numbers: np.ndarray,
) -> dict[str, Any]:
    """Per-atom-index MAE/RMSE averaged over frames."""
    pred = _as_3d_forces(predictions)
    targ = _as_3d_forces(targets)
    n_frames, n_atoms, _ = pred.shape
    z_batch = _atomic_numbers_batch(atomic_numbers, n_frames, n_atoms)

    err = pred - targ
    mae_per_atom = np.nanmean(np.linalg.norm(err, axis=-1), axis=0)
    rmse_per_atom = np.sqrt(np.nanmean(np.sum(err**2, axis=-1), axis=0))

    # Representative element per atom index (mode over frames).
    elem_per_atom: list[str] = []
    for j in range(n_atoms):
        zj = z_batch[:, j]
        zj = zj[zj > 0]
        if zj.size == 0:
            elem_per_atom.append("X")
            continue
        zi = int(np.bincount(zj.astype(int)).argmax())
        elem_per_atom.append(
            chemical_symbols[zi] if 0 <= zi < len(chemical_symbols) else f"Z{zi}"
        )

    return {
        "mae_per_atom": mae_per_atom.tolist(),
        "rmse_per_atom": rmse_per_atom.tolist(),
        "elements_per_atom": elem_per_atom,
        "n_atoms": n_atoms,
        "n_frames": n_frames,
    }


def compute_element_force_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    atomic_numbers: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Aggregate force errors by chemical element."""
    pred = _as_3d_forces(predictions)
    targ = _as_3d_forces(targets)
    n_frames, n_atoms, _ = pred.shape
    z_batch = _atomic_numbers_batch(atomic_numbers, n_frames, n_atoms)

    err = pred - targ
    err_mag = np.linalg.norm(err, axis=-1)

    by_elem: dict[str, list[float]] = {}
    for i in range(n_frames):
        for j in range(n_atoms):
            zi = int(z_batch[i, j])
            if zi <= 0:
                continue
            sym = (
                chemical_symbols[zi]
                if 0 <= zi < len(chemical_symbols)
                else f"Z{zi}"
            )
            by_elem.setdefault(sym, []).append(float(err_mag[i, j]))

    out: dict[str, dict[str, float | int]] = {}
    for sym, vals in sorted(by_elem.items()):
        arr = np.asarray(vals, dtype=np.float64)
        out[sym] = {
            "n": int(arr.size),
            "mae": float(np.mean(arr)),
            "rmse": float(np.sqrt(np.mean(arr**2))),
            "max": float(np.max(arr)),
        }
    return out


def align_npz_arrays(
    reference: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    *,
    max_frames: int | None = None,
    frame_indices: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Extract aligned E/F/D arrays from two NPZ-like dicts."""
    ref_e = reference.get("E")
    pred_e = predictions.get("E")
    if ref_e is None or pred_e is None:
        raise KeyError("Both reference and predictions must contain 'E'.")

    n = min(len(ref_e), len(pred_e))
    if frame_indices is not None:
        idx = np.asarray(frame_indices, dtype=int)
    elif max_frames is not None:
        idx = np.arange(min(n, int(max_frames)), dtype=int)
    else:
        idx = np.arange(n, dtype=int)

    out: dict[str, np.ndarray] = {
        "E_ref": _as_2d_energy(ref_e)[idx],
        "E_pred": _as_2d_energy(pred_e)[idx],
        "indices": idx,
    }

    z_ref = reference.get("Z")
    z_pred = predictions.get("Z")
    if z_ref is not None:
        z = np.asarray(z_ref)
        out["Z"] = z[idx] if z.ndim >= 2 else z

    for key, out_key in (("F", "F"), ("Dxyz", "D"), ("D", "D")):
        ref_arr = reference.get(key)
        if ref_arr is None:
            ref_arr = reference.get(out_key)
        pred_arr = predictions.get(key)
        if pred_arr is None:
            pred_arr = predictions.get(out_key)
        if ref_arr is not None and pred_arr is not None:
            out[f"{out_key}_ref"] = np.asarray(ref_arr)[idx]
            out[f"{out_key}_pred"] = np.asarray(pred_arr)[idx]
            break

    r_ref = reference.get("R")
    if r_ref is not None:
        out["R"] = np.asarray(r_ref)[idx]

    return out


def compare_npz_arrays(
    aligned: Mapping[str, np.ndarray],
    *,
    energy_unit_label: str = "eV",
    force_unit_label: str = "eV/Å",
) -> dict[str, Any]:
    """Compute full metrics dict from aligned arrays."""
    metrics: dict[str, Any] = {
        "n_frames": int(len(aligned["E_ref"])),
        "units": {"energy": energy_unit_label, "forces": force_unit_label},
    }

    metrics["energy"] = compute_scalar_metrics(
        aligned["E_pred"], aligned["E_ref"]
    ).to_dict()

    z = aligned.get("Z")
    if "F_ref" in aligned and "F_pred" in aligned:
        f_met = compute_force_metrics(aligned["F_pred"], aligned["F_ref"])
        metrics["forces"] = f_met
        if z is not None:
            metrics["per_atom_forces"] = compute_per_atom_force_metrics(
                aligned["F_pred"], aligned["F_ref"], z
            )
            metrics["per_element_forces"] = compute_element_force_metrics(
                aligned["F_pred"], aligned["F_ref"], z
            )

    d_ref = aligned.get("D_ref")
    d_pred = aligned.get("D_pred")
    if d_ref is not None and d_pred is not None:
        d_met = compute_scalar_metrics(
            _as_2d_dipole(d_pred).reshape(-1),
            _as_2d_dipole(d_ref).reshape(-1),
        )
        metrics["dipole"] = d_met.to_dict()
        for i, axis in enumerate("xyz"):
            metrics["dipole"][f"mae_{axis}"] = compute_scalar_metrics(
                _as_2d_dipole(d_pred)[:, i],
                _as_2d_dipole(d_ref)[:, i],
            ).mae

    return metrics


def _maybe_kcal_energy(values: np.ndarray, unit_label: str) -> tuple[np.ndarray, str]:
    v = np.asarray(values, dtype=np.float64)
    if unit_label.lower() in {"ev", "electronvolt", "electron-volt"}:
        return v * EV_TO_KCAL_MOL, "kcal/mol"
    if unit_label.lower() in {"hartree", "ha", "hartrees"}:
        return v * HARTREE_TO_KCAL_MOL, "kcal/mol"
    return v, unit_label


def _maybe_kcal_forces(values: np.ndarray, unit_label: str) -> tuple[np.ndarray, str]:
    v = np.asarray(values, dtype=np.float64)
    if unit_label.lower() in {"ev/å", "ev/angstrom", "ev/a", "ev/ang"}:
        return v * EV_TO_KCAL_MOL, "kcal/(mol·Å)"
    if unit_label.lower() in {"hartree/bohr", "ha/bohr"}:
        return v * HARTREE_TO_KCAL_MOL * 1.88973, "kcal/(mol·Å)"
    return v, unit_label


def plot_comparison(
    aligned: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any],
    output_dir: Path,
    *,
    title_prefix: str = "",
    energy_unit: str = "eV",
    force_unit: str = "eV/Å",
) -> list[Path]:
    """Write parity and breakdown plots; return paths of saved figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    e_ref, e_unit = _maybe_kcal_energy(aligned["E_ref"], energy_unit)
    e_pred, _ = _maybe_kcal_energy(aligned["E_pred"], energy_unit)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(e_ref, e_pred, alpha=0.45, s=12, edgecolors="none")
    lims = [
        min(e_ref.min(), e_pred.min()),
        max(e_ref.max(), e_pred.max()),
    ]
    ax.plot(lims, lims, "r--", lw=1.5, label="y = x")
    em = metrics["energy"]
    ax.set_xlabel(f"Reference energy ({e_unit})")
    ax.set_ylabel(f"Predicted energy ({e_unit})")
    ax.set_title(
        f"{title_prefix}Energy parity\n"
        f"MAE={em['mae'] * (EV_TO_KCAL_MOL if e_unit == 'kcal/mol' else 1):.4f} "
        f"{e_unit} | R²={em['r2']:.4f}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    p = output_dir / "energy_parity.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # Energy error histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    err_e = e_pred - e_ref
    ax.hist(err_e, bins=40, alpha=0.75, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="r", ls="--")
    ax.set_xlabel(f"Energy error ({e_unit})")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix}Energy error distribution")
    ax.grid(True, alpha=0.3)
    p = output_dir / "energy_error_hist.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    if "F_ref" in aligned and "forces" in metrics:
        f_ref, f_unit = _maybe_kcal_forces(aligned["F_ref"], force_unit)
        f_pred, _ = _maybe_kcal_forces(aligned["F_pred"], force_unit)
        flat_ref = f_ref.reshape(-1)
        flat_pred = f_pred.reshape(-1)
        mask = np.isfinite(flat_ref) & np.isfinite(flat_pred)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(flat_ref[mask], flat_pred[mask], alpha=0.25, s=8, edgecolors="none")
        lims = [
            min(flat_ref[mask].min(), flat_pred[mask].min()),
            max(flat_ref[mask].max(), flat_pred[mask].max()),
        ]
        ax.plot(lims, lims, "r--", lw=1.5)
        fm = metrics["forces"]
        ax.set_xlabel(f"Reference force ({f_unit})")
        ax.set_ylabel(f"Predicted force ({f_unit})")
        ax.set_title(
            f"{title_prefix}Force components\n"
            f"MAE={fm['mae'] * (EV_TO_KCAL_MOL if 'eV' in force_unit else 1):.4f} | R²={fm['r2']:.4f}"
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        p = output_dir / "force_parity.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

        # Per-component MAE bar chart
        fig, ax = plt.subplots(figsize=(5, 4))
        comps = ["x", "y", "z"]
        mae_vals = [fm.get(f"mae_{c}", 0.0) for c in comps]
        scale = EV_TO_KCAL_MOL if "eV" in force_unit else 1.0
        ax.bar(comps, [v * scale for v in mae_vals], color=["#4C72B0", "#55A868", "#C44E52"])
        ax.set_ylabel(f"MAE ({f_unit})")
        ax.set_title(f"{title_prefix}Force MAE by component")
        ax.grid(True, alpha=0.3, axis="y")
        p = output_dir / "force_component_mae.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

        if "per_atom_forces" in metrics:
            pa = metrics["per_atom_forces"]
            mae_pa = np.asarray(pa["mae_per_atom"], dtype=np.float64)
            elems = pa["elements_per_atom"]
            scale = EV_TO_KCAL_MOL if "eV" in force_unit else 1.0

            fig, ax = plt.subplots(figsize=(max(6, len(mae_pa) * 0.45), 4))
            x = np.arange(len(mae_pa))
            colors = plt.cm.tab10(np.linspace(0, 1, len(mae_pa)))
            ax.bar(x, mae_pa * scale, color=colors, edgecolor="black", linewidth=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{e}\n#{i}" for i, e in enumerate(elems)], fontsize=8
            )
            ax.set_ylabel(f"Force MAE ({f_unit})")
            ax.set_title(f"{title_prefix}Per-atom force MAE (‖ΔF‖)")
            ax.grid(True, alpha=0.3, axis="y")
            p = output_dir / "force_per_atom_mae.png"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved.append(p)

        if "per_element_forces" in metrics:
            pe = metrics["per_element_forces"]
            syms = list(pe.keys())
            mae_e = [pe[s]["mae"] for s in syms]
            scale = EV_TO_KCAL_MOL if "eV" in force_unit else 1.0
            fig, ax = plt.subplots(figsize=(max(5, len(syms) * 0.8), 4))
            ax.bar(syms, np.asarray(mae_e) * scale, color="#8172B2", edgecolor="black")
            ax.set_ylabel(f"MAE ‖ΔF‖ ({f_unit})")
            ax.set_title(f"{title_prefix}Per-element force MAE")
            ax.grid(True, alpha=0.3, axis="y")
            p = output_dir / "force_per_element_mae.png"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved.append(p)

    if "D_ref" in aligned and "dipole" in metrics:
        d_ref = _as_2d_dipole(aligned["D_ref"])
        d_pred = _as_2d_dipole(aligned["D_pred"])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(d_ref.reshape(-1), d_pred.reshape(-1), alpha=0.4, s=10)
        lims = [
            min(d_ref.min(), d_pred.min()),
            max(d_ref.max(), d_pred.max()),
        ]
        ax.plot(lims, lims, "r--")
        dm = metrics["dipole"]
        ax.set_xlabel("Reference dipole")
        ax.set_ylabel("Predicted dipole")
        ax.set_title(f"{title_prefix}Dipole components | R²={dm['r2']:.4f}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        p = output_dir / "dipole_parity.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # Combined summary grid
    n_plots = len(saved)
    if n_plots >= 2:
        import math

        ncol = min(3, n_plots)
        nrow = math.ceil(n_plots / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))
        axes_flat = np.atleast_1d(axes).ravel()
        for ax, img_path in zip(axes_flat, saved):
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(img_path.stem.replace("_", " "), fontsize=10)
        for ax in axes_flat[n_plots:]:
            ax.axis("off")
        fig.suptitle(f"{title_prefix}NPZ comparison summary", fontsize=12)
        p = output_dir / "comparison_summary.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    return saved


def write_comparison_report(
    metrics: Mapping[str, Any],
    output_dir: Path,
    *,
    reference: str | None = None,
    predictions: str | None = None,
    plot_paths: Sequence[Path] | None = None,
) -> Path:
    """Write metrics.json and a short markdown report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    md_path = output_dir / "comparison_report.md"
    lines = ["# NPZ comparison report\n"]
    if reference:
        lines.append(f"- Reference: `{reference}`\n")
    if predictions:
        lines.append(f"- Predictions: `{predictions}`\n")
    lines.append(f"- Frames: {metrics.get('n_frames', 'n/a')}\n\n")

    if "energy" in metrics:
        e = metrics["energy"]
        lines.append("## Energy\n\n")
        lines.append(
            f"| MAE | RMSE | R² | bias | max |err| |\n"
            f"|-----|------|----|------|----------|\n"
            f"| {e['mae']:.6g} | {e['rmse']:.6g} | {e['r2']:.4f} | "
            f"{e['bias']:.6g} | {e['max_abs_error']:.6g} |\n\n"
        )

    if "forces" in metrics:
        f = metrics["forces"]
        lines.append("## Forces\n\n")
        lines.append(
            f"| MAE | RMSE | R² | MAE ‖F‖ | MAE_x | MAE_y | MAE_z |\n"
            f"|-----|------|----|---------|-------|-------|-------|\n"
            f"| {f['mae']:.6g} | {f['rmse']:.6g} | {f['r2']:.4f} | "
            f"{f['mae_magnitude']:.6g} | {f['mae_x']:.6g} | "
            f"{f['mae_y']:.6g} | {f['mae_z']:.6g} |\n\n"
        )

    if "per_element_forces" in metrics:
        lines.append("## Per-element force MAE (‖ΔF‖)\n\n")
        lines.append("| Element | n | MAE | RMSE |\n|---------|---|-----|------|\n")
        for sym, row in sorted(metrics["per_element_forces"].items()):
            lines.append(
                f"| {sym} | {row['n']} | {row['mae']:.6g} | {row['rmse']:.6g} |\n"
            )
        lines.append("\n")

    if plot_paths:
        lines.append("## Plots\n\n")
        for pp in plot_paths:
            lines.append(f"- `{pp.name}`\n")

    md_path.write_text("".join(lines))
    return metrics_path
