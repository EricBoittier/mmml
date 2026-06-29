#!/usr/bin/env python3
"""
Extract and Plot Training Metrics from Orbax Checkpoints

This tool extracts loss values and metrics from Orbax checkpoint files
and creates comprehensive training plots with log-scale loss.

Usage:
    python -m mmml.cli.extract_checkpoint_metrics \
        checkpoints/run/run-uuid/ \
        --output training_plots.png \
        --log-loss
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union
import numpy as np
import re

from matplotlib.lines import Line2D

from mmml.utils.plotting.styles import (
    DEFAULT_PLOT_STYLE,
    PlotStyle,
    apply_plot_style,
    comparison_colors,
    get_plot_style,
    list_plot_styles,
)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("❌ Matplotlib required")
    sys.exit(1)

try:
    from orbax.checkpoint import PyTreeCheckpointer
    HAS_ORBAX = True
except ImportError:
    HAS_ORBAX = False
    print("⚠️  Orbax not available, will try pickle fallback")


def extract_metrics_from_orbax(epoch_dir: Path) -> Dict:
    """Extract metrics from Orbax checkpoint."""
    try:
        if HAS_ORBAX:
            checkpointer = PyTreeCheckpointer()
            restored = checkpointer.restore(str(epoch_dir.resolve()))
            
            metrics = {
                'epoch': restored.get('epoch', None),
                'best_loss': restored.get('best_loss', None),
                'objectives': restored.get('objectives', {}),
                'lr_eff': restored.get('lr_eff', None),
            }
            
            return metrics
    except Exception:
        return None


def collect_all_metrics(
    ckpt_dir: Path,
    verbose: bool = True,
    *,
    stride: int = 1,
    max_epochs: int | None = None,
) -> Dict[str, List]:
    """Collect metrics from epoch checkpoints (optionally subsampled)."""
    
    # Find epoch directories
    epoch_dirs = sorted([d for d in ckpt_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('epoch-')],
                       key=lambda x: int(re.search(r'(\d+)', x.name).group(1)))
    if stride > 1:
        epoch_dirs = epoch_dirs[::stride]
    if max_epochs is not None and max_epochs > 0:
        epoch_dirs = epoch_dirs[:max_epochs]
    
    if not epoch_dirs:
        raise ValueError(f"No epoch checkpoints found in {ckpt_dir}")
    
    if verbose:
        print(f"Found {len(epoch_dirs)} epoch checkpoints")
    
    # Collect metrics
    all_metrics = {
        'epochs': [],
        'train_loss': [],
        'valid_loss': [],
        'train_energy_mae': [],
        'valid_energy_mae': [],
        'train_forces_mae': [],
        'valid_forces_mae': [],
        'train_dipole_mae': [],
        'valid_dipole_mae': [],
        'lr_eff': [],
        'best_loss': [],
    }
    
    for epoch_dir in epoch_dirs:
        metrics = extract_metrics_from_orbax(epoch_dir)
        
        if metrics is None:
            continue
        
        epoch_num = metrics.get('epoch')
        if epoch_num is None:
            match = re.search(r'epoch-(\d+)', epoch_dir.name)
            if match:
                epoch_num = int(match.group(1))
        
        if epoch_num is not None:
            all_metrics['epochs'].append(epoch_num)
        
        # Extract from objectives dict
        obj = metrics.get('objectives', {})
        all_metrics['train_loss'].append(obj.get('train_loss', np.nan))
        all_metrics['valid_loss'].append(obj.get('valid_loss', np.nan))
        all_metrics['train_energy_mae'].append(obj.get('train_energy_mae', np.nan))
        all_metrics['valid_energy_mae'].append(obj.get('valid_energy_mae', np.nan))
        all_metrics['train_forces_mae'].append(obj.get('train_forces_mae', np.nan))
        all_metrics['valid_forces_mae'].append(obj.get('valid_forces_mae', np.nan))
        all_metrics['train_dipole_mae'].append(obj.get('train_dipole_mae', np.nan))
        all_metrics['valid_dipole_mae'].append(obj.get('valid_dipole_mae', np.nan))
        all_metrics['lr_eff'].append(metrics.get('lr_eff', np.nan))
        all_metrics['best_loss'].append(metrics.get('best_loss', np.nan))
    
    # Convert to numpy arrays and remove NaNs
    for key in all_metrics:
        all_metrics[key] = np.array(all_metrics[key])
    
    if verbose:
        valid_count = np.sum(~np.isnan(all_metrics['valid_loss']))
        print(f"Extracted metrics from {valid_count}/{len(epoch_dirs)} checkpoints")
    
    return all_metrics


EV_TO_KCAL_MOL = 23.0605


def _apply_training_plot_style(plot_style: str | PlotStyle | None = None) -> PlotStyle:
    return apply_plot_style(plot_style)


def _finite_series(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    return out


def _valid_metric_mask(metrics: Dict[str, np.ndarray]) -> np.ndarray:
    return ~np.isnan(_finite_series(metrics["valid_loss"]))


def _warmup_trim_mask(values: np.ndarray, *, factor: float = 40.0) -> np.ndarray:
    """Mask out early epochs with unphysically large loss spikes."""
    y = _finite_series(values)
    if y.size < 4:
        return np.ones(y.size, dtype=bool)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return np.ones(y.size, dtype=bool)
    tail = finite[max(0, finite.size // 2) :]
    ref = float(np.nanmedian(tail))
    if ref <= 0:
        ref = float(np.nanmedian(finite))
    if ref <= 0:
        return np.ones(y.size, dtype=bool)
    return y <= factor * ref


def _set_positive_log_ylim(ax, arrays: List[np.ndarray]) -> None:
    pooled = np.concatenate([_finite_series(a) for a in arrays if a is not None])
    pooled = pooled[np.isfinite(pooled) & (pooled > 0)]
    if pooled.size == 0:
        return
    lo = float(np.nanpercentile(pooled, 2))
    hi = float(np.nanpercentile(pooled, 98))
    if hi <= lo:
        hi = lo * 10.0
    ax.set_yscale("log")
    ax.set_ylim(max(lo * 0.85, np.finfo(float).tiny), hi * 1.15)


def _short_run_label(name: str, max_len: int = 14) -> str:
    label = name.replace("dcm1-", "")
    if len(label) > max_len:
        return label[: max_len - 1] + "…"
    return label


_REPEAT_LINESTYLES = ("-", "--", "-.", ":")


@dataclass(frozen=True)
class ComparisonRunSpec:
    name: str
    metrics: Dict[str, np.ndarray]
    group: str | None = None
    repeat: int | None = None


def _parse_run_group_repeat(run_name: str) -> tuple[str | None, int | None]:
    match = re.match(r"^(n\d+)/r(\d+)$", run_name.strip(), flags=re.IGNORECASE)
    if not match:
        return None, None
    return match.group(1).lower(), int(match.group(2))


def _normalize_comparison_runs(
    runs: Sequence[Union[tuple[str, Dict[str, np.ndarray]], ComparisonRunSpec]],
) -> list[ComparisonRunSpec]:
    normalized: list[ComparisonRunSpec] = []
    for item in runs:
        if isinstance(item, ComparisonRunSpec):
            normalized.append(item)
            continue
        name, metrics = item
        group, repeat = _parse_run_group_repeat(name)
        normalized.append(ComparisonRunSpec(name=name, metrics=metrics, group=group, repeat=repeat))
    return normalized


def _group_sort_key(group: str) -> tuple[int, str]:
    match = re.match(r"^n(\d+)$", group, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), group
    return 10**9, group


def _comparison_group_styles(
    specs: Sequence[ComparisonRunSpec],
    style: PlotStyle,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    groups = sorted({spec.group or spec.name for spec in specs}, key=_group_sort_key)
    palette = comparison_colors(style, max(len(groups), 1))
    group_colors = {group: palette[i % len(palette)] for i, group in enumerate(groups)}

    linestyles: dict[str, str] = {}
    for spec in specs:
        repeat = spec.repeat or 1
        linestyles[spec.name] = _REPEAT_LINESTYLES[(repeat - 1) % len(_REPEAT_LINESTYLES)]
    return group_colors, linestyles, groups


def _render_summary_table(
    ax,
    summary_table: Sequence[Sequence[str]],
    *,
    title: str,
    style: PlotStyle,
) -> None:
    ax.axis("off")
    if not summary_table:
        return
    headers = list(summary_table[0])
    rows = [list(row) for row in summary_table[1:]]
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        bbox=[0.02, 0.20, 0.96, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.15)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(fontweight="bold")
        elif col == 0:
            cell.set_text_props(fontweight="bold")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)


def _add_group_repeat_legend(ax, group_colors: dict[str, str], groups: Sequence[str]) -> None:
    handles: list[Line2D] = []
    for group in groups:
        handles.append(
            Line2D([0], [0], color=group_colors[group], lw=2.2, linestyle="-", label=group)
        )
    for repeat_idx, linestyle in enumerate(_REPEAT_LINESTYLES[:3], start=1):
        handles.append(
            Line2D(
                [0],
                [0],
                color="#333333",
                lw=1.8,
                linestyle=linestyle,
                label=f"repeat {repeat_idx}",
            )
        )
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=7, frameon=True)


def _plot_train_valid_series(
    ax,
    epochs: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    *,
    ylabel: str,
    title: str,
    log_scale: bool,
    train_scale: float = 1.0,
    valid_scale: float = 1.0,
    style: PlotStyle,
) -> int | None:
    """Plot train/valid curves; return epoch index of best validation point."""
    train_y = _finite_series(train) * train_scale
    valid_y = _finite_series(valid) * valid_scale
    ax.plot(
        epochs,
        train_y,
        color=style.colors["train"],
        linewidth=style.train_linewidth,
        alpha=0.75,
        label="Train",
    )
    ax.plot(
        epochs,
        valid_y,
        color=style.colors["valid"],
        linewidth=style.valid_linewidth,
        alpha=0.95,
        label="Valid",
    )
    best_idx = int(np.nanargmin(valid_y))
    ax.scatter(
        epochs[best_idx],
        valid_y[best_idx],
        s=style.best_marker_size,
        marker="*",
        color=style.colors["best"],
        edgecolors=style.best_marker_edge,
        linewidths=0.8,
        zorder=5,
        label=f"Best (ep {int(epochs[best_idx])})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        _set_positive_log_ylim(ax, [train_y, valid_y])
    ax.legend(loc="best")
    return best_idx


def _summary_text_box(metrics: Dict[str, np.ndarray], valid_idx: np.ndarray) -> str:
    eV_to_kcal = EV_TO_KCAL_MOL
    valid_loss = _finite_series(metrics["valid_loss"])[valid_idx]
    energy = _finite_series(metrics["valid_energy_mae"])[valid_idx] * eV_to_kcal
    forces = _finite_series(metrics["valid_forces_mae"])[valid_idx] * eV_to_kcal
    lines = [
        f"Epochs: {int(valid_idx.sum())}",
        f"Best valid loss: {np.nanmin(valid_loss):.4g}",
        f"Final valid loss: {valid_loss[-1]:.4g}",
    ]
    if np.any(np.isfinite(energy)):
        lines.append(f"Best E MAE: {np.nanmin(energy):.3f} kcal/mol")
    if np.any(np.isfinite(forces)):
        lines.append(f"Best F MAE: {np.nanmin(forces):.3f} kcal/mol/Å")
    return "\n".join(lines)


def plot_training_metrics(
    metrics: Dict[str, np.ndarray],
    output_path: Path,
    ckpt_name: str = "Training",
    log_loss: bool = True,
    verbose: bool = True,
    *,
    ef_only: bool = False,
    plot_style: str | PlotStyle | None = DEFAULT_PLOT_STYLE,
):
    """Create publication-style training curves from extracted checkpoint metrics."""
    style = _apply_training_plot_style(plot_style)

    valid_idx = _valid_metric_mask(metrics)
    if not valid_idx.any():
        raise ValueError(
            f"No metrics could be extracted from {ckpt_name}. "
            "Check GPU/CUDA availability for Orbax restore or reduce --stride."
        )

    epochs_all = np.asarray(metrics["epochs"], dtype=float)
    trim = _warmup_trim_mask(metrics["valid_loss"][valid_idx])
    plot_idx = valid_idx.copy()
    plot_idx[valid_idx] = trim

    epochs = epochs_all[plot_idx]
    eV_to_kcal = EV_TO_KCAL_MOL

    if ef_only:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        ax_loss, ax_best, ax_e, ax_f = axes.ravel()
    else:
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 0.9])
        ax_loss = fig.add_subplot(gs[0, :])
        ax_best = fig.add_subplot(gs[1, 0])
        ax_e = fig.add_subplot(gs[1, 1])
        ax_f = fig.add_subplot(gs[2, 0])
        ax_lr = fig.add_subplot(gs[2, 1])

    _plot_train_valid_series(
        ax_loss,
        epochs,
        metrics["train_loss"][plot_idx],
        metrics["valid_loss"][plot_idx],
        ylabel="Loss",
        title="Training / validation loss",
        log_scale=log_loss,
        style=style,
    )

    best_loss = _finite_series(metrics["best_loss"])[plot_idx]
    ax_best.plot(epochs, best_loss, color=style.colors["accent"], linewidth=2.2)
    ax_best.set_xlabel("Epoch")
    ax_best.set_ylabel("Best valid loss so far")
    ax_best.set_title("Best-model trace")
    if log_loss:
        _set_positive_log_ylim(ax_best, [best_loss])

    _plot_train_valid_series(
        ax_e,
        epochs,
        metrics["train_energy_mae"][plot_idx],
        metrics["valid_energy_mae"][plot_idx],
        ylabel="Energy MAE (kcal/mol)",
        title="Energy MAE",
        log_scale=log_loss,
        train_scale=eV_to_kcal,
        valid_scale=eV_to_kcal,
        style=style,
    )
    _plot_train_valid_series(
        ax_f,
        epochs,
        metrics["train_forces_mae"][plot_idx],
        metrics["valid_forces_mae"][plot_idx],
        ylabel="Forces MAE (kcal/mol/Å)",
        title="Forces MAE",
        log_scale=log_loss,
        train_scale=eV_to_kcal,
        valid_scale=eV_to_kcal,
        style=style,
    )

    if not ef_only:
        lr = _finite_series(metrics.get("lr_eff", np.array([])))[plot_idx]
        if lr.size and np.any(np.isfinite(lr)):
            ax_lr.plot(epochs, lr, color=style.colors["lr"], linewidth=2.0)
            ax_lr.set_xlabel("Epoch")
            ax_lr.set_ylabel("Learning rate")
            ax_lr.set_title("LR schedule")
            if np.all(lr[np.isfinite(lr)] > 0):
                ax_lr.set_yscale("log")
        else:
            ax_lr.axis("off")
            ax_lr.text(
                0.5,
                0.5,
                _summary_text_box(metrics, plot_idx),
                ha="center",
                va="center",
                fontsize=10,
                family=style.summary_font_family,
                transform=ax_lr.transAxes,
                bbox=dict(style.text_box),
            )

        dip_valid = _finite_series(metrics.get("valid_dipole_mae", np.array([np.nan])))
        if not np.all(np.isnan(dip_valid)):
            inset = fig.add_axes([0.68, 0.08, 0.28, 0.22])
            _plot_train_valid_series(
                inset,
                epochs,
                metrics["train_dipole_mae"][plot_idx],
                metrics["valid_dipole_mae"][plot_idx],
                ylabel="Dipole (e·Å)",
                title="Dipole MAE",
                log_scale=log_loss,
                style=style,
            )
    else:
        ax_best.text(
            0.03,
            0.97,
            _summary_text_box(metrics, plot_idx),
            transform=ax_best.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family=style.summary_font_family,
            bbox=dict(style.text_box),
        )

    mode = "energy & forces" if ef_only else "full training"
    suptitle_kw: dict = {
        "fontsize": 14,
        "fontweight": "bold",
        "y": 1.02 if not ef_only else 1.01,
    }
    if style.suptitle_color:
        suptitle_kw["color"] = style.suptitle_color
    fig.suptitle(
        f"{ckpt_name}\n{int(plot_idx.sum())} checkpoints · {mode}",
        **suptitle_kw,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"Saved: {output_path}")


def plot_training_comparison(
    runs: Sequence[Union[tuple[str, Dict[str, np.ndarray]], ComparisonRunSpec]],
    output_path: Path,
    *,
    log_scale: bool = True,
    ef_only: bool = True,
    title: str = "Training comparison",
    verbose: bool = True,
    plot_style: str | PlotStyle | None = DEFAULT_PLOT_STYLE,
    summary_table: Sequence[Sequence[str]] | None = None,
    summary_table_title: str = "Hold-out test MAE (kcal/mol)",
    color_by_group: bool = True,
    linestyle_by_repeat: bool = True,
) -> None:
    """Overlay valid metrics from multiple checkpoint runs."""
    style = _apply_training_plot_style(plot_style)
    specs = _normalize_comparison_runs(runs)
    if not specs:
        raise ValueError("No runs provided for comparison plot")

    panels = (
        ("valid_loss", "Validation loss", 1.0),
        ("valid_energy_mae", "Valid energy MAE (kcal/mol)", EV_TO_KCAL_MOL),
        ("valid_forces_mae", "Valid forces MAE (kcal/mol/Å)", EV_TO_KCAL_MOL),
    )
    if not ef_only:
        panels = panels + (("valid_dipole_mae", "Valid dipole MAE (e·Å)", 1.0),)

    use_table_panel = summary_table is not None
    ncols = 2
    if use_table_panel:
        nrows = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 4.8 * nrows), constrained_layout=True)
        axes_flat = np.atleast_1d(axes).ravel()
        curve_axes = axes_flat[: len(panels)]
        table_ax = axes_flat[len(panels)]
        for ax in axes_flat[len(panels) + 1 :]:
            ax.axis("off")
    else:
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows), constrained_layout=True)
        axes_flat = np.atleast_1d(axes).ravel()
        curve_axes = axes_flat
        table_ax = None

    group_colors, linestyles, groups = _comparison_group_styles(specs, style)
    fallback_colors = comparison_colors(style, len(specs))

    for ax, (key, panel_title, scale) in zip(curve_axes, panels):
        panel_arrays: List[np.ndarray] = []
        for idx, spec in enumerate(specs):
            metrics = spec.metrics
            mask = _valid_metric_mask(metrics)
            if not mask.any():
                continue
            epochs = np.asarray(metrics["epochs"], dtype=float)[mask]
            y = _finite_series(metrics[key])[mask] * scale
            trim = _warmup_trim_mask(y, factor=40.0)
            if trim.any():
                epochs = epochs[trim]
                y = y[trim]
            if y.size == 0:
                continue
            panel_arrays.append(y)
            group = spec.group or spec.name
            color = group_colors.get(group, fallback_colors[idx]) if color_by_group else fallback_colors[idx]
            linestyle = linestyles.get(spec.name, "-") if linestyle_by_repeat else "-"
            ax.plot(
                epochs,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=2.1,
                label=_short_run_label(spec.name),
            )
        ax.set_xlabel("Epoch")
        ax.set_title(panel_title)
        if log_scale and panel_arrays:
            _set_positive_log_ylim(ax, panel_arrays)

    if not use_table_panel:
        for ax in axes_flat[len(panels) :]:
            ax.axis("off")
        if specs:
            curve_axes[0].legend(fontsize=8, loc="best")
    else:
        assert table_ax is not None
        _render_summary_table(
            table_ax,
            summary_table or [],
            title=summary_table_title,
            style=style,
        )
        if color_by_group or linestyle_by_repeat:
            _add_group_repeat_legend(table_ax, group_colors, groups)

    suptitle_kw: dict = {"fontsize": 14, "fontweight": "bold"}
    if style.suptitle_color:
        suptitle_kw["color"] = style.suptitle_color
    fig.suptitle(title, **suptitle_kw)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"Saved: {output_path}")


@dataclass(frozen=True)
class ScalingPoint:
    n_train: int
    repeat: int
    inv_sqrt_n: float
    values: Dict[str, float]


def collect_scaling_points(runs: Sequence[dict]) -> list[ScalingPoint]:
    """Collect final validation/test metrics for 1/√n_train scaling plots."""
    points: list[ScalingPoint] = []
    for run in runs:
        summary = run.get("summary", {})
        n_train = summary.get("n_train")
        if n_train is None:
            parent = str(run.get("parent", ""))
            if parent.startswith("n"):
                n_train = int(parent[1:])
            else:
                continue
        repeat = summary.get("repeat")
        if repeat is None:
            name = str(run.get("name", ""))
            if name.startswith("r"):
                repeat = int(name[1:])
            else:
                repeat = 0
        train_final = summary.get("training_final", {})
        test_eval = summary.get("test_eval", {})
        values = {
            "valid_loss": train_final.get("valid_loss"),
            "valid_energy_mae": train_final.get("valid_energy_mae"),
            "valid_forces_mae": train_final.get("valid_forces_mae"),
            "test_energy_mae": test_eval.get("energy_mae_kcal_mol"),
            "test_forces_mae": test_eval.get("forces_mae_kcal_mol"),
        }
        clean = {}
        for key, val in values.items():
            if val is None:
                continue
            scale = EV_TO_KCAL_MOL if key in {"valid_energy_mae", "valid_forces_mae"} else 1.0
            number = float(val) * scale
            if np.isfinite(number) and number > 0:
                clean[key] = number
        if not clean:
            continue
        points.append(
            ScalingPoint(
                n_train=int(n_train),
                repeat=int(repeat),
                inv_sqrt_n=1.0 / np.sqrt(float(n_train)),
                values=clean,
            )
        )
    return points


def plot_learning_curve_scaling(
    runs: Sequence[dict],
    output_path: Path,
    *,
    title: str = "Learning-curve scaling",
    plot_style: str | PlotStyle | None = DEFAULT_PLOT_STYLE,
    verbose: bool = True,
) -> None:
    """Plot ln(final metrics) versus 1/√n_train for sweep summary."""
    style = _apply_training_plot_style(plot_style)
    points = collect_scaling_points(runs)
    if not points:
        raise ValueError("No scaling points available for plot")

    metric_panels = (
        ("valid_loss", "ln validation loss"),
        ("valid_energy_mae", "ln valid energy MAE (kcal/mol)"),
        ("valid_forces_mae", "ln valid forces MAE (kcal/mol/Å)"),
        ("test_energy_mae", "ln test energy MAE (kcal/mol)"),
        ("test_forces_mae", "ln test forces MAE (kcal/mol/Å)"),
    )

    ncols = 3
    nrows = int(np.ceil(len(metric_panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    repeats = sorted({point.repeat for point in points})
    repeat_colors = comparison_colors(style, max(len(repeats), 1))
    repeat_color = {rep: repeat_colors[i % len(repeat_colors)] for i, rep in enumerate(repeats)}

    for ax, (metric_key, ylabel) in zip(axes_flat, metric_panels):
        metric_points = [p for p in points if metric_key in p.values]
        if not metric_points:
            ax.axis("off")
            continue

        for point in metric_points:
            ax.scatter(
                point.inv_sqrt_n,
                np.log(point.values[metric_key]),
                color=repeat_color.get(point.repeat, "#333333"),
                s=42,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )

        by_n: dict[int, list[float]] = defaultdict(list)
        inv_sqrt_by_n: dict[int, float] = {}
        for point in metric_points:
            by_n[point.n_train].append(np.log(point.values[metric_key]))
            inv_sqrt_by_n[point.n_train] = point.inv_sqrt_n

        n_sorted = sorted(by_n)
        xs = [inv_sqrt_by_n[n] for n in n_sorted]
        means = [float(np.mean(by_n[n])) for n in n_sorted]
        stds = [float(np.std(by_n[n])) if len(by_n[n]) > 1 else 0.0 for n in n_sorted]
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            color=style.colors.get("valid", "#333333"),
            linewidth=2.0,
            marker="o",
            markersize=5,
            capsize=3,
            label="mean ± σ",
            zorder=4,
        )

        ax.set_xlabel("1/√n_train")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.replace("ln ", ""), fontsize=9)
        ax.invert_xaxis()
        if ax is axes_flat[0]:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=repeat_color[rep],
                    label=f"repeat {rep}",
                )
                for rep in repeats
            ]
            handles.append(
                Line2D([0], [0], color=style.colors.get("valid", "#333333"), marker="o", label="mean ± σ")
            )
            ax.legend(handles=handles, fontsize=7, loc="best")

    for ax in axes_flat[len(metric_panels) :]:
        ax.axis("off")

    suptitle_kw: dict = {"fontsize": 14, "fontweight": "bold"}
    if style.suptitle_color:
        suptitle_kw["color"] = style.suptitle_color
    fig.suptitle(title, **suptitle_kw)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"Saved: {output_path}")


def print_metrics_summary(metrics: Dict[str, np.ndarray], ckpt_dir: Path):
    """Print summary statistics."""
    
    valid_idx = ~np.isnan(metrics['valid_loss'])
    epochs_valid = metrics['epochs'][valid_idx]
    
    print("\n" + "="*80)
    print("TRAINING METRICS SUMMARY")
    print("="*80)
    print(f"\nCheckpoint: {ckpt_dir.name}")
    print(f"Epochs analyzed: {len(epochs_valid)}")
    print(f"Epoch range: {int(np.min(epochs_valid))} to {int(np.max(epochs_valid))}")
    print("")
    
    # Loss statistics
    train_loss = metrics['train_loss'][valid_idx]
    valid_loss = metrics['valid_loss'][valid_idx]
    
    print("LOSS STATISTICS")
    print("-"*80)
    print("Training Loss:")
    print(f"  Initial: {train_loss[0]:.6f}")
    print(f"  Final:   {train_loss[-1]:.6f}")
    print(f"  Best:    {np.nanmin(train_loss):.6f} (epoch {epochs_valid[np.nanargmin(train_loss)]})")
    print(f"  Improvement: {(train_loss[0] - train_loss[-1])/train_loss[0]*100:.1f}%")
    print("")
    print("Validation Loss:")
    print(f"  Initial: {valid_loss[0]:.6f}")
    print(f"  Final:   {valid_loss[-1]:.6f}")
    print(f"  Best:    {np.nanmin(valid_loss):.6f} (epoch {epochs_valid[np.nanargmin(valid_loss)]})")
    print(f"  Improvement: {(valid_loss[0] - valid_loss[-1])/valid_loss[0]*100:.1f}%")
    print("")
    
    # MAE statistics
    print("VALIDATION MAE METRICS")
    print("-"*80)
    
    # Unit conversion
    eV_to_kcal = 23.0605  # 1 eV = 23.0605 kcal/mol
    
    mae_metrics = [
        ('valid_energy_mae', 'Energy', 'kcal/mol', eV_to_kcal),
        ('valid_forces_mae', 'Forces', 'kcal/mol/Å', eV_to_kcal),
        ('valid_dipole_mae', 'Dipole', 'e·Å', 1.0),
    ]
    
    for key, name, unit, conversion in mae_metrics:
        if not np.all(np.isnan(metrics[key])):
            data = metrics[key][valid_idx] * conversion
            print(f"{name:8s} MAE: {data[-1]:.6f} {unit:12s} (best: {np.nanmin(data):.6f} @ epoch {epochs_valid[np.nanargmin(data)]})")
    
    print("")
    
    # Learning rate
    if not np.all(np.isnan(metrics['lr_eff'])):
        lr = metrics['lr_eff'][valid_idx]
        print("LEARNING RATE")
        print("-"*80)
        print(f"Initial: {lr[0]:.6e}")
        print(f"Final:   {lr[-1]:.6e}")
        if lr[-1] < lr[0]:
            print(f"Decay:   {(1 - lr[-1]/lr[0])*100:.1f}%")
        print("")
    
    # Convergence analysis
    print("CONVERGENCE ANALYSIS")
    print("-"*80)
    
    # Check if converged (last 10 epochs stable)
    if len(valid_loss) >= 10:
        last_10_std = np.std(valid_loss[-10:])
        last_10_mean = np.mean(valid_loss[-10:])
        relative_std = last_10_std / last_10_mean * 100
        
        print("Last 10 epochs:")
        print(f"  Mean loss: {last_10_mean:.6f}")
        print(f"  Std dev:   {last_10_std:.6f}")
        print(f"  Relative std: {relative_std:.2f}%")
        
        if relative_std < 1.0:
            print("  ✅ Converged (std < 1%)")
        elif relative_std < 5.0:
            print("  ⚠️  Nearly converged (std < 5%)")
        else:
            print("  ❌ Not converged (std > 5%) - consider more epochs")
    
    print("\n" + "="*80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and plot training metrics from Orbax checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot glycol training with log scale
  python -m mmml.cli.extract_checkpoint_metrics \\
      examples/glycol/checkpoints/glycol_production/glycol_production-*/ \\
      --output glycol_training.png \\
      --log-loss
  
  # Without log scale
  python -m mmml.cli.extract_checkpoint_metrics \\
      checkpoints/run/run-uuid/ \\
      --output training.png
        """
    )
    parser.add_argument('checkpoint_dir', type=Path,
                       help='Checkpoint directory containing epoch-* subdirectories')
    parser.add_argument('-o', '--output', type=Path, required=True,
                       help='Output plot file (PNG)')
    parser.add_argument('--log-loss', action='store_true',
                       help='Use log scale for loss axes (recommended)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Read every Nth epoch checkpoint (default: 1 = all). Use for large runs.',
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='Cap the number of epoch checkpoints read after stride (default: no cap).',
    )
    parser.add_argument(
        '--metrics-json',
        type=Path,
        default=None,
        help='Optional path to write extracted metrics as JSON arrays.',
    )
    parser.add_argument(
        '--ef-only',
        action='store_true',
        help='Plot energy/forces panels only (omit dipole inset from main layout).',
    )
    parser.add_argument(
        '--plot-style',
        choices=list_plot_styles(),
        default=DEFAULT_PLOT_STYLE,
        help=(
            'Matplotlib style preset '
            f'(default: {DEFAULT_PLOT_STYLE}). '
            'Options: nature, xmgrace, google, tron, mpl_classic.'
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main():
    args = parse_args()

    verbose = not args.quiet
    
    # Resolve checkpoint directory (handle wildcards)
    if '*' in str(args.checkpoint_dir):
        from glob import glob
        matches = glob(str(args.checkpoint_dir))
        if not matches:
            print(f"❌ No directories match: {args.checkpoint_dir}")
            return 1
        args.checkpoint_dir = Path(matches[0])
        if verbose:
            print(f"📂 Resolved to: {args.checkpoint_dir.name}")
    
    if not args.checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {args.checkpoint_dir}")
        return 1
    
    if not HAS_ORBAX:
        print("❌ Orbax checkpoint library required")
        print("   Install with: pip install orbax-checkpoint")
        return 1
    
    if verbose:
        print("\n" + "="*80)
        print("EXTRACTING TRAINING METRICS FROM CHECKPOINTS")
        print("="*80)
        print(f"\nCheckpoint directory: {args.checkpoint_dir.name}")
        print(f"Output: {args.output}")
        print("")
    
    # Extract metrics
    if verbose:
        print("📊 Extracting metrics from checkpoints...")
    
    stride = max(1, int(args.stride))
    metrics = collect_all_metrics(
        args.checkpoint_dir,
        verbose=verbose,
        stride=stride,
        max_epochs=args.max_epochs,
    )

    if args.metrics_json is not None:
        payload = {k: np.asarray(v).tolist() for k, v in metrics.items()}
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        import json

        with args.metrics_json.open("w") as handle:
            json.dump(payload, handle, indent=2)
        if verbose:
            print(f"Wrote metrics JSON: {args.metrics_json}")
    
    # Create plots
    if verbose:
        print("\n📈 Creating training plots...")
    
    plot_training_metrics(
        metrics,
        args.output,
        ckpt_name=args.checkpoint_dir.name,
        log_loss=args.log_loss,
        verbose=verbose,
        ef_only=args.ef_only,
        plot_style=args.plot_style,
    )
    
    # Print summary
    print_metrics_summary(metrics, args.checkpoint_dir)
    
    if verbose:
        print("\n✅ ANALYSIS COMPLETE!")
        print("\nTo evaluate the best checkpoint:")
        best_epoch = int(metrics['epochs'][np.nanargmin(metrics['valid_loss'])])
        print("  python -m mmml.cli.evaluate_model \\")
        print(f"      {args.checkpoint_dir}/epoch-{best_epoch} \\")
        print("      --test-data splits/data_test.npz")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

