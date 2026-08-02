#!/usr/bin/env python3
"""Figure: monomer skeleton deviation, healthy CHARMM relax versus a corrupted cache.

Consumes the JSON files written by ``scripts/validate_packmol_monomer_geometry.py``
(real cluster builds) and, optionally, a Packmol cache entry whose coordinates are
known bad, and shows why
``mmml.utils.monomer_internal_geometry.DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A``
sits where it does.

Example
-------
  python scripts/plot_packmol_monomer_geometry_validation.py \\
    --json results/*.json \\
    --cache-entry local_validation/meoh_fix/.packmol_cache/eb33b00d98a0e2fa5bb74407 \\
    --out docs/images/packmol-monomer-geometry-gate/deviation_distribution.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_runs(paths: list[Path]) -> list[dict]:
    runs = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        deviations = data.get("deviations_A")
        if deviations is None:
            # Older summaries carry statistics only; keep them for the bar panel.
            data["_deviations"] = None
        else:
            arr = np.array([np.nan if d is None else float(d) for d in deviations])
            data["_deviations"] = arr[np.isfinite(arr)]
        data["_label"] = Path(path).stem
        runs.append(data)
    return runs


def _cache_entry_deviations(entry: Path) -> tuple[np.ndarray, str]:
    """Per-monomer deviations for a Packmol cache entry on disk."""
    from mmml.utils.monomer_internal_geometry import scan_monomer_internal_geometry

    data = np.load(entry / "cluster.npz", allow_pickle=False)
    residue_names = [str(x).upper() for x in data["residue_names"]]
    atoms_per_list = [int(x) for x in data["atoms_per_list"]]
    templates = {}
    for monomer_npz in sorted(entry.glob("monomer_*.npz")):
        key = monomer_npz.stem.split("monomer_", 1)[1].upper()
        mon = np.load(monomer_npz, allow_pickle=False)
        templates[key] = (np.asarray(mon["coords"], float), np.asarray(mon["z"], int))
    if not templates:
        raise SystemExit(f"no monomer_*.npz templates in {entry}")
    deviations, _report = scan_monomer_internal_geometry(
        np.asarray(data["positions"], float),
        atoms_per_list,
        residue_names=residue_names,
        templates=templates,
    )
    finite = deviations[np.isfinite(deviations)]
    manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
    label = "+".join(f"{r}:{n}" for r, n in manifest.get("composition", []))
    return finite, label or entry.name


def _plot_worst_monomer(entry: Path, out: Path) -> None:
    """Draw the template monomer next to the worst monomer in a cache entry."""
    import matplotlib.pyplot as plt

    from mmml.utils.monomer_internal_geometry import (
        DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A as LIMIT,
        covalent_skeleton_pairs,
        scan_monomer_internal_geometry,
    )
    from mmml.utils.plotting.styles import status_color

    data = np.load(entry / "cluster.npz", allow_pickle=False)
    residue_names = [str(x).upper() for x in data["residue_names"]]
    atoms_per_list = [int(x) for x in data["atoms_per_list"]]
    positions = np.asarray(data["positions"], float)
    templates = {}
    names_by_residue = {}
    for monomer_npz in sorted(entry.glob("monomer_*.npz")):
        key = monomer_npz.stem.split("monomer_", 1)[1].upper()
        mon = np.load(monomer_npz, allow_pickle=False)
        templates[key] = (np.asarray(mon["coords"], float), np.asarray(mon["z"], int))
        names_by_residue[key] = [str(x) for x in mon["names"]]

    _dev, report = scan_monomer_internal_geometry(
        positions, atoms_per_list, residue_names=residue_names, templates=templates
    )
    worst = report.worst
    if worst is None:
        raise SystemExit(f"nothing to draw for {entry}")

    offsets = np.concatenate([[0], np.cumsum(np.asarray(atoms_per_list, int))])
    s, e = int(offsets[worst.monomer]), int(offsets[worst.monomer + 1])
    coords, numbers = templates[worst.residue]
    labels = names_by_residue[worst.residue]
    pairs = covalent_skeleton_pairs(coords, numbers)
    bonds = [
        (int(i), int(j))
        for i, j in pairs
        if np.linalg.norm(coords[i] - coords[j]) < 1.8  # 1-2 only, for drawing
    ]

    from mmml.utils.structure_align import align_positions

    template = coords - coords.mean(axis=0)
    broken = align_positions(positions[s:e] - positions[s:e].mean(axis=0), template)

    # Shared 2D frame: first axis along the offending pair so it cannot project
    # onto itself, second axis the largest orthogonal component of the structure.
    axis1 = template[worst.atom_i] - template[worst.atom_j]
    axis1 = axis1 / np.linalg.norm(axis1)
    residual = template - np.outer(template @ axis1, axis1)
    _u, _s_vals, vt = np.linalg.svd(residual - residual.mean(axis=0))
    axis2 = vt[0] - float(vt[0] @ axis1) * axis1
    axis2 = axis2 / np.linalg.norm(axis2)
    proj = np.column_stack([axis1, axis2])
    ok_color = status_color("good")
    bad_color = status_color("critical")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharex=True, sharey=True)
    for ax, xyz, color, title in (
        (axes[0], template, ok_color, f"{worst.residue} template placed by Packmol"),
        (
            axes[1],
            broken,
            bad_color,
            f"monomer {worst.monomer + 1} from the cached cluster",
        ),
    ):
        p2 = xyz @ proj
        for i, j in bonds:
            d = float(np.linalg.norm(xyz[i] - xyz[j]))
            stretched = abs(d - float(np.linalg.norm(template[i] - template[j]))) > LIMIT
            ax.plot(
                [p2[i, 0], p2[j, 0]],
                [p2[i, 1], p2[j, 1]],
                color=bad_color if stretched else "0.45",
                lw=2.4 if stretched else 1.6,
                ls=":" if stretched else "-",
                zorder=1,
            )
        # The pair the gate reports, in both panels, for a like-for-like read.
        wi, wj = worst.atom_i, worst.atom_j
        ax.plot(
            [p2[wi, 0], p2[wj, 0]],
            [p2[wi, 1], p2[wj, 1]],
            color=color, lw=1.4, ls="--", zorder=1,
        )
        ax.annotate(
            f"{float(np.linalg.norm(xyz[wi] - xyz[wj])):.2f} Å",
            (0.5 * (p2[wi, 0] + p2[wj, 0]), 0.5 * (p2[wi, 1] + p2[wj, 1])),
            fontsize=8, color=color, xytext=(0, -12), textcoords="offset points",
            ha="center", fontweight="bold",
        )
        sizes = 260.0 * (np.asarray(numbers, float) / numbers.max()) ** 0.5
        ax.scatter(p2[:, 0], p2[:, 1], s=sizes, color=color, zorder=2, edgecolor="white")
        for k, name in enumerate(labels):
            ax.annotate(name, (p2[k, 0], p2[k, 1]), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("Å")
    axes[0].set_ylabel("Å")
    fig.suptitle(
        f"Worst monomer: {labels[worst.atom_i]}–{labels[worst.atom_j]} "
        f"{worst.template_distance_A:.2f} Å → {worst.distance_A:.2f} Å "
        f"({worst.deviation_A:.2f} Å change; gate {LIMIT:g} Å)",
        fontsize=10,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", nargs="+", type=Path, required=True, help="Validation summaries")
    parser.add_argument(
        "--cache-entry",
        type=Path,
        default=None,
        help="Packmol cache dir with known-bad cluster.npz (adds the failure case)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output PNG")
    parser.add_argument(
        "--out-structure",
        type=Path,
        default=None,
        help="Second PNG: worst monomer of --cache-entry next to its template",
    )
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mmml.utils.monomer_internal_geometry import (
        DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A as LIMIT,
    )
    from mmml.utils.plotting.styles import apply_plot_style, status_color

    apply_plot_style()
    runs = _load_runs(list(args.json))
    bad: np.ndarray | None = None
    bad_label = ""
    if args.cache_entry is not None:
        bad, bad_label = _cache_entry_deviations(Path(args.cache_entry))

    ok_color = status_color("good")
    bad_color = status_color("critical")
    limit_color = status_color("warning")

    fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    bins = np.logspace(-3.2, 0.7, 60)
    for run in runs:
        dev = run["_deviations"]
        if dev is None or dev.size == 0:
            continue
        ax_hist.hist(
            dev,
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=ok_color,
            alpha=0.85,
        )
    if bad is not None and bad.size:
        ax_hist.hist(
            bad,
            bins=bins,
            histtype="stepfilled",
            linewidth=1.6,
            color=bad_color,
            alpha=0.35,
            edgecolor=bad_color,
        )
    ax_hist.axvline(LIMIT, color=limit_color, linestyle="--", linewidth=2.0)
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("per-monomer max 1-2/1-3 distance change vs template (Å)")
    ax_hist.set_ylabel("monomers")
    ax_hist.set_title("Monomer skeleton after the cluster MM relax")
    handles = [
        plt.Line2D([], [], color=ok_color, lw=1.8, label="healthy CHARMM (pc-studix)"),
        plt.Line2D([], [], color=limit_color, lw=2.0, ls="--", label=f"gate {LIMIT:g} Å"),
    ]
    if bad is not None:
        handles.insert(1, plt.Line2D([], [], color=bad_color, lw=6, alpha=0.45,
                                     label=f"corrupted cache ({bad_label})"))
    ax_hist.legend(handles=handles, fontsize=8, loc="upper left")

    labels: list[str] = []
    maxima: list[float] = []
    colors: list[str] = []
    for run in runs:
        labels.append(run.get("_label", "run"))
        maxima.append(float(run["report"]["max_deviation_A"]))
        colors.append(ok_color)
    if bad is not None and bad.size:
        labels.append(f"CORRUPTED {bad_label}")
        maxima.append(float(np.max(bad)))
        colors.append(bad_color)

    y = np.arange(len(labels))
    ax_bar.barh(y, maxima, color=colors, alpha=0.85)
    ax_bar.axvline(LIMIT, color=limit_color, linestyle="--", linewidth=2.0)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=7)
    ax_bar.invert_yaxis()
    ax_bar.set_xscale("log")
    ax_bar.set_xlabel("worst monomer in the build (Å)")
    ax_bar.set_title("Worst case per build vs the gate")
    for yi, value in zip(y, maxima):
        ax_bar.text(value * 1.12, yi, f"{value:.3f}", va="center", fontsize=7)
    ax_bar.set_xlim(right=max(maxima) * 3.0)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")

    healthy_max = max(
        (float(r["report"]["max_deviation_A"]) for r in runs), default=float("nan")
    )
    print(f"healthy worst monomer: {healthy_max:.4f} A  (gate {LIMIT:g} A)")
    if bad is not None and bad.size:
        print(
            f"corrupted worst monomer: {float(np.max(bad)):.4f} A, "
            f"{100.0 * float((bad > LIMIT).mean()):.0f}% of monomers over the gate"
        )
    if args.out_structure is not None:
        if args.cache_entry is None:
            raise SystemExit("--out-structure needs --cache-entry")
        _plot_worst_monomer(Path(args.cache_entry), Path(args.out_structure))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
