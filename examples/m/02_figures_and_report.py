#!/usr/bin/env python3
"""Build house-style figures + MkDocs report for the NH3–CH3Cl PhysNet example.

Reads ``artifacts/nh3_ch3cl/evaluate/`` (from ``01_evaluate.sh``) and optional
``md_summary.json`` files from the free-space MD smokes, then writes:

  - ``docs/images/examples/nh3-ch3cl/*.png``
  - ``docs/examples/nh3-ch3cl-results.md``
  - ``artifacts/nh3_ch3cl/report_summary.json``
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts" / "nh3_ch3cl"
IMG = REPO / "docs" / "images" / "examples" / "nh3-ch3cl"
RESULTS_MD = REPO / "docs" / "examples" / "nh3-ch3cl-results.md"
CKPT = REPO / "examples" / "m" / "kl.json"
DATA = REPO / "examples" / "m" / "nh3_ch3cl_filtered.npz"
COMMIT = "30eb7a01f7fcf1d42a795f188526a80e547110fd"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parity_plots(pred_npz: Path, out_png: Path) -> Path | None:
    if not pred_npz.is_file():
        return None
    data = np.load(pred_npz)
    e_ref = np.asarray(data["E_ref_kcal_mol"]).reshape(-1)
    e_pred = np.asarray(data["E_pred_kcal_mol"]).reshape(-1)
    f_ref = np.asarray(data["F_ref_kcal_mol"]).reshape(-1)
    f_pred = np.asarray(data["F_pred_kcal_mol"]).reshape(-1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mmml.utils.plotting.styles import apply_plot_style, comparison_colors

    style = apply_plot_style("icml")
    colors = comparison_colors(style, n=2)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), dpi=160)

    def _panel(ax, xref, xpred, label, color):
        ax.scatter(xref, xpred, s=8, alpha=0.35, color=color, rasterized=True)
        lo = float(min(xref.min(), xpred.min()))
        hi = float(max(xref.max(), xpred.max()))
        ax.plot([lo, hi], [lo, hi], color="0.35", lw=1.0, zorder=0)
        ax.set_xlabel(f"{label} ref")
        ax.set_ylabel(f"{label} pred")
        mae = float(np.mean(np.abs(xpred - xref)))
        rmse = float(np.sqrt(np.mean((xpred - xref) ** 2)))
        ax.set_title(f"MAE={mae:.3f}  RMSE={rmse:.3f}")

    _panel(axes[0], e_ref, e_pred, r"$E$ (kcal/mol)", colors[0])
    _panel(axes[1], f_ref, f_pred, r"$F$ (kcal/mol/Å)", colors[1])
    fig.suptitle("NH₃–CH₃Cl PhysNet (kl.json)", y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _md_trace_plot(summaries: list[dict], out_png: Path) -> Path | None:
    if not summaries:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mmml.utils.plotting.styles import apply_plot_style, comparison_colors, legend_outside

    style = apply_plot_style("icml")
    colors = comparison_colors(style, n=max(len(summaries), 1))
    nve = [s for s in summaries if s.get("ensemble") == "nve"]
    nvt = [s for s in summaries if s.get("ensemble") == "nvt"]
    ncols = int(bool(nve)) + int(bool(nvt))
    if ncols == 0:
        return None
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.4), dpi=160)
    if ncols == 1:
        axes = [axes]
    col = 0
    if nve:
        ax = axes[col]
        for i, s in enumerate(nve):
            y = np.asarray(s.get("E_trace_kcal_mol", []), dtype=float)
            if y.size == 0:
                continue
            y = y - y[0]
            ax.plot(y, color=colors[i % len(colors)], label=f"{s.get('backend')} ΔE")
        ax.set_xlabel("sample")
        ax.set_ylabel(r"$\Delta E$ (kcal/mol)")
        ax.set_title("NVE energy drift")
        legend_outside(ax)
        col += 1
    if nvt:
        ax = axes[col]
        for i, s in enumerate(nvt):
            y = np.asarray(s.get("T_trace_K", []), dtype=float)
            if y.size == 0:
                continue
            ax.plot(y, color=colors[i % len(colors)], label=f"{s.get('backend')} T")
        ax.set_xlabel("sample")
        ax.set_ylabel("T (K)")
        ax.set_title("NVT temperature")
        legend_outside(ax)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _collect_md_summaries(art: Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted(art.glob("*/md_summary.json")):
        out.append(_load_json(path))
    for path in sorted(art.glob("md_system_*/*/md_summary.json")):
        out.append(_load_json(path))
    # Also pick up md-system run manifests if present under free_* dirs.
    return [s for s in out if s]


def _write_results_md(summary: dict) -> None:
    ev = summary.get("evaluation", {})
    e = ev.get("energy", {})
    f = ev.get("forces", {})
    d = ev.get("dipole", {})
    md_rows = summary.get("md", [])
    figures = summary.get("figures", [])

    def row(k: str, v) -> str:
        return f"| {k} | {v} |"

    lines = [
        "# NH₃–CH₃Cl PhysNet results",
        "",
        f"Generated: {summary.get('generated_at', 'n/a')}  ",
        f"Checkpoint / data commit: `{COMMIT}`  ",
        f"Artifacts: `{summary.get('artifacts_dir', 'artifacts/nh3_ch3cl')}`",
        "",
        "Reproduce:",
        "",
        "```bash",
        "source examples/m/_env.sh",
        "bash examples/m/01_evaluate.sh",
        "bash examples/m/run_md_smokes.sh   # ASE / JAX-MD / PyCHARMM NVE+NVT",
        "uv run python examples/m/02_figures_and_report.py",
        "```",
        "",
        "## Checkpoint & dataset",
        "",
        "| Quantity | Value |",
        "|----------|-------|",
        row("Checkpoint", "`examples/m/kl.json`"),
        row("Dataset", "`examples/m/nh3_ch3cl_filtered.npz`"),
        row("Padded atoms", "9"),
        row("Frames (N=9 / N=4 / N=5)", "12000 / 2000 / 2000"),
        row("Eval samples", ev.get("n_eval", "—")),
        "",
        "## Validation metrics (PhysNet vs NPZ labels)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        row("Energy MAE", f"{e.get('mae_kcal_mol', '—')} kcal/mol"),
        row("Energy RMSE", f"{e.get('rmse_kcal_mol', '—')} kcal/mol"),
        row("Force MAE", f"{f.get('mae_kcal_mol', '—')} kcal/mol/Å"),
        row("Force RMSE", f"{f.get('rmse_kcal_mol', '—')} kcal/mol/Å"),
    ]
    if d:
        lines += [
            row("Dipole MAE", f"{d.get('mae_e_bohr', '—')} e·Bohr"),
            row("Dipole RMSE", f"{d.get('rmse_e_bohr', '—')} e·Bohr"),
        ]
    lines.append("")
    if any(Path(p).name == "parity_plots.png" for p in figures):
        lines += [
            "![Energy/force parity](../images/examples/nh3-ch3cl/parity_plots.png)",
            "",
        ]
    lines += [
        "## Free-space MD smokes",
        "",
        "ML-only vacuum runs from a dataset dimer frame (ASE / JAX-MD) and",
        "`md-system` Packmol `AMM1:1,CH3CL:1` (ASE / JAX-MD / PyCHARMM) with",
        "`--no-include-mm`.",
        "",
    ]
    if md_rows:
        lines += [
            "| Backend | Ensemble | Steps | ΔE or ⟨T⟩ | Artifact |",
            "|---------|----------|-------|-----------|----------|",
        ]
        for m in md_rows:
            if m.get("ensemble") == "nve":
                metric = f"ΔE={m.get('dE_kcal_mol', '—'):.4g} kcal/mol" if isinstance(
                    m.get("dE_kcal_mol"), (int, float)
                ) else f"ΔE={m.get('dE_kcal_mol', '—')}"
            else:
                metric = f"⟨T⟩={m.get('T_mean_K', '—')} K"
            lines.append(
                f"| {m.get('backend', '—')} | {m.get('ensemble', '—')} | "
                f"{m.get('n_steps', '—')} | {metric} | `{m.get('source', '')}` |"
            )
        lines.append("")
    if any(Path(p).name == "md_traces.png" for p in figures):
        lines += [
            "![MD traces](../images/examples/nh3-ch3cl/md_traces.png)",
            "",
        ]
    lines += [
        "## Related",
        "",
        "- Example README: [`examples/m/README.md`](../../examples/m/README.md)",
        "- CLI: [`physnet-evaluate`](../cli/commands/physnet-evaluate.md), "
        "[`md-system`](../cli/commands/md-system.md)",
        "",
    ]
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ART)
    args = parser.parse_args()
    art = Path(args.artifacts).resolve()
    eval_dir = art / "evaluate"
    metrics = _load_json(eval_dir / "metrics.json")
    if not metrics:
        print(
            f"Missing {eval_dir / 'metrics.json'}; run examples/m/01_evaluate.sh first.",
            file=sys.stderr,
        )
        return 1

    IMG.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []

    # Prefer house-style replot from predictions; fall back to physnet-evaluate PNG.
    pred = eval_dir / "predictions.npz"
    parity = _parity_plots(pred, IMG / "parity_plots.png")
    if parity is None:
        src = eval_dir / "parity_plots.png"
        if src.is_file():
            shutil.copy2(src, IMG / "parity_plots.png")
            parity = IMG / "parity_plots.png"
    if parity is not None:
        figures.append(parity.relative_to(REPO).as_posix())

    md_summaries = _collect_md_summaries(art)
    md_rows = []
    for s in md_summaries:
        row = dict(s)
        row["source"] = "artifacts/nh3_ch3cl/"
        md_rows.append(row)
    md_png = _md_trace_plot(md_summaries, IMG / "md_traces.png")
    if md_png is not None:
        figures.append(md_png.relative_to(REPO).as_posix())

    n_eval = int(metrics.get("n_batches", 0)) * int(metrics.get("batch_size", 0))
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "artifacts_dir": art.relative_to(REPO).as_posix(),
        "checkpoint": str(CKPT),
        "data": str(DATA),
        "commit": COMMIT,
        "evaluation": {
            **metrics,
            "n_eval": n_eval or metrics.get("n_batches"),
        },
        "md": md_rows,
        "figures": figures,
    }
    (art / "report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_results_md(summary)
    print(f"Wrote {RESULTS_MD}")
    print(f"Wrote {art / 'report_summary.json'}")
    for fig in figures:
        print(f"  figure: {fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
