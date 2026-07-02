#!/usr/bin/env python3
"""Run short md-embedding smoke and publish figures + metrics into MkDocs.

Example::

    export CHARMM_HOME=... CHARMM_LIB_DIR=... LD_LIBRARY_PATH=...
    JAX_PLATFORMS=cpu uv run python scripts/collect_md_embedding_docs_results.py

Writes:
  - ``artifacts/md_embedding/aaa_docs/`` — train/build/eval artifacts
  - ``docs/images/examples/md-embedding/`` — PNG figures for MkDocs
  - ``docs/examples/md-embedding-results.md`` — metrics tables + figure links
  - ``mmml/data/external/md_embedding_docs_summary.json`` — machine-readable summary
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts" / "md_embedding" / "aaa_docs"
IMG = REPO / "docs" / "images" / "examples" / "md-embedding"
SUMMARY_JSON = REPO / "mmml" / "data" / "external" / "md_embedding_docs_summary.json"
RESULTS_MD = REPO / "docs" / "examples" / "md-embedding-results.md"
SHORT_CONFIG = REPO / "mmml" / "cli" / "run" / "md_embedding_aaa_train_short.yaml"


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd or REPO)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _copy_figures(artifacts: Path) -> list[str]:
    IMG.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    patterns = [
        artifacts / "figures" / "*.png",
        artifacts / "eval" / "*.png",
    ]
    for parent in (artifacts / "figures", artifacts / "eval"):
        if not parent.is_dir():
            continue
        for src in sorted(parent.glob("*.png")):
            dest = IMG / src.name
            shutil.copy2(src, dest)
            copied.append(dest.relative_to(REPO).as_posix())
    return copied


def _training_loss_plot(ckpt_dir: Path, out: Path) -> Path | None:
    """Plot train/valid loss from orbax metrics if present."""
    try:
        from mmml.cli.misc.compare_training_runs import collect_all_metrics
    except ImportError:
        return None
    if not ckpt_dir.is_dir():
        return None
    runs = sorted(
        (p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("aaa_smoke-")),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        return None
    metrics = collect_all_metrics(runs[-1], verbose=False)
    if metrics is None:
        return None
    epochs = np.asarray(metrics.get("epochs", []))
    if epochs.size == 0:
        return None
    train_loss = np.asarray(metrics.get("train_loss", []))
    valid_loss = np.asarray(metrics.get("valid_loss", []))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=140)
    ax.plot(epochs, train_loss, label="train_loss", color="#2563eb", lw=1.5)
    ax.plot(epochs, valid_loss, label="valid_loss", color="#059669", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("aaa_smoke PhysNet (short run)")
    ax.legend(frameon=False)
    ax.set_facecolor("#f8fafc")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _write_results_md(summary: dict) -> None:
    train = summary.get("training", {})
    eval_m = summary.get("evaluation", {})
    build = summary.get("build", {})
    run_m = summary.get("run", {})
    figures = summary.get("figures", [])

    def _row(k: str, v) -> str:
        return f"| {k} | {v} |"

    lines = [
        "# md-embedding smoke results (aaa.ama)",
        "",
        f"Generated: {summary.get('generated_at', 'n/a')}  ",
        f"Artifacts: `{summary.get('artifacts_dir', 'artifacts/md_embedding/aaa_docs')}`",
        "",
        "Reproduce:",
        "",
        "```bash",
        "export CHARMM_HOME=... CHARMM_LIB_DIR=... LD_LIBRARY_PATH=...",
        "JAX_PLATFORMS=cpu uv run python scripts/collect_md_embedding_docs_results.py",
        "```",
        "",
        "## Training (PhysNet smoke)",
        "",
        "| Quantity | Value |",
        "|----------|-------|",
        _row("Epochs", train.get("num_epochs", "—")),
        _row("Train frames", train.get("train_frames", "—")),
        _row("Valid frames", train.get("valid_frames", "—")),
        _row("Best valid loss", f"{train.get('best_valid_loss', '—')}"),
        _row("Checkpoint JSON", f"`{train.get('checkpoint_json', '—')}`"),
        "",
    ]
    if "training_loss.png" in [Path(f).name for f in figures]:
        lines += [
            "![Training loss](../images/examples/md-embedding/training_loss.png)",
            "",
        ]
    lines += [
        "## Validation metrics (PhysNet vs NPZ labels)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        _row("Energy MAE", f"{eval_m.get('energy_mae_kcal_mol', '—')} kcal/mol"),
        _row("Energy RMSE", f"{eval_m.get('energy_rmse_kcal_mol', '—')} kcal/mol"),
        _row("Force MAE", f"{eval_m.get('force_mae_kcal_mol_A', '—')} kcal/mol/Å"),
        _row("Force RMSE", f"{eval_m.get('force_rmse_kcal_mol_A', '—')} kcal/mol/Å"),
        _row("Eval samples", eval_m.get("num_samples", "—")),
        "",
    ]
    for name in ("parity_plots.png", "training_loss.png"):
        if name in [Path(f).name for f in figures]:
            lines.append(f"![{name}](../images/examples/md-embedding/{name})")
            lines.append("")
    lines += [
        "## Build (CHARMM TRIA + TIP3)",
        "",
        "| Quantity | Value |",
        "|----------|-------|",
        _row("Peptide atoms (TRIA)", build.get("n_peptide_atoms", "—")),
        _row("Waters", build.get("n_waters", "—")),
        _row("Box side (Å)", build.get("box_side_A", "—")),
        _row("Bonded total (kcal/mol)", build.get("bonded_total_kcal_mol", "—")),
        "",
    ]
    for name in ("embedding_box.png", "embedding_peptide.png", "peptide_frame0.png"):
        if name in [Path(f).name for f in figures]:
            lines.append(f"![{name}](../images/examples/md-embedding/{name})")
            lines.append("")
    if run_m:
        lines += [
            "## Run (partial MLpot registration)",
            "",
            "| Quantity | Value |",
            "|----------|-------|",
            _row("CHARMM total ENER (kcal/mol)", run_m.get("charmm_total_energy_kcalmol", "—")),
            _row("ML segment", run_m.get("ml_seg_id", "PEPT")),
            "",
        ]
    lines.append(
        "See also: [md-embedding design](md-embedding-design.md), "
        "[aaa.ama workflow](aaa-ama-workflow.md)."
    )
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    best_valid = None
    if not args.skip_train:
        _run(
            [
                py,
                "-m",
                "mmml.cli.__main__",
                "md-embedding",
                "train",
                "-o",
                str(out),
                "--config",
                str(SHORT_CONFIG),
            ]
        )
        runs = sorted(
            (p for p in (out / "checkpoints").iterdir() if p.is_dir() and p.name.startswith("aaa_smoke-")),
            key=lambda p: p.stat().st_mtime,
        ) if (out / "checkpoints").is_dir() else []
        if runs:
            try:
                from mmml.cli.misc.compare_training_runs import collect_all_metrics

                m = collect_all_metrics(runs[-1], verbose=False)
                if m is not None and len(m.get("valid_loss", [])):
                    best_valid = float(np.nanmin(m["valid_loss"]))
            except Exception:
                pass

    manifest = _load_json(out / "train_manifest.json")
    if best_valid is not None:
        manifest["best_valid_loss"] = best_valid
    ckpt_json = manifest.get("checkpoint_json")
    if not ckpt_json or not Path(ckpt_json).is_file():
        cand = out / "aaa_smoke_params.json"
        ckpt_json = str(cand) if cand.is_file() else None

    eval_dir = out / "eval"
    eval_metrics: dict = {}
    if not args.skip_eval and ckpt_json:
        valid_npz = out / "valid.npz"
        if valid_npz.is_file():
            eval_dir.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    py,
                    "-m",
                    "mmml.cli.__main__",
                    "physnet-evaluate",
                    "--checkpoint",
                    ckpt_json,
                    "--data",
                    str(valid_npz),
                    "-o",
                    str(eval_dir),
                    "--natoms",
                    "34",
                    "--batch-size",
                    "32",
                    "--plots",
                    "--num-samples",
                    "200",
                ]
            )
            metrics_path = eval_dir / "metrics.json"
            if metrics_path.is_file():
                raw = _load_json(metrics_path)
                eval_metrics = {
                    "energy_mae_kcal_mol": (raw.get("energy") or {}).get("mae_kcal_mol"),
                    "energy_rmse_kcal_mol": (raw.get("energy") or {}).get("rmse_kcal_mol"),
                    "force_mae_kcal_mol_A": (raw.get("forces") or {}).get("mae_kcal_mol"),
                    "force_rmse_kcal_mol_A": (raw.get("forces") or {}).get("rmse_kcal_mol"),
                    "num_samples": raw.get("n_batches"),
                }
                parity = eval_dir / "parity_plots.png"
                if parity.is_file():
                    pass  # copied with glob below

    if not args.skip_build:
        _run(
            [
                py,
                "-m",
                "mmml.cli.__main__",
                "md-embedding",
                "build",
                "-o",
                str(out),
                "--n-waters",
                "10",
                "--box-side-A",
                "28",
            ]
        )

    run_metrics: dict = {}
    if not args.skip_run and ckpt_json and (out / "box.json").is_file():
        _run(
            [
                py,
                "-m",
                "mmml.cli.__main__",
                "md-embedding",
                "run",
                "-o",
                str(out),
                "--checkpoint",
                ckpt_json,
                "--mini-nstep",
                "20",
            ]
        )
        run_metrics = _load_json(out / "run_manifest.json")

    loss_png = _training_loss_plot(out / "checkpoints", out / "figures" / "training_loss.png")
    figures = _copy_figures(out)
    if loss_png is not None:
        dest = IMG / "training_loss.png"
        shutil.copy2(loss_png, dest)
        figures.append(dest.relative_to(REPO).as_posix())

    box_meta = _load_json(out / "box.json")
    bonded = box_meta.get("bonded_report") or {}
    train_cfg: dict = {}
    try:
        import yaml

        if (out / "train_config.yaml").is_file():
            train_cfg = yaml.safe_load((out / "train_config.yaml").read_text(encoding="utf-8"))
    except Exception:
        train_cfg = {}

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "artifacts_dir": out.relative_to(REPO).as_posix(),
        "training": {
            "num_epochs": train_cfg.get("num_epochs"),
            "train_frames": manifest.get("dataset_report", {}).get("n_frames"),
            "valid_frames": None,
            "checkpoint_json": ckpt_json,
            "best_valid_loss": manifest.get("best_valid_loss"),
        },
        "evaluation": {
            "energy_mae_kcal_mol": eval_metrics.get("energy_mae_kcal_mol"),
            "energy_rmse_kcal_mol": eval_metrics.get("energy_rmse_kcal_mol"),
            "force_mae_kcal_mol_A": eval_metrics.get("force_mae_kcal_mol_A"),
            "force_rmse_kcal_mol_A": eval_metrics.get("force_rmse_kcal_mol_A"),
            "num_samples": eval_metrics.get("num_samples"),
        },
        "build": {
            "n_peptide_atoms": box_meta.get("n_peptide_atoms"),
            "n_waters": box_meta.get("n_waters"),
            "box_side_A": box_meta.get("box_side_A"),
            "bonded_total_kcal_mol": bonded.get("total"),
        },
        "run": run_metrics,
        "figures": sorted(set(figures)),
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_results_md(summary)
    print(f"wrote {RESULTS_MD.relative_to(REPO)}")
    print(f"wrote {SUMMARY_JSON.relative_to(REPO)}")
    print(f"figures: {len(figures)} under {IMG.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
