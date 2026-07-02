#!/usr/bin/env python3
"""Finalize md-embedding docs from existing artifacts (no CHARMM rebuild)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.collect_md_embedding_docs_results import (  # noqa: E402
    ARTIFACTS,
    IMG,
    RESULTS_MD,
    SUMMARY_JSON,
    _copy_figures,
    _load_json,
    _training_loss_plot,
    _write_results_md,
)


def read_charmm_crd_positions(path: Path) -> np.ndarray:
    pos: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0].isdigit():
            pos.append([float(parts[4]), float(parts[5]), float(parts[6])])
    if not pos:
        raise ValueError(f"No coordinates parsed from {path}")
    return np.asarray(pos, dtype=float)


def count_peptide_atoms_psf(psf_path: Path) -> int:
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        n_peptide_atoms_in_trialanine_box,
    )

    return n_peptide_atoms_in_trialanine_box(psf_path)


def finalize(artifacts: Path = ARTIFACTS) -> int:
    out = Path(artifacts)
    psf = out / "model.psf"
    crd = out / "model.crd"
    if not psf.is_file() or not crd.is_file():
        raise FileNotFoundError(f"Need {psf} and {crd}")

    positions = read_charmm_crd_positions(crd)
    n_peptide = count_peptide_atoms_psf(psf)
    box_side = 28.0
    bonded = _load_json(out / "bonded_report.json")

    from mmml.interfaces.pycharmmInterface.mlpot.embedding_workflow import (
        plot_embedding_box_structures,
    )

    plot_embedding_box_structures(
        psf,
        positions,
        box_side_A=box_side,
        n_peptide_atoms=n_peptide,
        out_dir=out,
    )

    box_meta = {
        "workflow": "md-embedding",
        "peptide_resi": "TRIA",
        "ml_seg_id": "PEPT",
        "solvent_seg_id": "SOLV",
        "n_peptide_atoms": int(n_peptide),
        "n_waters": 10,
        "n_total_atoms": int(positions.shape[0]),
        "box_side_A": box_side,
        "training_n_atoms": 34,
        "psf": "model.psf",
        "crd": "model.crd",
        "bonded_report": bonded or None,
    }
    (out / "box.json").write_text(json.dumps(box_meta, indent=2) + "\n", encoding="utf-8")

    manifest = _load_json(out / "train_manifest.json")
    eval_raw = _load_json(out / "eval" / "metrics.json")
    eval_metrics = {
        "energy_mae_kcal_mol": (eval_raw.get("energy") or {}).get("mae_kcal_mol"),
        "energy_rmse_kcal_mol": (eval_raw.get("energy") or {}).get("rmse_kcal_mol"),
        "force_mae_kcal_mol_A": (eval_raw.get("forces") or {}).get("mae_kcal_mol"),
        "force_rmse_kcal_mol_A": (eval_raw.get("forces") or {}).get("rmse_kcal_mol"),
        "num_samples": eval_raw.get("n_batches"),
    }

    best_valid = None
    runs = sorted(
        (p for p in (out / "checkpoints").iterdir() if p.is_dir() and p.name.startswith("aaa_smoke-")),
        key=lambda p: p.stat().st_mtime,
    ) if (out / "checkpoints").is_dir() else []
    if runs:
        from mmml.cli.misc.compare_training_runs import collect_all_metrics

        m = collect_all_metrics(runs[-1], verbose=False)
        if m is not None and len(m.get("valid_loss", [])):
            best_valid = float(np.nanmin(m["valid_loss"]))

    loss_png = _training_loss_plot(out / "checkpoints", out / "figures" / "training_loss.png")
    figures = _copy_figures(out)
    if loss_png and loss_png.is_file():
        import shutil

        dest = IMG / "training_loss.png"
        shutil.copy2(loss_png, dest)
        figures.append(dest.relative_to(REPO).as_posix())

    # Copy peptide frame from train if present
    pep = out / "figures" / "peptide_frame0.png"
    if pep.is_file():
        import shutil

        dest = IMG / "peptide_frame0.png"
        shutil.copy2(pep, dest)
        if dest.relative_to(REPO).as_posix() not in figures:
            figures.append(dest.relative_to(REPO).as_posix())

    train_cfg = {}
    try:
        import yaml

        train_cfg = yaml.safe_load((out / "train_config.yaml").read_text(encoding="utf-8"))
    except Exception:
        pass

    valid_n = None
    if (out / "valid.npz").is_file():
        valid_n = int(len(np.load(out / "valid.npz")["E"]))

    summary = {
        "generated_at": manifest.get("generated_at") or "2026-07-02",
        "artifacts_dir": out.relative_to(REPO).as_posix(),
        "training": {
            "num_epochs": train_cfg.get("num_epochs"),
            "train_frames": manifest.get("dataset_report", {}).get("n_frames"),
            "valid_frames": valid_n,
            "checkpoint_json": manifest.get("checkpoint_json"),
            "best_valid_loss": best_valid,
        },
        "evaluation": eval_metrics,
        "build": {
            "n_peptide_atoms": n_peptide,
            "n_waters": 10,
            "box_side_A": box_side,
            "bonded_total_kcal_mol": (bonded or {}).get("total"),
        },
        "run": _load_json(out / "run_manifest.json"),
        "figures": sorted(set(figures)),
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_results_md(summary)
    print(f"wrote {RESULTS_MD}")
    print(f"wrote {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(finalize())
