#!/usr/bin/env python3
"""Train md-embedding PhysNet until valid E/F MAE < target (default 1 kcal/mol).

Each round trains for ``epochs_per_round`` additional epochs. On restart,
``physnet-train`` resumes at ``last_epoch + 1`` and stops at ``num_epochs``,
so ``num_epochs`` must be cumulative (not fixed at 40 every round).

Example::

    JAX_PLATFORMS=cpu uv run python scripts/train_md_embedding_converge.py \\
      -o artifacts/md_embedding/aaa_docs --target-mae 1.0 --max-rounds 15
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "artifacts" / "md_embedding" / "aaa_docs"
LONG_CFG = REPO / "mmml" / "cli" / "run" / "md_embedding_aaa_train_long.yaml"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def _epoch_index(epoch_dir: Path) -> int:
    return int(epoch_dir.name.split("-", 1)[1])


def _latest_run_dir(ckpt_root: Path, tag: str) -> Path | None:
    if not ckpt_root.is_dir():
        return None
    runs = sorted(
        (p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith(f"{tag}-")),
        key=lambda p: p.stat().st_mtime,
    )
    return runs[-1] if runs else None


def _latest_epoch(run_dir: Path) -> Path | None:
    if not run_dir.is_dir():
        return None
    epochs = sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("epoch-")),
        key=lambda p: _epoch_index(p),
    )
    return epochs[-1] if epochs else None


def _current_epoch(run_dir: Path | None) -> int:
    if run_dir is None:
        return 0
    latest = _latest_epoch(run_dir)
    return _epoch_index(latest) if latest is not None else 0


def _target_num_epochs(current_epoch: int, epochs_per_round: int) -> int:
    """Cumulative epoch ceiling passed to physnet-train."""
    if epochs_per_round <= 0:
        raise ValueError("epochs_per_round must be positive")
    return current_epoch + epochs_per_round


def _epochs_per_round(base_cfg: dict, cli_value: int | None) -> int:
    if cli_value is not None:
        return int(cli_value)
    if "epochs_per_round" in base_cfg:
        return int(base_cfg["epochs_per_round"])
    if "num_epochs" in base_cfg:
        return int(base_cfg["num_epochs"])
    return 40


def _export_json(epoch_dir: Path, out_json: Path) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "mmml.cli.__main__",
            "orbax-to-json",
            str(epoch_dir),
            "-o",
            str(out_json),
        ]
    )


def _evaluate(ckpt_json: Path, valid_npz: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "mmml.cli.__main__",
            "physnet-evaluate",
            "--checkpoint",
            str(ckpt_json),
            "--data",
            str(valid_npz),
            "-o",
            str(out_dir),
            "--natoms",
            "34",
            "--batch-size",
            "32",
            "--no-save-npz",
        ]
    )
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-mae", type=float, default=1.0)
    parser.add_argument("--max-rounds", type=int, default=15)
    parser.add_argument("--config", type=Path, default=LONG_CFG)
    parser.add_argument(
        "--epochs-per-round",
        type=int,
        default=None,
        help="Additional epochs per round (default: epochs_per_round or num_epochs from config)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a new training run instead of resuming the latest checkpoint",
    )
    parser.add_argument("--prepare-data", action="store_true", help="Run fix-and-split first")
    args = parser.parse_args(argv)

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    valid_npz = out / "valid.npz"
    train_npz = out / "train.npz"
    if args.prepare_data or not train_npz.is_file():
        _run(
            [
                sys.executable,
                "-m",
                "mmml.cli.__main__",
                "md-embedding",
                "train",
                "-o",
                str(out),
                "--skip-train",
                "--no-plot",
            ]
        )

    base_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tag = str(base_cfg.get("tag", "aaa_long"))
    epochs_per_round = _epochs_per_round(base_cfg, args.epochs_per_round)
    ckpt_root = out / "checkpoints"
    ckpt_json = out / f"{tag}_params.json"

    resume_run: Path | None = None
    if not args.fresh:
        resume_run = _latest_run_dir(ckpt_root, tag)
        if resume_run is not None:
            epoch = _current_epoch(resume_run)
            print(
                f"Resuming training run {resume_run} (epoch {epoch})",
                flush=True,
            )

    history: list[dict] = []

    for round_idx in range(1, int(args.max_rounds) + 1):
        cfg = dict(base_cfg)
        cfg["data"] = str(train_npz)
        cfg["valid_data"] = str(valid_npz)
        cfg["ckpt_dir"] = str(ckpt_root)
        cfg.pop("epochs_per_round", None)

        start_epoch = _current_epoch(resume_run)
        target_epochs = _target_num_epochs(start_epoch, epochs_per_round)
        cfg["num_epochs"] = target_epochs
        if resume_run is not None:
            cfg["restart"] = str(resume_run)

        round_cfg = out / f"train_config_round{round_idx:02d}.yaml"
        round_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        print(
            f"\n=== Training round {round_idx}/{args.max_rounds} "
            f"(epochs {start_epoch + 1}..{target_epochs}) ===",
            flush=True,
        )
        _run(
            [
                sys.executable,
                "-m",
                "mmml.cli.__main__",
                "physnet-train",
                "--config",
                str(round_cfg),
            ]
        )

        run_dir = _latest_run_dir(ckpt_root, tag)
        if run_dir is None:
            raise RuntimeError(f"No training run found under {ckpt_root}")
        end_epoch = _current_epoch(run_dir)
        if end_epoch <= start_epoch:
            raise RuntimeError(
                f"Training did not advance past epoch {start_epoch} "
                f"(still at {end_epoch}). Check restart and num_epochs."
            )

        epoch_dir = _latest_epoch(run_dir)
        if epoch_dir is None:
            raise RuntimeError(f"No orbax epoch found under {run_dir}")
        _export_json(epoch_dir, ckpt_json)
        resume_run = run_dir

        eval_dir = out / "eval" / f"round_{round_idx:02d}"
        metrics = _evaluate(ckpt_json, valid_npz, eval_dir)
        e_mae = float(metrics["energy"]["mae_kcal_mol"])
        f_mae = float(metrics["forces"]["mae_kcal_mol"])
        record = {
            "round": round_idx,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "target_epochs": target_epochs,
            "epoch_dir": str(epoch_dir),
            "energy_mae_kcal_mol": e_mae,
            "force_mae_kcal_mol_A": f_mae,
            "checkpoint_json": str(ckpt_json),
        }
        history.append(record)
        print(
            f"Round {round_idx}: epoch {end_epoch}  "
            f"E MAE={e_mae:.4f}  F MAE={f_mae:.4f} kcal/mol",
            flush=True,
        )

        manifest = {
            "tag": tag,
            "epochs_per_round": epochs_per_round,
            "target_mae_kcal_mol": float(args.target_mae),
            "rounds": history,
            "converged": e_mae < args.target_mae and f_mae < args.target_mae,
        }
        (out / "train_convergence.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        if e_mae < args.target_mae and f_mae < args.target_mae:
            print(
                f"\nConverged: E and F MAE < {args.target_mae} kcal/mol "
                f"after {round_idx} round(s).",
                flush=True,
            )
            return 0

    print(
        f"\nStopped after {args.max_rounds} rounds without reaching "
        f"E/F MAE < {args.target_mae} kcal/mol.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
