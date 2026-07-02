#!/usr/bin/env python3
"""Train md-embedding PhysNet until valid E/F MAE < target (default 1 kcal/mol).

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
        key=lambda p: int(p.name.split("-", 1)[1]),
    )
    return epochs[-1] if epochs else None


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
    ckpt_root = out / "checkpoints"
    ckpt_json = out / f"{tag}_params.json"
    restart: str | None = None
    existing = _latest_run_dir(ckpt_root, tag)
    if existing is not None:
        restart = str(existing)
        print(f"Resuming training run {restart}", flush=True)
    history: list[dict] = []

    for round_idx in range(1, int(args.max_rounds) + 1):
        cfg = dict(base_cfg)
        cfg["data"] = str(train_npz)
        cfg["valid_data"] = str(valid_npz)
        cfg["ckpt_dir"] = str(ckpt_root)
        if restart is not None:
            cfg["restart"] = restart
        round_cfg = out / f"train_config_round{round_idx:02d}.yaml"
        round_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        print(f"\n=== Training round {round_idx}/{args.max_rounds} ===", flush=True)
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
        epoch_dir = _latest_epoch(run_dir) if run_dir else None
        if epoch_dir is None:
            raise RuntimeError(f"No orbax epoch found under {ckpt_root}")
        _export_json(epoch_dir, ckpt_json)
        restart = str(run_dir)

        eval_dir = out / "eval" / f"round_{round_idx:02d}"
        metrics = _evaluate(ckpt_json, valid_npz, eval_dir)
        e_mae = float(metrics["energy"]["mae_kcal_mol"])
        f_mae = float(metrics["forces"]["mae_kcal_mol"])
        record = {
            "round": round_idx,
            "epoch_dir": str(epoch_dir),
            "energy_mae_kcal_mol": e_mae,
            "force_mae_kcal_mol_A": f_mae,
            "checkpoint_json": str(ckpt_json),
        }
        history.append(record)
        print(
            f"Round {round_idx}: E MAE={e_mae:.4f}  F MAE={f_mae:.4f} kcal/mol",
            flush=True,
        )

        manifest = {
            "tag": tag,
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
            # Also publish as default checkpoint alias for run phase
            alias = out / "aaa_smoke_params.json"
            if ckpt_json != alias:
                alias.write_bytes(ckpt_json.read_bytes())
            return 0

    print(
        f"\nStopped after {args.max_rounds} rounds without reaching "
        f"E/F MAE < {args.target_mae} kcal/mol.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
