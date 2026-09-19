"""Efield-train history / Orbax metadata and job audit (no datasets, no CHARMM)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.models.efield.args import build_train_parser
from mmml.models.efield.audit import audit_job, format_report, main as audit_main, parse_slurm_log
from mmml.models.efield.checkpointing import (
    append_history,
    epoch_record,
    is_orbax_checkpoint,
    load_params_orbax,
    read_history,
    save_params_orbax,
    write_best_valid,
    write_run_meta,
)

pytestmark = pytest.mark.data_loading


def test_parser_save_format_default_both():
    args = build_train_parser().parse_args([])
    assert args.save_format == "both"
    args = build_train_parser().parse_args(["--save-format", "orbax"])
    assert args.save_format == "orbax"


def test_history_and_best_valid_roundtrip(tmp_path: Path):
    rec1 = epoch_record(
        epoch=1,
        run_uuid="u1",
        improved=True,
        best_epoch=1,
        best_valid_loss=12.0,
        patience_counter=0,
        train_loss=15.0,
        valid_loss=12.0,
        train_energy_mae_kcal=1.0,
        valid_energy_mae_kcal=1.1,
        train_forces_mae_kcal=2.0,
        valid_forces_mae_kcal=2.1,
        train_polar_mae=34.8,
        valid_polar_mae=40.04,
        n_train=15975,
        n_valid=888,
        batch_size=4,
        polar_weight=100.0,
        epoch_wall_s=3600.0,
    )
    rec2 = epoch_record(
        epoch=2,
        run_uuid="u1",
        improved=False,
        best_epoch=1,
        best_valid_loss=12.0,
        patience_counter=1,
        train_loss=14.0,
        valid_loss=12.5,
        train_energy_mae_kcal=0.9,
        valid_energy_mae_kcal=1.0,
        train_forces_mae_kcal=1.8,
        valid_forces_mae_kcal=2.0,
        train_polar_mae=34.8,
        valid_polar_mae=40.04,
        n_train=15975,
        n_valid=888,
        batch_size=4,
        polar_weight=100.0,
    )
    append_history(tmp_path, rec1)
    append_history(tmp_path, rec2)
    write_best_valid(tmp_path, "u1", rec1)
    write_run_meta(tmp_path, {"uuid": "u1", "features": 64})

    rows = read_history(tmp_path)
    assert [r["epoch"] for r in rows] == [1, 2]
    assert rows[0]["valid_polar_mae_bohr3"] == pytest.approx(40.04)
    best = json.loads((tmp_path / "best-valid-u1.json").read_text(encoding="utf-8"))
    assert best["best_epoch"] == 1
    assert best["valid_polar_mae_bohr3"] == pytest.approx(40.04)
    assert (tmp_path / "best-valid.json").is_symlink()
    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["format"] == "mmml-efield-run-meta-v1"
    assert meta["features"] == 64


def test_audit_missing_symlink_is_not_plateau(tmp_path: Path):
    """Old trainers only created params-best.json on exit. Epoch-1 uuid files ≠ stalled."""
    write_best_valid(
        tmp_path,
        "441a9161-9eca-4563-9957-04c9d2ec5a34",
        {
            "best_valid_loss": 8.5,
            "best_epoch": 1,
            "valid_polar_mae_bohr3": 40.04,
        },
    )
    (tmp_path / "params-best-441a9161-9eca-4563-9957-04c9d2ec5a34.json").write_text(
        "{}", encoding="utf-8"
    )
    log = (
        "ready: n_train=15975 n_valid=888 B=4 valid_batches=222 epochs=100 "
        "polar_weight=100.0 (prints once per epoch + every 100 steps)\n"
        "Validation Batch[0]\n"
        "epoch:   1                    train:   valid:\n"
        "    weighted total loss        10.000000  8.500000\n"
        "    polar mae [Bohr³]          34.800000  40.040000\n"
        "    ✓ Best valid checkpoint: params-best-441a9161.json "
        "(weighted loss=8.500000, epoch 1)\n"
    )
    report = audit_job(ckpt_dir=tmp_path, log_text=log)
    assert report.verdict in {"improving", "first_epoch"}
    assert report.last_epoch == 1
    assert report.best_epoch == 1
    assert report.last_valid_polar_mae == pytest.approx(40.04)
    assert report.params_best_symlink is False
    assert "never improved" not in report.reason.lower()


def test_audit_plateau_from_history(tmp_path: Path):
    append_history(
        tmp_path,
        epoch_record(
            epoch=1,
            run_uuid="u",
            improved=True,
            best_epoch=1,
            best_valid_loss=1.0,
            patience_counter=0,
            train_loss=2.0,
            valid_loss=1.0,
            train_energy_mae_kcal=1.0,
            valid_energy_mae_kcal=1.0,
            train_forces_mae_kcal=1.0,
            valid_forces_mae_kcal=1.0,
            valid_polar_mae=40.0,
        ),
    )
    append_history(
        tmp_path,
        epoch_record(
            epoch=2,
            run_uuid="u",
            improved=False,
            best_epoch=1,
            best_valid_loss=1.0,
            patience_counter=1,
            train_loss=1.8,
            valid_loss=1.2,
            train_energy_mae_kcal=1.0,
            valid_energy_mae_kcal=1.0,
            train_forces_mae_kcal=1.0,
            valid_forces_mae_kcal=1.0,
            valid_polar_mae=40.0,
        ),
    )
    report = audit_job(ckpt_dir=tmp_path)
    assert report.verdict == "plateau"
    assert report.best_epoch == 1
    assert report.last_epoch == 2
    assert report.polar_delta == pytest.approx(0.0)


def test_audit_compiling_and_crash():
    compiling = parse_slurm_log(
        "ready: n_train=10 n_valid=4 B=2 valid_batches=2 epochs=2 polar_weight=1.0 (x)\n"
        "compiling train_step (first polar jacrev can take many minutes)\n"
        "Validation Batch[0]\n"
    )
    assert compiling["compiling"] is True
    report = audit_job(log_text="Validation Batch[0]\ncompiling train_step\n")
    assert report.verdict == "compiling"

    crashed = audit_job(log_text="JaxRuntimeError: Autotuning failed\n")
    assert crashed.verdict == "crashed"
    assert audit_main.__name__ == "main"


def test_audit_cli_json(tmp_path: Path, capsys):
    write_run_meta(tmp_path, {"uuid": "cli"})
    rc = audit_main(["--ckpt", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] in {"unknown", "first_epoch"}
    text = format_report(audit_job(ckpt_dir=tmp_path))
    assert "verdict:" in text


def test_orbax_params_roundtrip(tmp_path: Path):
    pytest.importorskip("orbax")
    from mmml.models.efield.training import load_params

    tree = {"params": {"w": np.array([1.0, 2.0], dtype=np.float32)}}
    dest = save_params_orbax(tmp_path / "orbax" / "best", tree, metadata={"epoch": 1})
    assert is_orbax_checkpoint(dest)
    restored = load_params_orbax(dest)
    np.testing.assert_allclose(np.asarray(restored["params"]["w"]), [1.0, 2.0])
    loaded = load_params(dest)
    np.testing.assert_allclose(np.asarray(loaded["params"]["w"]), [1.0, 2.0])
    meta = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
    assert meta["epoch"] == 1


def test_train_model_source_writes_history_and_orbax():
    import inspect

    from mmml.models.efield.training import train_model

    src = inspect.getsource(train_model)
    assert "append_history" in src
    assert "save_params_orbax" in src
    assert "write_best_valid" in src
    assert "history.jsonl" in src or "append_history" in src
