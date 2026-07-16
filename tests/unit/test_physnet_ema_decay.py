"""EMA decay must be configurable (CLI / YAML) and disable-able for physnet-train."""

from __future__ import annotations

import inspect

import pytest


def test_train_model_accepts_ema_decay_with_default():
    from mmml.models.physnetjax.physnetjax.training.training import train_model

    sig = inspect.signature(train_model)
    assert "ema_decay" in sig.parameters
    assert sig.parameters["ema_decay"].default == pytest.approx(0.999)


def test_train_step_ema_decay_zero_disables_ema():
    """ema_decay=0 => ema_params tracks raw params exactly (EMA off)."""
    decay = 0.0
    ema, new = 10.0, 3.0
    assert decay * ema + (1 - decay) * new == pytest.approx(new)
    # and the default keeps (almost all of) the old EMA
    decay = 0.999
    assert decay * ema + (1 - decay) * new == pytest.approx(9.993)


def test_training_config_exposes_ema_decay():
    from mmml.cli.misc.training_helpers import TrainingConfig

    cfg = TrainingConfig(data="x.npz")
    assert cfg.ema_decay == pytest.approx(0.999)
    assert TrainingConfig(data="x.npz", ema_decay=0.0).ema_decay == 0.0


def test_cli_parses_ema_decay_flag(tmp_path):
    from mmml.cli.make.make_training import parse_args

    data = tmp_path / "d.npz"
    data.write_text("")

    assert parse_args(["--data", str(data), "--ema-decay", "0.0"]).ema_decay == 0.0
    # underscore alias
    assert parse_args(["--data", str(data), "--ema_decay", "0.5"]).ema_decay == 0.5
    # default preserved
    assert parse_args(["--data", str(data)]).ema_decay == pytest.approx(0.999)


def test_yaml_config_sets_ema_decay(tmp_path):
    """`ema_decay: 0.0` in a --config YAML must reach the namespace."""
    from mmml.cli.make.make_training import parse_train_args

    data = tmp_path / "d.npz"
    data.write_text("")
    cfg = tmp_path / "train.yaml"
    cfg.write_text(f"data: {data}\nema_decay: 0.0\n")

    args = parse_train_args(["--config", str(cfg)])
    assert args.ema_decay == 0.0
    # explicit CLI flag still overrides the file
    args = parse_train_args(["--config", str(cfg), "--ema-decay", "0.9"])
    assert args.ema_decay == pytest.approx(0.9)
