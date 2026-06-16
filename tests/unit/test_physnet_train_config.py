"""Unit tests for PhysNet train CLI config parsing."""

from pathlib import Path

import pytest

from mmml.cli.make.make_training import (
    CONFIG_ALIASES,
    apply_mapping_to_namespace,
    namespace_from_yaml,
    parse_args,
    parse_train_args,
    save_train_config,
    validate_train_args,
)


def test_config_aliases_include_valid_data():
    assert CONFIG_ALIASES["valid"] == "valid_data"
    assert CONFIG_ALIASES["train"] == "data"


def test_namespace_from_yaml(tmp_path: Path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        """
data: train.npz
valid_data: valid.npz
ckpt_dir: ./ckpts/run
tag: demo
batch_size: 16
max_atomic_number: 35
""".strip()
    )
    args = namespace_from_yaml(cfg)
    assert args.data == "train.npz"
    assert args.valid_data == "valid.npz"
    assert args.ckpt_dir == "./ckpts/run"
    assert args.batch_size == 16
    assert args.max_atomic_number == 35


def test_cli_overrides_yaml(tmp_path: Path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text("data: from_yaml.npz\nbatch_size: 4\n")
    args = parse_train_args(["--config", str(cfg), "--batch-size", "32", "--data", "cli.npz"])
    assert args.data == "cli.npz"
    assert args.batch_size == 32


def test_validate_fixed_splits():
    args = parse_args([])
    args.data = "train.npz"
    args.valid_data = "valid.npz"
    args.n_train = 0
    args.n_valid = 0
    validate_train_args(args)

    args.n_train = 10
    with pytest.raises(ValueError, match="valid-data"):
        validate_train_args(args)


def test_save_train_config_roundtrip(tmp_path: Path):
    args = parse_args([])
    args.data = "train.npz"
    args.tag = "roundtrip"
    out = tmp_path / "saved.yaml"
    save_train_config(args, out)
    loaded = namespace_from_yaml(out)
    assert loaded.data == "train.npz"
    assert loaded.tag == "roundtrip"
