"""Unit tests for PhysNet train CLI config parsing."""

from pathlib import Path

import pytest

from mmml.cli.make.make_training import (
    CONFIG_ALIASES,
    apply_mapping_to_namespace,
    namespace_from_yaml,
    normalize_data_paths,
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


def test_legacy_underscore_cli_flags():
    args = parse_train_args(
        [
            "--data",
            "train.npz",
            "--ckpt_dir",
            "./ckpts/run",
            "--n_train",
            "100",
            "--n_valid",
            "10",
            "--batch_size",
            "8",
            "--num_epochs",
            "5",
            "--learning_rate",
            "0.01",
            "--energy_weight",
            "2.0",
            "--max_atomic_number",
            "35",
            "--num_iterations",
            "5",
            "--num_basis_functions",
            "64",
        ]
    )
    assert args.data == "train.npz"
    assert args.ckpt_dir == "./ckpts/run"
    assert args.n_train == 100
    assert args.batch_size == 8
    assert args.num_basis_functions == 64


def test_save_train_config_roundtrip(tmp_path: Path):
    args = parse_args([])
    args.data = "train.npz"
    args.tag = "roundtrip"
    out = tmp_path / "saved.yaml"
    save_train_config(args, out)
    loaded = namespace_from_yaml(out)
    assert loaded.data == "train.npz"
    assert loaded.tag == "roundtrip"


def test_new_config_and_cli_options(tmp_path: Path):
    # Test parsed CLI arguments
    args = parse_train_args(
        [
            "--optimizer", "adamw",
            "--transform", "reduce_on_plateau",
            "--schedule-fn", "cosine",
            "--early-stop-patience", "10",
            "--best",
            "--no-save-every-epoch",
            "--print-freq", "5",
            "--batch-method", "advanced",
            "--batch-args-dict", '{"batch_shape": 60, "batch_nbl_len": 120}',
            "--data-keys", "R", "Z", "F", "D",
            "--conversion", '{"energy": 2.0}',
            "--init-params", '{"params": {}}',
            "--rot-augment",
            "--rot-perturbation", "0.5",
            "--charges",
            "--total-charge", "1.0",
            "--no-electrostatics",
            "--efa",
            "--no-zbl",
            "--no-pbc",
            "--debug",
        ]
    )
    assert args.optimizer == "adamw"
    assert args.transform == "reduce_on_plateau"
    assert args.schedule_fn == "cosine"
    assert args.early_stop_patience == 10
    assert args.best is True
    assert args.save_every_epoch is False
    assert args.print_freq == 5
    assert args.batch_method == "advanced"
    assert args.batch_args_dict == '{"batch_shape": 60, "batch_nbl_len": 120}'
    assert args.data_keys == ["R", "Z", "F", "D"]
    assert args.conversion == '{"energy": 2.0}'
    assert args.init_params == '{"params": {}}'
    assert args.rot_augment is True
    assert args.rot_perturbation == 0.5
    assert args.charges is True
    assert args.total_charge == 1.0
    assert args.include_electrostatics is False
    assert args.efa is True
    assert args.zbl is False
    assert args.use_pbc is False
    assert args.debug is True

    # Test loading from YAML
    cfg = tmp_path / "extended.yaml"
    cfg.write_text(
        """
data: train.npz
optimizer: adamw
transform: reduce_on_plateau
schedule_fn: cosine
early_stop_patience: 15
best: true
save_every_epoch: false
print_freq: 2
batch_method: advanced
batch_args_dict:
  batch_shape: 40
  batch_nbl_len: 100
data_keys:
  - R
  - Z
conversion:
  energy: 0.5
init_params:
  params:
    layer: 1
rot_augment: true
rot_perturbation: 0.1
charges: true
total_charge: -1.0
include_electrostatics: true
efa: true
zbl: true
use_pbc: true
debug: false
""".strip()
    )
    loaded = namespace_from_yaml(cfg)
    assert loaded.optimizer == "adamw"
    assert loaded.early_stop_patience == 15
    assert loaded.best is True
    assert loaded.save_every_epoch is False
    assert loaded.print_freq == 2
    assert loaded.batch_method == "advanced"
    assert isinstance(loaded.batch_args_dict, dict)
    assert loaded.batch_args_dict["batch_shape"] == 40
    assert loaded.data_keys == ["R", "Z"]
    assert loaded.conversion == {"energy": 0.5}
    assert loaded.init_params == {"params": {"layer": 1}}
    assert loaded.rot_augment is True
    assert loaded.rot_perturbation == 0.1
    assert loaded.charges is True
    assert loaded.total_charge == -1.0
    assert loaded.include_electrostatics is True
    assert loaded.efa is True
    assert loaded.zbl is True
    assert loaded.use_pbc is True
    assert loaded.debug is False


def test_transfer_learning_yaml_keys(tmp_path: Path):
    cfg = tmp_path / "transfer.yaml"
    cfg.write_text(
        """
data:
  - train_a.npz
  - train_b.npz
valid_data: valid.npz
physnet_transfer_model: joint-training-defaults
match_checkpoint_architecture: true
distill: true
distill_alpha: 0.8
distill_targets:
  - energy
  - forces
metrics_plot: curves.png
log_loss: true
""".strip()
    )
    loaded = namespace_from_yaml(cfg)
    assert loaded.data == ["train_a.npz", "train_b.npz"]
    assert loaded.physnet_transfer_model == "joint-training-defaults"
    assert loaded.match_checkpoint_architecture is True
    assert loaded.distill is True
    assert loaded.distill_alpha == 0.8
    assert loaded.distill_targets == ["energy", "forces"]
    assert loaded.metrics_plot == "curves.png"
    assert loaded.log_loss is True


def test_normalize_data_paths():
    assert normalize_data_paths("a.npz") == ["a.npz"]
    assert normalize_data_paths("a.npz,b.npz") == ["a.npz", "b.npz"]
    assert normalize_data_paths(["a.npz", "b.npz"]) == ["a.npz", "b.npz"]


def test_validate_restart_transfer_conflict():
    args = parse_args([])
    args.data = "train.npz"
    args.restart = "run/epoch-1"
    args.physnet_checkpoint = "teacher.json"
    with pytest.raises(ValueError, match="restart"):
        validate_train_args(args)


def test_validate_distill_requires_teacher(monkeypatch):
    args = parse_args([])
    args.data = "train.npz"
    args.distill = True
    args.distill_alpha = 0.5
    args.physnet_transfer_model = None
    args.physnet_checkpoint = None
    args.teacher_checkpoint = None

    def _fail(_selection):
        raise KeyError("missing")

    monkeypatch.setattr(
        "mmml.cli.make.make_training.resolve_hf_physnet_model",
        _fail,
    )
    with pytest.raises(ValueError, match="teacher checkpoint"):
        validate_train_args(args)


def test_validate_transfer_model_checkpoint_conflict():
    args = parse_args([])
    args.data = "train.npz"
    args.physnet_transfer_model = "mmml-default"
    args.physnet_checkpoint = "teacher.json"
    with pytest.raises(ValueError, match="physnet-transfer-model"):
        validate_train_args(args)

