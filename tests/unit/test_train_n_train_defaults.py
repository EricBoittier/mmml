"""`n_train`/`n_valid` must be omittable when `valid_data` is set (as documented)."""

from __future__ import annotations

import pytest


def _validate(args):
    from mmml.cli.make.make_training import validate_train_args

    return validate_train_args(args)


def _args(tmp_path, **over):
    from mmml.cli.make.make_training import parse_args

    data = tmp_path / "train.npz"
    data.write_text("")
    argv = ["--data", str(data)]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return parse_args(argv)


def test_omitting_n_train_with_valid_data_is_allowed(tmp_path):
    """The documented usage: set valid_data, omit n_train/n_valid."""
    valid = tmp_path / "valid.npz"
    valid.write_text("")
    args = _args(tmp_path, valid_data=str(valid))
    assert args.n_train is None and args.n_valid is None
    _validate(args)  # must not raise


def test_zero_n_train_with_valid_data_still_allowed(tmp_path):
    """0 was the workaround while defaults were 1000/100 - keep it working."""
    valid = tmp_path / "valid.npz"
    valid.write_text("")
    args = _args(tmp_path, valid_data=str(valid), n_train=0, n_valid=0)
    _validate(args)  # must not raise


def test_positive_n_train_with_valid_data_still_errors(tmp_path):
    """An explicit positive value genuinely conflicts - keep the guard."""
    valid = tmp_path / "valid.npz"
    valid.write_text("")
    args = _args(tmp_path, valid_data=str(valid), n_train=5000)
    with pytest.raises(ValueError, match="do not set --n-train"):
        _validate(args)


def test_single_file_split_keeps_historical_defaults(tmp_path):
    """Without valid_data, omitted n_train/n_valid resolve to 1000/100."""
    args = _args(tmp_path)
    assert args.n_train is None
    _validate(args)
    assert args.n_train == 1000
    assert args.n_valid == 100


def test_single_file_split_respects_explicit_values(tmp_path):
    args = _args(tmp_path, n_train=50, n_valid=5)
    _validate(args)
    assert (args.n_train, args.n_valid) == (50, 5)


def test_single_file_split_rejects_empty_and_negative(tmp_path):
    args = _args(tmp_path, n_train=0, n_valid=0)
    with pytest.raises(ValueError, match="must be > 0"):
        _validate(args)
    args = _args(tmp_path, n_train=-1)
    with pytest.raises(ValueError, match="must be >= 0"):
        _validate(args)
