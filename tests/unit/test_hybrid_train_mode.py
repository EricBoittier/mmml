"""CLI/config wiring for the hybrid ML/MM training mode."""

from __future__ import annotations

import numpy as np
import pytest


def _npz(tmp_path, *, with_cgenff=True, n=4, natoms=6):
    payload = {
        "R": np.random.RandomState(0).randn(n, natoms, 3),
        "Z": np.tile(np.array([6, 1, 6, 1, 0, 0]), (n, 1)),
        "F": np.zeros((n, natoms, 3)),
        "E": np.zeros(n),
        "N": np.full(n, 4),
        "D": np.zeros((n, 3)),
    }
    if with_cgenff:
        payload.update(
            cgenff_type_idx=np.tile(np.array([0, 1, 0, 1, -1, -1]), (n, 1)),
            mol_id=np.tile(np.array([0, 0, 1, 1, -1, -1]), (n, 1)),
            cgenff_charge=np.zeros((n, natoms)),
            cgenff_master_sigmas=np.array([3.6, 2.4]),
            cgenff_master_epsilons=np.array([0.078, 0.024]),
        )
    p = tmp_path / "d.npz"
    np.savez(p, **payload)
    return p


def test_hybrid_mm_defaults_off_and_flags_match_the_md_side():
    from mmml.cli.make.make_training import parse_args
    from mmml.interfaces.pycharmmInterface.cutoffs import (
        DEFAULT_ML_SWITCH_WIDTH,
        DEFAULT_MM_SWITCH_ON,
        DEFAULT_MM_SWITCH_WIDTH,
    )

    assert parse_args(["--data", "x.npz"]).hybrid_mm is False
    a = parse_args(["--data", "x.npz", "--hybrid-mm"])
    assert a.hybrid_mm is True
    # The point of this test is that training and MD share ONE default, so assert
    # against the shared constant -- a literal here would just re-pin whatever
    # number happened to be current and would go stale on every retune.
    assert (a.ml_switch_width, a.mm_switch_on, a.mm_switch_width) == (
        DEFAULT_ML_SWITCH_WIDTH,
        DEFAULT_MM_SWITCH_ON,
        DEFAULT_MM_SWITCH_WIDTH,
    )


def test_config_builder_returns_none_when_off(tmp_path):
    from mmml.cli.make.make_training import _build_hybrid_mm_config, parse_args

    p = _npz(tmp_path)
    args = parse_args(["--data", str(p)])
    assert _build_hybrid_mm_config(args, [str(p)]) is None


def test_config_builder_loads_master_tables_and_switching(tmp_path):
    from mmml.cli.make.make_training import _build_hybrid_mm_config, parse_args
    from mmml.interfaces.pycharmmInterface.cutoffs import (
        DEFAULT_ML_SWITCH_WIDTH,
        DEFAULT_MM_SWITCH_ON,
        DEFAULT_MM_SWITCH_WIDTH,
    )

    p = _npz(tmp_path)
    args = parse_args(["--data", str(p), "--hybrid-mm", "--quiet"])
    cfg = _build_hybrid_mm_config(args, [str(p)])
    assert cfg is not None
    # tables come from the npz (they are (n_types,), so batching can't carry them)
    assert np.allclose(cfg["master_sigmas"], [3.6, 2.4])
    assert np.allclose(cfg["master_epsilons"], [0.078, 0.024])
    assert cfg["mm_switch_on"] == DEFAULT_MM_SWITCH_ON
    assert cfg["mm_switch_width"] == DEFAULT_MM_SWITCH_WIDTH
    assert cfg["ml_switch_width"] == DEFAULT_ML_SWITCH_WIDTH
    assert cfg["complementary_handoff"] is True


def test_no_complementary_handoff_flag_is_honoured(tmp_path):
    from mmml.cli.make.make_training import _build_hybrid_mm_config, parse_args

    p = _npz(tmp_path)
    args = parse_args(["--data", str(p), "--hybrid-mm", "--no-complementary-handoff", "--quiet"])
    assert _build_hybrid_mm_config(args, [str(p)])["complementary_handoff"] is False


def test_missing_cgenff_fields_fail_loudly(tmp_path):
    """A plain dataset + --hybrid-mm must error, not silently train ML-only."""
    from mmml.cli.make.make_training import _build_hybrid_mm_config, parse_args

    p = _npz(tmp_path, with_cgenff=False)
    args = parse_args(["--data", str(p), "--hybrid-mm", "--quiet"])
    with pytest.raises(ValueError, match="cgenff"):
        _build_hybrid_mm_config(args, [str(p)])


def test_train_model_and_train_step_accept_hybrid_mm():
    import inspect

    from mmml.models.physnetjax.physnetjax.training.training import train_model
    from mmml.models.physnetjax.physnetjax.training.trainstep import _forward

    assert "hybrid_mm" in inspect.signature(train_model).parameters
    assert "hybrid_mm" in inspect.signature(_forward).parameters


def test_batch_keys_are_the_per_atom_fields_only():
    """Master tables must NOT be batch keys: they are (n_types,), not per-sample."""
    from mmml.models.hybrid_energy import HYBRID_MM_BATCH_KEYS

    assert set(HYBRID_MM_BATCH_KEYS) == {"cgenff_type_idx", "mol_id", "cgenff_charge"}
    assert not any("master" in k for k in HYBRID_MM_BATCH_KEYS)


def test_charge_correction_flag_defaults_off_and_reaches_the_config(tmp_path):
    from mmml.cli.make.make_training import _build_hybrid_mm_config, parse_args

    p = _npz(tmp_path)
    args = parse_args(["--data", str(p), "--hybrid-mm", "--quiet"])
    assert args.mm_charge_correction is False
    assert args.mm_charge_mode is None
    assert _build_hybrid_mm_config(args, [str(p)])["mm_charge_mode"] == "fixed"

    args = parse_args(
        ["--data", str(p), "--hybrid-mm", "--mm-charge-correction", "--charges", "--quiet"]
    )
    assert _build_hybrid_mm_config(args, [str(p)])["mm_charge_mode"] == "fixed_plus_latent"

    args = parse_args(
        [
            "--data", str(p), "--hybrid-mm",
            "--mm-charge-mode", "latent", "--charges", "--quiet",
        ]
    )
    assert _build_hybrid_mm_config(args, [str(p)])["mm_charge_mode"] == "latent"


def test_charge_correction_without_a_charge_head_errors(tmp_path):
    """--mm-charge-correction / latent without --charges must fail loudly."""
    from mmml.cli.make.make_training import _build_hybrid_mm_config, parse_args

    p = _npz(tmp_path)
    args = parse_args(["--data", str(p), "--hybrid-mm", "--mm-charge-correction", "--quiet"])
    with pytest.raises(ValueError, match="charge head"):
        _build_hybrid_mm_config(args, [str(p)])
    args = parse_args(
        ["--data", str(p), "--hybrid-mm", "--mm-charge-mode", "latent", "--quiet"]
    )
    with pytest.raises(ValueError, match="charge head"):
        _build_hybrid_mm_config(args, [str(p)])
