"""The ML/MM handoff default moved 8.0 -> 6.0. Two things must stay true.

1. The ML radial basis must reach the handoff (mm_switch_on <= cutoff), or the
   dimer interaction is silently truncated inside the taper.
2. A checkpoint trained on one handoff must not be silently deployed on another:
   the ML term learned to COMPLEMENT MM under its trained taper, so a different
   one is a different PES. Every checkpoint trained before this change recorded
   8.0, so the new default would otherwise re-point them silently.
"""

from __future__ import annotations

import pytest


def test_defaults_are_self_consistent():
    """The shipped defaults must not violate their own coupling rule."""
    from mmml.cli.make.make_training import parse_args

    a = parse_args(["--data", "x.npz", "--hybrid-mm"])
    assert a.cutoff == 6.0
    assert a.mm_switch_on == 6.0
    assert a.mm_switch_on <= a.cutoff, "shipped defaults violate mm_switch_on <= cutoff"


def test_shared_default_moved_to_6():
    from mmml.interfaces.pycharmmInterface.cutoffs import (
        DEFAULT_ML_SWITCH_WIDTH,
        DEFAULT_MM_SWITCH_ON,
        DEFAULT_MM_SWITCH_WIDTH,
    )

    assert DEFAULT_MM_SWITCH_ON == 6.0
    # unchanged
    assert DEFAULT_MM_SWITCH_WIDTH == 5.0
    assert DEFAULT_ML_SWITCH_WIDTH == 1.5


def test_training_and_md_share_the_same_default():
    """One source of truth: a drift here is a silent train/deploy mismatch."""
    from mmml.cli.make.make_training import parse_args
    from mmml.interfaces.pycharmmInterface.cutoffs import DEFAULT_MM_SWITCH_ON

    a = parse_args(["--data", "x.npz"])
    assert a.mm_switch_on == DEFAULT_MM_SWITCH_ON


def test_handoff_beyond_the_basis_is_rejected():
    from mmml.cli.make.make_training import parse_args, validate_train_args

    a = parse_args(
        ["--data", "x.npz", "--hybrid-mm", "--cutoff", "5.0", "--mm-switch-on", "8.0"]
    )
    with pytest.raises(ValueError, match="exceeds --cutoff"):
        validate_train_args(a)


def test_handoff_within_the_basis_is_allowed():
    from mmml.cli.make.make_training import _validate_handoff_within_cutoff, parse_args

    a = parse_args(
        ["--data", "x.npz", "--hybrid-mm", "--cutoff", "8.0", "--mm-switch-on", "6.0"]
    )
    _validate_handoff_within_cutoff(a)  # must not raise


def test_guard_only_applies_to_hybrid_runs():
    """A plain ML run has no MM handoff to be consistent with."""
    from mmml.cli.make.make_training import _validate_handoff_within_cutoff, parse_args

    a = parse_args(["--data", "x.npz", "--cutoff", "5.0", "--mm-switch-on", "8.0"])
    _validate_handoff_within_cutoff(a)  # not hybrid -> no-op


def test_checkpoint_records_the_handoff_it_trained_with():
    """This metadata is what lets MD detect a mismatched taper."""
    from mmml.models.hybrid_energy import HybridMMConfig
    from mmml.models.mm_charge_mode import hybrid_mm_metadata_dict

    cfg = HybridMMConfig(
        master_sigmas=(3.6,), master_epsilons=(0.078,),
        mm_switch_on=8.0, mm_switch_width=5.0, ml_switch_width=1.5,
    )
    meta = hybrid_mm_metadata_dict(cfg)
    assert meta["hybrid_mm"] is True
    assert meta["mm_switch_on"] == 8.0
    assert meta["ml_switch_width"] == 1.5
    assert meta["mm_switch_width"] == 5.0
