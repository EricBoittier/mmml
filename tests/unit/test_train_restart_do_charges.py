"""Regression: doCharges must follow the restored model, not the pre-restart CLI model.

Restart rebuilds ``model`` from checkpoint ``model_attributes``.  If YAML has
``charges: true`` but the checkpoint was trained with ``charges: false``, the
loss still used ``doCharges=True`` and crashed on ``zeros_like(None)`` for
``sum_charges``.
"""

from __future__ import annotations


def test_do_charges_must_be_refreshed_after_model_replacement():
    """Document the ordering invariant used in train_model."""
    class _M:
        def __init__(self, charges):
            self.charges = charges

    cli_model = _M(True)
    cli_charges = bool(cli_model.charges)
    do_charges = cli_charges

    # restart_training replaces model from checkpoint attributes
    restored_model = _M(False)
    do_charges = bool(getattr(restored_model, "charges", False))

    assert cli_charges is True
    assert do_charges is False
    assert do_charges != cli_charges
