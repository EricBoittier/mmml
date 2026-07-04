"""Registry of canonical vs deprecated calculator paths."""

from __future__ import annotations

import warnings

from mmml.interfaces.pycharmmInterface.legacy_paths import (
    CANONICAL,
    DEPRECATED,
    warn_legacy,
)


def test_canonical_paths_are_importable_strings() -> None:
    assert "setup_calculator" in CANONICAL["hybrid_calculator_factory"]
    assert len(CANONICAL) >= 4
    assert "mmml_calculator" in DEPRECATED["mmml.models.physnetjax.physnetjax.calc.mmml_calculator"]


def test_warn_legacy_emits_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_legacy("legacy.foo", "canonical.bar", stacklevel=1)
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    assert "legacy.foo" in str(caught[0].message)
    assert "canonical.bar" in str(caught[0].message)


def test_pycharmm_interface_alias_warns_once() -> None:
    import mmml.pycharmmInterface as alias

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = alias.mmml_calculator
        _ = alias.calculator_utils
    alias_dep = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "mmml.pycharmmInterface" in str(w.message)
    ]
    assert len(alias_dep) == 1
