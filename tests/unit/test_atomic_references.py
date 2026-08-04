"""Loading and unit conversion for the atomic reference-energy table.

These references are subtracted from every total energy before training
(:func:`mmml.data.units.subtract_atom_refs`), so a wrong level, a wrong charge
state, or a silently mis-scaled unit shifts the entire learning target by a
constant per atom. That is the kind of error a model absorbs into its bias and
nobody notices.

The error paths -- unknown level, unknown unit, malformed table -- had no
coverage, which is exactly where a bad table would slip through.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.data import atomic_numbers

from mmml.data import atomic_references as ar
from mmml.data.atomic_references import (
    DEFAULT_CHARGE_STATE,
    DEFAULT_REFERENCE_LEVEL,
    DEFAULT_UNIT,
    _convert_value,
    _normalise_unit,
    get_atomic_reference_array,
    get_atomic_reference_dict,
    list_reference_levels,
)

# CODATA 2018, independent of the module's own table.
_HARTREE_EV = 27.211386245988
_HARTREE_KCAL = 627.5094740631
_HARTREE_KJ = 2625.4996394799


def _write_table(tmp_path: Path, payload: object, name: str = "refs.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _clear_table_cache():
    """``_load_reference_data`` is lru_cached on the path argument."""
    ar._load_reference_data.cache_clear()
    yield
    ar._load_reference_data.cache_clear()


# --- unit handling ----------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "factor"),
    [("hartree", 1.0), ("ev", _HARTREE_EV), ("kcal/mol", _HARTREE_KCAL), ("kj/mol", _HARTREE_KJ)],
)
def test_conversion_factors_match_codata(unit, factor):
    assert _convert_value(1.0, unit) == pytest.approx(factor, rel=1e-6)


@pytest.mark.parametrize("raw", ["Hartree", "EV", "eV", "KCAL/MOL", "kJ/mol"])
def test_unit_names_are_case_insensitive(raw):
    assert _normalise_unit(raw) == raw.lower()


def test_unknown_unit_is_rejected():
    """Defaulting an unrecognised unit would rescale every reference silently."""
    with pytest.raises(ValueError, match="Unknown energy unit"):
        _normalise_unit("rydberg")


def test_conversion_is_linear_and_signed():
    assert _convert_value(-2.5, "ev") == pytest.approx(-2.5 * _HARTREE_EV, rel=1e-6)
    assert _convert_value(0.0, "kj/mol") == 0.0


# --- table loading ----------------------------------------------------------


def test_missing_table_raises_with_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        get_atomic_reference_dict(data_path=tmp_path / "absent.json")


def test_non_mapping_table_is_rejected(tmp_path):
    path = _write_table(tmp_path, ["not", "a", "mapping"])
    with pytest.raises(ValueError, match="must be a mapping"):
        get_atomic_reference_dict(data_path=path)


def test_list_reference_levels_returns_the_table_keys(tmp_path):
    path = _write_table(tmp_path, {"lvl-a": {"H:0": -0.5}, "lvl-b": {"H:0": -0.4}})
    assert set(list_reference_levels(path)) == {"lvl-a", "lvl-b"}


def test_unknown_level_lists_the_available_ones(tmp_path):
    path = _write_table(tmp_path, {"lvl-a": {"H:0": -0.5}})
    with pytest.raises(ValueError, match="Unknown atomic reference level"):
        get_atomic_reference_dict(level="nope", data_path=path)


def test_malformed_entry_key_is_rejected(tmp_path):
    """Entries are ``SYMBOL:CHARGE``; anything else is a corrupt table."""
    path = _write_table(tmp_path, {"lvl": {"H": -0.5}})
    with pytest.raises(ValueError, match="Invalid entry"):
        get_atomic_reference_dict(level="lvl", data_path=path)


def test_unknown_element_symbol_is_rejected(tmp_path):
    path = _write_table(tmp_path, {"lvl": {"Xx:0": -0.5}})
    with pytest.raises(ValueError, match="Unknown chemical symbol"):
        get_atomic_reference_dict(level="lvl", data_path=path)


def test_no_entries_for_the_requested_charge_raises(tmp_path):
    path = _write_table(tmp_path, {"lvl": {"H:1": -0.4}})
    with pytest.raises(ValueError, match="No atomic reference energies found"):
        get_atomic_reference_dict(
            level="lvl", charge_state=-1, fallback_to_neutral=False, data_path=path
        )


# --- charge-state selection -------------------------------------------------


def test_entries_are_keyed_by_atomic_number_not_symbol(tmp_path):
    path = _write_table(tmp_path, {"lvl": {"H:0": -0.5, "O:0": -75.0}})
    got = get_atomic_reference_dict(level="lvl", data_path=path)
    assert got == {atomic_numbers["H"]: -0.5, atomic_numbers["O"]: -75.0}


def test_requested_charge_state_wins_over_neutral(tmp_path):
    path = _write_table(tmp_path, {"lvl": {"O:0": -75.0, "O:-1": -75.4}})
    got = get_atomic_reference_dict(level="lvl", charge_state=-1, data_path=path)
    assert got[atomic_numbers["O"]] == pytest.approx(-75.4)


def test_neutral_fallback_fills_a_missing_charge_state(tmp_path):
    path = _write_table(tmp_path, {"lvl": {"H:0": -0.5, "O:0": -75.0, "O:-1": -75.4}})
    got = get_atomic_reference_dict(level="lvl", charge_state=-1, data_path=path)
    # O has an anion entry; H does not, so it falls back to neutral.
    assert got[atomic_numbers["O"]] == pytest.approx(-75.4)
    assert got[atomic_numbers["H"]] == pytest.approx(-0.5)


def test_fallback_can_be_disabled(tmp_path):
    path = _write_table(tmp_path, {"lvl": {"H:0": -0.5, "O:-1": -75.4}})
    got = get_atomic_reference_dict(
        level="lvl", charge_state=-1, fallback_to_neutral=False, data_path=path
    )
    assert atomic_numbers["H"] not in got
    assert got[atomic_numbers["O"]] == pytest.approx(-75.4)


def test_units_are_applied_to_the_selected_entries(tmp_path):
    path = _write_table(tmp_path, {"lvl": {"H:0": -0.5}})
    got = get_atomic_reference_dict(level="lvl", unit="ev", data_path=path)
    assert got[atomic_numbers["H"]] == pytest.approx(-0.5 * _HARTREE_EV, rel=1e-6)


# --- array form -------------------------------------------------------------


def test_array_is_indexed_by_atomic_number(tmp_path):
    path = _write_table(tmp_path, {"lvl": {"H:0": -0.5, "O:0": -75.0}})
    arr = get_atomic_reference_array(level="lvl", data_path=path)
    assert arr[atomic_numbers["H"]] == pytest.approx(-0.5)
    assert arr[atomic_numbers["O"]] == pytest.approx(-75.0)


def test_array_is_zero_for_elements_absent_from_the_table(tmp_path):
    """Zero, not NaN -- subtract_atom_refs sums these, so a NaN would poison
    every energy containing an untabulated element rather than just offset it."""
    path = _write_table(tmp_path, {"lvl": {"H:0": -0.5}})
    arr = get_atomic_reference_array(level="lvl", data_path=path)
    assert arr[atomic_numbers["Fe"]] == 0.0
    assert np.all(np.isfinite(arr))


def test_array_spans_the_periodic_table_by_default(tmp_path):
    path = _write_table(tmp_path, {"lvl": {"H:0": -0.5}})
    assert len(get_atomic_reference_array(level="lvl", data_path=path)) >= 119


def test_array_honours_an_explicit_size(tmp_path):
    path = _write_table(tmp_path, {"lvl": {"H:0": -0.5}})
    assert len(get_atomic_reference_array(level="lvl", size=200, data_path=path)) == 200


def test_array_size_never_truncates_the_table(tmp_path):
    """A `size` smaller than the largest Z present must not drop entries."""
    path = _write_table(tmp_path, {"lvl": {"U:0": -1.0}})
    arr = get_atomic_reference_array(level="lvl", size=5, data_path=path)
    assert arr[atomic_numbers["U"]] == pytest.approx(-1.0)


def test_array_and_dict_agree(tmp_path):
    path = _write_table(tmp_path, {"lvl": {"H:0": -0.5, "C:0": -37.8}})
    as_dict = get_atomic_reference_dict(level="lvl", unit="ev", data_path=path)
    as_array = get_atomic_reference_array(level="lvl", unit="ev", data_path=path)
    for z, energy in as_dict.items():
        assert as_array[z] == pytest.approx(energy)


# --- the shipped table ------------------------------------------------------


def test_shipped_table_has_the_default_level():
    assert DEFAULT_REFERENCE_LEVEL in set(list_reference_levels())


def test_shipped_defaults_load_and_are_physical():
    """Atomic total energies are negative and grow in magnitude with Z."""
    got = get_atomic_reference_dict(
        level=DEFAULT_REFERENCE_LEVEL, charge_state=DEFAULT_CHARGE_STATE, unit=DEFAULT_UNIT
    )
    assert got, "shipped table produced no neutral references"
    assert all(e < 0.0 for e in got.values())
    h, c = atomic_numbers["H"], atomic_numbers["C"]
    if h in got and c in got:
        assert abs(got[c]) > abs(got[h])


def test_shipped_table_unit_scaling_is_consistent():
    hartree = get_atomic_reference_dict(unit="hartree")
    in_ev = get_atomic_reference_dict(unit="ev")
    z = next(iter(hartree))
    assert in_ev[z] == pytest.approx(hartree[z] * _HARTREE_EV, rel=1e-6)


def test_shipped_table_is_a_single_well_formed_json_document():
    """The shipped table must parse, and parse as ONE document.

    A merge once concatenated two versions of this file -- the second document's
    opening brace landed where a comma belonged, leaving 271 '{' against 270
    '}'. Nothing caught it until `import mmml.md.energy.terms` died in CI,
    because the table is read at module scope by `mmml.data.preprocessing`, so a
    malformed file takes down the whole package rather than one code path.

    Fork PRs get no CI in this repository, so this file is the guard.
    """
    from mmml.data.atomic_references import _DATA_PATH

    raw = Path(_DATA_PATH).read_text(encoding="utf-8")

    # json.load would accept only the first document if there were trailing
    # data, so assert on the balance too -- that is what actually broke.
    assert raw.count("{") == raw.count("}"), (
        f"unbalanced braces: {raw.count('{')} open vs {raw.count('}')} close; "
        "this is what two concatenated JSON documents look like"
    )

    decoder = json.JSONDecoder()
    obj, end = decoder.raw_decode(raw.lstrip())
    assert not raw.lstrip()[end:].strip(), (
        "trailing content after the first JSON document -- the file holds more "
        "than one document"
    )
    assert isinstance(obj, dict) and obj, "table must be a non-empty mapping"


def test_shipped_table_has_no_duplicate_levels():
    """Duplicate top-level keys parse fine and silently drop the earlier one."""
    from mmml.data.atomic_references import _DATA_PATH

    seen: list[str] = []
    json.loads(
        Path(_DATA_PATH).read_text(encoding="utf-8"),
        object_pairs_hook=lambda kv: seen.extend(k for k, _ in kv) or dict(kv),
    )
    # Only the outermost object's keys are level names; inner ones are species.
    levels = [k for k in seen if ":" not in k]
    assert len(levels) == len(set(levels)), "a level appears twice; one would win silently"
