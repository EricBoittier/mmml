"""Conversion factors, alias parsing, and unit-inference paths in ``mmml.data.units``.

``test_units.py`` covers the happy paths of this module; the branches left over
were the ones that matter most when something is wrong: rejecting an unknown
unit string, the dipole conversions, and the heuristics that guess a reference
NPZ's units when no manifest is embedded. A silently-wrong guess there rescales
a whole training set.

Conversion factors are checked against CODATA values computed in the test rather
than against the module's own constants, so a typo in a literal fails here
instead of propagating. That is not hypothetical: the DCMNet dipole chain
carried a transposed-digit Angstrom -> bohr factor for a long time because
nothing pinned it.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from mmml.data.units import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    DEBYE_TO_EANGSTROM,
    EANGSTROM_TO_DEBYE,
    EV_TO_HARTREE,
    EV_TO_KCAL_MOL,
    HARTREE_BOHR_TO_EV_ANGSTROM,
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_MOL,
    KCAL_MOL_TO_EV,
    UnitsManifestV2,
    _as_numeric_energy_array,
    _infer_reference_units_from_arrays,
    _units_from_npz_metadata_json,
    attach_units_to_npz_payload,
    convert_coords,
    convert_dipole,
    convert_energy,
    convert_forces,
    find_units_manifest,
    format_energy_ev_kcal,
    format_energy_kcal_ev,
    format_fmax_ev_kcal_a,
    format_grms_kcal_ev_a,
    infer_reference_energy_unit,
    infer_reference_force_unit,
    normalize_dipole_unit,
    normalize_energy_unit,
    normalize_force_unit,
    normalize_length_unit,
    normalize_to_canonical,
    pyscf_units_json,
    pyscf_units_metadata,
)

# CODATA 2018, spelled out here so the reference is independent of the module.
_BOHR_A = 0.529177210903
_HARTREE_EV = 27.211386245988
_HARTREE_KCAL = 627.5094740631
_E_C = 1.602176634e-19
_DEBYE_CM = 3.335640952e-30

_TOL = 2e-5  # the repo rounds its literals; nothing like a real typo


# --- physical constants -----------------------------------------------------


def test_length_factors_match_codata():
    assert BOHR_TO_ANGSTROM == pytest.approx(_BOHR_A, rel=_TOL)
    assert ANGSTROM_TO_BOHR == pytest.approx(1.0 / _BOHR_A, rel=_TOL)


def test_energy_factors_match_codata():
    assert HARTREE_TO_EV == pytest.approx(_HARTREE_EV, rel=_TOL)
    assert HARTREE_TO_KCAL_MOL == pytest.approx(_HARTREE_KCAL, rel=_TOL)
    assert EV_TO_KCAL_MOL == pytest.approx(_HARTREE_KCAL / _HARTREE_EV, rel=_TOL)


def test_dipole_factor_matches_codata():
    assert EANGSTROM_TO_DEBYE == pytest.approx(_E_C * 1e-10 / _DEBYE_CM, rel=_TOL)


def test_force_factor_is_the_ratio_of_its_parts():
    assert HARTREE_BOHR_TO_EV_ANGSTROM == pytest.approx(
        HARTREE_TO_EV / BOHR_TO_ANGSTROM, rel=1e-12
    )


@pytest.mark.parametrize(
    ("forward", "backward"),
    [
        (HARTREE_TO_EV, EV_TO_HARTREE),
        (EV_TO_KCAL_MOL, KCAL_MOL_TO_EV),
        (EANGSTROM_TO_DEBYE, DEBYE_TO_EANGSTROM),
        (ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM),
    ],
)
def test_reciprocal_pairs_multiply_to_one(forward, backward):
    assert forward * backward == pytest.approx(1.0, rel=1e-5)


# --- alias normalisation ----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("eV", "ev"), (" EV ", "ev"), ("Hartree", "hartree"), ("ha", "hartree"),
     ("kcal/mol", "kcal_mol"), ("KCAL", "kcal_mol")],
)
def test_energy_aliases(raw, expected):
    assert normalize_energy_unit(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("eV/Angstrom", "ev_angstrom"), ("ev/ang", "ev_angstrom"), ("eV/A", "ev_angstrom"),
     ("hartree/bohr", "hartree_bohr"), ("Ha/Bohr", "hartree_bohr"),
     ("kcal/mol/ang", "kcal_mol_angstrom")],
)
def test_force_aliases(raw, expected):
    assert normalize_force_unit(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Angstrom", "angstrom"), ("ang", "angstrom"), ("A", "angstrom"),
     ("Bohr", "bohr"), ("au", "bohr")],
)
def test_length_aliases(raw, expected):
    assert normalize_length_unit(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Debye", "debye"), ("D", "debye"), ("e*Angstrom", "e_angstrom"),
     ("e-angstrom", "e_angstrom"), ("E_Angstrom", "e_angstrom")],
)
def test_dipole_aliases(raw, expected):
    assert normalize_dipole_unit(raw) == expected


def _alias_cases():
    """Every alias in every table, paired with its normalizer."""
    from mmml.data import units as units_module

    tables = (
        ("energy", units_module._ENERGY_ALIASES, normalize_energy_unit),
        ("force", units_module._FORCE_ALIASES, normalize_force_unit),
        ("length", units_module._LENGTH_ALIASES, normalize_length_unit),
        ("dipole", units_module._DIPOLE_ALIASES, normalize_dipole_unit),
    )
    return [
        pytest.param(fn, raw, expected, id=f"{kind}:{raw}")
        for kind, table, fn in tables
        for raw, expected in table.items()
    ]


@pytest.mark.parametrize(("fn", "raw", "expected"), _alias_cases())
def test_every_listed_alias_is_actually_reachable(fn, raw, expected):
    """An alias the normalizer can never match is dead config that reads as support.

    ``_FORCE_ALIASES`` listed ``ev/a``, ``ha/bohr`` and ``kcal/mol/ang`` while
    ``normalize_force_unit`` replaces ``/`` with ``_`` *before* the lookup, so
    none of the three could ever be hit -- the table advertised units the
    parser rejected. Spot-checking a handful of aliases (above) cannot catch
    that; only walking the tables can.
    """
    assert fn(raw) == expected


@pytest.mark.parametrize(("fn", "raw", "expected"), _alias_cases())
def test_alias_lookup_survives_case_and_padding(fn, raw, expected):
    assert fn(f"  {raw.upper()} ") == expected


@pytest.mark.parametrize(
    ("fn", "bad"),
    [
        (normalize_energy_unit, "joules"),
        (normalize_force_unit, "newtons"),
        (normalize_length_unit, "nanometre"),
        (normalize_dipole_unit, "coulomb_metre"),
    ],
)
def test_unknown_units_are_rejected_rather_than_guessed(fn, bad):
    """Silently defaulting an unrecognised unit is how a dataset gets rescaled."""
    with pytest.raises(ValueError, match="Unsupported"):
        fn(bad)


# --- conversions ------------------------------------------------------------


def test_convert_energy_hartree_to_kcal_matches_codata():
    assert convert_energy(1.0, "hartree", "kcal_mol") == pytest.approx(
        _HARTREE_KCAL, rel=_TOL
    )


def test_convert_energy_is_identity_for_matching_units():
    assert convert_energy(3.5, "ev", "eV") == pytest.approx(3.5)


def test_convert_energy_preserves_array_shape():
    arr = np.arange(6.0).reshape(2, 3)
    assert convert_energy(arr, "hartree", "ev").shape == (2, 3)


def test_convert_forces_round_trips_through_hartree_bohr():
    arr = np.array([[1.0, -2.0, 0.5]])
    there = convert_forces(arr, "ev_angstrom", "hartree_bohr")
    back = convert_forces(there, "hartree_bohr", "ev_angstrom")
    assert back == pytest.approx(arr, rel=1e-10)


def test_convert_forces_kcal_path():
    assert convert_forces(1.0, "kcal_mol_angstrom", "ev_angstrom") == pytest.approx(
        KCAL_MOL_TO_EV, rel=1e-12
    )


def test_convert_coords_uses_the_bohr_radius():
    assert convert_coords(1.0, "angstrom", "bohr") == pytest.approx(1.0 / _BOHR_A, rel=_TOL)
    assert convert_coords(1.0, "bohr", "angstrom") == pytest.approx(_BOHR_A, rel=_TOL)
    assert convert_coords(2.0, "ang", "A") == pytest.approx(2.0)


def test_convert_dipole_both_directions():
    assert convert_dipole(1.0, "e_angstrom", "debye") == pytest.approx(
        _E_C * 1e-10 / _DEBYE_CM, rel=_TOL
    )
    assert convert_dipole(1.0, "debye", "e*Angstrom") == pytest.approx(
        _DEBYE_CM / (_E_C * 1e-10), rel=_TOL
    )
    assert convert_dipole(2.5, "debye", "D") == pytest.approx(2.5)


def test_scalar_in_scalar_out():
    for fn, args in (
        (convert_energy, ("hartree", "ev")),
        (convert_forces, ("hartree_bohr", "ev_angstrom")),
        (convert_coords, ("angstrom", "bohr")),
        (convert_dipole, ("debye", "e_angstrom")),
    ):
        assert isinstance(fn(1.0, *args), float)


# --- formatting helpers -----------------------------------------------------


def test_format_helpers_report_both_units():
    assert "eV" in format_energy_ev_kcal(1.0) and "kcal/mol" in format_energy_ev_kcal(1.0)
    assert "kcal/mol" in format_energy_kcal_ev(1.0) and "eV" in format_energy_kcal_ev(1.0)
    assert "kcal/mol/Å" in format_grms_kcal_ev_a(1.0)
    assert "eV/Å" in format_fmax_ev_kcal_a(1.0)


def test_format_energy_conversion_is_correct_not_just_present():
    text = format_energy_ev_kcal(1.0, ev_digits=3, kcal_digits=3)
    assert text.startswith("1.000 eV")
    assert f"{EV_TO_KCAL_MOL:.3f}" in text


# --- reference-energy loading ----------------------------------------------


def test_numeric_energy_array_rejects_unit_labels():
    """An 'E' array holding the string 'hartree' must fail loudly."""
    with pytest.raises(ValueError, match="non-numeric"):
        _as_numeric_energy_array(np.array([1.0, "hartree"], dtype=object))


def test_numeric_energy_array_accepts_mixed_numeric_objects():
    out = _as_numeric_energy_array(np.array([1, 2.5, np.float64(3)], dtype=object))
    assert out.tolist() == [1.0, 2.5, 3.0]


# --- embedded metadata parsing ---------------------------------------------


def test_metadata_json_reads_arrays_block():
    raw = json.dumps({"arrays": {"E": "hartree", "F": "hartree_bohr"}})
    assert _units_from_npz_metadata_json(raw) == ("hartree", "hartree_bohr")


def test_metadata_json_reads_flat_mapping():
    assert _units_from_npz_metadata_json({"E": "ev", "F": "ev_angstrom"}) == (
        "ev",
        "ev_angstrom",
    )


def test_metadata_json_decodes_bytes_and_zero_dim_arrays():
    payload = json.dumps({"arrays": {"E": "ev"}})
    assert _units_from_npz_metadata_json(payload.encode("utf-8"))[0] == "ev"
    assert _units_from_npz_metadata_json(np.array(payload, dtype=object))[0] == "ev"


@pytest.mark.parametrize("raw", [None, "not json at all", 42, b"\xff\xfe"[:1] + b"x"])
def test_metadata_json_returns_none_for_junk(raw):
    assert _units_from_npz_metadata_json(raw) == (None, None)


# --- unit inference from array magnitudes -----------------------------------


def _npz(tmp_path: Path, name: str = "ref.npz", **arrays) -> Path:
    path = tmp_path / name
    np.savez(path, **arrays)
    return path


def test_e_ev_key_short_circuits_inference(tmp_path):
    path = _npz(tmp_path, E_eV=np.array([-1.0, -2.0]))
    assert _infer_reference_units_from_arrays(path) == ("ev", "ev_angstrom")


def test_large_forces_imply_ev_angstrom(tmp_path):
    """|F| of order 1 is eV/Angstrom; Hartree/bohr forces are far smaller."""
    path = _npz(tmp_path, E=np.array([-100.0]), F=np.array([[[2.0, 0.0, 0.0]]]))
    assert _infer_reference_units_from_arrays(path) == ("ev", "ev_angstrom")


def test_small_nonzero_forces_imply_hartree_bohr(tmp_path):
    path = _npz(tmp_path, E=np.array([-100.0]), F=np.array([[[0.01, 0.0, 0.0]]]))
    assert _infer_reference_units_from_arrays(path) == ("hartree", "hartree_bohr")


def test_large_energies_imply_ev_when_no_forces(tmp_path):
    path = _npz(tmp_path, E=np.array([-2000.0, -2100.0]))
    assert _infer_reference_units_from_arrays(path) == ("ev", "ev_angstrom")


def test_small_energies_stay_unknown(tmp_path):
    """Hartree-scale totals are ambiguous, so the guess must abstain."""
    path = _npz(tmp_path, E=np.array([-40.0, -41.0]))
    assert _infer_reference_units_from_arrays(path) == (None, None)


def test_missing_energy_key_is_unknown(tmp_path):
    assert _infer_reference_units_from_arrays(_npz(tmp_path, R=np.zeros((1, 2, 3)))) == (
        None,
        None,
    )


def test_unreadable_file_is_unknown_not_an_exception(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_text("not an npz")
    assert _infer_reference_units_from_arrays(bad) == (None, None)


def test_infer_energy_unit_falls_back_to_the_declared_default(tmp_path):
    path = _npz(tmp_path, E=np.array([-40.0]))
    assert infer_reference_energy_unit(path, default="kcal_mol") == "kcal_mol"


def test_infer_force_unit_prefers_an_explicit_manifest(tmp_path):
    manifest = UnitsManifestV2(arrays={"E": "ev", "F": "ev_angstrom"})
    assert infer_reference_force_unit(None, manifest=manifest) == "ev_angstrom"


def test_find_units_manifest_searches_the_parent_directory(tmp_path):
    nested = tmp_path / "split"
    nested.mkdir()
    (tmp_path / "units_manifest.json").write_text(
        json.dumps({"schema_version": 2, "arrays": {"E": "hartree"}})
    )
    found = find_units_manifest(nested / "train.npz")
    assert found is not None and found.energy_unit() == "hartree"


def test_find_units_manifest_returns_none_when_absent(tmp_path):
    assert find_units_manifest(tmp_path / "train.npz") is None


# --- normalize_to_canonical -------------------------------------------------


def test_hartree_input_warns_and_converts():
    manifest = UnitsManifestV2(arrays={"E": "hartree", "F": "hartree_bohr", "R": "bohr"})
    data = {"E": np.array([1.0]), "F": np.array([[[1.0, 0.0, 0.0]]]), "R": np.array([[[1.0, 0.0, 0.0]]])}

    with pytest.warns(UserWarning, match="not canonical eV"):
        out = normalize_to_canonical(data, manifest)

    assert out["E"][0] == pytest.approx(_HARTREE_EV, rel=_TOL)
    assert out["F"][0, 0, 0] == pytest.approx(_HARTREE_EV / _BOHR_A, rel=_TOL)
    assert out["R"][0, 0, 0] == pytest.approx(_BOHR_A, rel=_TOL)


def test_allow_hartree_suppresses_the_warning_but_still_converts():
    manifest = UnitsManifestV2(arrays={"E": "hartree"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = normalize_to_canonical({"E": np.array([1.0])}, manifest, allow_hartree=True)
    assert out["E"][0] == pytest.approx(_HARTREE_EV, rel=_TOL)


def test_canonical_input_is_left_alone():
    data = {"E": np.array([1.0]), "F": np.array([[[1.0, 0.0, 0.0]]])}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = normalize_to_canonical(data, UnitsManifestV2())
    assert out["E"][0] == pytest.approx(1.0)


def test_efield_energy_arrays_follow_the_energy_unit():
    manifest = UnitsManifestV2(arrays={"E": "hartree"})
    with pytest.warns(UserWarning):
        out = normalize_to_canonical(
            {"efield_energy": np.array([1.0]), "efield_scf_energy": np.array([2.0])},
            manifest,
        )
    assert out["efield_energy"][0] == pytest.approx(_HARTREE_EV, rel=_TOL)
    assert out["efield_scf_energy"][0] == pytest.approx(2 * _HARTREE_EV, rel=_TOL)


def test_dipole_is_converted_when_the_manifest_declares_debye():
    manifest = UnitsManifestV2(dipole_out="debye")
    out = normalize_to_canonical({"Dxyz": np.array([[1.0, 0.0, 0.0]])}, manifest)
    assert out["Dxyz"][0, 0] == pytest.approx(DEBYE_TO_EANGSTROM, rel=1e-9)


def test_normalize_without_a_manifest_assumes_canonical():
    data = {"E": np.array([5.0])}
    assert normalize_to_canonical(data)["E"][0] == pytest.approx(5.0)


def test_normalize_does_not_mutate_its_input():
    data = {"E": np.array([1.0])}
    with pytest.warns(UserWarning):
        normalize_to_canonical(data, UnitsManifestV2(arrays={"E": "hartree"}))
    assert data["E"][0] == pytest.approx(1.0)


# --- PySCF export metadata --------------------------------------------------


def test_pyscf_metadata_is_a_defensive_copy():
    first = pyscf_units_metadata()
    first["E"] = "tampered"
    assert pyscf_units_metadata()["E"] == "hartree"


def test_pyscf_units_json_round_trips():
    assert json.loads(pyscf_units_json()) == pyscf_units_metadata()


def test_attach_units_embeds_a_readable_payload():
    out = attach_units_to_npz_payload({"E": np.array([1.0])})
    assert "E" in out
    assert json.loads(str(out["_mmml_units"]))["Dxyz"] == "debye"
