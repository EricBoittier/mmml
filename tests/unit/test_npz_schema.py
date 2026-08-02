"""Tests for :mod:`mmml.data.npz_schema`.

The NPZ schema is the contract every dataset in the repo is written and read
against, and it had no tests: a validator that silently accepts a malformed file
is worse than no validator, because downstream code then fails somewhere far
from the cause.

Each shape rule below is asserted in both directions -- a conforming array must
pass and a specific deformation of it must be reported -- so a rule that stops
being enforced shows up here rather than as a confusing training crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.data import npz_schema
from mmml.data.npz_schema import (
    METADATA_KEYS,
    OPTIONAL_KEYS,
    REQUIRED_KEYS,
    NPZSchema,
    create_empty_npz,
    main,
    validate_npz,
)


def _good_dataset(n_structures: int = 3, n_atoms: int = 5) -> dict[str, np.ndarray]:
    """A minimal dataset that satisfies every required rule."""
    return {
        "R": np.zeros((n_structures, n_atoms, 3)),
        "Z": np.ones((n_structures, n_atoms), dtype=np.int32) * 6,
        "E": np.zeros(n_structures),
        "N": np.full(n_structures, n_atoms, dtype=np.int32),
    }


# --- schema tables ----------------------------------------------------------


def test_key_namespaces_do_not_overlap():
    """A key in two tables would make `strict` behaviour depend on lookup order."""
    assert set(REQUIRED_KEYS) & set(OPTIONAL_KEYS) == set()
    assert set(REQUIRED_KEYS) & set(METADATA_KEYS) == set()
    assert set(OPTIONAL_KEYS) & set(METADATA_KEYS) == set()


def test_required_keys_are_the_four_documented_ones():
    assert set(REQUIRED_KEYS) == {"R", "Z", "E", "N"}


def test_default_schema_mirrors_the_module_tables():
    schema = NPZSchema()
    assert schema.required_keys == set(REQUIRED_KEYS)
    assert schema.optional_keys == set(OPTIONAL_KEYS)
    assert schema.metadata_keys == set(METADATA_KEYS)
    assert schema.strict is False


# --- required keys ----------------------------------------------------------


def test_a_conforming_dataset_validates():
    is_valid, errors = NPZSchema().validate(_good_dataset())
    assert is_valid, errors
    assert errors == []


@pytest.mark.parametrize("missing", ["R", "Z", "E", "N"])
def test_each_missing_required_key_is_reported(missing):
    data = _good_dataset()
    del data[missing]
    is_valid, errors = NPZSchema().validate(data)
    assert not is_valid
    assert any("Missing required keys" in e and missing in e for e in errors)


# --- unknown keys and strict mode -------------------------------------------


def test_unknown_key_is_only_a_warning_by_default(capsys):
    data = _good_dataset()
    data["totally_made_up"] = np.zeros(3)

    is_valid, errors = NPZSchema().validate(data)

    assert is_valid, errors
    assert "totally_made_up" in capsys.readouterr().out


def test_unknown_key_is_an_error_in_strict_mode():
    data = _good_dataset()
    data["totally_made_up"] = np.zeros(3)

    is_valid, errors = NPZSchema(strict=True).validate(data)

    assert not is_valid
    assert any("Unknown keys" in e for e in errors)


def test_metadata_keys_are_accepted_even_in_strict_mode():
    data = _good_dataset()
    data["basis_set"] = np.array("aug-cc-pVTZ")
    assert NPZSchema(strict=True).validate(data)[0]


def test_optional_keys_are_accepted_even_in_strict_mode():
    data = _good_dataset()
    data["mono"] = np.zeros((3, 5))
    assert NPZSchema(strict=True).validate(data)[0]


# --- shape rules ------------------------------------------------------------


def test_r_must_be_three_dimensional_with_a_trailing_three():
    data = _good_dataset()
    data["R"] = np.zeros((3, 5))  # dropped the xyz axis
    is_valid, errors = NPZSchema().validate(data)
    assert not is_valid
    assert any("'R' must have shape" in e for e in errors)


def test_r_trailing_axis_must_be_xyz_not_some_other_width():
    data = _good_dataset()
    data["R"] = np.zeros((3, 5, 4))
    assert any("'R' must have shape" in e for e in NPZSchema().validate(data)[1])


def test_z_must_be_two_dimensional():
    data = _good_dataset()
    data["Z"] = np.ones(5, dtype=np.int32)
    assert any("'Z' must have shape" in e for e in NPZSchema().validate(data)[1])


def test_r_and_z_leading_axes_must_agree():
    data = _good_dataset()
    data["Z"] = np.ones((3, 4), dtype=np.int32)  # 4 atoms vs R's 5
    assert any("shape mismatch" in e for e in NPZSchema().validate(data)[1])


def test_energy_may_be_flat_or_a_column():
    for shape in [(3,), (3, 1)]:
        data = _good_dataset()
        data["E"] = np.zeros(shape)
        assert NPZSchema().validate(data)[0], f"shape {shape} should be accepted"


def test_energy_rank_three_is_rejected():
    data = _good_dataset()
    data["E"] = np.zeros((3, 1, 1))
    assert any("'E' must have shape" in e for e in NPZSchema().validate(data)[1])


def test_forces_must_match_coordinates_exactly():
    data = _good_dataset()
    data["F"] = np.zeros((3, 4, 3))
    assert any("'F' and 'R' shape mismatch" in e for e in NPZSchema().validate(data)[1])

    data["F"] = np.zeros((3, 5, 3))
    assert NPZSchema().validate(data)[0]


def test_dipole_must_be_n_by_three():
    data = _good_dataset()
    data["D"] = np.zeros((3, 2))
    assert any("'D' must have shape" in e for e in NPZSchema().validate(data)[1])

    data["D"] = np.zeros((3, 3))
    assert NPZSchema().validate(data)[0]


def test_esp_grid_must_be_a_point_cloud():
    data = _good_dataset()
    data["esp"] = np.zeros((3, 100))
    data["esp_grid"] = np.zeros((3, 100))  # missing the xyz axis
    assert any("'esp_grid' must have shape" in e for e in NPZSchema().validate(data)[1])


def test_esp_and_grid_point_counts_must_agree():
    data = _good_dataset()
    data["esp"] = np.zeros((3, 100))
    data["esp_grid"] = np.zeros((3, 80, 3))
    assert any("n_grid mismatch" in e for e in NPZSchema().validate(data)[1])

    data["esp_grid"] = np.zeros((3, 100, 3))
    assert NPZSchema().validate(data)[0]


def test_atom_counts_may_not_exceed_the_padded_width():
    data = _good_dataset(n_atoms=5)
    data["N"] = np.array([5, 6, 5], dtype=np.int32)
    assert any("exceed array dimensions" in e for e in NPZSchema().validate(data)[1])


def test_padded_structures_with_fewer_atoms_are_fine():
    """Ragged datasets are stored padded, so N < n_atoms is the normal case."""
    data = _good_dataset(n_atoms=5)
    data["N"] = np.array([5, 3, 1], dtype=np.int32)
    assert NPZSchema().validate(data)[0]


def test_several_violations_are_all_reported_at_once():
    data = _good_dataset()
    data["R"] = np.zeros((3, 5))
    data["D"] = np.zeros((3, 2))
    del data["E"]
    is_valid, errors = NPZSchema().validate(data)
    assert not is_valid
    assert len(errors) >= 3


# --- get_info ---------------------------------------------------------------


def test_get_info_summarises_sizes_and_ranges():
    data = _good_dataset(n_structures=4, n_atoms=6)
    data["E"] = np.array([-1.0, 0.0, 1.0, 2.0])
    data["R"] = np.linspace(-2.0, 3.0, 4 * 6 * 3).reshape(4, 6, 3)

    info = NPZSchema().get_info(data)

    assert info["n_structures"] == 4
    assert info["n_atoms"] == 6
    assert info["energy_range"]["min"] == pytest.approx(-1.0)
    assert info["energy_range"]["max"] == pytest.approx(2.0)
    assert info["energy_range"]["mean"] == pytest.approx(0.5)
    assert info["coordinate_range"]["min"] == pytest.approx(-2.0)
    assert info["coordinate_range"]["max"] == pytest.approx(3.0)


def test_get_info_counts_elements_ignoring_padding_zeros():
    data = _good_dataset(n_structures=2, n_atoms=4)
    data["Z"] = np.array([[8, 1, 1, 0], [6, 1, 0, 0]], dtype=np.int32)

    info = NPZSchema().get_info(data)

    assert info["unique_elements"] == [1, 6, 8]
    assert info["element_counts"] == {1: 3, 6: 1, 8: 1}
    assert 0 not in info["element_counts"], "padding must not be reported as an element"


def test_get_info_lists_which_schema_keys_are_present():
    data = _good_dataset()
    data["F"] = np.zeros_like(data["R"])

    info = NPZSchema().get_info(data)

    assert set(info["required_keys_present"]) == {"R", "Z", "E", "N"}
    assert info["optional_keys_present"] == ["F"]


# --- create_empty_npz -------------------------------------------------------


def test_created_dataset_validates_against_the_schema():
    data = create_empty_npz(7, 11)
    assert set(data) == set(REQUIRED_KEYS)
    assert NPZSchema(strict=True).validate(data)[0]


def test_created_dataset_has_the_requested_dimensions():
    data = create_empty_npz(7, 11)
    assert data["R"].shape == (7, 11, 3)
    assert data["Z"].shape == (7, 11)
    assert data["E"].shape == (7,)
    assert data["N"].shape == (7,)
    assert np.issubdtype(data["Z"].dtype, np.integer)
    assert np.issubdtype(data["N"].dtype, np.integer)


@pytest.mark.parametrize(
    ("prop", "shape"),
    [
        ("F", (2, 3, 3)),
        ("D", (2, 3)),
        ("Dxyz", (2, 3)),
        ("mono", (2, 3)),
        ("polar", (2, 3, 3)),
        ("quadrupole", (2, 3, 3)),
    ],
)
def test_optional_properties_get_their_documented_shapes(prop, shape):
    data = create_empty_npz(2, 3, properties=[prop])
    assert data[prop].shape == shape
    assert NPZSchema(strict=True).validate(data)[0]


def test_unhandled_optional_property_is_silently_dropped():
    """`esp` is a valid schema key but `create_empty_npz` has no branch for it.

    Pinned deliberately: callers that ask for it get a dataset without it and no
    warning, so this test is the record of that gap rather than an endorsement.
    """
    data = create_empty_npz(2, 3, properties=["esp"])
    assert "esp" not in data


# --- validate_npz on real files ---------------------------------------------


def test_validate_npz_round_trips_a_written_file(tmp_path, capsys):
    path = tmp_path / "good.npz"
    np.savez(path, **_good_dataset(n_structures=3, n_atoms=5))

    is_valid, info = validate_npz(str(path), verbose=True)

    assert is_valid
    assert info["n_structures"] == 3
    assert info["n_atoms"] == 5
    assert "is valid" in capsys.readouterr().out


def test_validate_npz_reports_a_malformed_file(tmp_path, capsys):
    path = tmp_path / "bad.npz"
    np.savez(path, R=np.zeros((3, 5, 3)))  # missing Z, E, N

    is_valid, info = validate_npz(str(path), verbose=True)

    assert not is_valid
    assert info is None
    assert "has errors" in capsys.readouterr().out


def test_validate_npz_returns_false_for_a_missing_file(tmp_path, capsys):
    is_valid, info = validate_npz(str(tmp_path / "absent.npz"), verbose=True)
    assert (is_valid, info) == (False, None)
    assert "Error loading NPZ file" in capsys.readouterr().out


def test_validate_npz_returns_false_for_a_file_that_is_not_an_npz(tmp_path):
    path = tmp_path / "not-really.npz"
    path.write_text("this is not a zip archive")
    assert validate_npz(str(path), verbose=False) == (False, None)


def test_validate_npz_can_be_quiet(tmp_path, capsys):
    path = tmp_path / "good.npz"
    np.savez(path, **_good_dataset())
    validate_npz(str(path), verbose=False)
    assert capsys.readouterr().out == ""


# --- CLI --------------------------------------------------------------------


def test_cli_without_arguments_prints_the_schema_and_returns_nonzero(
    monkeypatch, capsys
):
    """Regression: ``sys`` was imported inside ``main`` only, so the module-level
    ``sys.exit(main())`` raised NameError instead of exiting."""
    monkeypatch.setattr(npz_schema.sys, "argv", ["npz_schema.py"])

    assert main() == 1

    out = capsys.readouterr().out
    assert "Usage:" in out
    assert "Required keys:" in out


def test_cli_exits_zero_for_a_valid_file(tmp_path, monkeypatch):
    path = tmp_path / "good.npz"
    np.savez(path, **_good_dataset())
    monkeypatch.setattr(npz_schema.sys, "argv", ["npz_schema.py", str(path)])

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


def test_cli_exits_nonzero_for_an_invalid_file(tmp_path, monkeypatch):
    path = tmp_path / "bad.npz"
    np.savez(path, R=np.zeros((3, 5, 3)))
    monkeypatch.setattr(npz_schema.sys, "argv", ["npz_schema.py", str(path)])

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_module_can_be_run_as_a_script(tmp_path):
    """``python -m mmml.data.npz_schema`` must not die on an import error."""
    import subprocess
    import sys as _sys

    proc = subprocess.run(
        [_sys.executable, "-m", "mmml.data.npz_schema"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stderr
    assert "NameError" not in proc.stderr
    assert "Usage:" in proc.stdout
