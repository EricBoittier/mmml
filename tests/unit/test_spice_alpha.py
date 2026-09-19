"""Synthetic SPICE-α HDF5 → PhysNet NPZ (no Zenodo download)."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from mmml.data.spice_alpha import (
    DEFAULT_CHARGE_TOL,
    SPICE_ALPHA_CANONICAL_UNITS,
    TRAIN_NPZ_UNITS,
    assert_train_npz_contract,
    classify_units_map,
    convert_spice_alpha_hdf5,
    iter_spice_alpha_frames,
    main,
    max_atomic_number,
    normalize_unit_token,
    pad_frames,
    parse_units_attr,
    read_units_map,
    write_physnet_npz,
)
from mmml.models.physnetjax.physnetjax.data.read_h5 import _detect_natoms, load_h5
from spice_alpha_fixtures import write_spice_h5

pytestmark = pytest.mark.data_loading

_write_spice_h5 = write_spice_h5


def test_normalize_unit_token_collapses_spelling():
    assert normalize_unit_token("e·Angstrom^2/volt") == normalize_unit_token(
        "e*Angstrom^2/volt"
    )
    assert "angstrom" in normalize_unit_token("Ångstrom")


def test_classify_units_map_canonical_and_atomic():
    assert classify_units_map(SPICE_ALPHA_CANONICAL_UNITS) == "canonical"
    assert (
        classify_units_map(
            {
                "conformations": "bohr",
                "dft_total_energy": "hartree",
                "dft_total_gradient": "hartree/bohr",
            }
        )
        == "atomic"
    )
    assert classify_units_map(None) == "unknown"
    assert classify_units_map({}) == "unknown"


def test_parse_units_attr_json_bytes_and_dict():
    payload = {"dft_total_energy": "eV"}
    assert parse_units_attr(json.dumps(payload)) == payload
    assert parse_units_attr(json.dumps(payload).encode()) == payload
    assert parse_units_attr(payload) == payload
    assert parse_units_attr(None) == {}


def test_iter_frames_flips_gradient_and_skips_non_molecule_groups(tmp_path):
    path = _write_spice_h5(tmp_path / "spice.hdf5")
    with h5py.File(path, "r") as handle:
        assert classify_units_map(read_units_map(handle)) == "canonical"
        frames = list(iter_spice_alpha_frames(handle))
    assert len(frames) == 3
    assert frames[0].F[0, 0] == pytest.approx(-1.5)
    assert frames[0].E == pytest.approx(-100.0)
    assert frames[0].D[0] == pytest.approx(0.2)
    assert frames[2].Z.tolist() == [8, 1, 1]
    assert frames[2].group == "O"


def test_iter_frames_can_keep_raw_gradient(tmp_path):
    path = _write_spice_h5(tmp_path / "spice.hdf5")
    with h5py.File(path, "r") as handle:
        frames = list(iter_spice_alpha_frames(handle, flip_gradient=False))
    assert frames[0].F[0, 0] == pytest.approx(1.5)


def test_neutral_only_drops_charged_conformer(tmp_path):
    path = _write_spice_h5(tmp_path / "spice.hdf5", charged=True)
    with h5py.File(path, "r") as handle:
        all_frames = list(iter_spice_alpha_frames(handle))
        neutral = list(iter_spice_alpha_frames(handle, neutral_only=True))
    assert len(all_frames) == 3
    assert len(neutral) == 2
    assert all(
        fr.Q is None or abs(fr.Q) <= DEFAULT_CHARGE_TOL for fr in neutral
    )


def test_pad_and_contract_variable_atom_counts(tmp_path):
    path = _write_spice_h5(tmp_path / "spice.hdf5")
    with h5py.File(path, "r") as handle:
        data = pad_frames(list(iter_spice_alpha_frames(handle)))
    assert_train_npz_contract(data)
    assert data["R"].shape == (3, 9, 3)
    assert data["N"].tolist() == [9, 9, 3]
    assert int(data["Z"][2, 3:].sum()) == 0
    assert data["Z"][2, :3].tolist() == [8, 1, 1]
    assert max_atomic_number(data) == 8
    units = json.loads(str(data["_mmml_units"]))
    assert units["E"] == TRAIN_NPZ_UNITS["E"]
    assert units["force"] == "negated dft_total_gradient"


def test_pad_too_small_raises(tmp_path):
    path = _write_spice_h5(tmp_path / "spice.hdf5", extra_group=False)
    with h5py.File(path, "r") as handle:
        frames = list(iter_spice_alpha_frames(handle))
    with pytest.raises(ValueError, match="pad_atoms"):
        pad_frames(frames, pad_atoms=4)


def test_empty_frames_raise():
    with pytest.raises(ValueError, match="no frames"):
        pad_frames([])


def test_missing_energy_group_is_skipped(tmp_path):
    path = tmp_path / "partial.hdf5"
    with h5py.File(path, "w") as handle:
        handle.attrs["units_map"] = json.dumps(SPICE_ALPHA_CANONICAL_UNITS)
        g = handle.create_group("bare")
        g.create_dataset("atomic_numbers", data=np.array([1, 1], np.int32))
        g.create_dataset("conformations", data=np.zeros((1, 2, 3)))
    with h5py.File(path, "r") as handle:
        assert list(iter_spice_alpha_frames(handle)) == []


def test_shape_mismatch_is_an_error(tmp_path):
    path = tmp_path / "bad.hdf5"
    with h5py.File(path, "w") as handle:
        g = handle.create_group("bad")
        g.create_dataset("atomic_numbers", data=np.array([1, 1], np.int32))
        g.create_dataset("conformations", data=np.zeros((2, 2, 3)))
        g.create_dataset("dft_total_energy", data=np.array([-1.0]))
        g.create_dataset("dft_total_gradient", data=np.zeros((2, 2, 3)))
    with h5py.File(path, "r") as handle:
        with pytest.raises(ValueError, match="energy/gradient"):
            list(iter_spice_alpha_frames(handle))


def test_convert_writes_npz_and_honors_max_frames(tmp_path):
    src = _write_spice_h5(tmp_path / "spice.hdf5")
    out = tmp_path / "train.npz"
    data = convert_spice_alpha_hdf5([src], out, max_frames=1)
    assert out.is_file()
    loaded = np.load(out, allow_pickle=True)
    assert loaded["E"].shape == (1,)
    assert loaded["F"][0, 0, 0] == pytest.approx(-1.5)
    assert_train_npz_contract(loaded)
    assert data["E"][0] == pytest.approx(-100.0)


def test_atomic_units_refused_unless_allowed(tmp_path):
    src = _write_spice_h5(
        tmp_path / "spice.hdf5",
        units={
            "conformations": "bohr",
            "dft_total_energy": "hartree",
            "dft_total_gradient": "hartree/bohr",
        },
    )
    with pytest.raises(ValueError, match="Bohr/Hartree"):
        convert_spice_alpha_hdf5([src], tmp_path / "out.npz")
    data = convert_spice_alpha_hdf5(
        [src], tmp_path / "out.npz", require_canonical_units=False, max_frames=1
    )
    assert data["E"][0] == pytest.approx(-100.0)


def test_read_h5_mol_star_layout_does_not_see_spice_groups(tmp_path):
    src = _write_spice_h5(tmp_path / "spice.hdf5")
    with pytest.raises(ValueError, match="No molecule groups"):
        _detect_natoms(src)
    with pytest.raises(ValueError, match="No structures loaded"):
        load_h5(src, natoms=9, cache=False)


def test_qcell_mol_star_file_still_detected(tmp_path):
    path = tmp_path / "qcell.h5"
    with h5py.File(path, "w") as handle:
        g = handle.create_group("mol_000001")
        g.create_dataset("atomic_numbers", data=np.array([1, 1], np.int32))
        g.create_dataset("positions", data=np.zeros((2, 3)))
        g.create_dataset("formation_energy", data=np.array(-1.0))
        g.create_dataset("total_forces", data=np.zeros((2, 3)))
    assert _detect_natoms(path) == 2


def test_cli_main_writes_npz(tmp_path):
    src = _write_spice_h5(tmp_path / "spice.hdf5")
    out = tmp_path / "cli.npz"
    assert main([str(src), "-o", str(out), "--max-frames", "2"]) == 0
    loaded = np.load(out)
    assert loaded["R"].shape[0] == 2


def test_write_physnet_npz_rejects_bad_shapes(tmp_path):
    with pytest.raises(ValueError, match="missing train keys"):
        write_physnet_npz({"R": np.zeros((1, 2, 3))}, tmp_path / "bad.npz")


def test_missing_dipole_is_nan(tmp_path):
    src = _write_spice_h5(tmp_path / "spice.hdf5", include_dipole=False, extra_group=False)
    with h5py.File(src, "r") as handle:
        frames = list(iter_spice_alpha_frames(handle))
    assert np.isnan(frames[0].D).all()


def test_assert_train_npz_contract_rejects_stale_units():
    data = {
        "R": np.zeros((1, 2, 3)),
        "Z": np.ones((1, 2), np.int32),
        "N": np.array([2], np.int32),
        "E": np.array([-1.0]),
        "F": np.zeros((1, 2, 3)),
        "_mmml_units": np.array(json.dumps({"E": "hartree", "F": "hartree_bohr"})),
    }
    with pytest.raises(ValueError, match="train units"):
        assert_train_npz_contract(data)


def test_assert_train_npz_contract_rejects_bad_n_and_d():
    data = {
        "R": np.zeros((1, 2, 3)),
        "Z": np.ones((1, 2), np.int32),
        "N": np.array([9], np.int32),
        "E": np.array([-1.0]),
        "F": np.zeros((1, 2, 3)),
    }
    with pytest.raises(ValueError, match="N must satisfy"):
        assert_train_npz_contract(data)
    data["N"] = np.array([2], np.int32)
    data["D"] = np.zeros((1, 2))
    with pytest.raises(ValueError, match="D must be"):
        assert_train_npz_contract(data)


def test_nonfinite_energy_and_gradient_are_skipped(tmp_path):
    path = _write_spice_h5(
        tmp_path / "spice.hdf5", extra_group=False, nan_energy=True
    )
    with h5py.File(path, "r") as handle:
        frames = list(iter_spice_alpha_frames(handle))
    assert len(frames) == 1
    assert frames[0].E == pytest.approx(-100.0)

    path = _write_spice_h5(
        tmp_path / "spice_g.hdf5", extra_group=False, nan_gradient=True
    )
    with h5py.File(path, "r") as handle:
        frames = list(iter_spice_alpha_frames(handle))
    assert len(frames) == 1


def test_group_level_units_map(tmp_path):
    path = _write_spice_h5(
        tmp_path / "spice.hdf5", extra_group=False, file_level_units=False
    )
    with h5py.File(path, "r") as handle:
        assert classify_units_map(read_units_map(handle)) == "canonical"


def test_convert_concatenates_two_files(tmp_path):
    a = _write_spice_h5(tmp_path / "a.hdf5", extra_group=False)
    b = _write_spice_h5(tmp_path / "b.hdf5", extra_group=False)
    data = convert_spice_alpha_hdf5([a, b], tmp_path / "both.npz")
    assert data["E"].shape == (4,)
    assert_train_npz_contract(data)


def test_polar_is_stored_when_present(tmp_path):
    src = _write_spice_h5(tmp_path / "spice.hdf5", extra_group=False)
    data = convert_spice_alpha_hdf5([src], tmp_path / "out.npz")
    assert "polar" in data
    assert data["polar"].shape == (2, 3, 3)
    assert data["polar"][0, 0, 0] == pytest.approx(1.2)


def test_unknown_units_are_allowed_by_default(tmp_path):
    src = _write_spice_h5(tmp_path / "spice.hdf5", units={"note": "missing"}, extra_group=False)
    data = convert_spice_alpha_hdf5([src], tmp_path / "out.npz", max_frames=1)
    assert data["E"][0] == pytest.approx(-100.0)


def test_missing_units_attr_is_unknown(tmp_path):
    src = _write_spice_h5(tmp_path / "spice.hdf5", units=None, extra_group=False)
    with h5py.File(src, "r") as handle:
        assert classify_units_map(read_units_map(handle)) == "unknown"


def test_bad_conformations_rank_is_an_error(tmp_path):
    path = tmp_path / "bad.hdf5"
    with h5py.File(path, "w") as handle:
        g = handle.create_group("bad")
        g.create_dataset("atomic_numbers", data=np.array([1, 1], np.int32))
        g.create_dataset("conformations", data=np.zeros((2, 3)))
        g.create_dataset("dft_total_energy", data=np.array([-1.0, -1.1]))
        g.create_dataset("dft_total_gradient", data=np.zeros((2, 2, 3)))
    with h5py.File(path, "r") as handle:
        with pytest.raises(ValueError, match="conformations must be"):
            list(iter_spice_alpha_frames(handle))


def test_dipole_shape_mismatch_is_an_error(tmp_path):
    path = tmp_path / "bad.hdf5"
    with h5py.File(path, "w") as handle:
        g = handle.create_group("bad")
        g.create_dataset("atomic_numbers", data=np.array([1, 1], np.int32))
        g.create_dataset("conformations", data=np.zeros((2, 2, 3)))
        g.create_dataset("dft_total_energy", data=np.array([-1.0, -1.1]))
        g.create_dataset("dft_total_gradient", data=np.zeros((2, 2, 3)))
        g.create_dataset("scf_dipole", data=np.zeros((2, 2)))
    with h5py.File(path, "r") as handle:
        with pytest.raises(ValueError, match="scf_dipole"):
            list(iter_spice_alpha_frames(handle))


def test_cli_neutral_only_and_no_flip(tmp_path):
    src = _write_spice_h5(tmp_path / "spice.hdf5", charged=True, extra_group=False)
    out = tmp_path / "neutral.npz"
    assert main([str(src), "-o", str(out), "--neutral-only"]) == 0
    loaded = np.load(out)
    assert loaded["E"].shape == (1,)
    out2 = tmp_path / "rawg.npz"
    assert main([str(src), "-o", str(out2), "--no-flip-gradient", "--max-frames", "1"]) == 0
    assert np.load(out2)["F"][0, 0, 0] == pytest.approx(1.5)


def test_cli_allow_atomic_units(tmp_path):
    src = _write_spice_h5(
        tmp_path / "spice.hdf5",
        extra_group=False,
        units={
            "conformations": "bohr",
            "dft_total_energy": "hartree",
            "dft_total_gradient": "hartree/bohr",
        },
    )
    out = tmp_path / "atomic.npz"
    with pytest.raises(ValueError, match="Bohr/Hartree"):
        main([str(src), "-o", str(out)])
    assert main([str(src), "-o", str(out), "--allow-atomic-units", "--max-frames", "1"]) == 0


def test_max_atomic_number_empty_n_is_zero():
    data = {
        "R": np.zeros((1, 2, 3)),
        "Z": np.array([[6, 1]], np.int32),
        "N": np.array([0], np.int32),
        "E": np.array([0.0]),
        "F": np.zeros((1, 2, 3)),
    }
    assert max_atomic_number(data) == 0


def test_package_export_round_trip(tmp_path):
    from mmml.data import convert_spice_alpha_hdf5 as exported

    src = _write_spice_h5(tmp_path / "spice.hdf5", extra_group=False)
    data = exported([src], tmp_path / "pkg.npz", max_frames=1)
    assert_train_npz_contract(data)
