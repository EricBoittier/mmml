"""PhysNet train NPZ contract: units, gradient sign, data_keys (synthetic only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.cli.make.make_training import _parse_list_option
from mmml.data.spice_alpha import (
    assert_train_npz_contract,
    convert_spice_alpha_hdf5,
    max_atomic_number,
)
from mmml.data.units import HARTREE_TO_EV
from spice_alpha_fixtures import write_spice_h5 as _write_spice_h5

pytestmark = pytest.mark.data_loading


def _minimal_train_npz(n=4, n_atoms=3, energy=-100.0):
    rng = np.random.default_rng(0)
    r = rng.random((n, n_atoms, 3)) + 1.0
    return {
        "R": r,
        "Z": np.tile(np.array([6, 1, 1], np.int32), (n, 1)),
        "N": np.full(n, n_atoms, np.int32),
        "E": np.full(n, energy),
        "F": np.zeros((n, n_atoms, 3)),
        "D": np.zeros((n, 3)),
        "_mmml_units": np.array(
            json.dumps({"R": "angstrom", "E": "ev", "F": "ev_angstrom", "D": "e_angstrom"})
        ),
    }


def test_preserve_units_keeps_already_ev_energies(tmp_path, monkeypatch):
    from mmml.cli.misc import fix_and_split as fas

    data = _minimal_train_npz()
    efd = tmp_path / "efd.npz"
    np.savez_compressed(efd, **data)
    monkeypatch.setattr(fas, "validate_fixed_data", lambda *a, **k: True)
    out = tmp_path / "splits"
    ok = fas.fix_and_split_data(
        efd_file=efd,
        grid_file=None,
        output_dir=out,
        verbose=False,
        skip_validation=True,
        preserve_units=True,
        train_frac=0.5,
        valid_frac=0.25,
        test_frac=0.25,
    )
    assert ok
    train = np.load(out / "energies_forces_dipoles_train.npz")
    assert train["E"][0] == pytest.approx(-100.0)


def test_default_fix_and_split_treats_ev_as_hartree(tmp_path, monkeypatch):
    """The trap: already-eV SPICE-α run through PySCF defaults."""
    from mmml.cli.misc import fix_and_split as fas

    data = _minimal_train_npz()
    efd = tmp_path / "efd.npz"
    np.savez_compressed(efd, **data)
    monkeypatch.setattr(fas, "validate_fixed_data", lambda *a, **k: True)
    out = tmp_path / "splits"
    ok = fas.fix_and_split_data(
        efd_file=efd,
        grid_file=None,
        output_dir=out,
        verbose=False,
        skip_validation=True,
        preserve_units=False,
        energy_in="hartree",
        energy_out="ev",
        force_in="hartree_bohr",
        force_out="ev_angstrom",
        coords_in="angstrom",
        coords_out="same",
        train_frac=0.5,
        valid_frac=0.25,
        test_frac=0.25,
    )
    assert ok
    train = np.load(out / "energies_forces_dipoles_train.npz")
    assert train["E"][0] == pytest.approx(-100.0 * HARTREE_TO_EV)
    assert abs(train["E"][0]) > 1000.0


def test_physnet_default_data_keys_are_efd_not_polar():
    keys = ("R", "Z", "F", "N", "E", "D", "batch_segments")
    assert "polar" not in keys
    assert set(keys) >= {"R", "Z", "N", "E", "F"}
    assert _parse_list_option(None) is None


def test_iodine_requires_max_atomic_number_53(tmp_path):
    src = _write_spice_h5(tmp_path / "spice.hdf5", extra_group=False)
    with __import__("h5py").File(src, "r+") as handle:
        handle["CCO"]["atomic_numbers"][8] = 53
    data = convert_spice_alpha_hdf5([src], tmp_path / "out.npz", max_frames=1)
    assert max_atomic_number(data) == 53
    assert max_atomic_number(data) > 35


def test_help_text_mentions_preserve_units():
    from mmml.cli.help_text import EXAMPLE_BLOCKS

    examples = [line for _, block in EXAMPLE_BLOCKS for line in block]
    assert any("preserve-units" in line for line in examples)


def test_assert_contract_accepts_distill_shaped_npz():
    data = _minimal_train_npz()
    data["kind"] = np.array([0, 0, 1, 1], np.int32)
    data["E_int"] = np.array([np.nan, np.nan, -0.1, -0.2])
    assert_train_npz_contract(data)


def test_example_yaml_documents_the_contract():
    text = Path("mmml/cli/misc/physnet_train.example.yaml").read_text()
    assert "training-npz-contract" in text
    assert "preserve-units" in text
    assert "−∇E" in text or "-∇E" in text or "force" in text.lower()
