"""Cache keying and flat-layout bookkeeping in the PhysNetJAX HDF5 reader.

The flat (concatenated-atom) layout stores every molecule's atoms end to end and
recovers molecule boundaries from ``mol_offsets``. An off-by-one there does not
crash: it silently attributes atoms to the wrong molecule, and training proceeds
on corrupted data. The same is true of the cache key -- if two different load
configurations hash to the same directory, the second one silently trains on the
first one's arrays.

Both were untested. The assertions below reconstruct molecule boundaries by hand
rather than trusting the offsets the code produced.
"""

from __future__ import annotations


import numpy as np
import pytest

from mmml.models.physnetjax.physnetjax.data.read_h5 import (
    _cache_key,
    _cache_key_flat,
    _concatenate_data_dicts,
    _concatenate_flat_data_dicts,
    _get_cache_dir,
    _get_cache_dir_flat,
    _restore_from_cache,
    _subset_flat_dataset,
)

_KEY_KWARGS = dict(
    natoms=8,
    energy_key="formation_energy",
    force_key="total_forces",
    dipole_key="dipole",
    max_structures=None,
    charge_filter=None,
    spin_key=None,
)


# --- cache keying -----------------------------------------------------------


def test_cache_key_is_deterministic(tmp_path):
    p = tmp_path / "d.h5"
    assert _cache_key(p, **_KEY_KWARGS) == _cache_key(p, **_KEY_KWARGS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("natoms", 16),
        ("energy_key", "total_energy"),
        ("force_key", "forces"),
        ("dipole_key", "dipole_xyz"),
        ("max_structures", 100),
        ("charge_filter", 0.0),
        ("spin_key", "spin"),
    ],
)
def test_every_load_parameter_changes_the_cache_key(tmp_path, field, value):
    """A parameter that does not affect the key lets one config read another's
    cached arrays -- silent, and invisible in the training logs."""
    p = tmp_path / "d.h5"
    kwargs = dict(_KEY_KWARGS)
    kwargs[field] = value
    assert _cache_key(p, **kwargs) != _cache_key(p, **_KEY_KWARGS)


def test_cache_key_depends_on_the_file(tmp_path):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    assert _cache_key(a, **_KEY_KWARGS) != _cache_key(b, **_KEY_KWARGS)


def test_flat_and_padded_keys_never_collide(tmp_path):
    """The two layouts produce differently-shaped arrays for identical
    parameters, so they must not share a cache directory."""
    p = tmp_path / "d.h5"
    assert _cache_key(p, **_KEY_KWARGS) != _cache_key_flat(p, **_KEY_KWARGS)
    assert _get_cache_dir(p, None, **_KEY_KWARGS) != _get_cache_dir_flat(
        p, None, **_KEY_KWARGS
    )


def test_cache_dir_defaults_beside_the_dataset(tmp_path):
    p = tmp_path / "d.h5"
    got = _get_cache_dir(p, None, **_KEY_KWARGS)
    assert got.parent == tmp_path / ".h5_cache"
    assert got.name.startswith("d_")


def test_explicit_cache_dir_is_honoured(tmp_path):
    p = tmp_path / "d.h5"
    elsewhere = tmp_path / "scratch"
    assert _get_cache_dir(p, elsewhere, **_KEY_KWARGS).parent == elsewhere


def test_flat_cache_dir_is_tagged_flat(tmp_path):
    got = _get_cache_dir_flat(tmp_path / "d.h5", None, **_KEY_KWARGS)
    assert "_flat_" in got.name


def test_restore_from_a_missing_cache_returns_none(tmp_path):
    assert _restore_from_cache(tmp_path / "absent", verbose=False) is None


def test_restore_from_a_corrupt_cache_returns_none_and_clears_it(tmp_path):
    """A stale cache must not wedge every future run of the same config."""
    bad = tmp_path / "cache"
    bad.mkdir()
    (bad / "junk").write_text("not a checkpoint")

    assert _restore_from_cache(bad, verbose=False) is None
    assert not bad.exists(), "incompatible cache should be removed"


# --- padded-layout concatenation -------------------------------------------


def _padded(n: int, natoms: int = 3, offset: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "R": np.full((n, natoms, 3), offset),
        "Z": np.ones((n, natoms), dtype=np.int32),
        "E": np.arange(n, dtype=np.float64) + offset,
        "N": np.full(n, natoms, dtype=np.int32),
    }


def test_concatenate_joins_along_the_structure_axis():
    out = _concatenate_data_dicts([_padded(2), _padded(3, offset=10.0)])
    assert out["R"].shape == (5, 3, 3)
    assert out["E"].tolist() == pytest.approx([0.0, 1.0, 10.0, 11.0, 12.0])


def test_concatenate_keeps_only_keys_present_everywhere():
    """``D`` is absent from some archives; a partial key would misalign."""
    a = _padded(2)
    a["D"] = np.zeros((2, 3))
    out = _concatenate_data_dicts([a, _padded(2)])
    assert "D" not in out
    assert set(out) == {"R", "Z", "E", "N"}


def test_concatenate_rejects_an_empty_list():
    with pytest.raises(ValueError, match="empty list"):
        _concatenate_data_dicts([])


# --- flat-layout bookkeeping ------------------------------------------------


def _flat(sizes: list[int], tag: float = 0.0) -> dict[str, np.ndarray]:
    """A flat dict whose atom rows encode (molecule index + tag) for tracing."""
    offsets = np.cumsum([0, *sizes]).astype(np.int32)
    total = int(offsets[-1])
    r = np.zeros((total, 3))
    z = np.zeros(total, dtype=np.int32)
    for i, n in enumerate(sizes):
        a0, a1 = int(offsets[i]), int(offsets[i + 1])
        r[a0:a1] = tag + i
        z[a0:a1] = i + 1
    return {
        "R": r,
        "Z": z,
        "F": r.copy(),
        "mol_offsets": offsets,
        "E": np.arange(len(sizes), dtype=np.float64) + tag,
        "N": np.array(sizes, dtype=np.int32),
        "Q": np.zeros((len(sizes), 1)),
        "S": np.ones((len(sizes), 1)),
    }


def test_flat_concatenation_rebases_the_second_files_offsets():
    a = _flat([2, 3])          # 5 atoms
    b = _flat([4], tag=100.0)  # 4 atoms

    out = _concatenate_flat_data_dicts([a, b])

    assert out["mol_offsets"].tolist() == [0, 2, 5, 9]
    assert len(out["R"]) == 9


def test_flat_concatenation_keeps_each_molecules_atoms_together():
    """Reconstruct every molecule from the merged offsets and check its atoms
    still carry the tag they were created with."""
    a = _flat([2, 3])
    b = _flat([4], tag=100.0)
    expected = [0.0, 1.0, 100.0]

    out = _concatenate_flat_data_dicts([a, b])
    mo = out["mol_offsets"]

    for i, want in enumerate(expected):
        block = out["R"][int(mo[i]) : int(mo[i + 1])]
        assert np.all(block == want), f"molecule {i} picked up the wrong atoms"


def test_flat_concatenation_offsets_are_monotonic_and_total_correct():
    out = _concatenate_flat_data_dicts([_flat([1, 2]), _flat([3]), _flat([1])])
    mo = out["mol_offsets"]
    assert mo[0] == 0
    assert np.all(np.diff(mo) > 0)
    assert int(mo[-1]) == len(out["R"]) == 7


def test_flat_concatenation_rejects_an_empty_list():
    with pytest.raises(ValueError, match="empty list"):
        _concatenate_flat_data_dicts([])


# --- flat subsetting --------------------------------------------------------


def test_subset_selects_the_requested_molecules():
    data = _flat([2, 3, 4])
    out = _subset_flat_dataset(data, np.array([0, 2]))

    assert out["mol_offsets"].tolist() == [0, 2, 6]
    assert out["E"].tolist() == pytest.approx([0.0, 2.0])
    assert out["N"].tolist() == [2, 4]


def test_subset_keeps_atoms_with_their_molecule():
    """The off-by-one that matters: atoms must follow the molecule they belong
    to, not the slot it happens to land in."""
    data = _flat([2, 3, 4])
    out = _subset_flat_dataset(data, np.array([2, 0]))  # deliberately reordered
    mo = out["mol_offsets"]

    assert np.all(out["R"][int(mo[0]) : int(mo[1])] == 2.0)
    assert np.all(out["R"][int(mo[1]) : int(mo[2])] == 0.0)


def test_subset_preserves_the_optional_dipole_when_present():
    data = _flat([2, 3])
    data["D"] = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    out = _subset_flat_dataset(data, np.array([1]))
    assert out["D"].shape == (1, 3)
    assert out["D"][0].tolist() == pytest.approx([0.0, 2.0, 0.0])


def test_subset_omits_the_dipole_when_the_source_has_none():
    assert "D" not in _subset_flat_dataset(_flat([2, 3]), np.array([0]))


def test_subset_of_everything_round_trips():
    data = _flat([2, 3, 4])
    out = _subset_flat_dataset(data, np.arange(3))
    assert out["mol_offsets"].tolist() == data["mol_offsets"].tolist()
    assert out["R"] == pytest.approx(data["R"])


def test_subset_of_a_single_molecule():
    out = _subset_flat_dataset(_flat([2, 3, 4]), np.array([1]))
    assert out["mol_offsets"].tolist() == [0, 3]
    assert np.all(out["Z"] == 2)


def test_subset_then_concatenate_is_consistent():
    """Splitting a dataset and rejoining the halves must restore the original
    molecule boundaries -- the train/valid split relies on exactly this."""
    data = _flat([2, 3, 4, 1])
    left = _subset_flat_dataset(data, np.array([0, 1]))
    right = _subset_flat_dataset(data, np.array([2, 3]))

    rejoined = _concatenate_flat_data_dicts([left, right])

    assert rejoined["mol_offsets"].tolist() == data["mol_offsets"].tolist()
    assert rejoined["R"] == pytest.approx(data["R"])
    assert rejoined["E"] == pytest.approx(data["E"])
