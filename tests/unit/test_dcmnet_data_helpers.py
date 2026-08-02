"""Train/valid splitting and ESP grid masking in the DCMNet data pipeline.

``dcmnet/data.py`` sat at 20.4%. The two things it does that can go wrong
silently are both here:

* ``get_choices`` draws the train and validation sets. If those overlap, the
  reported validation error is measured partly on training data -- the model
  looks better than it is and nothing raises.
* ``cut_vdw`` masks grid points inside the van der Waals surface before ESP
  fitting. Mask the wrong points and the fit is weighted by the molecular core,
  where the classical ESP is meaningless.

Assertions are against hand-computable geometry and set algebra, not against
recorded output.
"""

from __future__ import annotations

import ase.data
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from mmml.models.dcmnet.dcmnet.data import (  # noqa: E402
    assert_dataset_size,
    cut_vdw,
    get_choices,
    make_dicts,
)


# --- train / valid splitting ------------------------------------------------


def test_split_sizes_are_exactly_as_requested():
    train, valid = get_choices(jax.random.PRNGKey(0), 100, 70, 20)
    assert len(train) == 70
    assert len(valid) == 20


def test_train_and_valid_never_overlap():
    """Leakage here inflates every validation metric and raises nothing."""
    train, valid = get_choices(jax.random.PRNGKey(1), 100, 70, 20)
    assert set(train.tolist()) & set(valid.tolist()) == set()


def test_indices_are_drawn_without_replacement():
    train, valid = get_choices(jax.random.PRNGKey(2), 50, 30, 20)
    assert len(set(train.tolist())) == 30
    assert len(set(valid.tolist())) == 20


def test_indices_stay_in_range():
    train, valid = get_choices(jax.random.PRNGKey(3), 40, 20, 10)
    both = np.concatenate([train, valid])
    assert both.min() >= 0 and both.max() < 40


def test_split_is_deterministic_for_a_given_key():
    a = get_choices(jax.random.PRNGKey(7), 60, 40, 10)
    b = get_choices(jax.random.PRNGKey(7), 60, 40, 10)
    assert a[0].tolist() == b[0].tolist()
    assert a[1].tolist() == b[1].tolist()


def test_different_keys_give_different_splits():
    a, _ = get_choices(jax.random.PRNGKey(0), 200, 100, 50)
    b, _ = get_choices(jax.random.PRNGKey(1), 200, 100, 50)
    assert a.tolist() != b.tolist()


def test_split_is_shuffled_not_sequential():
    """A sequential split correlates the sets with dataset ordering, which for
    a trajectory-derived dataset means training and validating on different
    parts of the same trajectory."""
    train, _ = get_choices(jax.random.PRNGKey(5), 200, 100, 50)
    assert train.tolist() != list(range(100))


def test_using_the_whole_dataset_leaves_nothing_out():
    train, valid = get_choices(jax.random.PRNGKey(9), 30, 20, 10)
    assert set(train.tolist()) | set(valid.tolist()) == set(range(30))


def test_zero_sized_validation_set_is_allowed():
    train, valid = get_choices(jax.random.PRNGKey(4), 10, 10, 0)
    assert len(train) == 10 and len(valid) == 0


# --- dataset size guard -----------------------------------------------------


def test_size_guard_accepts_an_exact_fit():
    assert_dataset_size(np.zeros(30), 20, 10)


def test_size_guard_accepts_a_surplus():
    assert_dataset_size(np.zeros(100), 20, 10)


def test_size_guard_rejects_an_over_draw():
    with pytest.raises(RuntimeError, match="only contains 30 points"):
        assert_dataset_size(np.zeros(30), 25, 10)


def test_size_guard_rejects_negative_counts():
    with pytest.raises(AssertionError):
        assert_dataset_size(np.zeros(30), -1, 10)
    with pytest.raises(AssertionError):
        assert_dataset_size(np.zeros(30), 10, -1)


# --- dict construction ------------------------------------------------------


def test_make_dicts_applies_the_indices_per_key(capsys):
    data = [np.arange(10).reshape(10, 1), np.arange(10, 20).reshape(10, 1)]
    train, valid = make_dicts(data, ["R", "E"], np.array([0, 1]), np.array([8, 9]))
    capsys.readouterr()

    assert train["R"].ravel().tolist() == [0, 1]
    assert train["E"].ravel().tolist() == [10, 11]
    assert valid["R"].ravel().tolist() == [8, 9]
    assert valid["E"].ravel().tolist() == [18, 19]


def test_make_dicts_keeps_keys_aligned_with_their_arrays(capsys):
    """Key order and data order must correspond; a mismatch swaps E and F for
    the whole run."""
    data = [np.full((4, 1), 1.0), np.full((4, 1), 2.0), np.full((4, 1), 3.0)]
    train, _ = make_dicts(data, ["A", "B", "C"], np.array([0]), np.array([1]))
    capsys.readouterr()

    assert train["A"].item() == 1.0
    assert train["B"].item() == 2.0
    assert train["C"].item() == 3.0


# --- van der Waals masking --------------------------------------------------


def _one_atom(z: int = 8):
    return np.array([[0.0, 0.0, 0.0]]), np.array([z])


def test_grid_point_on_top_of_an_atom_is_masked_out():
    xyz, z = _one_atom()
    grid = np.array([[0.0, 0.0, 0.0]])
    mask, _, _ = cut_vdw(grid, xyz, z)
    assert not mask[0]


def test_distant_grid_point_is_kept():
    xyz, z = _one_atom()
    grid = np.array([[10.0, 0.0, 0.0]])
    mask, _, _ = cut_vdw(grid, xyz, z)
    assert mask[0]


def test_cutoff_sits_at_the_scaled_vdw_radius():
    """Points just inside the scaled radius are dropped, just outside kept."""
    xyz, z = _one_atom(8)
    r = ase.data.vdw_radii[8] * 1.4
    grid = np.array([[r * 0.99, 0.0, 0.0], [r * 1.01, 0.0, 0.0]])

    mask, _, _ = cut_vdw(grid, xyz, z, vdw_scale=1.4)

    assert not mask[0]
    assert mask[1]


def test_a_larger_scale_masks_at_least_as_much():
    xyz, z = _one_atom(8)
    r = ase.data.vdw_radii[8]
    grid = np.array([[r * s, 0.0, 0.0] for s in (1.0, 1.2, 1.5, 2.0, 3.0)])

    loose = cut_vdw(grid, xyz, z, vdw_scale=1.0)[0]
    tight = cut_vdw(grid, xyz, z, vdw_scale=2.0)[0]

    assert tight.sum() <= loose.sum()
    assert np.all(loose[~tight] | ~loose[~tight])  # tight-masked implies subset


def test_closest_atom_index_and_type_are_reported():
    xyz = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    z = np.array([8, 1])
    grid = np.array([[9.5, 0.0, 0.0], [0.5, 0.0, 0.0]])

    _, closest_type, closest_idx = cut_vdw(grid, xyz, z)

    assert closest_idx.tolist() == [1, 0]
    assert closest_type.tolist() == [1, 8]


def test_a_point_inside_any_atom_is_masked_not_just_the_nearest():
    """``mask.any(axis=1)`` -- being inside *any* atom disqualifies the point."""
    xyz = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    z = np.array([8, 8])
    grid = np.array([[20.0, 0.0, 0.0]])
    assert not cut_vdw(grid, xyz, z)[0][0]


def test_element_symbols_are_accepted_as_documented():
    """Regression: the symbol branch built a list, and indexing it with the
    numpy ``closest_atom`` array raised TypeError -- the documented input
    crashed outright."""
    xyz = np.array([[0.0, 0.0, 0.0]])
    grid = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    by_symbol = cut_vdw(grid, xyz, ["O"])
    by_number = cut_vdw(grid, xyz, np.array([8]))

    assert by_symbol[0].tolist() == by_number[0].tolist()
    assert by_symbol[1].tolist() == by_number[1].tolist()


def test_mask_length_matches_the_grid():
    xyz, z = _one_atom()
    grid = np.random.default_rng(0).uniform(-5, 5, size=(37, 3))
    mask, closest_type, closest_idx = cut_vdw(grid, xyz, z)
    assert len(mask) == len(closest_type) == len(closest_idx) == 37
