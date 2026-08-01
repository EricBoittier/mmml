"""The PhysNetJAX copy of ``cut_vdw``, and the wiring that calls it.

``cut_vdw`` exists twice: ``dcmnet/data.py`` and
``physnetjax/data/cut_grid.py``. The DCMNet copy was fixed after the units
audit -- it kept ``elements`` as a plain list on the element-symbol path, so
``elements[closest_atom]`` (indexing with a numpy array) raised "only integer
scalar arrays can be converted to a scalar index" for exactly the input its
docstring advertises. The PhysNetJAX copy was left holding the same bug, and
``physnetjax/data/data.py`` called ``cut_vdw`` without importing it at all, so
``prepare_multiple_datasets(..., esp_mask=True)`` raised NameError.

Neither showed up anywhere: the DCMNet tests only import the DCMNet copy, and
the ESP-mask branch has no CI caller. The tests below pin both copies to the
same behaviour so a fix to one cannot silently skip the other.
"""

from __future__ import annotations

import ase.data
import numpy as np
import pytest

from mmml.models.dcmnet.dcmnet.data import cut_vdw as dcmnet_cut_vdw
from mmml.models.physnetjax.physnetjax.data.cut_grid import cut_vdw


def _one_atom(z: int = 8):
    return np.array([[0.0, 0.0, 0.0]]), np.array([z])


# --- the documented symbol path ---------------------------------------------


def test_element_symbols_are_accepted():
    """The regression. This raised TypeError before the fix."""
    xyz = np.array([[0.0, 0.0, 0.0]])
    grid = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    mask, closest_type, closest_idx = cut_vdw(grid, xyz, ["O"])

    assert list(mask) == [False, True]
    assert list(closest_type) == [8, 8]
    assert list(closest_idx) == [0, 0]


def test_symbols_and_atomic_numbers_agree():
    xyz = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    grid = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [12.0, 0.0, 0.0]])

    by_symbol = cut_vdw(grid, xyz, ["O", "H"])
    by_number = cut_vdw(grid, xyz, np.array([8, 1]))

    for from_symbols, from_numbers in zip(by_symbol, by_number):
        assert np.array_equal(from_symbols, from_numbers)


def test_a_python_list_of_atomic_numbers_is_accepted():
    """A list of ints hit the same array-indexing failure as symbols."""
    xyz = np.array([[0.0, 0.0, 0.0]])
    grid = np.array([[10.0, 0.0, 0.0]])

    mask, closest_type, _ = cut_vdw(grid, xyz, [8])

    assert bool(mask[0])
    assert int(closest_type[0]) == 8


def test_closest_atom_type_is_atomic_numbers_even_for_symbol_input():
    """Callers index element tables with this; symbols would break them."""
    xyz = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    grid = np.array([[4.9, 0.0, 0.0]])

    _, closest_type, closest_idx = cut_vdw(grid, xyz, ["O", "N"])

    assert np.issubdtype(np.asarray(closest_type).dtype, np.integer)
    assert int(closest_type[0]) == 7
    assert int(closest_idx[0]) == 1


# --- masking behaviour, checked against the radii table ----------------------


def test_grid_point_on_top_of_an_atom_is_masked_out():
    xyz, z = _one_atom()
    assert not cut_vdw(np.array([[0.0, 0.0, 0.0]]), xyz, z)[0][0]


def test_cutoff_sits_at_the_scaled_vdw_radius():
    xyz, z = _one_atom(8)
    r = ase.data.vdw_radii[8] * 1.4
    grid = np.array([[r * 0.99, 0.0, 0.0], [r * 1.01, 0.0, 0.0]])

    mask, _, _ = cut_vdw(grid, xyz, z, vdw_scale=1.4)

    assert not mask[0]
    assert mask[1]


def test_a_larger_scale_masks_at_least_as_much():
    xyz, z = _one_atom(8)
    grid = np.linspace(0.0, 12.0, 40).reshape(-1, 1) * np.array([[1.0, 0.0, 0.0]])

    loose = cut_vdw(grid, xyz, z, vdw_scale=1.0)[0]
    tight = cut_vdw(grid, xyz, z, vdw_scale=2.0)[0]

    assert np.all(tight <= loose)
    assert tight.sum() < loose.sum()


def test_a_point_inside_any_atom_is_masked_not_just_the_nearest():
    """``mask.any(axis=1)`` -- a point can be outside its nearest atom's
    radius and inside a larger neighbour's."""
    xyz = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    grid = np.array([[3.0, 0.0, 0.0]])

    mask, _, closest_idx = cut_vdw(grid, xyz, np.array([8, 8]))

    assert not mask[0]
    assert int(closest_idx[0]) == 1


def test_mask_length_matches_the_grid():
    xyz = np.array([[0.0, 0.0, 0.0]])
    grid = np.random.default_rng(0).normal(size=(37, 3)) * 6.0

    mask, closest_type, closest_idx = cut_vdw(grid, xyz, np.array([6]))

    assert mask.shape == (37,)
    assert closest_type.shape == (37,)
    assert closest_idx.shape == (37,)


# --- the two copies must not diverge again -----------------------------------


@pytest.mark.parametrize("elements", [["O", "H"], [8, 1], np.array([8, 1])])
def test_the_two_implementations_return_the_same_thing(elements):
    xyz = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    grid = np.array([[0.0, 0.0, 0.0], [2.5, 0.5, 0.0], [9.0, 9.0, 9.0]])

    mine = cut_vdw(grid, xyz, elements)
    theirs = dcmnet_cut_vdw(grid, xyz, elements)

    for a, b in zip(mine, theirs):
        assert np.array_equal(a, b)


def test_the_esp_mask_caller_can_reach_cut_vdw():
    """``physnetjax/data/data.py`` called ``cut_vdw`` with no import of it.

    The ESP-mask branch that uses it raised NameError for every caller. Assert
    the name resolves in that module rather than re-running the whole dataset
    builder.
    """
    from mmml.models.physnetjax.physnetjax.data import data as physnetjax_data

    assert physnetjax_data.cut_vdw is cut_vdw
