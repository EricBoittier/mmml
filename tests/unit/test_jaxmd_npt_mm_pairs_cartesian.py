"""MM pair updates follow the updater's configured frame, not ``is_npt``."""
from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    mm_pair_fractional_to_cartesian,
    mm_pair_positions_for_update,
    refresh_mm_pairs,
)

_L = 12.0
_CUTOFF = 5.0


def _mic(d: np.ndarray, box: np.ndarray) -> np.ndarray:
    return d - box * np.round(d / box)


def _pair_updater(*, fractional_coordinates: bool):
    """Brute-force interatomic updater that honors the real frame contract."""
    n = 6
    ii, jj = np.triu_indices(n, k=1)

    def update_fn(positions, box=None, **_):
        box_np = np.asarray(box, dtype=np.float64).reshape(-1)[:3]
        P = np.asarray(positions, dtype=np.float64)
        if fractional_coordinates:
            P = P * box_np
        r = np.linalg.norm(_mic(P[jj] - P[ii], box_np), axis=1)
        mask = r < _CUTOFF
        return (
            jnp.asarray(np.stack([ii, jj], axis=1), dtype=jnp.int32),
            jnp.asarray(mask, dtype=jnp.float32),
        )

    update_fn.fractional_coordinates = fractional_coordinates
    return update_fn


def _energy_from_pairs(positions_cart, pair_idx, pair_mask, box):
    idx = np.asarray(pair_idx)
    mask = np.asarray(pair_mask)
    d = positions_cart[idx[:, 1]] - positions_cart[idx[:, 0]]
    d = _mic(d, box)
    r = np.linalg.norm(d, axis=1)
    return float(np.sum(mask * np.exp(-r)))


def test_refresh_mm_pairs_reads_updater_attribute_not_ensemble() -> None:
    rng = np.random.default_rng(4)
    R = rng.uniform(0.5, _L - 0.5, size=(6, 3))
    box = np.array([_L, _L, _L], dtype=np.float64)
    R_frac = R / box

    cart_fn = _pair_updater(fractional_coordinates=False)
    frac_fn = _pair_updater(fractional_coordinates=True)

    idx_c, mask_c = refresh_mm_pairs(
        cart_fn, R, box, positions_are_cartesian=True
    )
    idx_f, mask_f = refresh_mm_pairs(
        frac_fn, R, box, positions_are_cartesian=True
    )
    idx_c_from_frac, mask_c_from_frac = refresh_mm_pairs(
        cart_fn, R_frac, box, positions_are_cartesian=False
    )
    idx_f_from_frac, mask_f_from_frac = refresh_mm_pairs(
        frac_fn, R_frac, box, positions_are_cartesian=False
    )

    np.testing.assert_array_equal(np.asarray(idx_c), np.asarray(idx_f))
    np.testing.assert_array_equal(np.asarray(mask_c), np.asarray(mask_f))
    np.testing.assert_array_equal(np.asarray(mask_c), np.asarray(mask_c_from_frac))
    np.testing.assert_array_equal(np.asarray(mask_f), np.asarray(mask_f_from_frac))

    e_c = _energy_from_pairs(R, idx_c, mask_c, box)
    e_f = _energy_from_pairs(R, idx_f, mask_f, box)
    assert e_c != 0.0
    np.testing.assert_allclose(e_c, e_f, rtol=1e-12)


def test_mismatched_frame_without_helper_changes_the_pair_set() -> None:
    """Feeding fractional coords to a Cartesian updater is the #231 failure."""
    rng = np.random.default_rng(5)
    R = rng.uniform(0.5, _L - 0.5, size=(6, 3))
    box = np.array([_L, _L, _L], dtype=np.float64)
    cart_fn = _pair_updater(fractional_coordinates=False)
    _, mask_ok = cart_fn(R, box=box)
    _, mask_bad = cart_fn(R / box, box=box)
    assert int(np.sum(np.asarray(mask_ok))) != int(np.sum(np.asarray(mask_bad)))


def test_empty_pair_list_is_legitimate_for_sparse_system() -> None:
    """A single isolated molecule can have zero intermolecular pairs."""
    R = np.array([[1.0, 1.0, 1.0], [1.2, 1.0, 1.0]], dtype=np.float64)
    box = np.array([_L, _L, _L], dtype=np.float64)

    def empty_fn(positions, box=None, **_):
        return (
            jnp.zeros((0, 2), dtype=jnp.int32),
            jnp.zeros((0,), dtype=jnp.float32),
        )

    empty_fn.fractional_coordinates = False
    idx, mask = refresh_mm_pairs(empty_fn, R, box, positions_are_cartesian=True)
    assert int(np.asarray(mask).size) == 0
    assert int(np.asarray(idx).shape[0]) == 0


def test_mm_pair_positions_for_update_both_directions() -> None:
    R = np.array([[3.0, 0.0, 0.0], [0.0, 6.0, 0.0]], dtype=np.float64)
    box = np.array([_L, _L, _L], dtype=np.float64)
    frac = R / box
    np.testing.assert_allclose(
        mm_pair_positions_for_update(
            R, box, positions_are_cartesian=True, fractional_coordinates=True
        ),
        frac,
    )
    np.testing.assert_allclose(
        mm_pair_positions_for_update(
            frac, box, positions_are_cartesian=False, fractional_coordinates=False
        ),
        mm_pair_fractional_to_cartesian(frac, box),
    )
    assert (
        mm_pair_positions_for_update(
            R, box, positions_are_cartesian=True, fractional_coordinates=False
        )
        is R
    )


def test_nl_valid_pair_count_always_returns_a_bound_value() -> None:
    """Mask-sum failures must keep the fallback, not leave the name unset."""
    from mmml.cli.run.jaxmd_runner import _nl_valid_pair_count

    assert _nl_valid_pair_count(np.array([1, 0, 1, 1])) == 3
    assert _nl_valid_pair_count(None, fallback=7) == 7

    class _Unsummable:
        def __array__(self, dtype=None):
            raise TypeError("cannot form an array")

    assert _nl_valid_pair_count(_Unsummable(), fallback=4) == 4
    assert _nl_valid_pair_count(_Unsummable(), fallback=None) is None


def test_jaxmd_npt_init_refreshes_from_fractional_integrator_state() -> None:
    """NPT init_fn takes md_pos_frac; the pair refresh must use that same frame."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[2] / "mmml/cli/run/jaxmd_runner.py"
    ).read_text(encoding="utf-8")
    block = src.split("md_pos_frac = as_jaxmd_dtype", 1)[1]
    block = block.split("state = init_fn", 1)[0]
    refresh = block.split("refresh_mm_pairs(", 1)[1].split(")", 1)[0]
    assert "md_pos_frac" in refresh
    assert "positions_are_cartesian=False" in refresh
    assert "md_pos_wrapped" not in refresh
