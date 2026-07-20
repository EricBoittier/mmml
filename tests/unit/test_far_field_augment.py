"""Tests for synthetic far-field composite construction (size-extensivity
training augmentation) -- see mmml/models/physnetjax/physnetjax/training/
far_field_augment.py for the physics rationale.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.physnetjax.physnetjax.training.far_field_augment import (
    append_far_field_composites_to_data,
    build_far_field_composites,
    build_one_far_field_composite,
    compute_safe_separation,
    eligible_source_indices,
)
from scripts.train_so3lr_spooky_extxyz import _per_fragment_charge_conservation_mse


def _make_flat_data(n_structures: int = 6, rng: np.random.Generator | None = None) -> dict:
    """Minimal flat cache dict: 3 atoms/structure, alternating eligibility."""
    rng = rng or np.random.default_rng(0)
    n_atoms_per = 3
    mol_offsets = np.arange(0, (n_structures + 1) * n_atoms_per, n_atoms_per, dtype=np.int64)
    n_atoms = n_structures * n_atoms_per
    R = rng.normal(scale=0.5, size=(n_atoms, 3))
    Z = np.tile([8, 1, 1], n_structures).astype(np.int32)
    F = rng.normal(scale=0.1, size=(n_atoms, 3))
    E = rng.normal(loc=-76.0, scale=1.0, size=n_structures)
    D = rng.normal(scale=0.5, size=(n_structures, 3))
    N = np.full(n_structures, n_atoms_per, dtype=np.int32)
    # Even indices: neutral singlet (eligible). Odd indices: charged or
    # non-singlet (ineligible) -- exercises the eligibility filter.
    Q = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n_structures)])
    S = np.array([1.0 if i % 2 == 0 else 2.0 for i in range(n_structures)])
    return {
        "mol_offsets": mol_offsets,
        "R": R,
        "Z": Z,
        "F": F,
        "E": E,
        "Q": Q,
        "S": S,
        "D": D,
        "N": N,
    }


def test_eligible_source_indices_filters_neutral_singlet_only():
    data = _make_flat_data(n_structures=6)
    eligible = eligible_source_indices(data)
    np.testing.assert_array_equal(eligible, np.array([0, 2, 4]))


def test_eligible_source_indices_empty_when_none_qualify():
    data = _make_flat_data(n_structures=1)
    data["Q"] = np.array([1.0])
    data["S"] = np.array([2.0])
    assert eligible_source_indices(data).size == 0


def test_compute_safe_separation_uses_binding_constraint():
    # electrostatics_off_end dominates
    assert compute_safe_separation(cutoff=6.0, electrostatics_off_end=10.0) == pytest.approx(12.0)
    # cutoff dominates when it's the larger of the two
    assert compute_safe_separation(cutoff=15.0, electrostatics_off_end=10.0) == pytest.approx(17.0)
    # safety margin is additive and overridable
    assert compute_safe_separation(
        cutoff=6.0, electrostatics_off_end=10.0, safety_margin=5.0
    ) == pytest.approx(15.0)


def test_build_one_far_field_composite_enforces_minimum_separation():
    data = _make_flat_data(n_structures=6)
    safe_separation = compute_safe_separation(cutoff=6.0, electrostatics_off_end=10.0)
    composite = build_one_far_field_composite(
        data, np.array([0, 2, 4]), safe_separation=safe_separation
    )
    r = composite["R"]
    mol_id = composite["mol_id"]
    min_cross_fragment_dist = np.inf
    for i, j in itertools.combinations(range(len(r)), 2):
        if mol_id[i] == mol_id[j]:
            continue
        d = np.linalg.norm(r[i] - r[j])
        min_cross_fragment_dist = min(min_cross_fragment_dist, d)
    assert min_cross_fragment_dist >= safe_separation


def test_build_one_far_field_composite_sums_are_exact():
    data = _make_flat_data(n_structures=6)
    frag_indices = np.array([0, 2, 4])
    composite = build_one_far_field_composite(data, frag_indices, safe_separation=12.0)

    expected_E = sum(float(data["E"][i]) for i in frag_indices)
    expected_Q = sum(float(data["Q"][i]) for i in frag_indices)
    expected_D = sum(data["D"][i] for i in frag_indices)
    expected_N = 3 * len(frag_indices)

    assert composite["E"] == pytest.approx(expected_E)
    assert composite["Q"] == pytest.approx(expected_Q)
    assert composite["S"] == 1.0
    assert composite["N"] == expected_N
    np.testing.assert_allclose(composite["D"], expected_D)

    # mol_id has exactly len(frag_indices) distinct values, each covering
    # exactly one fragment's atoms (3 atoms/fragment in this fixture).
    unique, counts = np.unique(composite["mol_id"], return_counts=True)
    np.testing.assert_array_equal(unique, np.arange(len(frag_indices)))
    np.testing.assert_array_equal(counts, np.full(len(frag_indices), 3))


def test_build_far_field_composites_only_draws_eligible_fragments():
    data = _make_flat_data(n_structures=6)
    rng = np.random.default_rng(1)
    composites = build_far_field_composites(
        data, rng, n_composites=4, k_min=2, k_max=5, safe_separation=12.0
    )
    assert len(composites) == 4
    for comp in composites:
        n_fragments = len(np.unique(comp["mol_id"]))
        assert 2 <= n_fragments <= 5
        # Every composite must be built entirely from Q=0 fragments, so its
        # total charge must be exactly zero (sum of exact zeros).
        assert comp["Q"] == pytest.approx(0.0)


def test_build_far_field_composites_raises_when_no_eligible_source():
    data = _make_flat_data(n_structures=1)
    data["Q"] = np.array([1.0])
    data["S"] = np.array([2.0])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="No exactly-neutral"):
        build_far_field_composites(data, rng, n_composites=1)


def test_append_far_field_composites_to_data_preserves_originals_and_flags_composites():
    data = _make_flat_data(n_structures=6)
    rng = np.random.default_rng(2)
    composites = build_far_field_composites(
        data, rng, n_composites=3, k_min=2, k_max=3, safe_separation=12.0
    )
    n_orig = 6
    out = append_far_field_composites_to_data(data, composites)

    n_atoms_orig = int(data["mol_offsets"][-1])
    np.testing.assert_allclose(out["R"][:n_atoms_orig], data["R"])
    np.testing.assert_allclose(out["E"].reshape(-1)[:n_orig], data["E"])

    assert out["is_far_field_composite"].shape == (n_orig + 3,)
    assert not out["is_far_field_composite"][:n_orig].any()
    assert out["is_far_field_composite"][n_orig:].all()

    # mol_id for every pre-existing real structure's atoms must be all-zero
    # (single implicit fragment), not colliding with composite fragment ids.
    assert (out["mol_id"][:n_atoms_orig] == 0).all()

    # mol_offsets strictly increasing and spans the full concatenated R.
    offsets = out["mol_offsets"]
    assert offsets.shape == (n_orig + 3 + 1,)
    assert np.all(np.diff(offsets) > 0)
    assert offsets[-1] == out["R"].shape[0]

    # cgenff_type_idx added (all zero) purely to satisfy the has_mm gate in
    # build_spooky_batch_from_flat_data -- must never reference a real vdW
    # table entry.
    assert (out["cgenff_type_idx"] == 0).all()


def test_per_fragment_charge_mse_excludes_single_fragment_structures():
    """A single-fragment structure with nonzero net charge must NOT contribute
    -- that signal already comes from the whole-structure charge_mse term, so
    including it here would silently double its effective weight."""
    k_max = 3
    atomic_charges = jnp.array([0.6, 0.4], dtype=jnp.float32)  # sums to 1.0
    mol_id = jnp.array([0, 0], dtype=jnp.int32)
    atom_mask = jnp.ones(2, dtype=jnp.float32)
    batch_segments = jnp.array([0, 0], dtype=jnp.int32)

    mse = _per_fragment_charge_conservation_mse(
        atomic_charges, mol_id, atom_mask, batch_segments, batch_size=1, k_max=k_max
    )
    assert float(mse) == pytest.approx(0.0)


def test_per_fragment_charge_mse_penalizes_multi_fragment_imbalance():
    """Two-fragment composite with fragment sums [+0.3, -0.3]: MSE must equal
    the mean of the two squared per-fragment sums, computed independently."""
    k_max = 3
    # Structure 0: fragment 0 -> atoms 0,1 (sum 0.3); fragment 1 -> atom 2 (sum -0.3)
    atomic_charges = jnp.array([0.2, 0.1, -0.3], dtype=jnp.float32)
    mol_id = jnp.array([0, 0, 1], dtype=jnp.int32)
    atom_mask = jnp.ones(3, dtype=jnp.float32)
    batch_segments = jnp.array([0, 0, 0], dtype=jnp.int32)

    mse = _per_fragment_charge_conservation_mse(
        atomic_charges, mol_id, atom_mask, batch_segments, batch_size=1, k_max=k_max
    )
    expected = ((0.3**2) + (0.3**2)) / 2.0
    assert float(mse) == pytest.approx(expected, abs=1e-6)


def test_per_fragment_charge_mse_ignores_masked_padding_atoms():
    """Padding atoms (atom_mask=0) must not perturb the per-fragment sum:
    fragment 0 (atoms 0,1) is exactly balanced; fragment 1 is a real,
    exactly-neutral atom plus a masked padding atom carrying a huge charge
    that must be excluded from the sum."""
    k_max = 2
    atomic_charges = jnp.array([0.3, -0.3, 0.0, 999.0], dtype=jnp.float32)
    mol_id = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    atom_mask = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)  # atom 3 is padding
    batch_segments = jnp.array([0, 0, 0, 0], dtype=jnp.int32)

    mse = _per_fragment_charge_conservation_mse(
        atomic_charges, mol_id, atom_mask, batch_segments, batch_size=1, k_max=k_max
    )
    # Both fragments' real (unmasked) atoms sum to exactly 0.
    assert float(mse) == pytest.approx(0.0, abs=1e-6)


def test_per_fragment_charge_mse_averages_across_batch_and_structures():
    """Mixed batch: one single-fragment structure (excluded) and one
    two-fragment composite (included) -- verifies batch_segments correctly
    routes each structure's atoms to its own k_max-sized fragment block."""
    k_max = 2
    atomic_charges = jnp.array(
        [1.0, 0.5, -0.4],  # struct 0 (single fragment, excluded), struct 1 frags
        dtype=jnp.float32,
    )
    mol_id = jnp.array([0, 0, 1], dtype=jnp.int32)
    atom_mask = jnp.ones(3, dtype=jnp.float32)
    batch_segments = jnp.array([0, 1, 1], dtype=jnp.int32)

    mse = _per_fragment_charge_conservation_mse(
        atomic_charges, mol_id, atom_mask, batch_segments, batch_size=2, k_max=k_max
    )
    # Struct 1's fragments sum to 0.5 and -0.4 respectively.
    expected = ((0.5**2) + (0.4**2)) / 2.0
    assert float(mse) == pytest.approx(expected, abs=1e-6)
