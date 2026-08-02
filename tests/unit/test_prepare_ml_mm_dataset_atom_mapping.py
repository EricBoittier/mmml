"""Regression tests for CGenFF template-to-geometry atom correspondence."""

from __future__ import annotations

import numpy as np

# The template-to-geometry mapping now lives in the shared core; the Orbax
# script (scripts/prepare_ml_mm_dataset.py) and `mmml prepare-mm-dataset` both
# call it from here.
from mmml.data.cgenff_dataset import load_reference, match_cgenff_template

_REF = load_reference()


def match_cgenff_template_fast(z_sub, pos_sub=None, target_charge=0.0, canonical_smiles=None):
    return match_cgenff_template(
        _REF, z_sub, pos_sub, target_charge=target_charge, canonical_smiles=canonical_smiles
    )


def _water_ohh() -> tuple[np.ndarray, np.ndarray]:
    z = np.array([8, 1, 1], dtype=np.int32)
    r = np.array(
        [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]],
        dtype=np.float64,
    )
    return z, r


def test_tip3_parameters_follow_permuted_geometry_atom_order() -> None:
    z, r = _water_ohh()
    permutation = np.array([1, 2, 0])  # observed H,H,O instead of template O,H,H

    residue, _, charges = match_cgenff_template_fast(z[permutation], r[permutation])

    assert residue == "TIP3"
    np.testing.assert_allclose(charges, [0.417, 0.417, -0.834])


def test_tip3_parameters_preserve_template_order_geometry() -> None:
    z, r = _water_ohh()

    residue, _, charges = match_cgenff_template_fast(z, r)

    assert residue == "TIP3"
    np.testing.assert_allclose(charges, [-0.834, 0.417, 0.417])
