"""Unit tests for CHARMM CMAP parsing and bicubic interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.cgenff_cmap import (
    calc_map_derivatives,
    parse_cmap_types_from_prm,
)


def test_parse_cmap_types_from_prm_minimal(tmp_path) -> None:
    prm = tmp_path / "mini.prm"
    prm.write_text(
        "\n".join(
            [
                "CMAP",
                "! alanine-like header",
                "C NH1 CT1 C NH1 CT1 C NH1 2",
                "0.0 1.0",
                "2.0 3.0",
                "END",
            ]
        ),
        encoding="utf-8",
    )
    types = parse_cmap_types_from_prm(prm)
    key = ("C", "NH1", "CT1", "C", "NH1", "CT1", "C", "NH1")
    assert key in types
    assert types[key].resolution == 2
    assert types[key].energies == (0.0, 1.0, 2.0, 3.0)


def test_calc_map_derivatives_shape_and_finite() -> None:
    size = 4
    base = np.linspace(0.0, 1.0, size, dtype=np.float64)
    grid = np.tile(base, size)
    grid[0::size] = base
    coeffs = calc_map_derivatives(size, grid)
    assert coeffs.shape == (size * size, 16)
    assert np.all(np.isfinite(coeffs))


def test_bicubic_patch_evaluates_finite() -> None:
    from mmml.interfaces.pycharmmInterface.cgenff_cmap import _eval_bicubic
    import jax.numpy as jnp

    size = 4
    base = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    grid = np.tile(base, size)
    coeffs = calc_map_derivatives(size, grid)
    coeff = jnp.asarray(coeffs[0])
    val = _eval_bicubic(coeff, jnp.array(0.25), jnp.array(0.5))
    assert np.isfinite(float(val))


def test_resolve_cmap_key_forward_and_reverse() -> None:
    from mmml.interfaces.pycharmmInterface.cgenff_cmap import (
        CmapType,
        _resolve_cmap_type_key,
    )

    cmap_types = {
        ("C", "NH1", "CT1", "C", "NH1", "CT1", "C", "NH1"): CmapType(
            2, (0.0, 1.0, 2.0, 3.0)
        )
    }
    row = (0, 1, 2, 3, 4, 5, 6, 7)
    atom_types = ["C", "NH1", "CT1", "C", "NH1", "CT1", "C", "NH1"]
    assert _resolve_cmap_type_key(atom_types, row, cmap_types) is not None
    assert (
        _resolve_cmap_type_key(list(reversed(atom_types)), row, cmap_types) is not None
    )
    assert _resolve_cmap_type_key(atom_types, row, {}) is None


def test_parse_protein_prm_skips_cmap_blocks() -> None:
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        have_protein_toppar,
        protein_toppar_paths,
    )
    from mmml.interfaces.pycharmmInterface.cgenff_topology import (
        _merge_charmm_prm_parameters,
    )

    if not have_protein_toppar():
        pytest.skip("protein toppar not available")
    _, protein_prm = protein_toppar_paths()
    bonds, angles, dihedrals = _merge_charmm_prm_parameters(protein_prm)
    assert bonds
    assert angles
    assert dihedrals
