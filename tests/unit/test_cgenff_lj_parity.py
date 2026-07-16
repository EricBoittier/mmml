"""CGenFF LJ parity: dataset npz fields vs the MD calculator's MM parameters.

Hybrid ML/MM training is only meaningful if the MM term it optimises against is
the *same* MM term pycharmm/jax-md evaluate at MD time.  Both sides ultimately
parse ``par_all36_cgenff.prm``, but with different conventions:

* dataset (``scripts/prepare_ml_mm_dataset.py``): sigma (Angstrom), epsilon > 0
* MD (``mm_energy_forces``):                      Rmin/2 (Angstrom), epsilon < 0

These tests pin that relationship so the two cannot silently drift, and so the
sign/length conventions are explicit rather than folklore.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Rmin/2 -> sigma:  sigma = 2 * (Rmin/2) / 2^(1/6)
RMIN_HALF_TO_SIGMA = 2.0 / (2.0 ** (1.0 / 6.0))


def _dataset_table():
    from scripts.prepare_ml_mm_dataset import DEF_PRM_PATH, load_cgenff_nonbonded_table

    nb_map, sigmas, epsilons = load_cgenff_nonbonded_table(Path(DEF_PRM_PATH))
    return DEF_PRM_PATH, nb_map, sigmas, epsilons


def _md_table():
    """Replicates the parser in mm_energy_forces (name -> (epsilon, Rmin/2))."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM

    out = {}
    for line in Path(CGENFF_PRM).read_text(errors="replace").splitlines():
        parts = line.split()
        if len(line) > 5 and len(parts) > 4 and parts[1] == "0.0" and line[0] != "!":
            name, _, ep, rmin_half = parts[:4]
            out[name] = (float(ep), float(rmin_half))
    return CGENFF_PRM, out


def test_both_paths_read_the_same_prm_file():
    ds_prm, *_ = _dataset_table()
    md_prm, _ = _md_table()
    assert Path(ds_prm).resolve() == Path(md_prm).resolve()


def test_sigma_matches_md_rmin_half_exactly():
    """Length convention reconciles for every shared type."""
    _, nb_map, sigmas, _ = _dataset_table()
    _, md = _md_table()
    shared = sorted(set(nb_map) & set(md))
    assert len(shared) > 100
    for name in shared:
        ds_sigma = float(sigmas[nb_map[name]])
        _, md_rmin_half = md[name]
        assert ds_sigma == pytest.approx(md_rmin_half * RMIN_HALF_TO_SIGMA, abs=1e-6), name


def test_epsilon_magnitude_matches_and_sign_convention_is_pinned():
    """|eps| is identical; dataset stores >=0, CHARMM PRM stores <=0.

    Feeding one into the other's LJ form without flipping turns the well into a
    barrier -- pin it so the conversion is never implicit.
    """
    _, nb_map, _, epsilons = _dataset_table()
    _, md = _md_table()
    shared = sorted(set(nb_map) & set(md))
    for name in shared:
        ds_eps = float(epsilons[nb_map[name]])
        md_eps = md[name][0]
        assert abs(ds_eps) == pytest.approx(abs(md_eps), abs=1e-9), name
        assert ds_eps >= 0.0, f"dataset epsilon must be non-negative: {name}={ds_eps}"
        assert md_eps <= 0.0, f"CHARMM PRM epsilon must be non-positive: {name}={md_eps}"


def test_dcm_and_acetone_types_are_consistent():
    """The types actually used by the DCM/ACO datasets."""
    _, nb_map, sigmas, epsilons = _dataset_table()
    _, md = _md_table()
    for name in ("CG331", "CG321", "CG2O5", "OG2D3", "CLGA1", "HGA2", "HGA3"):
        assert name in nb_map and name in md, name
        i = nb_map[name]
        md_eps, md_rmin_half = md[name]
        assert float(epsilons[i]) == pytest.approx(-md_eps, abs=1e-9)
        assert float(sigmas[i]) == pytest.approx(md_rmin_half * RMIN_HALF_TO_SIGMA, abs=1e-6)


def test_dataset_table_is_a_superset_of_md_types():
    """Dataset adds water/placeholder types; it must not be missing any MD type."""
    _, nb_map, _, _ = _dataset_table()
    _, md = _md_table()
    assert not (set(md) - set(nb_map)), sorted(set(md) - set(nb_map))
