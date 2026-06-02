"""PyCHARMM integration tests for COMP set/get and selective force-damp recipe."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
    apply_selective_force_damp_recipe,
    force_magnitudes_kcalmol_A,
    get_comparison_array,
    prepare_comp_for_iasvel0,
    run_charmm_script,
    set_comparison_array,
    zero_comparison_scalars,
)


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def charmm_aco_dimer():
    if not _can_import("pycharmm"):
        pytest.skip("pycharmm not available")

    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        build_ase_cluster,
        setup_charmm_nbonds,
    )

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.energy as energy

    build_ase_cluster("ACO", 2, spacing=6.0)
    setup_charmm_nbonds()
    energy.show()
    yield


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_comparison_array_roundtrip(charmm_aco_dimer):
    import pycharmm.coor as coor

    n = coor.get_natom()
    values = np.stack(
        [
            np.linspace(0.1, 0.4, n),
            np.linspace(-0.2, 0.5, n),
            np.linspace(0.3, -0.1, n),
            np.zeros(n),
        ],
        axis=1,
    )
    set_comparison_array(values[:, :3])
    out = get_comparison_array()
    assert out.shape == (n, 4)
    assert np.allclose(out[:, :3], values[:, :3], atol=1e-6)
    assert np.allclose(out[:, 3], 0.0, atol=1e-6)


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_scalar_zero_clears_comparison_array(charmm_aco_dimer):
    import pycharmm.coor as coor

    n = coor.get_natom()
    set_comparison_array(np.ones((n, 3)))
    zero_comparison_scalars()
    out = get_comparison_array()
    assert np.allclose(out, 0.0, atol=1e-8)


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_selective_force_damp_respects_threshold(charmm_aco_dimer):
    run_charmm_script("ENER")
    mags = force_magnitudes_kcalmol_A()
    high_threshold = float(np.max(mags) + 1.0)
    apply_selective_force_damp_recipe(
        min_force_kcalmol_A=high_threshold,
        force_scale=0.01,
    )
    comp_high = get_comparison_array()
    assert np.allclose(comp_high[:, :3], 0.0, atol=1e-8)

    run_charmm_script("ENER")
    low_threshold = float(np.percentile(mags, 50))
    scale = 0.01
    apply_selective_force_damp_recipe(
        min_force_kcalmol_A=low_threshold,
        force_scale=scale,
    )
    comp = get_comparison_array()
    import pycharmm.coor as coor

    forces = coor.get_forces()[["dx", "dy", "dz"]].to_numpy(dtype=float)
    expected = scale * forces
    high_mask = mags >= low_threshold
    assert np.any(high_mask)
    assert np.allclose(comp[high_mask, :3], expected[high_mask], rtol=1e-4, atol=1e-6)
    assert np.allclose(comp[~high_mask, :3], 0.0, atol=1e-8)


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_lower_threshold_includes_more_atoms(charmm_aco_dimer):
    run_charmm_script("ENER")
    mags = force_magnitudes_kcalmol_A()
    p90 = float(np.percentile(mags, 90))
    p50 = float(np.percentile(mags, 50))

    apply_selective_force_damp_recipe(min_force_kcalmol_A=p90, force_scale=0.01)
    comp_strict = get_comparison_array()
    n_strict = int(np.count_nonzero(np.linalg.norm(comp_strict[:, :3], axis=1)))

    run_charmm_script("ENER")
    apply_selective_force_damp_recipe(min_force_kcalmol_A=p50, force_scale=0.01)
    comp_loose = get_comparison_array()
    n_loose = int(np.count_nonzero(np.linalg.norm(comp_loose[:, :3], axis=1)))

    assert n_loose >= n_strict


@pytest.mark.skipif(not _can_import("pycharmm"), reason="pycharmm not available")
def test_prepare_comp_zero_only(charmm_aco_dimer):
    n = prepare_comp_for_iasvel0(zero_only=True)
    assert n == 0
    comp = get_comparison_array()
    assert np.allclose(comp, 0.0, atol=1e-8)
