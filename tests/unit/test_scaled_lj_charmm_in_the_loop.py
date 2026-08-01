"""CHARMM-in-the-loop: deployed LJ scales must reach CHARMM's own VDW energy.

`test_scaled_cgenff_prm.py` proves the rewritten prm carries the right numbers.
This proves CHARMM *reads* them, and — more importantly — that deploying twice
does not apply the scales twice. Silent double-application is the failure this
whole mechanism exists to avoid: it produces a plausible-looking energy from a
force field nobody fitted.

The assertions are exact rather than approximate. Pair epsilons combine as
``eps_ij = sqrt(eps_i * eps_j)``, so scaling *every* type's epsilon by ``f``
scales every pair epsilon by exactly ``f``; LJ energy is linear in eps_ij, and
CHARMM's switching function depends only on r. Therefore::

    E_vdw(all eps x f)  ==  f * E_vdw(base)

to float precision, through CHARMM's full nonbond machinery. Sigma has no such
identity (it sits inside r^-12/r^-6), so it is checked for *effect* instead.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.conftest import can_import_pycharmm

if not can_import_pycharmm():
    pytest.skip("PyCHARMM is not available", allow_module_level=True)

pytestmark = pytest.mark.pycharmm


def _sidecar(tmp_path, names, *, eps_factor=1.0, sig_factor=1.0, name="hybrid_mm.json"):
    p = tmp_path / name
    p.write_text(json.dumps({
        "learn_mm_lj_scales": True,
        "cgenff_type_names": list(names),
        "mm_lj_sigma_scale": [sig_factor] * len(names),
        "mm_lj_epsilon_scale": [eps_factor] * len(names),
        # Widen bounds: this is a deliberate probe, not a trained sidecar.
        "mm_lj_sigma_scale_bounds": [0.5, 2.0],
        "mm_lj_epsilon_scale_bounds": [0.1, 10.0],
    }))
    return p


@pytest.fixture
def charmm_dimer(pycharmm_workdir):
    """Fresh TIP3+MEOH system; returns a callable giving CHARMM's VDW energy."""
    import pandas as pd
    import pycharmm
    import pycharmm.coor as coor
    import pycharmm.energy as energy
    import pycharmm.generate as gen
    import pycharmm.psf as psf
    import pycharmm.read as read
    import pycharmm.settings as settings

    from mmml.data.cgenff_dataset import load_reference, reorder_to_cgenff_template
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        CGENFF_PRM, CGENFF_RTF, pycharmm_quiet, reset_block,
    )
    from mmml.analysis.dimer_molecules import make_oriented_scan_geometries

    pycharmm_quiet()
    reset_block()
    settings.set_bomb_level(-2)
    read.rtf(CGENFF_RTF)
    read.prm(CGENFF_PRM)

    dimer = list(make_oriented_scan_geometries("TIP3", "MEOH", [4.0], [0.0]))[0].atoms
    z = np.asarray(dimer.get_atomic_numbers())
    pos = np.asarray(dimer.get_positions())
    ref = load_reference()
    _, pa = reorder_to_cgenff_template(ref, z[:3], pos[:3], "TIP3")
    _, pb = reorder_to_cgenff_template(ref, z[3:], pos[3:], "MEOH")
    coords = np.vstack([pa, pb])

    def vdw_energy():
        psf.delete_atoms()
        for seg, resi in (("A", "TIP3"), ("B", "MEOH")):
            read.sequence_string(resi)
            gen.new_segment(seg)
        coor.set_positions(pd.DataFrame(coords, columns=["x", "y", "z"]))
        pycharmm.lingo.charmm_script("ENER")
        return float(energy.get_vdw())

    return vdw_energy


def _live_types():
    """CGenFF type names present in the parameter files."""
    from mmml.models.mm_lj_scales import cgenff_type_names_from_prm

    return [n for n in cgenff_type_names_from_prm() if n != "DEFAULT"]


def test_charmm_vdw_picks_up_the_deployed_epsilon_scale(tmp_path, charmm_dimer):
    """The headline: CHARMM's own VDW must change by exactly the scale factor."""
    from mmml.interfaces.pycharmmInterface.mlpot.scaled_cgenff_prm import (
        deploy_scaled_lj_into_charmm,
    )

    base = charmm_dimer()
    assert abs(base) > 1e-6, "degenerate fixture: base VDW is ~0, nothing to scale"

    deploy_scaled_lj_into_charmm(
        _sidecar(tmp_path, _live_types(), eps_factor=2.0),
        out_dir=tmp_path / "scaled",
        verbose=False,
    )
    scaled = charmm_dimer()

    assert scaled == pytest.approx(2.0 * base, rel=1e-6), (
        f"CHARMM VDW {scaled} is not 2x the base {base}; the deployed prm "
        "either did not reach CHARMM or did not carry the scale"
    )


def test_deploying_twice_does_not_double_apply(tmp_path, charmm_dimer):
    """The bug this mechanism exists to prevent.

    Re-deploying must re-derive from the pristine parameter files, not scale the
    already-scaled ones. If it compounded, the second call would give 4x.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.scaled_cgenff_prm import (
        deploy_scaled_lj_into_charmm,
    )

    base = charmm_dimer()
    sidecar = _sidecar(tmp_path, _live_types(), eps_factor=2.0)

    deploy_scaled_lj_into_charmm(sidecar, out_dir=tmp_path / "s1", verbose=False)
    once = charmm_dimer()
    deploy_scaled_lj_into_charmm(sidecar, out_dir=tmp_path / "s2", verbose=False)
    twice = charmm_dimer()

    assert once == pytest.approx(2.0 * base, rel=1e-6)
    assert twice == pytest.approx(once, rel=1e-9), (
        f"double application: {twice} vs {once} (would be "
        f"{4.0 * base} if scales compounded)"
    )


def test_deploy_order_relative_to_psf_build_does_not_matter(tmp_path, charmm_dimer):
    """Parameters are read into CHARMM globally; the PSF is rebuilt per call.

    Deploying before or after a PSF build must give the same energy, or the
    wiring becomes order-sensitive in a way callers cannot reason about.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.scaled_cgenff_prm import (
        deploy_scaled_lj_into_charmm,
    )

    sidecar = _sidecar(tmp_path, _live_types(), eps_factor=1.5)

    # Deploy first, then build+evaluate.
    deploy_scaled_lj_into_charmm(sidecar, out_dir=tmp_path / "a", verbose=False)
    before = charmm_dimer()

    # Build+evaluate once more (PSF rebuilt), then re-deploy and evaluate.
    _ = charmm_dimer()
    deploy_scaled_lj_into_charmm(sidecar, out_dir=tmp_path / "b", verbose=False)
    after = charmm_dimer()

    assert before == pytest.approx(after, rel=1e-9)


def test_sigma_scale_changes_the_energy(tmp_path, charmm_dimer):
    """Sigma has no linear identity, so assert effect rather than a factor."""
    from mmml.interfaces.pycharmmInterface.mlpot.scaled_cgenff_prm import (
        deploy_scaled_lj_into_charmm,
    )

    base = charmm_dimer()
    deploy_scaled_lj_into_charmm(
        _sidecar(tmp_path, _live_types(), sig_factor=1.1),
        out_dir=tmp_path / "scaled",
        verbose=False,
    )
    scaled = charmm_dimer()

    assert scaled != pytest.approx(base, rel=1e-6), "sigma scale had no effect"


def test_unit_scales_leave_charmm_vdw_untouched(tmp_path, charmm_dimer):
    """A sidecar of all-1.0 must be a genuine no-op end to end."""
    from mmml.interfaces.pycharmmInterface.mlpot.scaled_cgenff_prm import (
        deploy_scaled_lj_into_charmm,
    )

    base = charmm_dimer()
    deploy_scaled_lj_into_charmm(
        _sidecar(tmp_path, _live_types()),
        out_dir=tmp_path / "scaled",
        verbose=False,
    )
    assert charmm_dimer() == pytest.approx(base, rel=1e-9)
