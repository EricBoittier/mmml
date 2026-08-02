"""CHARMM-in-the-loop: deployed LJ scales must reach CHARMM's own VDW energy.

`test_scaled_cgenff_prm.py` proves the rewritten prm carries the right numbers.
This proves CHARMM *reads* them, and pins the session contract that makes the
wiring safe.

Everything lives in one test because of a hard CHARMM constraint: a **second**
non-append parameter read into a live session silently zeroes the VDW energy.
Measured on this fixture:

    base 1.11008  ->  2.22016 after the first deploy  ->  0.00000 after a
    second unguarded one

So `deploy_scaled_lj_into_charmm` allows one real deploy per process, and this
test spends that one deploy exercising every assertion that genuinely needs
CHARMM. Split into separate test functions these pass individually and fail as
a file, which is worse than useless in CI.

The epsilon assertion is exact, not approximate. Pair epsilons combine as
``eps_ij = sqrt(eps_i * eps_j)``, so scaling *every* type's epsilon by ``f``
scales every pair epsilon by exactly ``f``; LJ energy is linear in eps_ij and
CHARMM's switching function depends only on r. Hence
``E_vdw(all eps x f) == f * E_vdw(base)`` through the full nonbond machinery.
Sigma has no such identity (it sits inside r^-12/r^-6), so sigma behaviour is
covered by the pure-Python table tests instead.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.conftest import can_import_pycharmm

if not can_import_pycharmm():
    pytest.skip("PyCHARMM is not available", allow_module_level=True)

pytestmark = pytest.mark.pycharmm

EPS_FACTOR = 2.0


def _sidecar(tmp_path, names, *, eps_factor=1.0, name="hybrid_mm.json"):
    p = tmp_path / name
    p.write_text(json.dumps({
        "learn_mm_lj_scales": True,
        "cgenff_type_names": list(names),
        "mm_lj_sigma_scale": [1.0] * len(names),
        "mm_lj_epsilon_scale": [eps_factor] * len(names),
        # Widened: this is a deliberate probe, not a trained sidecar.
        "mm_lj_sigma_scale_bounds": [0.5, 2.0],
        "mm_lj_epsilon_scale_bounds": [0.1, 10.0],
    }))
    return p


def _live_types():
    from mmml.models.mm_lj_scales import cgenff_type_names_from_prm

    return [n for n in cgenff_type_names_from_prm() if n != "DEFAULT"]


def _build_vdw_probe():
    """Return a callable giving CHARMM's VDW energy for a fresh TIP3+MEOH PSF."""
    import pandas as pd
    import pycharmm
    import pycharmm.coor as coor
    import pycharmm.energy as energy
    import pycharmm.generate as gen
    import pycharmm.psf as psf
    import pycharmm.read as read
    import pycharmm.settings as settings

    from mmml.analysis.dimer_molecules import make_oriented_scan_geometries
    from mmml.data.cgenff_dataset import load_reference, reorder_to_cgenff_template
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        CGENFF_PRM, CGENFF_RTF, pycharmm_quiet, reset_block,
    )

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


def test_scaled_lj_reaches_charmm_and_the_session_contract_holds(
    tmp_path, pycharmm_workdir
):
    from mmml.interfaces.pycharmmInterface.mlpot import scaled_cgenff_prm
    from mmml.interfaces.pycharmmInterface.mlpot.scaled_cgenff_prm import (
        deploy_scaled_lj_into_charmm,
    )

    scaled_cgenff_prm.reset_deployed_lj_scales()
    vdw = _build_vdw_probe()
    types = _live_types()

    base = vdw()
    assert abs(base) > 1e-6, "degenerate fixture: base VDW ~0, nothing to scale"

    sidecar = _sidecar(tmp_path, types, eps_factor=EPS_FACTOR)
    deploy_scaled_lj_into_charmm(sidecar, out_dir=tmp_path / "s1", verbose=False)
    once = vdw()

    # 1. CHARMM's own VDW picked up the deployed scale, exactly.
    assert once == pytest.approx(EPS_FACTOR * base, rel=1e-6), (
        f"CHARMM VDW {once} is not {EPS_FACTOR}x the base {base}; the deployed "
        "prm either did not reach CHARMM or did not carry the scale. "
        "(A ratio of exactly 1.0 means the parameter read was append-only, "
        "which does not override existing NONBONDED entries.)"
    )

    # 2. Parameters survive a PSF rebuild.
    assert vdw() == pytest.approx(once, rel=1e-9), "deploy did not survive PSF rebuild"

    # 3. Re-deploying the same sidecar is a guarded no-op, not a second read.
    deploy_scaled_lj_into_charmm(sidecar, out_dir=tmp_path / "s2", verbose=False)
    twice = vdw()
    assert twice == pytest.approx(once, rel=1e-9), (
        f"repeat deploy changed the energy: {twice} vs {once}. 0.0 means the "
        f"second parameter read wiped the VDW; {EPS_FACTOR**2 * base} would "
        "mean the scales compounded."
    )
    assert abs(twice) > 1e-9, "second read zeroed the VDW"

    # 4. A *different* sidecar must refuse rather than silently zero the VDW.
    other = _sidecar(tmp_path, types, eps_factor=3.0, name="other.json")
    with pytest.raises(RuntimeError, match="already deployed"):
        deploy_scaled_lj_into_charmm(other, out_dir=tmp_path / "s3", verbose=False)

    # ...and the refusal left the session intact.
    assert vdw() == pytest.approx(once, rel=1e-9)

    scaled_cgenff_prm.reset_deployed_lj_scales()
