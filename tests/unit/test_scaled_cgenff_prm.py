"""Baking LJ scales into a CGenFF prm must reproduce the JAX effective tables.

The whole point is exactness: `periodic_external` gets its VDW from CHARMM, so
the scaled prm must give bit-for-bit the same per-type LJ that the JAX path
computes as ``master * scale``. The load-back test below asserts that through
the real production parser rather than trusting the algebra in the module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.scaled_cgenff_prm import (
    scale_nonbonded_block,
    write_scaled_cgenff_prm,
)

PRM = """\
BONDS
CG1 CG2  300.0  1.5

NONBONDED nbxmod  5 atom cdiel fshift vatom vdistance vfswitch -
cutnb 14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5

!a comment inside the block
HGA1     0.0       -0.0450     1.3400 ! alkane, igor, 6/05
OT       0.0       -0.1521     1.7682
CG331    0.0       -0.0780     2.0500   0.0 -0.0100 1.9000

NBFIX
SOD  CLA  -0.0839  3.7310
"""


def _nb_values(text, atom_type):
    """(epsilon, rmin_half) for a type, read back out of a prm string."""
    in_nb = False
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("NONBONDED"):
            in_nb = True
            continue
        if in_nb and s.upper().startswith("NBFIX"):
            in_nb = False
        if not in_nb or not s or s.startswith("!"):
            continue
        parts = s.split("!")[0].split()
        if parts and parts[0] == atom_type:
            return float(parts[2]), float(parts[3])
    raise KeyError(atom_type)


def test_scales_epsilon_and_rmin_half_by_type():
    out, stats = scale_nonbonded_block(PRM, {"HGA1": 1.05}, {"HGA1": 2.0})
    eps, rmin = _nb_values(out, "HGA1")
    assert eps == pytest.approx(-0.0450 * 2.0)
    assert rmin == pytest.approx(1.3400 * 1.05)
    assert stats.scaled == ["HGA1"]


def test_types_absent_from_the_sidecar_are_untouched():
    out, _ = scale_nonbonded_block(PRM, {"HGA1": 1.05}, {"HGA1": 2.0})
    assert _nb_values(out, "OT") == _nb_values(PRM, "OT")


def test_nbfix_and_other_sections_are_untouched():
    out, _ = scale_nonbonded_block(PRM, {"SOD": 0.5, "CG1": 0.5}, {"SOD": 0.5, "CG1": 0.5})
    assert "SOD  CLA  -0.0839  3.7310" in out, "NBFIX must not be rewritten"
    assert "CG1 CG2  300.0  1.5" in out, "BONDS must not be rewritten"


def test_comments_and_header_survive():
    out, _ = scale_nonbonded_block(PRM, {"HGA1": 1.05}, {"HGA1": 2.0})
    assert "! alkane, igor, 6/05" in out
    assert "!a comment inside the block" in out
    assert "cutnb 14.0 ctofnb 12.0" in out


def test_one_four_columns_untouched_by_default():
    out, _ = scale_nonbonded_block(PRM, {"CG331": 1.05}, {"CG331": 2.0})
    line = [l for l in out.splitlines() if l.strip().startswith("CG331")][0]
    nums = line.split("!")[0].split()
    assert float(nums[2]) == pytest.approx(-0.0780 * 2.0)   # primary eps scaled
    assert float(nums[3]) == pytest.approx(2.0500 * 1.05)   # primary rmin scaled
    assert float(nums[5]) == pytest.approx(-0.0100)         # 1-4 eps unchanged
    assert float(nums[6]) == pytest.approx(1.9000)          # 1-4 rmin unchanged


def test_one_four_columns_scaled_when_requested():
    out, _ = scale_nonbonded_block(
        PRM, {"CG331": 1.05}, {"CG331": 2.0}, scale_14=True
    )
    nums = [l for l in out.splitlines() if l.strip().startswith("CG331")][0].split()
    assert float(nums[5]) == pytest.approx(-0.0100 * 2.0)
    assert float(nums[6]) == pytest.approx(1.9000 * 1.05)


def test_unit_scales_are_a_no_op():
    out, stats = scale_nonbonded_block(PRM, {"HGA1": 1.0}, {"HGA1": 1.0})
    assert out == PRM
    assert stats.scaled == []


def test_sidecar_types_missing_from_prm_are_reported():
    _, stats = scale_nonbonded_block(PRM, {"NOPE": 1.1}, {"NOPE": 1.1})
    assert stats.missing_from_prm == ["NOPE"]


# --- the one that actually matters -------------------------------------------

def test_roundtrip_through_production_parser_matches_master_times_scale(tmp_path):
    """Parse the scaled prm with the real CGenFF reader; it must equal master*scale.

    This is the guarantee `periodic_external` depends on: CHARMM reads these
    per-type values and combines them, so they must already equal what the JAX
    path would have produced as ``master * scale``.
    """
    from mmml.data.cgenff_dataset import (
        DEF_PRM_PATH, DEF_RTF_PATH, DEF_EXTRA_TOPPAR, load_reference,
    )

    ref = load_reference(str(DEF_PRM_PATH), str(DEF_RTF_PATH))
    names = [""] * len(ref.nb_map)
    for name, idx in ref.nb_map.items():
        names[int(idx)] = str(name)

    rng = np.random.default_rng(0)
    sig_scale = rng.uniform(0.95, 1.05, size=len(names))
    eps_scale = rng.uniform(0.25, 4.0, size=len(names))
    # DEFAULT is a synthetic sentinel with no prm line, so it can never be
    # scaled -- leave it at 1.0 or the missing-type guard (correctly) fires.
    sig_scale[names.index("DEFAULT")] = 1.0
    eps_scale[names.index("DEFAULT")] = 1.0
    scaled_types = [n for n in names if n != "DEFAULT"]

    sidecar = tmp_path / "hybrid_mm.json"
    sidecar.write_text(json.dumps({
        "learn_mm_lj_scales": True,
        "cgenff_type_names": names,
        "mm_lj_sigma_scale": sig_scale.tolist(),
        "mm_lj_epsilon_scale": eps_scale.tolist(),
        "mm_lj_sigma_scale_bounds": [0.95, 1.05],
        "mm_lj_epsilon_scale_bounds": [0.25, 4.0],
    }))

    out_dir = tmp_path / "scaled"
    results = write_scaled_cgenff_prm(sidecar, out_dir)
    total = sum(len(s.scaled) for s in results.values())
    assert total > 100, f"too few types scaled: {total}"

    # Read back through the SAME merge path (base prm + streams) as production.
    ref2 = load_reference(
        str(out_dir / DEF_PRM_PATH.name), str(DEF_RTF_PATH),
        extra_toppar=tuple(out_dir / Path(q).name for q in DEF_EXTRA_TOPPAR),
    )

    checked = 0
    for name in scaled_types:
        i, j = ref.nb_map[name], ref2.nb_map[name]
        s, e = sig_scale[i], eps_scale[i]
        if ref.sigmas[i] == 0.0 and ref.epsilons[i] == 0.0:
            continue  # LPH / DUM: zero LJ by design
        assert ref2.sigmas[j] == pytest.approx(ref.sigmas[i] * s, rel=1e-4), name
        assert ref2.epsilons[j] == pytest.approx(ref.epsilons[i] * e, rel=1e-4), name
        checked += 1
    assert checked > 100, f"only {checked} types actually verified"


def test_refuses_to_overwrite_without_permission(tmp_path):
    sidecar = tmp_path / "s.json"
    sidecar.write_text(json.dumps({
        "learn_mm_lj_scales": True,
        "cgenff_type_names": ["HGA1"],
        "mm_lj_sigma_scale": [1.0],
        "mm_lj_epsilon_scale": [1.0],
    }))
    out_dir = tmp_path / "scaled"
    out_dir.mkdir()
    (out_dir / "par_all36_cgenff.prm").write_text("existing")
    with pytest.raises(FileExistsError):
        write_scaled_cgenff_prm(sidecar, out_dir)
