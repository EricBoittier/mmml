"""Training-side CGenFF LJ must reproduce the MD calculator's formula exactly.

The parity test (test_cgenff_lj_parity) pins the *parameters*; this pins the
*formula*: combining rules, the epsilon sign cancellation, and the sigma <->
Rmin/2 conversion, by replicating mm_energy_forces' math independently.
"""

from __future__ import annotations

import numpy as np
import pytest


def _md_reference_lj(r, rmin_half_a, rmin_half_b, eps_a_prm, eps_b_prm):
    """Replicates mm_energy_forces verbatim.

        at_ep   = -1 * abs(atc_epsilons)      (per-atom, forced negative)
        pair_rm = rm_a + rm_b
        pair_ep = (ep_a * ep_b) ** 0.5
        E       = ep * (r6**2 - 2*r6),  r6 = (sig/r)**6
    """
    ep_a = -1.0 * abs(eps_a_prm)
    ep_b = -1.0 * abs(eps_b_prm)
    pair_rm = rmin_half_a + rmin_half_b
    pair_ep = (ep_a * ep_b) ** 0.5
    r6 = (pair_rm / max(r, 1e-10)) ** 6
    return pair_ep * (r6**2 - 2 * r6)


def test_pair_lj_matches_md_formula_for_real_types():
    """Dataset (sigma, eps>=0) -> same energy as MD (Rmin/2, eps<=0)."""
    from scripts.prepare_ml_mm_dataset import DEF_PRM_PATH, load_cgenff_nonbonded_table
    from mmml.models.cgenff_mm import cgenff_pair_lj, sigma_to_rmin_half

    nb_map, sigmas, epsilons = load_cgenff_nonbonded_table(__import__("pathlib").Path(DEF_PRM_PATH))

    # CHARMM PRM values (epsilon negative, Rmin/2) for the DCM/ACO types.
    md_prm = {
        "CG331": (-0.0780, 2.0500),
        "CG321": (-0.0560, 2.0100),
        "CG2O5": (-0.0900, 2.0000),
        "OG2D3": (-0.0500, 1.7000),
        "CLGA1": (-0.3430, 1.9100),
        "HGA2": (-0.0350, 1.3400),
        "HGA3": (-0.0240, 1.3400),
    }
    names = sorted(md_prm)
    for a in names:
        for b in names:
            for r in (2.5, 3.0, 3.6, 5.0, 8.0):
                ia, ib = nb_map[a], nb_map[b]
                mine = float(
                    cgenff_pair_lj(
                        np.float64(r),
                        sigma_to_rmin_half(np.float64(sigmas[ia]))
                        + sigma_to_rmin_half(np.float64(sigmas[ib])),
                        np.sqrt(np.float64(epsilons[ia]) * np.float64(epsilons[ib])),
                    )
                )
                ref = _md_reference_lj(r, md_prm[a][1], md_prm[b][1], md_prm[a][0], md_prm[b][0])
                assert mine == pytest.approx(ref, rel=1e-9, abs=1e-12), (a, b, r)


def test_epsilon_sign_cancels_in_geometric_mean():
    """sqrt((-a)(-b)) == sqrt((+a)(+b)) -- why the dataset's +eps is safe."""
    a, b = 0.078, 0.056
    assert np.sqrt((-a) * (-b)) == pytest.approx(np.sqrt(a * b))


def test_well_minimum_is_negative_at_pair_rmin():
    """Physical sanity: E(Rmin) = -eps (a well, not a barrier)."""
    from mmml.models.cgenff_mm import cgenff_pair_lj

    eps, rmin = 0.0661, 4.06
    assert float(cgenff_pair_lj(np.float64(rmin), np.float64(rmin), np.float64(eps))) == pytest.approx(-eps)
    # repulsive inside, attractive outside, decaying to 0
    assert float(cgenff_pair_lj(np.float64(rmin * 0.7), np.float64(rmin), np.float64(eps))) > 0.0
    assert float(cgenff_pair_lj(np.float64(rmin * 1.3), np.float64(rmin), np.float64(eps))) < 0.0
    # r^-6 tail: ~1e-8 at 50 A for Rmin~4 A, so decayed but not denormal
    far = float(cgenff_pair_lj(np.float64(50.0), np.float64(rmin), np.float64(eps)))
    assert far == pytest.approx(0.0, abs=1e-6) and far < 0.0


def test_sigma_rmin_roundtrip():
    from mmml.models.cgenff_mm import RMIN_HALF_TO_SIGMA, sigma_to_rmin_half

    # HGA3: Rmin/2 = 1.34 -> sigma 2.3876 (as the parity test asserts)
    assert float(sigma_to_rmin_half(np.float64(1.34 * RMIN_HALF_TO_SIGMA))) == pytest.approx(1.34)


def test_lj_energy_padding_and_intermolecular_mask():
    """Padding (-1) contributes nothing; intra-monomer pairs are excluded."""
    from mmml.models.cgenff_mm import cgenff_lj_energy

    sig = np.array([3.6, 2.4], dtype=np.float64)
    eps = np.array([0.078, 0.024], dtype=np.float64)
    # 2 atoms in monomer 0, 2 in monomer 1, 2 padded
    pos = np.array(
        [[0.0, 0, 0], [1.0, 0, 0], [4.0, 0, 0], [5.0, 0, 0], [0, 0, 0], [0, 0, 0]],
        dtype=np.float64,
    )
    tidx = np.array([0, 1, 0, 1, -1, -1])
    mid = np.array([0, 0, 1, 1, -1, -1])

    inter = float(cgenff_lj_energy(pos, tidx, mid, sig, eps, intermolecular_only=True))
    allp = float(cgenff_lj_energy(pos, tidx, mid, sig, eps, intermolecular_only=False))
    assert inter != 0.0
    assert allp != inter          # intra pairs add something
    assert np.isfinite(inter) and np.isfinite(allp)

    # padding must not contribute: same answer with more padding
    pos2 = np.concatenate([pos, np.zeros((4, 3))])
    t2 = np.concatenate([tidx, -np.ones(4, dtype=int)])
    m2 = np.concatenate([mid, -np.ones(4, dtype=int)])
    assert float(cgenff_lj_energy(pos2, t2, m2, sig, eps)) == pytest.approx(inter)


def test_lj_energy_is_vmappable_and_differentiable():
    import jax
    import jax.numpy as jnp
    from mmml.models.cgenff_mm import cgenff_lj_energy

    sig = jnp.array([3.6, 2.4]); eps = jnp.array([0.078, 0.024])
    tidx = jnp.array([0, 1, 0, 1]); mid = jnp.array([0, 0, 1, 1])
    f = lambda p: cgenff_lj_energy(p, tidx, mid, sig, eps)

    pos = jnp.array([[0.0, 0, 0], [1.0, 0, 0], [4.0, 0, 0], [5.0, 0, 0]])
    assert jnp.isfinite(f(pos))
    g = jax.grad(f)(pos)
    assert g.shape == pos.shape and bool(jnp.all(jnp.isfinite(g)))
    batch = jnp.stack([pos, pos + 0.1])
    assert jax.vmap(f)(batch).shape == (2,)


# --------------------------------------------------------------------------
# TOTAL-energy gate: switched LJ + switched electrostatics vs the MD math.
# --------------------------------------------------------------------------

def _md_reference_total(pos, tidx, mid, q, sig, eps, *, on, width, mlw):
    """Independent replication of mm_energy_forces' switched MM total.

        at_ep    = -abs(eps);  pair_ep = sqrt(ep_a*ep_b);  pair_rm = rm_a+rm_b
        E_pair   = ep*[(rm/r)^12 - 2(rm/r)^6] + 332.063711*qq/r
        handoff  = sharpstep(r_com, on-mlw, on, gamma=GAMMA_ON)
        mm_taper = 1 - sharpstep(r_com, on, on+width, gamma=GAMMA_OFF)
        E        = handoff*mm_taper * sum_inter(E_pair)
    """
    from mmml.interfaces.pycharmmInterface.calculator_utils import _sharpstep
    from mmml.interfaces.pycharmmInterface.cutoffs import GAMMA_OFF, GAMMA_ON
    from mmml.models.cgenff_mm import RMIN_HALF_TO_SIGMA

    n = len(tidx)
    rm = np.array([sig[t] / RMIN_HALF_TO_SIGMA if t >= 0 else 0.0 for t in tidx])
    ep = np.array([-abs(eps[t]) if t >= 0 else 0.0 for t in tidx])  # MD forces negative
    tot = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if tidx[i] < 0 or tidx[j] < 0 or mid[i] == mid[j]:
                continue
            r = float(np.linalg.norm(pos[i] - pos[j]))
            pair_rm = rm[i] + rm[j]
            pair_ep = (ep[i] * ep[j]) ** 0.5          # two negatives -> positive
            r6 = (pair_rm / max(r, 1e-10)) ** 6
            tot += pair_ep * (r6**2 - 2 * r6)
            tot += 3.32063711e2 * q[i] * q[j] / max(r, 1e-10)
    c0 = pos[np.array(mid) == 0].mean(axis=0)
    c1 = pos[np.array(mid) == 1].mean(axis=0)
    r_com = float(np.linalg.norm(c1 - c0))
    handoff = float(_sharpstep(np.float64(r_com), on - mlw, on, gamma=GAMMA_ON))
    taper = 1.0 - float(_sharpstep(np.float64(r_com), on, on + width, gamma=GAMMA_OFF))
    return handoff * taper * tot


def _dimer(sep):
    """DCM-like 2+2 dimer separated along x by `sep`."""
    pos = np.array(
        [[0.0, 0, 0], [1.0, 0.2, 0], [sep, 0, 0], [sep + 1.0, 0.2, 0], [0, 0, 0]],
        dtype=np.float64,
    )
    tidx = np.array([0, 1, 0, 1, -1])
    mid = np.array([0, 0, 1, 1, -1])
    q = np.array([-0.3, 0.15, -0.3, 0.15, 0.0])
    return pos, tidx, mid, q


def test_total_mm_energy_matches_md_across_the_whole_switching_range():
    """The gate: switched LJ + electrostatics total == MD math, at every regime."""
    from mmml.models.cgenff_mm import cgenff_mm_energy

    sig = np.array([3.6527, 2.3876]); eps = np.array([0.0780, 0.0240])
    on, width, mlw = 8.0, 5.0, 1.5
    # spans ML-only (<6.5), handoff (6.5-8), MM tail (8-13), and beyond (>13)
    for sep in (4.0, 5.0, 6.4, 7.0, 7.9, 8.5, 10.0, 12.9, 14.0, 20.0):
        pos, tidx, mid, q = _dimer(sep)
        mine = float(
            cgenff_mm_energy(pos, tidx, mid, q, sig, eps,
                             mm_switch_on=on, mm_switch_width=width, ml_switch_width=mlw)
        )
        ref = _md_reference_total(pos, tidx, mid, q, sig, eps, on=on, width=width, mlw=mlw)
        assert mine == pytest.approx(ref, rel=1e-8, abs=1e-10), f"sep={sep}"


def test_mm_is_off_inside_ml_region_and_beyond_the_tail():
    """MM must vanish where ML is fully on (<6.5) and past the tail (>=13)."""
    from mmml.models.cgenff_mm import cgenff_mm_energy

    sig = np.array([3.6527, 2.3876]); eps = np.array([0.0780, 0.0240])
    kw = dict(mm_switch_on=8.0, mm_switch_width=5.0, ml_switch_width=1.5)
    for sep in (3.0, 5.0, 6.4):
        pos, tidx, mid, q = _dimer(sep)
        assert float(cgenff_mm_energy(pos, tidx, mid, q, sig, eps, **kw)) == pytest.approx(0.0, abs=1e-12)
    for sep in (13.5, 20.0):
        pos, tidx, mid, q = _dimer(sep)
        assert float(cgenff_mm_energy(pos, tidx, mid, q, sig, eps, **kw)) == pytest.approx(0.0, abs=1e-12)
    # ...and non-zero in the handoff/tail
    pos, tidx, mid, q = _dimer(9.0)
    assert abs(float(cgenff_mm_energy(pos, tidx, mid, q, sig, eps, **kw))) > 0.0


def test_mm_scale_is_complement_of_ml_scale_at_handoff():
    """'Complementary handoff': MM ramps up exactly as ML ramps down."""
    from mmml.interfaces.pycharmmInterface.calculator_utils import ml_switch_scale, mm_switch_scale

    for r in (6.5, 6.9, 7.3, 7.7, 8.0):
        ml = float(ml_switch_scale(np.float64(r), mm_switch_on=8.0, ml_switch_width=1.5))
        mm = float(mm_switch_scale(np.float64(r), mm_switch_on=8.0, mm_switch_width=5.0,
                                   ml_switch_width=1.5))
        # inside the handoff the MM tail is still 1, so mm == 1 - ml exactly
        assert mm == pytest.approx(1.0 - ml, abs=1e-9), r


def test_monomer_has_no_mm_term():
    """A single monomer has no intermolecular pairs -> exactly 0."""
    from mmml.models.cgenff_mm import cgenff_mm_energy

    sig = np.array([3.6527, 2.3876]); eps = np.array([0.0780, 0.0240])
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 0, 0]])
    tidx = np.array([0, 1, -1]); mid = np.array([0, 0, -1]); q = np.array([-0.3, 0.3, 0.0])
    assert float(cgenff_mm_energy(pos, tidx, mid, q, sig, eps,
                                  mm_switch_on=8.0, mm_switch_width=5.0,
                                  ml_switch_width=1.5)) == 0.0


def test_mm_switch_scale_works_under_jit_with_a_traced_flag():
    """complementary_handoff must survive being traced (hybrid train_step is jitted)."""
    import jax
    import jax.numpy as jnp
    from mmml.interfaces.pycharmmInterface.calculator_utils import mm_switch_scale

    def f(r, flag):
        return mm_switch_scale(r, mm_switch_on=8.0, mm_switch_width=5.0,
                               ml_switch_width=1.5, complementary_handoff=flag)

    jf = jax.jit(f)
    for flag in (True, False):
        out = float(jf(jnp.float64(9.0), jnp.asarray(flag)))
        assert np.isfinite(out)
    # complementary vs legacy genuinely differ in the handoff
    a = float(jf(jnp.float64(7.2), jnp.asarray(True)))
    b = float(jf(jnp.float64(7.2), jnp.asarray(False)))
    assert a > 0.0 and b == pytest.approx(0.0)   # legacy MM starts only at 8.0
