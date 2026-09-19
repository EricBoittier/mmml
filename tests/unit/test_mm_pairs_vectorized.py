"""Vectorized MM pair filter and eterm split match a per-pair reference."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mm_energy_forces import decompose_mlpot_mm_nb_eterms_kcalmol
from mmml.interfaces.pycharmmInterface.nl_reference import (
    apply_mm_pair_filters,
    mm_pair_filter_mask,
    unique_mic_orthorhombic,
)

L = 12.0
APM = 3


def _system(seed=0, n_mol=20):
    rng = np.random.default_rng(seed)
    R = rng.uniform(0, L, (n_mol * APM, 3))
    mid = np.repeat(np.arange(n_mol), APM)
    offs = np.arange(0, n_mol * APM + 1, APM)
    i, j = np.triu_indices(len(R), 1)
    return rng, R, mid, offs, i, j


def _mic(d):
    return d - L * np.round(d / L)


def test_filter_mask_matches_bruteforce():
    _, R, mid, offs, i, j = _system()
    com = R.reshape(-1, APM, 3).mean(1)
    ref = [(a, b) for a, b in zip(i, j) if mid[a] != mid[b] and np.linalg.norm(_mic(com[mid[b]] - com[mid[a]])) >= 4.0]
    mask = mm_pair_filter_mask(i, j, monomer_id=mid, positions=R, cell=np.eye(3) * L, mm_r_min=4.0, monomer_offsets=offs)
    assert set(zip(i[mask].tolist(), j[mask].tolist())) == set(ref)
    assert apply_mm_pair_filters(zip(i, j), monomer_id=mid, positions=R, cell=np.eye(3) * L, mm_r_min=4.0, monomer_offsets=offs) == set(ref)


@pytest.mark.parametrize("complementary", [True, False])
def test_eterm_split_matches_per_pair_reference(complementary):
    rng, R, mid, _, i, j = _system(1)
    n = len(R)
    q, rm, ep = rng.normal(0, 0.3, n), rng.uniform(0.2, 2.1, n), -rng.uniform(0.02, 0.2, n)
    pidx = np.c_[i, j]
    mask = np.ones(len(i), dtype=bool)
    kw = dict(charges_e=q, rmins_A=rm, epsilons_kcal=ep, monomer_id=mid, mm_switch_on=6.0,
              mm_switch_width=5.0, ml_switch_width=1.5, complementary_handoff=complementary)
    got = decompose_mlpot_mm_nb_eterms_kcalmol(R, pidx, mask, np.eye(3) * L, **kw)

    def step(r, x0, x1, g):
        s = min(max((r - x0) / (x1 - x0), 0.0), 1.0) ** g
        return s * s * (3 - 2 * s)

    from mmml.interfaces.pycharmmInterface.cutoffs import GAMMA_OFF, GAMMA_ON

    ref = dict(vdw_primary=0.0, vdw_image=0.0, elec_primary=0.0, elec_image=0.0)
    for a, b in zip(i, j):
        if mid[a] == mid[b]:
            continue
        d = R[b] - R[a]
        prim = bool(np.all(np.round(d / L) == 0))
        r = float(np.linalg.norm(_mic(d)))
        if complementary:
            s = step(r, 4.5, 6.0, GAMMA_ON) * (1 - step(r, 6.0, 11.0, GAMMA_OFF))
        else:
            s = step(r, 6.0, 11.0, GAMMA_ON) * (1 - step(r, 11.0, 16.0, GAMMA_OFF))
        sig = (rm[a] + rm[b]) / 2 ** (1 / 6)
        e = np.sqrt(ep[a] * ep[b])
        ref["vdw_primary" if prim else "vdw_image"] += e * ((sig / r) ** 12 - 2 * (sig / r) ** 6) * s
        ref["elec_primary" if prim else "elec_image"] += 332.063711 * q[a] * q[b] / r * s
    for k, v in ref.items():
        assert got[k] == pytest.approx(v, rel=1e-10, abs=1e-10)
    assert got["mm_total"] == pytest.approx(sum(ref.values()), rel=1e-10, abs=1e-10)


def _step(r, x0, x1, g):
    s = min(max((r - x0) / (x1 - x0), 0.0), 1.0) ** g
    return s * s * (3 - 2 * s)


@pytest.mark.parametrize("complementary", [True, False])
def test_eterm_split_com_weighted_matches_per_pair_reference(complementary):
    """Production path: switch at the dimer COM distance, atom-distance fallback for dimer index -1."""
    from mmml.interfaces.pycharmmInterface.cutoffs import GAMMA_OFF, GAMMA_ON

    rng = np.random.default_rng(2)
    n_mol = 16
    # Rigid-ish 3-atom monomers around random centers; wrap atoms into the box so some pairs are images.
    centers = rng.uniform(0, L, (n_mol, 3))
    unwrapped = centers[:, None, :] + rng.normal(0, 0.6, (n_mol, APM, 3))
    R = (unwrapped % L).reshape(-1, 3)
    com = unwrapped.mean(1)
    n = len(R)
    mid = np.repeat(np.arange(n_mol), APM)
    i, j = np.triu_indices(n, 1)

    # Dimer list over monomer pairs (a < b), COM distance by MIC; drop every third dimer (index -1).
    ma, mb = np.triu_indices(n_mol, 1)
    dimer_of = -np.ones((n_mol, n_mol), dtype=np.int64)
    dimer_of[ma, mb] = np.arange(len(ma))
    com_dist = np.linalg.norm(_mic(com[mb] - com[ma]), axis=1)
    lo, hi = np.minimum(mid[i], mid[j]), np.maximum(mid[i], mid[j])
    pair_dimer = np.where(mid[i] != mid[j], dimer_of[lo, hi], -1)
    pair_dimer[(pair_dimer >= 0) & (pair_dimer % 3 == 0)] = -1
    assert np.any(pair_dimer >= 0) and np.any((pair_dimer == -1) & (mid[i] != mid[j]))

    q, rm, ep = rng.normal(0, 0.3, n), rng.uniform(0.2, 2.1, n), -rng.uniform(0.02, 0.2, n)
    kw = dict(charges_e=q, rmins_A=rm, epsilons_kcal=ep, monomer_id=mid, mm_switch_on=6.0,
              mm_switch_width=5.0, ml_switch_width=1.5, complementary_handoff=complementary)
    pidx, mask, cell = np.c_[i, j], np.ones(len(i), dtype=bool), np.eye(3) * L
    got = decompose_mlpot_mm_nb_eterms_kcalmol(
        R, pidx, mask, cell, pair_dimer_idx=pair_dimer, com_distances_A=com_dist, **kw
    )

    ref = dict(vdw_primary=0.0, vdw_image=0.0, elec_primary=0.0, elec_image=0.0)
    n_com_switched = 0
    for a, b, di in zip(i, j, pair_dimer):
        if mid[a] == mid[b]:
            continue
        d = R[b] - R[a]
        prim = bool(np.all(np.round(d / L) == 0))
        r = float(np.linalg.norm(_mic(d)))
        rs = float(com_dist[di]) if di >= 0 else r
        if di >= 0 and 4.5 < rs < 16.0:  # COM distance inside a switching window
            n_com_switched += 1
        if complementary:
            s = _step(rs, 4.5, 6.0, GAMMA_ON) * (1 - _step(rs, 6.0, 11.0, GAMMA_OFF))
        else:
            s = _step(rs, 6.0, 11.0, GAMMA_ON) * (1 - _step(rs, 11.0, 16.0, GAMMA_OFF))
        sig = (rm[a] + rm[b]) / 2 ** (1 / 6)
        e = np.sqrt(ep[a] * ep[b])
        ref["vdw_primary" if prim else "vdw_image"] += e * ((sig / r) ** 12 - 2 * (sig / r) ** 6) * s
        ref["elec_primary" if prim else "elec_image"] += 332.063711 * q[a] * q[b] / r * s
    assert n_com_switched > 0
    assert ref["elec_primary"] != 0.0 and ref["elec_image"] != 0.0
    for k, v in ref.items():
        assert got[k] == pytest.approx(v, rel=1e-10, abs=1e-10)
    assert got["mm_total"] == pytest.approx(sum(ref.values()), rel=1e-10, abs=1e-10)

    # The COM path must actually differ from the atom-distance fallback.
    fallback = decompose_mlpot_mm_nb_eterms_kcalmol(R, pidx, mask, cell, **kw)
    assert fallback["mm_total"] != pytest.approx(got["mm_total"], rel=1e-6)


def test_unique_mic_orthorhombic_is_strict_at_two_cutoff():
    cell = np.eye(3) * 12.0
    assert unique_mic_orthorhombic(cell, 5.0)  # 12 > 10
    assert not unique_mic_orthorhombic(cell, 6.0)  # 12 == 12: boundary
    assert not unique_mic_orthorhombic(cell, 7.0)  # 12 < 14
    assert unique_mic_orthorhombic(np.eye(3) * 12.0001, 6.0)


@pytest.mark.parametrize("cutoff", [5.0, 6.0, 7.0])
def test_vesin_mic_pair_arrays_sorted_unique_and_complete(cutoff):
    """5 Å: unique-MIC (L > 2c). 6 Å: L == 2c boundary. 7 Å: two-image box."""
    pytest.importorskip("vesin")
    from mmml.interfaces.pycharmmInterface.nl_reference import vesin_mic_pair_arrays

    _, R, mid, offs, i, j = _system(seed=3)
    d = np.linalg.norm(_mic(R[j] - R[i]), axis=1)
    ref = {(a, b) for a, b, r in zip(i.tolist(), j.tolist(), d) if mid[a] != mid[b] and r < cutoff}
    pi, pj = vesin_mic_pair_arrays(R, np.eye(3) * L, cutoff, mid, monomer_offsets=offs)
    key = pi * (len(R) + 1) + pj
    assert np.all(pi < pj)
    assert np.all(np.diff(key) > 0)  # lexicographically sorted, no duplicates
    assert set(zip(pi.tolist(), pj.tolist())) == ref


def test_vesin_excludes_pair_exactly_at_cutoff_on_L_equals_2c_boundary():
    """L = 2*cutoff: atoms L/2 apart have d == cutoff and must stay out (strict <)."""
    pytest.importorskip("vesin")
    from mmml.interfaces.pycharmmInterface.nl_reference import vesin_mic_pair_arrays

    cutoff = 6.0
    R = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [6.1, 0.0, 0.0],
            [6.2, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    mid = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    offs = np.array([0, 3, 6], dtype=np.int64)
    pi, pj = vesin_mic_pair_arrays(R, np.eye(3) * 12.0, cutoff, mid, monomer_offsets=offs)
    pairs = set(zip(pi.tolist(), pj.tolist()))
    assert (0, 3) not in pairs  # |6.0 - 0.0| == cutoff
    for a, b in pairs:
        d = float(np.linalg.norm(_mic(R[b] - R[a])))
        assert a < b
        assert d < cutoff


def _etoh_like_liquid_frame(n_mol=181, apm=9, L=26.0, seed=19):
    """Deterministic ETOH:181 / 26 Å stand-in (9-atom monomers on a cubic lattice)."""
    rng = np.random.default_rng(seed)
    n_side = int(np.ceil(n_mol ** (1.0 / 3.0)))
    spacing = L / n_side
    coms = np.array(
        [
            [(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing]
            for i in range(n_side)
            for j in range(n_side)
            for k in range(n_side)
        ],
        dtype=np.float64,
    )[:n_mol]
    local = rng.normal(scale=0.35, size=(n_mol, apm, 3))
    R = (coms[:, None, :] + local).reshape(-1, 3)
    mid = np.repeat(np.arange(n_mol), apm)
    offs = np.arange(0, n_mol * apm + 1, apm)
    return R, mid, offs


def _brute_intermonomer_pairs(R, mid, L, cutoff):
    d = R[:, None, :] - R[None, :, :]
    d -= L * np.round(d / L)
    dist = np.linalg.norm(d, axis=2)
    ii, jj = np.triu_indices(len(R), k=1)
    keep = (mid[ii] != mid[jj]) & (dist[ii, jj] < cutoff)
    return set(zip(ii[keep].tolist(), jj[keep].tolist()))


@pytest.mark.parametrize("cutoff", [7.5, 13.0])
def test_vesin_etoh181_26A_rebuild_matches_bruteforce(cutoff):
    """Rebuilt pair set on an ETOH:181 26 Å frame, including the L=2c boundary."""
    pytest.importorskip("vesin")
    from mmml.interfaces.pycharmmInterface.nl_reference import (
        unique_mic_orthorhombic,
        vesin_mic_pair_arrays,
    )

    L_box = 26.0
    R, mid, offs = _etoh_like_liquid_frame()
    # 7.5 Å: unique-MIC (26 > 15). 13 Å: L == 2c, two images possible.
    assert unique_mic_orthorhombic(np.eye(3) * L_box, cutoff) == (L_box > 2.0 * cutoff)
    # Plant one inter-monomer pair exactly at the cutoff along x.
    R = R.copy()
    R[0] = [0.0, 0.5, 0.5]
    R[9] = [cutoff, 0.5, 0.5]
    ref = _brute_intermonomer_pairs(R, mid, L_box, cutoff)
    assert (0, 9) not in ref
    pi, pj = vesin_mic_pair_arrays(R, np.eye(3) * L_box, cutoff, mid, monomer_offsets=offs)
    got = set(zip(pi.tolist(), pj.tolist()))
    assert (0, 9) not in got
    assert np.all(pi < pj)
    assert np.all(np.diff(pi * (len(R) + 1) + pj) > 0)
    assert got == ref
