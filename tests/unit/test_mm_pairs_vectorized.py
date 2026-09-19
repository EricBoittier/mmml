"""Vectorized MM pair filter and eterm split match a per-pair reference."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mm_energy_forces import decompose_mlpot_mm_nb_eterms_kcalmol
from mmml.interfaces.pycharmmInterface.nl_reference import apply_mm_pair_filters, mm_pair_filter_mask

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
