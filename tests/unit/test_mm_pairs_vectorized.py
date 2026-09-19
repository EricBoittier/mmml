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


def _charmm_lj(r, rmin, eps):
    return eps * ((rmin / r) ** 12 - 2 * (rmin / r) ** 6)


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

    from mmml.interfaces.pycharmmInterface.cutoffs import GAMMA_OFF, GAMMA_ON

    ref = dict(vdw_primary=0.0, vdw_image=0.0, elec_primary=0.0, elec_image=0.0)
    for a, b in zip(i, j):
        if mid[a] == mid[b]:
            continue
        d = R[b] - R[a]
        prim = bool(np.all(np.round(d / L) == 0))
        r = float(np.linalg.norm(_mic(d)))
        if complementary:
            s = _step(r, 4.5, 6.0, GAMMA_ON) * (1 - _step(r, 6.0, 11.0, GAMMA_OFF))
        else:
            s = _step(r, 6.0, 11.0, GAMMA_ON) * (1 - _step(r, 11.0, 16.0, GAMMA_OFF))
        e = np.sqrt(ep[a] * ep[b])
        ref["vdw_primary" if prim else "vdw_image"] += _charmm_lj(r, rm[a] + rm[b], e) * s
        ref["elec_primary" if prim else "elec_image"] += 332.063711 * q[a] * q[b] / r * s
    for k, v in ref.items():
        assert got[k] == pytest.approx(v, rel=1e-10, abs=1e-10)
    assert got["mm_total"] == pytest.approx(sum(ref.values()), rel=1e-10, abs=1e-10)


def _step(r, x0, x1, g):
    s = min(max((r - x0) / (x1 - x0), 0.0), 1.0) ** g
    return s**3 * (10 - 15 * s + 6 * s * s)  # quintic smootherstep, as the JAX switch


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
        e = np.sqrt(ep[a] * ep[b])
        ref["vdw_primary" if prim else "vdw_image"] += _charmm_lj(r, rm[a] + rm[b], e) * s
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


def test_eterm_split_vdw_minimum_at_charmm_rmin():
    """Issue #218: one CGenFF pair (CG331/OG311) has its VDW minimum -sqrt(eps_i eps_j) at Rmin_i/2 + Rmin_j/2."""
    rmh, eps = np.array([2.05, 1.765]), np.array([-0.078, -0.192])
    r = np.linspace(3.0, 5.0, 2001)
    kw = dict(charges_e=np.zeros(2), rmins_A=rmh, epsilons_kcal=eps, monomer_id=np.array([0, 1]),
              mm_switch_on=6.0, mm_switch_width=5.0, pair_dimer_idx=np.array([0]), com_distances_A=np.array([6.0]))
    vdw = np.array([
        decompose_mlpot_mm_nb_eterms_kcalmol(np.array([[0, 0, 0], [x, 0, 0]]), np.array([[0, 1]]), np.array([True]), None,
                                             **kw)["vdw_primary"] for x in r
    ])
    assert r[np.argmin(vdw)] == pytest.approx(rmh.sum(), abs=1e-3)
    assert vdw.min() == pytest.approx(-np.sqrt(eps[0] * eps[1]), rel=1e-6)
    assert vdw == pytest.approx(_charmm_lj(r, rmh.sum(), np.sqrt(eps[0] * eps[1])), rel=1e-12)


def test_hybrid_jax_split_matches_mm_energy_and_numpy():
    """The routed split is the hybrid's own switched MM, bucketed; numpy fallback agrees on the same inputs."""
    pytest.importorskip("vesin")
    import jax.numpy as jnp

    from tests.unit.test_mm_pair_list_completeness import APM as n_per, L as box, Q, _box, _build

    R = _box(5.3, seed=4)
    R[:, 0] += box - 2.65 - R[:n_per, 0].mean()  # edge dimer (molecules 0, 1) straddles the x face
    R %= box
    mm_fn, update = _build(R)
    pidx, pmask = update(R, force_rebuild=True)
    e_mm, _ = mm_fn(jnp.asarray(R), pidx, pmask)
    got = np.asarray(update.mm_eterm_split(jnp.asarray(R), pidx, pmask, None))
    assert got.sum() == pytest.approx(float(e_mm), rel=1e-9)
    assert got[1] < 0.0 and got[3] != 0.0  # the in-window edge dimer straddles the face -> image buckets

    n = len(R)
    mid = np.repeat(np.arange(n // n_per), n_per)
    com = np.stack([R[mid == m].mean(0) for m in range(n // n_per)])
    pi, pj = np.asarray(pidx).T
    dcom = com[mid[pj]] - com[mid[pi]]
    kw = dict(charges_e=np.tile(Q, n // n_per), rmins_A=np.full(n, 1.6), epsilons_kcal=np.full(n, -0.05),
              monomer_id=mid, mm_switch_on=5.0, mm_switch_width=1.5, ml_switch_width=1.0,
              pair_dimer_idx=np.arange(len(pi)), com_distances_A=np.linalg.norm(dcom - box * np.round(dcom / box), axis=1))
    ref = decompose_mlpot_mm_nb_eterms_kcalmol(R, np.asarray(pidx), np.asarray(pmask) > 0, np.eye(3) * box, **kw)
    keys = ["vdw_primary", "vdw_image", "elec_primary", "elec_image"]
    assert got == pytest.approx([ref[k] for k in keys], rel=1e-6, abs=1e-9)


# --- GPU (Vesin + CuPy) rebuild: identical pair set and order to the CPU path ---


def _gpu_pairlist_or_skip():
    pytest.importorskip("vesin")
    pytest.importorskip("cupy")
    import jax

    from mmml.interfaces.pycharmmInterface import nl_gpu

    try:
        if not jax.devices("gpu"):
            pytest.skip("no JAX GPU device")
    except RuntimeError:
        pytest.skip("no JAX GPU device")
    if not nl_gpu.gpu_nl_path_available("gpu", positions=jax.device_put(np.zeros((1, 3)), jax.devices("gpu")[0])):
        pytest.skip("GPU pair-list path unavailable (CuPy JIT / vesin>=0.6.1)")
    return nl_gpu


def _mixed_size_frame(L_box=26.0, seed=5):
    """ETOH (9) + water (3) + DCM-like (5) monomers: exercises the non-uniform COM path."""
    rng = np.random.default_rng(seed)
    sizes = np.tile([9, 3, 5], 50)
    centers = rng.uniform(0.0, L_box, (len(sizes), 3))
    R = np.concatenate([c + rng.normal(scale=0.5, size=(k, 3)) for c, k in zip(centers, sizes)])
    offs = np.concatenate([[0], np.cumsum(sizes)])
    return R % L_box, np.repeat(np.arange(len(sizes)), sizes), offs


@pytest.mark.parametrize("frame", ["etoh181", "mixed"])
@pytest.mark.parametrize("cutoff", [7.5, 12.47, 13.0])
@pytest.mark.parametrize("mm_r_min", [None, 4.05])
def test_gpu_rebuild_identical_to_cpu(frame, cutoff, mm_r_min):
    """GPU pairs == CPU pairs element-wise (same set, same lexicographic order), host or device input."""
    nl_gpu = _gpu_pairlist_or_skip()
    import jax

    from mmml.interfaces.pycharmmInterface.nl_backend import build_mm_pairs_with_backend

    L_box = 26.0
    R, mid, offs = _etoh_like_liquid_frame() if frame == "etoh181" else _mixed_size_frame()
    if frame == "etoh181":
        R = R.copy()
        R[0] = [0.0, 0.5, 0.5]
        R[9] = [cutoff, 0.5, 0.5]  # exactly at the cutoff: excluded by both
    cell = np.eye(3) * L_box
    ci, cj, cmask, n_cpu, cap, used = build_mm_pairs_with_backend(
        "vesin", positions=R, box=cell, cutoff=cutoff, monomer_offsets=offs, mm_r_min=mm_r_min,
        total_atoms=len(R),
    )
    assert used == "vesin" and n_cpu > 0
    for pos in (R, jax.device_put(R, jax.devices("gpu")[0])):
        idx, mask, label = nl_gpu.rebuild_vesin_pairs_gpu(
            pos, cell, cutoff=cutoff, monomer_offsets=offs, mm_r_min=mm_r_min, max_pairs=cap,
            total_atoms=len(R),
        )
        assert label == "vesin_gpu"
        assert list(idx.devices())[0].platform == "gpu"  # stays on device
        idx, mask = np.asarray(idx), np.asarray(mask)
        assert idx.shape == (cap, 2) and mask.shape == (cap,)
        np.testing.assert_array_equal(mask, cmask)
        np.testing.assert_array_equal(idx[:, 0], ci)
        np.testing.assert_array_equal(idx[:, 1], cj)
    bf = _brute_intermonomer_pairs(R, mid, L_box, cutoff)
    if mm_r_min is None:
        assert set(zip(ci[cmask].tolist(), cj[cmask].tolist())) == bf


def test_gpu_rebuild_truncation_raises():
    nl_gpu = _gpu_pairlist_or_skip()
    from mmml.interfaces.pycharmmInterface.cell_list import PairListTruncationError

    R, _, offs = _etoh_like_liquid_frame()
    with pytest.raises(PairListTruncationError):
        nl_gpu.rebuild_vesin_pairs_gpu(R, np.eye(3) * 26.0, cutoff=7.5, monomer_offsets=offs, max_pairs=16)


def test_mm_nl_device_request_resolution(monkeypatch):
    from mmml.interfaces.pycharmmInterface import nl_gpu

    monkeypatch.delenv("MMML_MM_NL_DEVICE", raising=False)
    assert nl_gpu.resolve_mm_nl_device_request() == "auto"
    monkeypatch.setenv("MMML_MM_NL_DEVICE", "CPU")
    assert nl_gpu.resolve_mm_nl_device_request() == "cpu"
    assert nl_gpu.resolve_mm_nl_device() == "cpu"
    assert not nl_gpu.gpu_nl_path_available()  # never probes CuPy
    assert nl_gpu.resolve_mm_nl_device_request("gpu") == "gpu"
    monkeypatch.setenv("MMML_MM_NL_DEVICE", "tpu")
    with pytest.raises(ValueError):
        nl_gpu.resolve_mm_nl_device_request()


def test_auto_falls_back_to_cpu_without_cupy_or_jax_gpu(monkeypatch):
    import jax

    from mmml.interfaces.pycharmmInterface import nl_gpu

    monkeypatch.delenv("MMML_MM_NL_DEVICE", raising=False)
    monkeypatch.setattr(nl_gpu, "have_cupy", lambda: False)
    assert not nl_gpu.gpu_nl_path_available()
    assert nl_gpu.resolve_mm_nl_device() == "cpu"
    # CuPy present but JAX runs on CPU: pairs must stay on the CPU path.
    monkeypatch.setattr(nl_gpu, "have_cupy", lambda: True)
    monkeypatch.setattr(nl_gpu, "_jax_target_device", lambda positions=None: jax.devices("cpu")[0])
    monkeypatch.setattr(nl_gpu, "cupy_runtime_ok", lambda **kw: pytest.fail("must not probe CuPy"))
    assert not nl_gpu.gpu_nl_path_available()
