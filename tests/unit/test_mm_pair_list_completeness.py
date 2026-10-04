"""COM-switched MM pair list: completeness, energy/force consistency, radius checks.

The MM weight follows the monomer COM distance, so every atom pair of a dimer
whose COM is inside ``mm_switch_on + mm_switch_width`` must be listed, even when
the atom-atom distance is larger than that (up to ``+ 2 x extent``).
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    DEFAULT_JAX_MD_SKIN_DISTANCE_A,
    build_mm_energy_forces_fn,
    have_vesin,
    max_monomer_extent_A,
)

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(not have_vesin(), reason="vesin not installed")

ON, WIDTH, ML_W = 5.0, 1.5, 1.0
OLD_LIST_RADIUS = ON + WIDTH + DEFAULT_JAX_MD_SKIN_DISTANCE_A  # atom cutoff before the fix
L = 24.0
APM, HALF = 3, 1.2  # linear triatomics, atoms at -HALF, 0, +HALF (extent = HALF)
Q = np.array([0.4, -0.8, 0.4])


def _molecule(center, axis):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return np.asarray(center) + np.outer([-HALF, 0.0, HALF], axis)


def _box(edge_com_distance: float, seed: int = 0) -> np.ndarray:
    """3x3x3 jittered lattice; molecules 0 and 1 form the edge dimer along x.

    Their axes point along x, so the outer atoms are ``edge_com_distance + 2 HALF``
    apart: beyond the old atom-cutoff list radius.
    """
    rng = np.random.default_rng(seed)
    sp = L / 3.0
    centers = [
        np.array([i, j, k]) * sp - L / 2 + sp / 2 + rng.uniform(-0.9, 0.9, 3)
        for i in range(3)
        for j in range(3)
        for k in range(3)
    ]
    centers[1] = centers[0] + np.array([edge_com_distance, 0.0, 0.0])
    mols = [_molecule(c, rng.normal(size=3)) for c in centers]
    mols[0] = _molecule(centers[0], [1.0, 0.0, 0.0])
    mols[1] = _molecule(centers[1], [1.0, 0.0, 0.0])
    return np.concatenate(mols, axis=0)


@contextmanager
def _fake_charmm(n_atoms: int):
    charges = np.tile(Q, n_atoms // APM)
    fake_psf = MagicMock()
    fake_psf.get_charges.return_value = charges
    fake_psf.get_iac.return_value = np.zeros(n_atoms, dtype=np.int32)
    fake_param = MagicMock()
    fake_param.get_atc.return_value = ["CG321"]
    rtf_mock = MagicMock()
    rtf_mock.readlines.return_value = ["ATOM C1 CG321 -0.1\n"]
    prm_mock = MagicMock()
    prm_mock.readlines.return_value = ["CG321 0.0 -0.05 1.6 0.0 -0.01 1.9\n"]  # >4 fields: parsed
    mod = "mmml.interfaces.pycharmmInterface.mm_energy_forces"
    with patch("pycharmm.psf", fake_psf), patch("pycharmm.param", fake_param), patch(
        f"{mod}.open", side_effect=[rtf_mock, prm_mock]
    ), patch(f"{mod}._get_actual_psf_charges", return_value=charges), patch(
        f"{mod}.CGENFF_PRM", "/dev/null"
    ), patch(f"{mod}.CGENFF_RTF", "/dev/null"):
        yield


def _build(R, *, box_L=L, **kw):
    n = R.shape[0]
    n_mono = n // APM
    kw.setdefault("use_jax_md_neighbor_list", False)
    kw.setdefault("mm_nl_backend", "vesin")
    with _fake_charmm(n):
        return build_mm_energy_forces_fn(
            R,
            total_atoms=n,
            n_monomers=n_mono,
            monomer_offsets=np.arange(0, n + 1, APM, dtype=np.int32),
            atoms_per_monomer_list=[APM] * n_mono,
            lambda_monomer=np.ones(n_mono),
            ml_switch_width=ML_W,
            mm_switch_on=ON,
            mm_switch_width=WIDTH,
            pbc_cell=np.diag([box_L] * 3),
            lr_solver="mic",
            defer_xla_gpu_warmup=True,
            **kw,
        )


def _mic(d):
    return d - L * np.round(d / L)


@pytest.mark.parametrize("edge_com", [ON + WIDTH - 0.01, ON + 0.3, 4.4])
def test_every_atom_pair_of_weighted_dimers_is_listed(edge_com):
    R = _box(edge_com)
    _, update = _build(R)
    pair_idx, pair_mask = update(R, force_rebuild=True)
    pi = np.asarray(pair_idx)
    keep = np.asarray(pair_mask) > 0
    listed = {(min(a, b), max(a, b)) for a, b in pi[keep]}

    n_mono = R.shape[0] // APM
    coms = R.reshape(n_mono, APM, 3).mean(axis=1)
    required, beyond_old = 0, 0
    for mi in range(n_mono):
        for mj in range(mi + 1, n_mono):
            if np.linalg.norm(_mic(coms[mj] - coms[mi])) >= ON + WIDTH:
                continue  # MM weight is exactly zero
            for a in range(mi * APM, (mi + 1) * APM):
                for b in range(mj * APM, (mj + 1) * APM):
                    required += 1
                    beyond_old += np.linalg.norm(_mic(R[b] - R[a])) >= OLD_LIST_RADIUS
                    assert (a, b) in listed, (mi, mj, a, b)
    assert required >= APM * APM
    # The edge dimer has atom pairs the old atom-cutoff list could not hold.
    assert beyond_old > 0


def _energy_forces(mm_fn, update, R):
    pair_idx, pair_mask = update(np.asarray(R), force_rebuild=True)
    e, f = mm_fn(jnp.asarray(R), pair_idx, pair_mask)
    return float(e), np.asarray(f)


def test_energy_is_consistent_with_forces_across_old_list_edge():
    """Slide molecule 1 so its outer-atom pairs cross the old atom cutoff.

    The COM stays inside the MM window, so those pairs carry weight. With an
    atom-cutoff list they would pop in/out and the path energy would jump
    relative to the force work; with the COM-complete list it does not.
    """
    R0 = _box(ON + 0.1)
    mm_fn, update = _build(R0)
    idx = np.arange(APM, 2 * APM)  # molecule 1
    direction = np.zeros_like(R0)
    direction[idx, 0] = 1.0

    # Molecule-1 COM runs over ON + 0.1 + s. Pairs at COM + HALF (center/outer)
    # and COM + 2 HALF (outer/outer) cross OLD_LIST_RADIUS inside this window.
    s = np.linspace(-0.9, 0.6, 101)
    crossing = OLD_LIST_RADIUS - (ON + 0.1) - HALF
    crossing_outer = OLD_LIST_RADIUS - (ON + 0.1) - 2 * HALF
    assert s[0] < crossing_outer < crossing < s[-1]

    energies, work_rate = [], []
    for si in s:
        e, f = _energy_forces(mm_fn, update, R0 + si * direction)
        energies.append(e)
        work_rate.append(-np.sum(f * direction))  # dE/ds
    energies = np.asarray(energies)
    work_rate = np.asarray(work_rate)

    # Energy change must equal integrated dE/ds (trapezoid) along the path.
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (work_rate[1:] + work_rate[:-1]) * np.diff(s))])
    scale = max(1.0, float(np.max(np.abs(energies - energies[0]))))
    assert np.max(np.abs((energies - energies[0]) - integ)) < 2e-3 * scale

    # Pointwise central finite difference at the crossing (list rebuilt at each side).
    h = 1e-4
    e_p, _ = _energy_forces(mm_fn, update, R0 + (crossing + h) * direction)
    e_m, _ = _energy_forces(mm_fn, update, R0 + (crossing - h) * direction)
    _, f_c = _energy_forces(mm_fn, update, R0 + crossing * direction)
    fd = (e_p - e_m) / (2 * h)
    assert fd == pytest.approx(-np.sum(f_c * direction), rel=1e-4, abs=1e-6)


def test_radius_reaching_half_box_raises():
    R = _box(ON + 0.3)
    # 5 + 1.5 + 2 x (1.2 + 0.25) + 0.25 = 9.65 A >= 9 A = L/2 for an 18 A box.
    with pytest.raises(ValueError, match="half the box"):
        _build(R, box_L=18.0)


def _stretch(R, mol, by):
    """Stretch linear molecule ``mol`` symmetrically: both bonds grow by ``by``."""
    out = R.copy()
    a = mol * APM
    axis = (R[a + 2] - R[a]) / np.linalg.norm(R[a + 2] - R[a])
    out[a] -= by * axis
    out[a + 2] += by * axis
    return out


def _lattice(box_L, seed=0):
    """2x2x2 molecules with random axes in a ``box_L`` box (no close contacts)."""
    rng = np.random.default_rng(seed)
    sp = box_L / 2.0
    centers = [
        np.array([i, j, k]) * sp + sp / 2 + rng.uniform(-0.5, 0.5, 3)
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
    return np.concatenate([_molecule(c, rng.normal(size=3)) for c in centers], axis=0)


def _mm(mm_fn, pair_idx, pair_mask, R):
    e, f = mm_fn(jnp.asarray(R), pair_idx, pair_mask)
    return float(e), np.asarray(f)


def test_outgrown_extent_refits_list_and_matches_larger_margin(monkeypatch):
    """Extent past the assumed margin, room below L/2: refit + rebuild, same MM as a wider list."""
    monkeypatch.setenv("MMML_MM_EXTENT_MARGIN_A", "0.25")
    R = _box(ON + WIDTH - 0.01)
    mm_fn, update = _build(R)
    old_list = update.get_stats()["radius"]["list_radius_A"]  # 6.5 + 2 x 1.45 + 0.25
    update(R, force_rebuild=True)
    # Edge dimer, both molecules stretched along x: extent 1.6 > 1.2 + 0.25, bonds
    # x1.33 (not broken); its outer atoms are now beyond the old list radius.
    stretched = _stretch(_stretch(R, 0, 0.4), 1, 0.4)
    assert np.linalg.norm(stretched[5] - stretched[0]) > old_list
    pidx, pmask = update(stretched, force_rebuild=True)
    stats = update.get_stats()
    assert stats["list_refits"] == 1
    assert stats["radius"]["assumed_extent_A"] >= 1.6
    assert old_list < stats["radius"]["list_radius_A"] < L / 2
    listed = {tuple(p) for p in np.asarray(pidx)[np.asarray(pmask) > 0]}
    assert (0, 5) in listed
    e, f = _mm(mm_fn, pidx, pmask, stretched)

    monkeypatch.setenv("MMML_MM_EXTENT_MARGIN_A", "1.0")
    mm_ref, update_ref = _build(R)
    e_ref, f_ref = _mm(mm_ref, *update_ref(stretched, force_rebuild=True), stretched)
    assert update_ref.get_stats()["list_refits"] == 0
    assert e == pytest.approx(e_ref, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(f, f_ref, rtol=1e-10, atol=1e-12)


def test_jax_md_backend_refits_outgrown_extent(monkeypatch):
    """``mm_nl_backend=jax_md`` closes over the setup cutoff; a refit reallocates it."""
    pytest.importorskip("jax_md")
    monkeypatch.setenv("MMML_MM_EXTENT_MARGIN_A", "0.25")
    R = _box(ON + WIDTH - 0.01)
    kw = dict(use_jax_md_neighbor_list=True, mm_nl_backend="jax_md")
    mm_fn, update = _build(R, **kw)
    old_list = update.get_stats()["radius"]["list_radius_A"]
    stretched = _stretch(_stretch(R, 0, 0.4), 1, 0.4)
    pidx, pmask = update(stretched)
    stats = update.get_stats()
    assert stats["list_refits"] == 1
    assert stats["radius"]["assumed_extent_A"] >= 1.6
    assert old_list < stats["radius"]["list_radius_A"] < L / 2
    listed = {tuple(int(a) for a in p) for p in np.asarray(pidx)[np.asarray(pmask) > 0]}
    assert (0, 5) in listed
    e, f = _mm(mm_fn, pidx, pmask, stretched)

    monkeypatch.setenv("MMML_MM_EXTENT_MARGIN_A", "1.0")
    mm_ref, update_ref = _build(R, **kw)
    e_ref, f_ref = _mm(mm_ref, *update_ref(stretched), stretched)
    assert update_ref.get_stats()["list_refits"] == 0
    assert e == pytest.approx(e_ref, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(f, f_ref, rtol=1e-10, atol=1e-12)


def test_outgrown_extent_trades_skin_when_half_box_is_tight(monkeypatch):
    """No L/2 headroom left: keep the list radius, shrink the skin; MM unchanged."""
    monkeypatch.setenv("MMML_MM_EXTENT_MARGIN_A", "0.25")
    box_L = 2 * (ON + WIDTH + 2 * 1.5 + 0.201)  # stretched extent 1.5 leaves 0.2 A
    R = _lattice(box_L)
    mm_fn, update = _build(R, box_L=box_L)
    stretched = _stretch(R, 3, 0.3)
    pidx, pmask = update(stretched, force_rebuild=True)
    stats = update.get_stats()
    assert stats["list_refits"] == 1
    assert 0.0 < stats["skin_distance"] < DEFAULT_JAX_MD_SKIN_DISTANCE_A
    assert stats["radius"]["list_radius_A"] < box_L / 2
    e, f = _mm(mm_fn, pidx, pmask, stretched)

    monkeypatch.setenv("MMML_MM_EXTENT_MARGIN_A", "0.0")
    mm_ref, update_ref = _build(stretched, box_L=box_L, jax_md_skin_distance=0.1)
    e_ref, f_ref = _mm(mm_ref, *update_ref(stretched, force_rebuild=True), stretched)
    assert e == pytest.approx(e_ref, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(f, f_ref, rtol=1e-10, atol=1e-12)


def test_outgrown_extent_beyond_half_box_still_raises(monkeypatch):
    monkeypatch.setenv("MMML_MM_EXTENT_MARGIN_A", "0.25")
    box_L = 20.0  # start list 9.65 < 10; extent 1.75 needs 6.5 + 3.5 + skin >= 10
    R = _lattice(box_L)
    _, update = _build(R, box_L=box_L)
    with pytest.raises(ValueError, match="no longer fits the box"):
        update(_stretch(R, 2, 0.55), force_rebuild=True)  # bonds x1.46: not broken


def test_dissociated_molecule_raises(monkeypatch):
    monkeypatch.setenv("MMML_MM_EXTENT_MARGIN_A", "0.25")
    R = _box(ON + 0.3)
    _, update = _build(R)
    stretched = R.copy()
    stretched[5, 0] += 0.2  # within the margin: fine
    update(stretched, force_rebuild=True)
    stretched[5, 0] += 0.5  # outer atom of molecule 1 now 1.9 A from its neighbour (1.2)
    with pytest.raises(ValueError, match="Molecule 1 has dissociated"):
        update(stretched, force_rebuild=True)


def test_refit_mm_pair_list_dcm_32A():
    """26 Sep DCM:308 / 32 A trip: extent 2.376 vs 2.3745 assumed, list 15.999 A."""
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
        MM_REFIT_MIN_SKIN_A,
        refit_mm_pair_list,
    )

    fit = refit_mm_pair_list(extent_A=2.376, com_switch_end_A=11.0, skin_A=0.25, box_half_min_A=16.0)
    assert fit["list_radius_A"] == pytest.approx(15.999)
    assert fit["assumed_extent_A"] == pytest.approx(2.426)
    assert MM_REFIT_MIN_SKIN_A <= fit["skin_A"] < 0.25
    # big box: skin kept, margin capped at 1 A
    fit = refit_mm_pair_list(extent_A=2.376, com_switch_end_A=11.0, skin_A=0.25, box_half_min_A=25.0)
    assert fit["skin_A"] == 0.25 and fit["extent_margin_A"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no longer fits"):
        refit_mm_pair_list(extent_A=2.48, com_switch_end_A=11.0, skin_A=0.25, box_half_min_A=16.0)


def test_refit_keeps_box_headroom_for_variable_cell():
    """NpT: a refit leaves L/2 headroom, so a slightly smaller box needs no second refit."""
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import refit_mm_pair_list

    fixed = refit_mm_pair_list(extent_A=1.806, com_switch_end_A=11.0, skin_A=0.25, box_half_min_A=15.995)
    assert fixed["list_radius_A"] == pytest.approx(15.994)  # all room to the margin
    npt = refit_mm_pair_list(
        extent_A=1.806, com_switch_end_A=11.0, skin_A=0.25, box_half_min_A=15.995,
        box_headroom_fraction=0.5,
    )
    assert npt["skin_A"] == 0.25 and npt["assumed_extent_A"] > 1.806 + 0.25
    assert 15.995 - npt["list_radius_A"] == pytest.approx(0.5 * (fixed["list_radius_A"] - 11.0 - 2 * 1.806 - 0.25), abs=1e-3)


def test_shrinking_box_refits_once_with_headroom(monkeypatch):
    """Box passed in shrinks below the list radius (NpT): one refit, then rebuilds reuse it."""
    monkeypatch.setenv("MMML_MM_EXTENT_MARGIN_A", "0.25")
    box_L = 2 * (ON + WIDTH + 2 * 1.2 + 0.25 + 2 * 0.25) + 0.002  # list = L/2 - 1e-3
    R = _lattice(box_L)
    mm_fn, update = _build(R, box_L=box_L)
    assert update.get_stats()["list_refits"] == 0
    for k in range(1, 6):  # 5 x 0.02 A shrinks (affine), each rebuilt
        s_ = (box_L - 0.02 * k) / box_L
        pidx, pmask = update(R * s_, np.full(3, box_L * s_), force_rebuild=True)
    stats = update.get_stats()
    assert stats["list_refits"] == 1, stats
    assert stats["radius"]["list_radius_A"] < 0.5 * (box_L - 0.1)
    # the refit list holds every pair the COM switch weights: same MM as a fresh build
    Rs, Ls = R * s_, box_L * s_
    e, f = mm_fn(jnp.asarray(Rs), pidx, pmask, box_override=jnp.diag(jnp.full(3, Ls)))
    monkeypatch.setenv("MMML_MM_EXTENT_MARGIN_A", "0.0")
    mm_ref, update_ref = _build(Rs, box_L=Ls)
    e_ref, f_ref = _mm(mm_ref, *update_ref(Rs, force_rebuild=True), Rs)
    assert float(e) == pytest.approx(e_ref, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(np.asarray(f), f_ref, rtol=1e-10, atol=1e-12)


def _spread(spacing, box_L=40.0, seed=0):
    """2x2x2 molecules ``spacing`` apart (random axes) in a ``box_L`` box."""
    rng = np.random.default_rng(seed)
    centers = [
        np.array([i, j, k]) * spacing + 5.0 + rng.uniform(-0.3, 0.3, 3)
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
    return np.concatenate([_molecule(c, rng.normal(size=3)) for c in centers], axis=0)


def test_pair_capacity_growth_keeps_forces(monkeypatch):
    """A rebuild that overflows the pair capacity grows it (new list length), same MM.

    The switching weights read the setup-time pair list, so the first grow
    crashed with ``Incompatible shapes for broadcasting`` (WIP DCM:308 NVT).
    """
    monkeypatch.setenv("MMML_MM_EXTENT_MARGIN_A", "0.25")
    sparse, dense = _spread(20.0), _spread(5.5)  # no MM pairs -> COM 5.5 A dimers
    mm_fn, update = _build(sparse, box_L=40.0, max_pairs=16)
    assert update(sparse, force_rebuild=True)[0].shape[0] == 16
    pidx, pmask = update(dense, force_rebuild=True)
    stats = update.get_stats()
    assert pidx.shape[0] > 16 and stats["capacity_grows"] >= 1, stats
    e, f = _mm(mm_fn, pidx, pmask, dense)
    assert e != 0.0
    mm_ref, update_ref = _build(dense, box_L=40.0)
    e_ref, f_ref = _mm(mm_ref, *update_ref(dense, force_rebuild=True), dense)
    assert e == pytest.approx(e_ref, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(f, f_ref, rtol=1e-10, atol=1e-12)


def test_max_monomer_extent_is_image_invariant():
    R = _box(ON + 0.3)
    offsets = np.arange(0, R.shape[0] + 1, APM)
    cell = np.diag([L] * 3)
    ref = max_monomer_extent_A(R, offsets, cell)
    split = R.copy()
    split[3] += np.array([L, 0.0, -L])  # one atom of molecule 1 in another image
    assert max_monomer_extent_A(split, offsets, cell) == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("edge_com", [ON + WIDTH - 0.01, 4.4])
def test_update_mm_pairs_gpu_rebuild_identical_to_cpu(edge_com, monkeypatch):
    """The closure's GPU rebuild (auto device) gives the CPU pair list bit-for-bit (and the same MM energy/forces)."""
    pytest.importorskip("cupy")
    from mmml.interfaces.pycharmmInterface import nl_gpu

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("no JAX GPU device")
    if not nl_gpu.gpu_nl_path_available("gpu", positions=jax.device_put(jnp.zeros((1, 3)), gpu)):
        pytest.skip("GPU pair-list path unavailable")
    R = _box(edge_com, seed=7)
    out = {}
    for device in ("cpu", "auto"):
        monkeypatch.setenv("MMML_MM_NL_DEVICE", device)
        with jax.default_device(gpu):
            mm_fn, update = _build(R)
            for pos in (R, jax.device_put(jnp.asarray(R), gpu)):  # host (MLpot) and device input
                pidx, pmask = update(pos, force_rebuild=True)
                e, f = mm_fn(jnp.asarray(R), pidx, pmask)
                out.setdefault(device, []).append((np.asarray(pidx), np.asarray(pmask), float(e), np.asarray(f)))
            stats = update.get_stats()
        assert stats["gpu_rebuilds" if device == "auto" else "cpu_rebuilds"] >= 2, stats
    for (ic, mc, ec, fc), (ig, mg, eg, fg) in zip(out["cpu"], out["auto"]):
        assert mc.dtype == mg.dtype
        np.testing.assert_array_equal(mc, mg)
        np.testing.assert_array_equal(ic[mc > 0], ig[mg > 0])
        # Identical pair list + inputs; XLA GPU scatter-add order may flip the last ulp.
        assert eg == pytest.approx(ec, rel=1e-13, abs=1e-13)
        np.testing.assert_allclose(fg, fc, rtol=1e-12, atol=1e-13)


def test_extent_margin_grows_into_half_box_headroom(monkeypatch):
    """The assumed extent uses the room left below L/2 (capped), never less than requested."""
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import resolve_mm_extent_margin_A

    monkeypatch.delenv("MMML_MM_EXTENT_MARGIN_A", raising=False)
    kw = dict(mm_switch_on=6.0, mm_switch_width=5.0, skin_distance=0.25)
    # DCM:308 in 32 A: 16 - 1e-3 - (11.25 + 2 * 1.727) = 1.295 of radius -> 0.6475 of extent
    m = resolve_mm_extent_margin_A(0.25, measured_extent_A=1.727, cell=np.eye(3) * 32.0, **kw)
    assert m == pytest.approx(0.6475, abs=1e-6)
    assert 11.25 + 2 * (1.727 + m) < 16.0
    # no headroom: the requested floor
    assert resolve_mm_extent_margin_A(0.25, measured_extent_A=2.4, cell=np.eye(3) * 32.0, **kw) == 0.25
    # large box: capped
    assert resolve_mm_extent_margin_A(0.25, measured_extent_A=1.0, cell=np.eye(3) * 80.0, **kw) == 1.0
    monkeypatch.setenv("MMML_MM_EXTENT_MARGIN_A", "0.1")
    assert resolve_mm_extent_margin_A(0.25, measured_extent_A=1.0, cell=np.eye(3) * 80.0, **kw) == 0.1
