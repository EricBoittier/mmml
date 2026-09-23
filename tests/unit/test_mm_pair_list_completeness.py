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
            use_jax_md_neighbor_list=False,
            mm_nl_backend="vesin",
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


def test_rebuild_raises_when_molecule_outgrows_assumed_extent(monkeypatch):
    # Pin the margin: the default grows into free L/2 headroom (resolve_mm_extent_margin_A).
    monkeypatch.setenv("MMML_MM_EXTENT_MARGIN_A", "0.25")
    R = _box(ON + 0.3)
    extent0 = max_monomer_extent_A(R, np.arange(0, R.shape[0] + 1, APM), np.diag([L] * 3))
    assert extent0 == pytest.approx(HALF)
    mm_fn, update = _build(R, mm_extent_margin_A=0.25)

    stretched = R.copy()
    stretched[5, 0] += 0.2  # within the margin: fine
    update(stretched, force_rebuild=True)

    stretched[5, 0] += 0.5  # molecule 1 now well past extent + margin
    with pytest.raises(ValueError, match="Molecule extent"):
        update(stretched, force_rebuild=True)


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
