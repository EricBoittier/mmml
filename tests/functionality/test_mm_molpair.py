"""MolPairMM (molecule-pair MM tail) vs a brute-force all-atom-pair reference.

Reference = the jax_mic switched tail written out directly: every inter-molecular
atom pair (exact MIC), soft core, LJ + Coulomb scaled by lambda_a lambda_b and by
the complementary COM handoff of the dimer, plus the short-range wall.  Mixed
box (3-atom and 6-atom molecules) so the class-pair buckets are exercised.
"""
import itertools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from mmml.interfaces.pycharmmInterface import mm_molpair as M  # noqa: E402

GAMMA_ON, GAMMA_OFF = None, None


def _sharp_np(r, x0, x1, gamma):
    s = np.clip((r - x0) / max(x1 - x0, 1e-6), 0.0, 1.0) ** gamma
    return s**3 * (s * (6 * s - 15) + 10)


def _system(seed=0, n_w=24, n_m=16, L=15.0):
    rng = np.random.default_rng(seed)
    # grid-ish centres so molecules do not overlap
    n = n_w + n_m
    g = int(np.ceil(n ** (1 / 3)))
    cells = np.array(list(itertools.product(range(g), repeat=3)))[:n] * (L / g)
    cells = cells + rng.uniform(0.5, 1.0, size=cells.shape)
    pos, off, Z = [], [0], []
    order = rng.permutation(n)
    for k in order:
        c = cells[k]
        if k < n_w:
            local = np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
            Z += [8, 1, 1]
        else:
            local = np.array([[0, 0, 0], [1.43, 0, 0], [-0.3, 0.9, 0.0], [1.8, 1.0, 0], [1.8, -0.5, 0.9], [1.8, -0.5, -0.9]])
            Z += [8, 1, 6, 1, 1, 1]
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        pos.append(c + local @ R.T)
        off.append(off[-1] + len(local))
    pos = np.concatenate(pos)  # whole molecules (may stick out of the box; MIC)
    N = len(pos)
    spec = dict(
        q=rng.normal(scale=0.4, size=N),
        rmin_half=rng.uniform(0.2, 2.0, size=N),
        eps=-rng.uniform(0.0, 0.2, size=N),
        lambda_monomer=np.ones(n),
        mm_switch_on=3.5,
        mm_switch_width=3.0,
        ml_switch_width=1.5,
        complementary_handoff=True,
        mm_r_min=None,
        hybrid_hamiltonian="handoff",
        use_jax_pme=False,
    )
    return pos, np.array(off), np.array(Z), spec, L


def _reference(pos, off, spec, L, wall=True):
    from mmml.interfaces.pycharmmInterface.cutoffs import GAMMA_OFF, GAMMA_ON
    from mmml.models.short_range_wall import DEFAULT_WALL_K_EV_A2 as K, DEFAULT_WALL_R_ON_A as RON

    nm = len(off) - 1
    whole = pos.copy()
    for m in range(nm):
        x = whole[off[m]:off[m + 1]]
        whole[off[m]:off[m + 1]] = x - L * np.round((x - x[0]) / L)
    com = np.array([whole[off[m]:off[m + 1]].mean(0) for m in range(nm)])
    on, w, mlw = spec["mm_switch_on"], spec["mm_switch_width"], spec["ml_switch_width"]
    e_mm = 0.0
    e_w = 0.0
    q, rm, ep, lam = spec["q"], spec["rmin_half"], spec["eps"], spec["lambda_monomer"]
    for a in range(nm):
        for b in range(a + 1, nm):
            dc = com[b] - com[a]
            dc -= L * np.round(dc / L)
            R = np.linalg.norm(dc)
            s = _sharp_np(R, on - mlw, on, GAMMA_ON) * (1 - _sharp_np(R, on, on + w, GAMMA_OFF))
            ia = np.arange(off[a], off[a + 1])
            ib = np.arange(off[b], off[b + 1])
            d = whole[ib][None] - whole[ia][:, None]
            d -= L * np.round(d / L)
            r = np.linalg.norm(d, axis=-1)
            if wall:
                rr = np.maximum(r, 1e-6)
                e_w += np.sum(np.where(rr < RON, K * (RON - rr) ** 3 / rr, 0.0))
            rs = np.logaddexp(20 * (r - 0.6), 0) / 20 + 0.6
            rmin = rm[ia][:, None] + rm[ib][None]
            eij = np.sqrt(ep[ia][:, None] * ep[ib][None])
            r6 = (rmin / rs) ** 6
            e = eij * (r6**2 - 2 * r6) + M.COULOMB_K * q[ia][:, None] * q[ib][None] / rs
            e_mm += s * lam[a] * lam[b] * e.sum()
    return e_mm * M.KCALMOL2EV + e_w


@pytest.mark.parametrize("seed", [0, 1])
def test_energy_forces_match_bruteforce(seed):
    pos, off, Z, spec, L = _system(seed)
    topo = M.build_topology(off, Z)
    assert topo.buckets == [(0, 0), (0, 1), (1, 1)]
    mp = M.MolPairMM(spec, topo, skin_A=1.0, ml_active_radius_A=3.5, dtype=jnp.float64)
    box = jnp.array([L, L, L])
    p = jnp.asarray(pos)
    mp.size_caps(p, box)
    nl = mp.build(p, box)
    assert not bool(nl.overflow)
    e, f = jax.jit(lambda x: mp.energy_and_forces(x, box, nl))(p)
    e_ref = _reference(pos, off, spec, L)
    assert abs(float(e) - e_ref) < 1e-8 * max(1.0, abs(e_ref))
    # forces vs central finite differences of the reference on a few atoms
    for k in [0, 7, len(pos) - 1]:
        for c in range(3):
            h = 1e-5
            pp, pm = pos.copy(), pos.copy()
            pp[k, c] += h
            pm[k, c] -= h
            fd = -(_reference(pp, off, spec, L) - _reference(pm, off, spec, L)) / (2 * h)
            assert abs(float(f[k, c]) - fd) < 1e-5 * max(1.0, abs(fd))


def test_verlet_rebuild_and_candidates():
    pos, off, Z, spec, L = _system(2)
    topo = M.build_topology(off, Z)
    mp = M.MolPairMM(spec, topo, skin_A=1.0, ml_active_radius_A=3.5, dtype=jnp.float64)
    box = jnp.array([L, L, L])
    p = jnp.asarray(pos)
    mp.size_caps(p, box, margin=2.0)
    nl = mp.build(p, box)
    # small move: no rebuild, energy still exact
    rng = np.random.default_rng(3)
    p2 = p + 0.05 * rng.normal(size=pos.shape)
    nl2 = mp.maybe_update(p2, box, nl)
    assert int(nl2.n_builds) == int(nl.n_builds)
    e2 = float(mp.energy(p2, box, nl2))
    assert abs(e2 - _reference(np.asarray(p2), off, spec, L)) < 1e-8 * max(1, abs(e2))
    # large move of one molecule: rebuild
    p3 = p.at[off[0]:off[1]].add(0.8)
    nl3 = mp.maybe_update(p3, box, nl)
    assert int(nl3.n_builds) == int(nl.n_builds) + 1
    # ML candidates: sorted global ids of all pairs with centroid < r_ml_list
    nm = len(off) - 1
    com = np.array([pos[off[m]:off[m + 1]].mean(0) for m in range(nm)])  # molecules whole here? use MIC-whole
    cand = np.asarray(nl.ml_cand)
    cand = cand[cand < topo.n_dimers]
    assert np.all(np.diff(cand) > 0)
    ids = {i * (2 * nm - i - 1) // 2 + (j - i - 1) for i in range(nm) for j in range(i + 1, nm)}
    assert set(cand.tolist()) <= ids
    # buckets: every active dimer exactly once, class-ordered
    bk, ovf = M.ml_dimer_buckets(mp, nl, p, box, [mp.caps.ml] * 3)
    assert not bool(ovf)
    seen = []
    for (ca, cb), v in bk.items():
        va = np.asarray(v["valid"])
        a = np.asarray(v["mol_a"])[va]
        b = np.asarray(v["mol_b"])[va]
        assert np.all(topo.mol_class[a] == ca) and np.all(topo.mol_class[b] == cb)
        seen += np.asarray(v["dimer_id"])[va].tolist()
    assert len(seen) == len(set(seen))
    assert sum(int(v["n"]) for v in bk.values()) == len(seen)
