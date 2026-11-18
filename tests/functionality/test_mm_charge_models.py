"""Alternative MM charges for the acetone liquid: neutrality, exact reduction, float64 finite differences of forces and strain virial.

MM tail only (MolPairMM, CPU, float64); the ML part is untouched by the charges. Checks:
  * every provider is exactly neutral per molecule; the DCMnet projection reproduces the site dipole and is rotation/translation invariant
  * switched off (charge_fn None, static:cgenff, charge_fn returning the CGenFF q) -> identical energy (1e-10 relative)
  * forces and the atomic-strain virial of E(x, h) with q = q(x) agree with central finite differences (<= 1e-6 relative), fixed neighbour list;
    the frozen-charge force (dq/dx dropped) differs, i.e. the dq/dx term is really in the force
  * engine guard and acetone-only guard (fail closed)
"""
import itertools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from mmml.interfaces.pycharmmInterface import mm_charge_models as mcm  # noqa: E402
from mmml.interfaces.pycharmmInterface import mm_molpair as M  # noqa: E402

ACE0 = np.array([  # an acetone geometry (A), order O, C(=O), CH3 C, CH3 C, H6 (MP2-like equilibrium)
    [0.0000, 1.2200, 0.0000], [0.0000, 0.0000, 0.0000], [1.2900, -0.7800, 0.0000], [-1.2900, -0.7800, 0.0000],
    [1.2200, -1.4300, 0.8800], [1.2200, -1.4300, -0.8800], [2.1700, -0.1400, 0.0000],
    [-1.2200, -1.4300, 0.8800], [-1.2200, -1.4300, -0.8800], [-2.1700, -0.1400, 0.0000]])
Z10 = np.array(mcm.ACETONE_Z)


def _rot(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _box(seed=0, n_side=3, L=16.5, jitter=0.04):
    """n_side^3 thermally distorted acetones (whole molecules, may stick out of the cell), random orientations."""
    rng = np.random.default_rng(seed)
    n = n_side ** 3
    cells = np.array(list(itertools.product(range(n_side), repeat=3))) * (L / n_side) + L / n_side / 2
    pos = []
    for c in cells:
        x = ACE0 + rng.normal(scale=jitter, size=ACE0.shape)
        pos.append(c + (x - x.mean(0)) @ _rot(rng).T)
    pos = np.concatenate(pos)
    off = np.arange(n + 1) * 10
    Zs = np.tile(Z10, n)
    rm = np.tile([1.7, 2.0, 2.0, 2.0, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3], n) / 2 + 0.5
    ep = -np.tile([0.12, 0.07, 0.07, 0.07, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], n)
    spec = dict(q=np.tile(mcm.STATIC_SETS["cgenff"], n), rmin_half=rm, eps=ep, lambda_monomer=np.ones(n), mm_switch_on=3.5,
                mm_switch_width=3.0, ml_switch_width=1.5, complementary_handoff=True, mm_r_min=None,
                hybrid_hamiltonian="handoff", use_jax_pme=False)
    return pos, off, Zs, spec, L


def _setup(charge_fn=None, seed=0):
    pos, off, Zs, spec, L = _box(seed)
    topo = M.build_topology(off, Zs)
    mp = M.MolPairMM(spec, topo, skin_A=1.0, ml_active_radius_A=3.5, dtype=jnp.float64, charge_fn=charge_fn, include_wall=False)
    box = jnp.array([L, L, L])
    p = jnp.asarray(pos)
    mp.size_caps(p, box)
    nl = mp.build(p, box)
    assert not bool(nl.overflow)
    return mp, p, box, nl, off, Zs, spec


def _providers():
    out = {"lincd": mcm.parse_spec("lincd")}
    try:
        import e3x  # noqa: F401
        import flax  # noqa: F401
        out["dcm"] = mcm.parse_spec("dcm:" + __import__("os").environ.get("DCM_TEST_SPEC", "dcmnet_acetone_n3_cgenff") + ";dtype=float64")
    except Exception:
        pass
    return out


@pytest.fixture(scope="module", params=sorted(_providers()))
def provider(request):
    _, _, _, _, off, Zs, _ = _setup()
    return request.param, mcm.make_charge_fn(_providers()[request.param], Zs, off)


def test_static_sets_neutral_and_cgenff_reduction():
    for name, q in mcm.STATIC_SETS.items():
        sp = mcm.parse_spec(f"static:{name}")
        qq = mcm.static_charges(sp)
        assert abs(qq.sum()) < 1e-15, name
    assert abs(mcm.static_charges(mcm.parse_spec("static:cgenff*1.0")) - np.array(mcm.STATIC_SETS["cgenff"])).max() < 1e-15
    pos, off, Zs, spec, L = _box()
    q0 = spec["q"].copy()
    assert mcm.apply_static_charge_set(q0, Zs, off, spec=None) is q0                       # switched off: same object
    q1 = mcm.apply_static_charge_set(q0, Zs, off, spec=mcm.parse_spec("static:cgenff"))
    assert np.abs(q1 - q0).max() < 1e-15
    q2 = mcm.apply_static_charge_set(q0, Zs, off, spec=mcm.parse_spec("static:refit_equiv*1.18"))
    assert np.allclose(q2.reshape(-1, 10).sum(1), 0, atol=1e-13) and abs(q2[0] / mcm.static_charges(mcm.parse_spec("static:refit_equiv"))[0] - 1.18) < 1e-12


def test_neutral_per_molecule(provider):
    name, fn = provider
    pos, *_ = _box(seed=3)
    q = np.asarray(fn(jnp.asarray(pos))).reshape(-1, 10)
    assert np.abs(q.sum(1)).max() < 1e-12
    assert q[:, 0].mean() < -0.3 and q[:, 1].mean() > 0.3            # O negative, carbonyl C positive
    assert np.std(q, axis=0).max() < 0.1                              # thermal-size distortions (0.04 A per coordinate) change charges by < 0.1 e


def test_dcm_projection_dipole_and_invariance():
    if "dcm" not in _providers():
        pytest.skip("e3x/flax not available")
    from mmml.interfaces.pycharmmInterface.mm_charge_dcmnet import make_dcm_atom_charge_fn
    fn = make_dcm_atom_charge_fn(__import__("os").environ.get("DCM_TEST_SPEC", "dcmnet_acetone_n3_cgenff"), mcm.ACETONE_Z, dtype="float64", return_sites=True)
    rng = np.random.default_rng(5)
    X = jnp.asarray(np.stack([ACE0 + rng.normal(scale=0.04, size=ACE0.shape) for _ in range(6)]))
    qa, qs, ps = fn(X)
    mu_atoms = np.einsum("ma,mac->mc", np.asarray(qa), np.asarray(X))
    mu_sites = np.einsum("ms,msc->mc", np.asarray(qs), np.asarray(ps))
    assert np.abs(mu_atoms - mu_sites).max() < 1e-12 and np.abs(np.asarray(qa).sum(1)).max() < 1e-12
    Rm = _rot(rng)
    qb, _, _ = fn(X @ jnp.asarray(Rm).T + jnp.array([3.0, -2.0, 7.5]))
    assert np.abs(np.asarray(qa) - np.asarray(qb)).max() < 1e-9      # rotation + translation invariant


def test_switched_off_is_identical():
    mp0, p, box, nl, off, Zs, spec = _setup(None)
    e0 = float(mp0.energy(p, box, nl))
    mp1, *_ = _setup(lambda pos: jnp.asarray(spec["q"], pos.dtype))
    e1 = float(mp1.energy(p, box, nl))
    assert abs(e1 - e0) <= 1e-10 * max(1.0, abs(e0)) and abs(e0) > 1e-3
    mp2, *_ = _setup(mcm.make_charge_fn(mcm.parse_spec("lincd*0.0"), Zs, off))   # all charges zero: no Coulomb
    _, _, _, e_el0 = mp2.energy_terms(p, box, nl)
    assert abs(float(e_el0)) < 1e-14
    _, _, _, e_el = mp0.energy_terms(p, box, nl)
    assert abs(float(e_el)) > 1e-4                                            # CGenFF Coulomb is not zero


def _fd_dir(f, x, d, h):
    return (f(x + h * d) - f(x - h * d)) / (2 * h)


def test_forces_and_strain_virial_finite_difference(provider):
    name, fn = provider
    mp, p, box, nl, off, Zs, spec = _setup(fn, seed=1)
    E = lambda x: mp.energy(x, box, nl)
    g = jax.grad(E)(p)
    rng = np.random.default_rng(11)
    for _ in range(3):
        d = jnp.asarray(rng.normal(size=p.shape)); d = d / jnp.linalg.norm(d)
        ana = float(jnp.sum(g * d))
        fds = [float(_fd_dir(E, p, d, h)) for h in (2e-4, 1e-4)]
        fd = (4 * fds[1] - fds[0]) / 3          # Richardson
        assert abs(ana - fd) <= 1e-6 * max(abs(ana), float(jnp.linalg.norm(g))), (name, ana, fd)
    # atomic affine strain: positions and cell scaled together, neighbour list fixed
    Es = lambda s: mp.energy((1 + s) * p, (1 + s) * box, nl)
    ana = float(jax.grad(Es)(0.0))
    fds = [(float(Es(h)) - float(Es(-h))) / (2 * h) for h in (2e-5, 1e-5)]
    fd = (4 * fds[1] - fds[0]) / 3
    assert abs(ana - fd) <= 1e-6 * max(abs(ana), 1e-3), (name, ana, fd)
    # frozen charges (dq/dx dropped): the force and the strain derivative are different => the dq/dx terms are in the autodiff
    q0 = jax.lax.stop_gradient(fn(p))
    gf = jax.grad(lambda x: mp.energy(x, box, nl, charges=q0))(p)
    assert float(jnp.linalg.norm(g - gf)) > 1e-3 * float(jnp.linalg.norm(g)), name
    anaf = float(jax.grad(lambda s: mp.energy((1 + s) * p, (1 + s) * box, nl, charges=q0))(0.0))
    assert abs(anaf - ana) > 1e-4 * abs(ana), (name, anaf, ana)


def test_guards():
    sp = mcm.parse_spec("dcm:dcmnet_acetone_n3_cgenff")
    with pytest.raises(ValueError):
        mcm.assert_engine_supports(sp, "mlpot")
    mcm.assert_engine_supports(sp, "jaxmd_molpair")
    mcm.assert_engine_supports(mcm.parse_spec("static:refit_equiv"), "mlpot")      # static is fine for every engine
    with pytest.raises(ValueError):
        mcm.parse_spec("bogus:x")
    Zw = np.tile([8, 1, 1], 4)
    with pytest.raises(ValueError):                                                # water is refused
        mcm.make_charge_fn(mcm.parse_spec("lincd"), Zw, np.arange(5) * 3)
    with pytest.raises(ValueError):
        mcm.apply_static_charge_set(np.zeros(12), Zw, np.arange(5) * 3, spec=mcm.parse_spec("static:refit_equiv"))


def test_dcm_compact_graph_equals_padded_graph():
    """One weighted dummy (multiplicity 50 / 49 as MessagePass weights) == the literal 60-atom graph, sites, charges and gradients (float64)."""
    if "dcm" not in _providers():
        pytest.skip("e3x/flax not available")
    from mmml.interfaces.pycharmmInterface.mm_charge_dcmnet import make_dcm_atom_charge_fn
    kw = dict(dtype="float64", return_sites=True)
    fp = make_dcm_atom_charge_fn("dcmnet_acetone_n3_cgenff", mcm.ACETONE_Z, padded=True, **kw)
    fc = make_dcm_atom_charge_fn("dcmnet_acetone_n3_cgenff", mcm.ACETONE_Z, padded=False, **kw)
    rng = np.random.default_rng(2)
    X = jnp.asarray(np.stack([ACE0 + rng.normal(scale=0.05, size=ACE0.shape) for _ in range(4)]))
    a, b = fp(X), fc(X)
    for u, v in zip(a, b):
        assert float(jnp.abs(u - v).max()) < 1e-11
    w = jnp.asarray(rng.normal(size=(4, 10)))
    gp = jax.grad(lambda x: jnp.sum(w * fp(x)[0]))(X); gc = jax.grad(lambda x: jnp.sum(w * fc(x)[0]))(X)
    assert float(jnp.abs(gp - gc).max()) < 1e-10
