"""CPU unit tests of the polarisable (field-conditioned) DCMnet.  No checkpoints: random init + zero-initialised field kernels."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import e3x
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from karml.models.dcmnet.polarizable import (
    NATOMS,
    FieldMessagePassingModel,
    MessagePassingModel,
    field_inputs,
    merged_charges,
    merged_graph,
    pad_field,
    postprocess,
)

# architecture of the acetone start model (small enough for a CPU test); dc_scale as in the n3 fine-tune
CFG = dict(
    features=16,
    max_degree=2,
    num_iterations=2,
    num_basis_functions=16,
    cutoff=4.0,
    max_atomic_number=17,
    n_dcm=3,
    include_pseudotensors=True,
    dc_scale=0.3,
)
Z10 = np.array([6, 6, 8, 6, 1, 1, 1, 1, 1, 1])


@pytest.fixture(scope="module", autouse=True)
def _float32():
    """The model is float32 (e3x rejects mixed dtypes); importing karml may switch jax to x64."""
    with jax.enable_x64(False):
        yield


@pytest.fixture(scope="module")
def ctx(_float32):
    rng = np.random.default_rng(0)
    R10 = rng.normal(size=(10, 3)) * 1.3
    R10 -= R10.mean(0)
    R10 = R10.astype(np.float32)
    Zp = jnp.zeros(NATOMS, jnp.int32).at[:10].set(jnp.asarray(Z10, jnp.int32))
    Rp = jnp.zeros((NATOMS, 3), jnp.float32).at[:10].set(R10)
    dst, src = e3x.ops.sparse_pairwise_indices(NATOMS)
    fm = FieldMessagePassingModel(**CFG)
    base = MessagePassingModel(**CFG)
    z60, z603 = jnp.zeros(NATOMS), jnp.zeros((NATOMS, 3))
    init = fm.init(
        jax.random.PRNGKey(1), atomic_numbers=Zp, positions=Rp, dst_idx=dst, src_idx=src, field_V=z60, field_E=z603
    )
    p_base = {"params": {k: v for k, v in init["params"].items() if not k.startswith("field_")}}
    # trained-like field kernels
    pr = jax.tree_util.tree_map(lambda x: x, init)
    for i, k in enumerate(("field_s", "field_v")):
        pr["params"][k] = {
            "kernel": jax.random.normal(jax.random.PRNGKey(7 + i), init["params"][k]["kernel"].shape) * 0.03
        }
    q_src = jnp.asarray(rng.normal(size=12) * 0.3, jnp.float32)
    p_src = jnp.asarray(rng.normal(size=(12, 3)) * 2.0 + np.array([5.0, 0, 0]), jnp.float32)
    return dict(
        rng=rng,
        R10=R10,
        Zp=Zp,
        Rp=Rp,
        dst=dst,
        src=src,
        fm=fm,
        base=base,
        init=init,
        pr=pr,
        p_base=p_base,
        q_src=q_src,
        p_src=p_src,
    )


def _apply(c, params, R=None, V=None, E=None):
    return c["fm"].apply(
        params,
        atomic_numbers=c["Zp"],
        positions=c["Rp"] if R is None else R,
        dst_idx=c["dst"],
        src_idx=c["src"],
        field_V=V,
        field_E=E,
    )


def _field(c):
    V, E = field_inputs(jnp.asarray(c["R10"]), c["q_src"], c["p_src"], 10)
    return pad_field(V, E)


def test_param_tree_is_base_plus_field_maps(ctx):
    extra = set(ctx["init"]["params"]) - set(ctx["p_base"]["params"])
    assert extra == {"field_s", "field_v"}
    for k in extra:
        assert float(jnp.abs(ctx["init"]["params"][k]["kernel"]).max()) == 0.0
        assert set(ctx["init"]["params"][k]) == {"kernel"}  # bias-free


def test_zero_field_identity(ctx):
    c = ctx
    mb, db = c["base"].apply(
        c["p_base"], atomic_numbers=c["Zp"], positions=c["Rp"], dst_idx=c["dst"], src_idx=c["src"]
    )
    z60, z603 = jnp.zeros(NATOMS), jnp.zeros((NATOMS, 3))
    for params in (c["init"], c["pr"]):  # zero kernels, and random (trained-like) kernels
        for V, E in ((z60, z603), (None, None)):
            if V is None and params is c["pr"]:
                pass
            m, d = _apply(c, params, V=V, E=E)
            np.testing.assert_allclose(m, mb, atol=1e-6)
            np.testing.assert_allclose(d, db, atol=1e-6)


def test_field_changes_output_and_e_only_changes_output(ctx):
    c = ctx
    Vp, Ep = _field(c)
    m0, d0 = _apply(c, c["pr"], V=jnp.zeros(NATOMS), E=jnp.zeros((NATOMS, 3)))
    m1, _ = _apply(c, c["pr"], V=Vp, E=Ep)
    assert float(jnp.abs(m1 - m0).max()) > 1e-4
    # regression for the silently dropped vector field: E alone must act
    mE, dE = _apply(c, c["pr"], V=jnp.zeros(NATOMS), E=Ep)
    assert float(jnp.abs(mE - m0).max()) > 1e-4 or float(jnp.abs(dE - d0).max()) > 1e-4


def _rot(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def _mirror(rng):
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    return (np.eye(3) - 2 * np.outer(n, n)).astype(np.float32)


@pytest.mark.parametrize("kind", ["rotation", "inversion", "mirror"])
@pytest.mark.parametrize("e_only", [False, True])
def test_equivariance(ctx, kind, e_only):
    c = ctx
    rng = np.random.default_rng(3)
    M = {"rotation": _rot(rng), "inversion": -_rot(rng), "mirror": _mirror(rng)}[kind]
    Vp, Ep = _field(c)
    if e_only:
        Vp = jnp.zeros_like(Vp)
    m1, d1 = _apply(c, c["pr"], V=Vp, E=Ep)
    Rr = jnp.asarray(np.asarray(c["Rp"]) @ M.T)
    Er = jnp.asarray(np.asarray(Ep) @ M.T)
    m2, d2 = _apply(c, c["pr"], R=Rr, V=Vp, E=Er)
    disp1 = np.asarray(d1) - np.asarray(c["Rp"])[:, :, None]
    disp2 = np.asarray(d2) - np.asarray(Rr)[:, :, None]
    np.testing.assert_allclose(m2, m1, atol=5e-5)  # charges invariant
    np.testing.assert_allclose(np.einsum("ij,njk->nik", M, disp1), disp2, atol=5e-5)  # displacements equivariant


def test_field_inputs_translation_invariant(ctx):
    c = ctx
    V, E = field_inputs(jnp.asarray(c["R10"]), c["q_src"], c["p_src"], 10)
    V2, E2 = field_inputs(jnp.asarray(c["R10"] + 3.0), c["q_src"], c["p_src"] + 3.0, 10)
    np.testing.assert_allclose(V, V2, atol=1e-4)
    np.testing.assert_allclose(E, E2, atol=1e-4)


def test_net_charge_zero(ctx):
    c = ctx
    Vp, Ep = _field(c)
    m, d = _apply(c, c["pr"], V=Vp, E=Ep)
    q, _ = postprocess(m, d, 10, CFG["n_dcm"])
    assert abs(float(q.sum())) < 1e-5
    q2, _ = merged_charges(
        c["fm"], c["pr"], Z10, jnp.asarray(c["R10"]), *field_inputs(jnp.asarray(c["R10"]), c["q_src"], c["p_src"], 10)
    )
    assert abs(float(q2.sum())) < 1e-5


def test_merged_graph_equals_padded_model(ctx):
    c = ctx
    V, E = field_inputs(jnp.asarray(c["R10"]), c["q_src"], c["p_src"], 10)
    Vp, Ep = pad_field(V, E)
    m, d = _apply(c, c["pr"], V=Vp, E=Ep)
    q60, p60 = postprocess(m, d, 10, CFG["n_dcm"])
    q60, p60 = np.asarray(q60)[:30], np.asarray(p60)[:30]
    qm, pm = merged_charges(c["fm"], c["pr"], Z10, jnp.asarray(c["R10"]), V, E)
    np.testing.assert_allclose(qm, q60, atol=1e-4)
    np.testing.assert_allclose(pm, p60, atol=1e-4)
    # also without a field
    m0, d0 = _apply(c, c["pr"])
    q60, p60 = postprocess(m0, d0, 10, CFG["n_dcm"])
    qm, pm = merged_charges(c["fm"], c["pr"], Z10, jnp.asarray(c["R10"]))
    np.testing.assert_allclose(qm, np.asarray(q60)[:30], atol=1e-4)
    np.testing.assert_allclose(pm, np.asarray(p60)[:30], atol=1e-4)


def test_merged_graph_shape():
    P, dst, src, w = merged_graph(10)
    assert P == 11 and len(dst) == len(src) == len(w) == 11 * 10 + 1
    assert w[-1] == NATOMS - 10 - 1  # self edge of the merged padding node


def test_merged_charges_inside_jit(ctx):
    c = ctx
    V, E = field_inputs(jnp.asarray(c["R10"]), c["q_src"], c["p_src"], 10)
    f = jax.jit(lambda R, V, E: merged_charges(c["fm"], c["pr"], Z10, R, V, E))
    q, p = f(jnp.asarray(c["R10"]), V, E)
    q_ref, p_ref = merged_charges(c["fm"], c["pr"], Z10, jnp.asarray(c["R10"]), V, E)
    np.testing.assert_allclose(q, q_ref, atol=1e-5)
    np.testing.assert_allclose(p, p_ref, atol=1e-5)
    g = jax.jit(jax.grad(lambda R: jnp.sum(merged_charges(c["fm"], c["pr"], Z10, R, V, E)[1] ** 2)))(
        jnp.asarray(c["R10"])
    )
    assert bool(jnp.all(jnp.isfinite(g)))
