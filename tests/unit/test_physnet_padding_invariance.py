"""Padding atoms must not influence real atoms.

PhysNet is translation-invariant BY CONSTRUCTION, but the message-passing basis
was built for every pair with no ``batch_mask``.  ``atom_mask`` only zeroes a
padding atom's own atomic energy; it never stopped padding from *sending
messages*.  Padding sits at the origin, so any real atom within ``cutoff`` of
(0,0,0) had corrupted features and the energy moved with the molecule's absolute
position -- 23 eV for a real acetone 0.74 A from the origin, which was 96% of a
whole validation set's energy MSE.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.physnetjax.physnetjax.models.model import PhysNet
from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet

NA, NREAL = 10, 4


def _physnet(**kw):
    # zbl off: isolates the MessagePass padding leak.  ZBL has its own ~0.1 meV
    # float32 translation noise that is unrelated to this bug.
    return PhysNet(
        features=8,
        max_degree=0,
        num_iterations=2,
        num_basis_functions=4,
        cutoff=8.0,
        max_atomic_number=10,
        charges=True,
        max_padded_atoms=NA,
        n_refinement_blocks=1,
        total_charge=0.0,
        zbl=False,
        **kw,
    )


def _spooky(**kw):
    return SpookyPhysNet(
        features=8,
        max_degree=0,
        num_iterations=2,
        num_basis_functions=4,
        cutoff=8.0,
        max_atomic_number=10,
        charges=True,
        max_padded_atoms=NA,
        n_refinement_blocks=1,
        total_charge=0.0,
        zbl=False,
        **kw,
    )


def _inputs(offset, *, spooky: bool = False, pad_pos: float = 0.0):
    """A 4-atom molecule at `offset`; 6 padding slots at ``pad_pos``.

    Intentionally keeps real↔padding edges in ``dst_idx``/``src_idx`` with
    ``batch_mask=0`` — that is exactly the training/MD padding layout that
    used to leak into MessagePass.
    """
    rng = np.random.RandomState(0)
    R = np.zeros((NA, 3), dtype=np.float32)
    R[:NREAL] = rng.uniform(-1.2, 1.2, (NREAL, 3)) + np.asarray(
        offset, dtype=np.float32
    )
    R[NREAL:] = pad_pos
    Z = np.zeros(NA, dtype=np.int32)
    Z[:NREAL] = np.array([6, 1, 1, 8])
    am = (Z > 0).astype(np.float32)
    i = np.arange(NA)
    dst, src = np.meshgrid(i, i, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    edge_keep = dst != src
    batch_mask = (am[dst] > 0) & (am[src] > 0) & edge_keep
    dst, src = dst[edge_keep], src[edge_keep]
    batch_mask = batch_mask[edge_keep]
    out = dict(
        atomic_numbers=jnp.asarray(Z),
        positions=jnp.asarray(R),
        dst_idx=jnp.asarray(dst),
        src_idx=jnp.asarray(src),
        batch_segments=jnp.zeros(NA, dtype=jnp.int32),
        batch_size=1,
        batch_mask=jnp.asarray(batch_mask.astype(np.float32)),
        atom_mask=jnp.asarray(am),
    )
    if spooky:
        out["charges"] = jnp.zeros(NA, dtype=jnp.float32)
        out["spins"] = jnp.zeros(NA, dtype=jnp.float32)
    return out


@pytest.fixture(scope="module", params=["physnet", "spooky"])
def fitted(request):
    spooky = request.param == "spooky"
    m = _spooky() if spooky else _physnet()
    b = _inputs(np.array([0.0, 0.0, 0.0]), spooky=spooky)
    params = m.init(jax.random.PRNGKey(0), **b)
    return m, params, spooky


def _apply(m, params, b):
    return m.apply(params, **b, compute_forces=True)


def _energy(m, params, b):
    return float(np.asarray(_apply(m, params, b)["energy"]).reshape(-1)[0])


def test_energy_is_translation_invariant_even_at_the_origin(fitted):
    """The regression: a molecule ON the origin sits among the padding."""
    m, params, spooky = fitted
    e_origin = _energy(m, params, _inputs(np.array([0.0, 0.0, 0.0]), spooky=spooky))
    e_far = _energy(m, params, _inputs(np.array([50.0, 0.0, 0.0]), spooky=spooky))
    assert e_origin == pytest.approx(e_far, abs=1e-5), (
        f"energy moved {e_far - e_origin:.4f} with absolute position: "
        "padding atoms are leaking into the message passing"
    )


def test_moving_only_the_padding_changes_nothing(fitted):
    """Nothing physical can depend on where padding sits."""
    m, params, spooky = fitted
    b0 = _inputs(np.array([0.0, 0.0, 0.0]), spooky=spooky, pad_pos=0.0)
    b1 = _inputs(np.array([0.0, 0.0, 0.0]), spooky=spooky, pad_pos=1000.0)
    assert _energy(m, params, b0) == pytest.approx(_energy(m, params, b1), abs=1e-5)


def test_forces_are_finite_at_the_origin(fitted):
    """Padding-padding pairs are exactly coincident (r=0).

    Zeroing a basis whose gradient is singular there gives 0 * NaN = NaN in
    every force, so the masked pairs must be pushed off r=0 BEFORE the basis.
    """
    m, params, spooky = fitted
    b = _inputs(np.array([0.0, 0.0, 0.0]), spooky=spooky)
    forces = np.asarray(_apply(m, params, b)["forces"])
    assert np.isfinite(forces).all(), "NaN/Inf force from coincident padding"


def test_forces_are_translation_invariant(fitted):
    m, params, spooky = fitted

    def forces(off):
        b = _inputs(np.array(off), spooky=spooky)
        return np.asarray(_apply(m, params, b)["forces"])[:NREAL]

    assert np.allclose(
        forces([0.0, 0.0, 0.0]), forces([50.0, 0.0, 0.0]), atol=1e-4
    )
