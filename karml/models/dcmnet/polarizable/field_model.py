"""Field-conditioned DCMnet: a local, polarisable monomer model whose charges respond to a background field.

Same architecture and parameter tree as dcmnet_model.MessagePassingModel plus two zero-initialised, bias-free linear maps
(field_s, field_v) that inject the electrostatic potential and field at each atom into the features after message-passing
iteration 0 (when the features already carry l <= max_degree):
    scalar  (l=0, even)  += V_i * field_s(x_i)         V_i  = potential at atom i, molecule-mean removed   (invariant)
    vector  (l=1, odd )  += E_i (x) field_v(x_i)       E_i  = electric field at atom i                      (polar vector: true-parity l=1 slot,
                                                                                                           the same slot the model reads its dipoles from)
Invariants: (1) field = 0 (or field=None) reproduces the base model exactly, for any trained field_s/field_v (no biases on the field path);
            (2) mono is invariant and the dipole displacement is equivariant under rotations and reflections (tested in tests/unit/test_polarizable_dcmnet.py).
History: before 24 Nov the vector field E was silently dropped (the parity index was 1 on a parity axis of size 1, an out-of-range
scatter that JAX ignores); fixed below by indexing parity 0 when the axis has size 1.  Models trained earlier are potential-only.
Units: V in Hartree/e, E in Hartree/(e Bohr) times the module constants VSCALE/ESCALE (inputs are rescaled to O(1)).
"""

import functools

import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .base import BOHR_PER_ANG, NATOMS, MessagePassingModel

VSCALE = 20.0  # typical |V - mean| ~ 0.01-0.05 Hartree/e
ESCALE = 50.0  # typical |E| ~ 0.005-0.02 Hartree/(e Bohr)
SIGMA = 1.0  # Gaussian smearing (A) of the source charges when computing V and E at the atoms


class FieldMessagePassingModel(MessagePassingModel):
    inject_all: bool = False  # inject the field after every iteration (extra bias-free maps field_s{i}, field_v{i}); default: after iteration 0 only

    def mono(
        self,
        atomic_numbers,
        positions,
        dst_idx,
        src_idx,
        batch_segments,
        batch_size,
        field_V=None,
        field_E=None,
        edge_w=None,
    ):
        positions_dst = e3x.ops.gather_dst(positions, dst_idx=dst_idx)
        positions_src = e3x.ops.gather_src(positions, src_idx=src_idx)
        displacements = positions_src - positions_dst
        basis = e3x.nn.basis(
            displacements,
            num=self.num_basis_functions,
            max_degree=self.max_degree,
            radial_fn=e3x.nn.reciprocal_bernstein,
            cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=self.cutoff),
        )
        if edge_w is not None:
            basis = basis * edge_w.reshape(
                (-1,) + (1,) * (basis.ndim - 1)
            )  # multiplicity of identical (merged) padding nodes: messages are linear in the basis
        x = e3x.nn.Embed(num_embeddings=self.max_atomic_number + 1, features=self.features)(atomic_numbers)
        for i in range(self.num_iterations):
            if i == self.num_iterations - 1:
                y = e3x.nn.MessagePass(max_degree=self.max_degree, include_pseudotensors=False)(
                    x, basis, dst_idx=dst_idx, src_idx=src_idx
                )
            else:
                y = e3x.nn.MessagePass(include_pseudotensors=self.include_pseudotensors)(
                    x, basis, dst_idx=dst_idx, src_idx=src_idx
                )
            y = e3x.nn.add(x, y)
            y = e3x.nn.Dense(self.features)(y)
            y = e3x.nn.silu(y)
            y = e3x.nn.Dense(self.features)(y)
            x = e3x.nn.add(x, y)
            if (i == 0 or self.inject_all) and field_V is not None:
                s0 = x[:, 0, 0, :]
                sfx = "" if i == 0 else str(i)
                gs = nn.Dense(self.features, use_bias=False, kernel_init=nn.initializers.zeros, name="field_s" + sfx)(
                    s0
                )
                gv = nn.Dense(self.features, use_bias=False, kernel_init=nn.initializers.zeros, name="field_v" + sfx)(
                    s0
                )
                x = x.at[:, 0, 0, :].add(field_V[:, None] * gs)
                par = (
                    1 if x.shape[1] == 2 else 0
                )  # parity axis has size 1 for the true-parity-only features used here (an index of 1 is out of range and the update is silently dropped)
                x = x.at[:, par, 1:4, :].add(field_E[:, :, None] * gv[:, None, :])
        x = e3x.nn.TensorDense(features=self.n_dcm, max_degree=1, include_pseudotensors=False)(x)
        atomic_mono = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
        element_bias = self.param("element_bias", lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number + 1))
        atomic_mono = nn.Dense(self.n_dcm, use_bias=False)(atomic_mono)
        atomic_mono = atomic_mono.squeeze(axis=1)
        atomic_mono = atomic_mono.squeeze(axis=1)
        atomic_mono += element_bias[atomic_numbers][:, None]
        x = e3x.nn.hard_tanh(x) * self.dc_scale
        atomic_dipo = x[:, 1, 1:4, :]
        atomic_dipo += positions[:, :, None]
        return atomic_mono, atomic_dipo

    @nn.compact
    def __call__(
        self,
        atomic_numbers,
        positions,
        dst_idx,
        src_idx,
        batch_segments=None,
        batch_size=None,
        field_V=None,
        field_E=None,
        edge_w=None,
    ):
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1
        return self.mono(
            atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size, field_V, field_E, edge_w
        )


def source_field(R_atoms, q_src, p_src, sigma=SIGMA):
    """Potential V (Hartree/e) and field E (Hartree/(e Bohr)) at atom positions R_atoms (n,3, A) from Gaussian-smeared point charges
    q_src (m,) e at p_src (m,3) A (width sigma A).  Pure jnp (differentiable in R_atoms, p_src, q_src)."""
    d = R_atoms[:, None, :] - p_src[None, :, :]
    r = jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-12)
    rb = r * BOHR_PER_ANG
    t = r / sigma
    erf_t = jax.scipy.special.erf(t)
    gauss = (2.0 / jnp.sqrt(jnp.pi)) * t * jnp.exp(-t * t)
    V = jnp.sum(q_src[None, :] * erf_t / rb, axis=-1)
    Emag = q_src[None, :] * (erf_t - gauss) / (rb * rb)
    E = jnp.sum(Emag[:, :, None] * (d / r[:, :, None]), axis=1)
    return V, E


def field_inputs(R_atoms, q_src, p_src, nvalid, sigma=SIGMA):
    """Rescaled, molecule-mean-removed V and E for one molecule of nvalid atoms (R_atoms (nvalid,3))."""
    V, E = source_field(R_atoms, q_src, p_src, sigma)
    return (V - V.mean()) * VSCALE, E * ESCALE


def pad_field(V, E, natoms=NATOMS):
    n = V.shape[0]
    return jnp.zeros(natoms, jnp.float32).at[:n].set(V), jnp.zeros((natoms, 3), jnp.float32).at[:n].set(E)


def merged_graph(n, total=NATOMS):
    """Exact (n real + 1 merged padding node) equivalent of the `total`-atom padded graph: the (total - n) identical padding nodes (Z=0, origin) are ONE node whose outgoing
    edges to real atoms are weighted by (total - n), plus a self edge weighted (total - n - 1) (messages are linear in the edge basis).  Pure numpy (safe inside jit).
    Returns P = n + 1, dst, src, edge_w (numpy); use with model.apply(..., edge_w=edge_w) and arrays padded to P."""
    nd_ = total - n
    P = n + 1
    ii, jj = np.nonzero(~np.eye(P, dtype=bool))
    d0 = ii.astype(np.int32)
    s0 = jj.astype(np.int32)
    w0 = np.ones(len(d0), np.float32)
    w0[s0 == n] = nd_
    return (
        P,
        np.concatenate([d0, [n]]).astype(np.int32),
        np.concatenate([s0, [n]]).astype(np.int32),
        np.concatenate([w0, [nd_ - 1]]).astype(np.float32),
    )


@functools.lru_cache(maxsize=None)
def _graph(n, total):
    return merged_graph(n, total)


def merged_charges(model, params, Z, R, field_V=None, field_E=None, total=NATOMS):
    """Charges q (n*n_dcm,) and site positions p (n*n_dcm, 3) of ONE molecule of n real atoms (Z (n,), R (n,3), caller's frame; optional field_V (n,), field_E (n,3)).
    Uses the exact merged padding graph (n real nodes + one weighted node for the identical dummy atoms of the 60-atom padded model): identical results to the padded
    model (about 1e-6) without carrying dummy nodes.  Net charge is zero."""
    n = len(Z)
    nd = model.n_dcm
    P, dst, src, EW = _graph(n, total)
    Zp = jnp.zeros(P, jnp.int32).at[:n].set(jnp.asarray(Z, jnp.int32))
    Rp = jnp.zeros((P, 3), jnp.float32).at[:n].set(R)
    if field_V is None:
        Vp = Ep = None
    else:
        Vp, Ep = pad_field(field_V, field_E, P)
    mono, dipo = model.apply(
        params, atomic_numbers=Zp, positions=Rp, dst_idx=dst, src_idx=src, field_V=Vp, field_E=Ep, edge_w=EW
    )
    d = jnp.moveaxis(dipo, -1, -2)[:n].reshape(n * nd, 3)
    m = mono[:n].reshape(n * nd)
    return m - m.mean(), d
