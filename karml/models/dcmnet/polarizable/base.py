"""Standalone DCMnet (QM9, 2024) model definition + electrostatics helpers.

Kept standalone (rather than reusing ``DCMNetCharges`` elsewhere in karml) so that the parameter tree of the
released DCMnet checkpoints is reproduced exactly.

Copy of ``MessagePassingModel`` from
  github.com/EricBoittier/dcmnet  dcmnet/modules.py  (commit a31b41b2; identical to
  jaxeq commit fe51151e, Oct 2024), same author as this repository, with ONE change: the hard-coded scale of the
  hard_tanh that bounds the distributed-charge displacement (0.175 A in fe51151/a31b41b,
  0.3 A in jaxeq 6b7724e used for the `test4` series) is exposed as the attribute
  `dc_scale` so every checkpoint generation can be rebuilt.  Param tree is unchanged.
  dc_scale per generation (jaxeq git history): 0.173 (Jul 2024 - 1 Oct 2024; `qm9_esp/` models),
  0.3 (6b7724e / 8305ebf, 2-9 Oct 2024; `qm9_esp_dipole/` models), 0.175 (fe51151 / dcmnet a31b41b).
  The value for each released model is in its model_config.json.

Units: positions in Angstrom, charges in e, ESP in Hartree/e (atomic units),
dipole returned in e*Angstrom (multiply by 1/0.529177 for a.u., by 4.80320 for Debye).
"""

import functools

import e3x
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

NATOMS = 60
BOHR_PER_ANG = 1.88973  # used by dcmnet.electrostatics.coulomb_potential


class MessagePassingModel(nn.Module):
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 3
    num_basis_functions: int = 8
    cutoff: float = 5.0
    max_atomic_number: int = 17
    n_dcm: int = 4
    include_pseudotensors: bool = False
    dc_scale: float = 0.175  # NOT in the original dataclass (hard-coded there)

    def mono(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size):
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
        x = e3x.nn.TensorDense(features=self.n_dcm, max_degree=1, include_pseudotensors=False)(x)
        atomic_mono = e3x.nn.change_max_degree_or_type(x, max_degree=0, include_pseudotensors=False)
        element_bias = self.param("element_bias", lambda rng, shape: jnp.zeros(shape), (self.max_atomic_number + 1))
        atomic_mono = nn.Dense(self.n_dcm, use_bias=False)(atomic_mono)
        atomic_mono = atomic_mono.squeeze(axis=1)
        atomic_mono = atomic_mono.squeeze(axis=1)
        atomic_mono += element_bias[atomic_numbers][:, None]
        x = e3x.nn.hard_tanh(x) * self.dc_scale
        # NB: original code indexes x[:, 1, ...] on a parity axis of size 1; JAX clamps
        # out-of-bounds gather indices, so this is x[:, 0, 1:4, :] (the l=1 block).
        atomic_dipo = x[:, 0, 1:4, :]
        atomic_dipo += positions[:, :, None]
        return atomic_mono, atomic_dipo

    @nn.compact
    def __call__(self, atomic_numbers, positions, dst_idx, src_idx, batch_segments=None, batch_size=None):
        if batch_segments is None:
            batch_segments = jnp.zeros_like(atomic_numbers)
            batch_size = 1
        return self.mono(atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size)


def pad_molecule(Z, R, natoms=NATOMS):
    """Pad a molecule to the fixed 60-atom layout used in training (dummy atoms Z=0 at origin)."""
    Z = np.asarray(Z, dtype=np.int32)
    R = np.asarray(R, dtype=np.float32)
    n = len(Z)
    Zp = np.zeros(natoms, np.int32)
    Rp = np.zeros((natoms, 3), np.float32)
    Zp[:n] = Z
    Rp[:n] = R
    return Zp, Rp, n


def predict_charges(model, params, Z, R, n_atoms=None):
    """Forward pass on one molecule. Z (60,), R (60,3) padded (or unpadded -> padded here).
    Returns (q (n_atoms*n_dcm,), pos (n_atoms*n_dcm,3)) after the training-time post-processing:
    charges on dummy atoms zeroed and the net charge constrained to 0 by subtracting the mean
    (exactly as in dcmnet.loss.dipo_esp_mono_loss)."""
    if n_atoms is None:
        Z, R, n_atoms = pad_molecule(Z, R)
    dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(NATOMS)
    mono, dipo = model.apply(
        params, atomic_numbers=jnp.asarray(Z), positions=jnp.asarray(R), dst_idx=dst_idx, src_idx=src_idx
    )
    return postprocess(mono, dipo, n_atoms, model.n_dcm)


def postprocess(mono, dipo, n_atoms, n_dcm):
    d = jnp.moveaxis(dipo, -1, -2).reshape(NATOMS * n_dcm, 3)
    m = mono.reshape(NATOMS * n_dcm)
    NDC = n_atoms * n_dcm
    valid = jnp.where(jnp.arange(NATOMS * n_dcm) < NDC, 1, 0)
    m = m * valid
    m = (m - m.sum() / NDC) * valid
    return m, d


def esp(q, pos, grid):
    """ESP (Hartree/e) of point charges q (e) at pos (A) on grid points (A)."""
    r = jnp.linalg.norm(grid[:, None, :] - pos[None, :, :], axis=-1)
    return jnp.sum(q[None, :] / (r * BOHR_PER_ANG), axis=-1)


def dipole(q, pos, com):
    """Molecular dipole in e*Angstrom relative to com (A)."""
    return jnp.sum(q[:, None] * (pos - com[None, :]), axis=0)
