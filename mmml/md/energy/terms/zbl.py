"""Intermolecular Ziegler–Biersack–Littmark (ZBL) short-range repulsion.

Uses PhysNet/Spooky default cutoffs (``cuton=0.1`` Å, ``cutoff=0.6`` Å) and the
same universal screening constants as ``physnetjax.models.zbl.ZBLRepulsion``.
Only intermolecular pairs contribute.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mmml.md.energy.capacity import COMPUTE_DTYPE
from mmml.md.energy.registry import EnergyContext, NeighborRequest, TermFns, register_term
from mmml.md.system import MolecularSystem

__all__ = ["ZBLTerm", "DEFAULT_ZBL_CUTON_A", "DEFAULT_ZBL_CUTOFF_A"]

DEFAULT_ZBL_CUTON_A = 0.1
DEFAULT_ZBL_CUTOFF_A = 0.6

_BOHR_TO_ANGSTROM = 0.529177249
_COULOMB_EV_ANGSTROM = 14.3996454784255
_A_COEFFICIENT = 0.8854 * _BOHR_TO_ANGSTROM
_A_EXPONENT = 0.23
_PHI_COEFFICIENTS = (0.18175, 0.50986, 0.28022, 0.02817)
_PHI_EXPONENTS = (3.19980, 0.94229, 0.40290, 0.20162)


@register_term("zbl")
class ZBLTerm:
    """Intermolecular ZBL repulsion over a padded pair list."""

    name = "zbl"

    def __init__(
        self,
        cuton_A: float | None = None,
        cutoff_A: float | None = None,
    ):
        self.cuton_A = cuton_A
        self.cutoff_A = cutoff_A

    def neighbor_request(self, system: MolecularSystem):
        cutoff = float(self.cutoff_A) if self.cutoff_A is not None else DEFAULT_ZBL_CUTOFF_A
        return NeighborRequest(cutoff_A=cutoff, kind="intermolecular")

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        import jax.numpy as jnp

        opts = dict(ctx.options)
        cuton = float(
            self.cuton_A
            if self.cuton_A is not None
            else opts.get("zbl_cuton", DEFAULT_ZBL_CUTON_A)
        )
        cutoff = float(
            self.cutoff_A
            if self.cutoff_A is not None
            else opts.get("zbl_cutoff", DEFAULT_ZBL_CUTOFF_A)
        )
        if not (0.0 <= cuton < cutoff):
            raise ValueError(f"ZBL cuton/cutoff invalid: cuton={cuton}, cutoff={cutoff}")

        Z = jnp.asarray(system.Z, dtype=COMPUTE_DTYPE)
        mol_id = jnp.asarray(system.mol_id, dtype=jnp.int32)
        box0 = (
            None
            if system.box is None
            else jnp.asarray(np.diag(np.asarray(system.box)), dtype=COMPUTE_DTYPE)
        )

        def energy_fn(R, *, pair_i=None, pair_j=None, pair_mask=None, box=None, **kwargs) -> Any:
            if pair_i is None or pair_j is None:
                raise ValueError("zbl jax face requires pair_i/pair_j (neighbor list)")
            pi = jnp.asarray(pair_i, dtype=jnp.int32)
            pj = jnp.asarray(pair_j, dtype=jnp.int32)
            pos = jnp.asarray(R, dtype=COMPUTE_DTYPE)
            if box is not None:
                cell = jnp.asarray(box, dtype=COMPUTE_DTYPE)
                box_diag = jnp.diag(cell) if cell.ndim == 2 else cell
            elif box0 is not None:
                box_diag = box0
            else:
                box_diag = None

            d = pos[pj] - pos[pi]
            if box_diag is not None:
                d = d - box_diag * jnp.round(d / box_diag)
            r = jnp.sqrt(jnp.maximum(jnp.sum(d * d, axis=-1), 1e-16))

            inter = (mol_id[pi] != mol_id[pj]).astype(COMPUTE_DTYPE)
            undirected = (pi < pj).astype(COMPUTE_DTYPE)
            mask = inter * undirected
            if pair_mask is not None:
                mask = mask * jnp.asarray(pair_mask, dtype=COMPUTE_DTYPE)

            cuton_d = jnp.asarray(cuton, dtype=COMPUTE_DTYPE)
            cutoff_d = jnp.asarray(cutoff, dtype=COMPUTE_DTYPE)
            switch_range = jnp.maximum(cutoff_d - cuton_d, 1e-12)
            x_sw = (cutoff_d - r) / switch_range
            s = ((6.0 * x_sw - 15.0) * x_sw + 10.0) * x_sw**3
            sw = jnp.where(
                r < cuton_d,
                jnp.ones_like(r),
                jnp.where(r >= cutoff_d, jnp.zeros_like(r), s),
            )
            sw = jnp.clip(sw, 0.0, 1.0)

            a_exp = jnp.abs(jnp.asarray(_A_EXPONENT, dtype=COMPUTE_DTYPE))
            za_i = jnp.abs(Z[pi]) ** a_exp
            za_j = jnp.abs(Z[pj]) ** a_exp
            a_ij = jnp.abs(jnp.asarray(_A_COEFFICIENT, dtype=COMPUTE_DTYPE)) / jnp.maximum(
                za_i + za_j, 1e-12
            )
            x = r / jnp.maximum(a_ij, 1e-12)
            coeffs = jnp.asarray(_PHI_COEFFICIENTS, dtype=COMPUTE_DTYPE)
            coeffs = jnp.abs(coeffs) / jnp.linalg.norm(jnp.abs(coeffs))
            exps = jnp.abs(jnp.asarray(_PHI_EXPONENTS, dtype=COMPUTE_DTYPE))
            phi = jnp.sum(coeffs[None, :] * jnp.exp(-exps[None, :] * x[:, None]), axis=1)
            return jnp.sum(
                _COULOMB_EV_ANGSTROM * (Z[pi] * Z[pj]) / r * phi * sw * mask
            )

        return TermFns(
            jax_energy_fn=energy_fn,
            neighbor_request=NeighborRequest(cutoff_A=cutoff, kind="intermolecular"),
        )
