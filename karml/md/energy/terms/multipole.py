"""Fixed-fragment multipole electrostatics for rigid-body sampling.

Neural multipoles are predicted once (or injected via ``EnergyContext.options``)
in each monomer's body frame. During sampling only classical charge–dipole
pair energies are evaluated: origins follow each fragment COM and body-frame
dipoles are rotated via Kabsch alignment of the current geometry to the
reference monomer template.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mmml.md.energy.capacity import COMPUTE_DTYPE
from mmml.md.energy.registry import EnergyContext, TermFns, register_term
from mmml.md.system import MolecularSystem

__all__ = ["MultipoleTerm", "ANGSTROM_TO_BOHR", "HARTREE_TO_EV"]

ANGSTROM_TO_BOHR = 1.0 / 0.529177249
HARTREE_TO_EV = 27.211386245988


def _kabsch_rotation(P, Q):
    """Rotation ``R`` such that ``P @ R ≈ Q`` for centered point sets (N, 3)."""
    import jax.numpy as jnp

    H = P.T @ Q
    U, _, Vt = jnp.linalg.svd(H, full_matrices=False)
    d = jnp.sign(jnp.linalg.det(Vt.T @ U.T))
    # Guard against det≈0 numerical noise
    d = jnp.where(d == 0, 1.0, d)
    return Vt.T @ jnp.diag(jnp.array([1.0, 1.0, d], dtype=P.dtype)) @ U.T


def charge_dipole_pair_energy_au(r_vec, q_a, p_a, q_b, p_b, softening_bohr: float = 0.0):
    """Charge + dipole pair energy in Hartree (symmetric)."""
    import jax.numpy as jnp

    r2 = jnp.sum(r_vec * r_vec) + softening_bohr**2
    r = jnp.sqrt(jnp.maximum(r2, 1e-24))
    inv_r = 1.0 / r
    inv_r3 = inv_r**3
    inv_r5 = inv_r**5
    p_a_r = jnp.dot(p_a, r_vec)
    p_b_r = jnp.dot(p_b, r_vec)
    e_00 = q_a * q_b * inv_r
    e_01 = (q_b * p_a_r - q_a * p_b_r) * inv_r3
    e_11 = jnp.dot(p_a, p_b) * inv_r3 - 3.0 * p_a_r * p_b_r * inv_r5
    return e_00 + e_01 + e_11


@register_term("multipole")
class MultipoleTerm:
    """Fixed body-frame multipoles (charge + dipole) with Kabsch rotation."""

    name = "multipole"

    def neighbor_request(self, system: MolecularSystem):
        return None

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        import jax.numpy as jnp

        opts = dict(ctx.options)
        fixed = opts.get("fixed_multipoles")
        if fixed is None:
            raise ValueError(
                "multipole term requires ctx.options['fixed_multipoles'] "
                "(charges, dipoles_body_bohr, ref_positions_A, fragment_indices)"
            )

        charges = jnp.asarray(fixed["charges"], dtype=COMPUTE_DTYPE)
        dipoles_body = jnp.asarray(fixed["dipoles_body_bohr"], dtype=COMPUTE_DTYPE)
        ref_pos = jnp.asarray(fixed["ref_positions_A"], dtype=COMPUTE_DTYPE)
        frag_idx = [np.asarray(ix, dtype=np.int32) for ix in fixed["fragment_indices"]]
        softening = float(fixed.get("softening_bohr", 0.0))
        n_frag = len(frag_idx)
        if charges.shape[0] != n_frag or dipoles_body.shape[0] != n_frag:
            raise ValueError("fixed_multipoles charges/dipoles must match fragment_indices")

        # Pad fragments to a common atom count for vmap-friendly Kabsch.
        max_n = max(int(ix.size) for ix in frag_idx)
        idx_pad = np.zeros((n_frag, max_n), dtype=np.int32)
        mask_pad = np.zeros((n_frag, max_n), dtype=np.float64)
        for f, ix in enumerate(frag_idx):
            idx_pad[f, : ix.size] = ix
            mask_pad[f, : ix.size] = 1.0
        idx_pad_j = jnp.asarray(idx_pad, dtype=jnp.int32)
        mask_pad_j = jnp.asarray(mask_pad, dtype=COMPUTE_DTYPE)

        ref_local = []
        for f in range(n_frag):
            ix = frag_idx[f]
            pts = ref_pos[ix]
            com = jnp.mean(pts, axis=0)
            local = pts - com
            pad = jnp.zeros((max_n, 3), dtype=COMPUTE_DTYPE)
            pad = pad.at[: ix.size].set(local)
            ref_local.append(pad)
        ref_local_j = jnp.stack(ref_local)

        def _fragment_frame(R):
            pos = jnp.asarray(R, dtype=COMPUTE_DTYPE)

            def one(f):
                ix = idx_pad_j[f]
                m = mask_pad_j[f]
                pts = pos[ix]
                n = jnp.maximum(jnp.sum(m), 1.0)
                com = jnp.sum(pts * m[:, None], axis=0) / n
                local = (pts - com) * m[:, None]
                # Row-vector Kabsch: ref_local @ R ≈ local ⇒ dipole_lab = dipole_body @ R
                R_rot = _kabsch_rotation(ref_local_j[f], local)
                p_lab = dipoles_body[f] @ R_rot
                return com * ANGSTROM_TO_BOHR, p_lab

            coms = []
            dips = []
            for f in range(n_frag):
                c, p = one(f)
                coms.append(c)
                dips.append(p)
            return jnp.stack(coms), jnp.stack(dips)

        def energy_fn(R, **kwargs) -> Any:
            origins, dips = _fragment_frame(R)
            total = jnp.asarray(0.0, dtype=COMPUTE_DTYPE)
            for i in range(n_frag):
                for j in range(i + 1, n_frag):
                    r_vec = origins[j] - origins[i]
                    total = total + charge_dipole_pair_energy_au(
                        r_vec,
                        charges[i],
                        dips[i],
                        charges[j],
                        dips[j],
                        softening_bohr=softening,
                    )
            return total * HARTREE_TO_EV

        return TermFns(jax_energy_fn=energy_fn)
