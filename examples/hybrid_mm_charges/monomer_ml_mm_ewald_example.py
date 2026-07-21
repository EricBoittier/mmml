#!/usr/bin/env python3
"""Minimal "ML monomers + MM(Ewald)" example: ``fixed`` vs ``latent`` charges.

Exercises the assembly the capability matrix calls **Monomer ML + MM**
(``doML=True``, ``doML_dimer`` effectively inert, ``doMM=True`` — see
"Hybrid energy assembly modes" in ``docs/calculator-capabilities.md``) together
with the pure-JAX native Ewald long-range solver (``lr_solver="ewald"``), for
both MM charge modes that do not need a liquid box:

* **Mode A / ``fixed``**  — ``q_MM = q_CGenFF``
* **Mode B / ``latent``** (``q1``) — ``q_MM = neutralize_per_monomer(q_ML)``
  from the AB-dimer forward

See ``docs/hybrid-mm-charges.md`` for the full charge-mode taxonomy.

Why this script exists
-----------------------
``scripts/check_ewald_train_md_pme_parity.py`` already validates train<->MD
Ewald parity for Mode A (``fixed``); it has no charge head, so it cannot
exercise Mode B. This script adds the ``latent`` leg using a tiny analytic
stand-in model (predicts a fixed per-atom charge -- the same pattern as
``tests/unit/test_hybrid_energy.py::_charged_model``), so it needs **no
checkpoint and no CHARMM**.

"ML monomers + MM" in spirit, not via ``--skip-ml-dimers``
------------------------------------------------------------
``hybrid_forward`` (the training-time assembly ``mmml_calculator`` mirrors)
has no standalone flag to disable the switched ML-dimer term -- that MD-only
diagnostic knob (``doML_dimer=False`` / ``--skip-ml-dimers``) lives on the
deployed calculator, not this training-side function, and Mode B needs a
live AB-dimer forward for its charges regardless. Instead this script places
the two monomers past the ML->MM handoff tail (``mm_switch_on +
mm_switch_width``), where the switch weight ``s(r_com) -> 0`` and the dimer
correction ``s * dE_ML`` vanishes on its own -- ``E_total`` collapses to
``E_ML(A) + E_ML(B) + E_MM`` for both modes, which *is* "ML monomers + MM".
Per the request this responds to: this script does not try to zero the
residual dimer term exactly, only to make it negligible by separation.

No CHARMM, no checkpoint required::

    python examples/hybrid_mm_charges/monomer_ml_mm_ewald_example.py

Registered in ``docs/hybrid-mm-charges.md`` and
``examples/hybrid_mm_charges/README.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Canonical ML<->MM handoff defaults (mmml/interfaces/pycharmmInterface/cutoffs.py) --
# reused here rather than re-hardcoding the same numbers.
from mmml.interfaces.pycharmmInterface.cutoffs import (
    DEFAULT_ML_SWITCH_WIDTH,
    DEFAULT_MM_SWITCH_ON,
    DEFAULT_MM_SWITCH_WIDTH,
)

# Small synthetic system: two 2-atom "monomers" (+1 padding slot), mirroring
# the fixture already unit-tested in tests/unit/test_hybrid_energy.py.
ATOMIC_NUMBERS = jnp.array([6, 1, 6, 1, 0])  # C, H, C, H, padding
MOL_ID = jnp.array([0, 0, 1, 1, -1])
CGENFF_CHARGE = jnp.array([-0.3, 0.15, -0.3, 0.15, 0.0])
CGENFF_TYPE_IDX = jnp.array([0, 1, 0, 1, -1])
MASTER_SIGMAS = jnp.array([3.6527, 2.3876])
MASTER_EPSILONS = jnp.array([0.0780, 0.0240])

# Separation past the handoff tail (mm_switch_on + mm_switch_width) so the
# switched ML-dimer correction s(r_com)*dE_ML is negligible: the system is
# effectively "ML monomers + MM" without needing doML_dimer=False.
MONOMER_SEPARATION_A = 20.0
assert MONOMER_SEPARATION_A > DEFAULT_MM_SWITCH_ON + DEFAULT_MM_SWITCH_WIDTH
# Cubic box for the periodic Ewald sum; matches the 30 A boxes used by the
# other small-system Ewald smoke tests in this repo (e.g. --pme-box-length 30
# in scripts/check_ewald_train_md_pme_parity.py).
PME_BOX_LENGTH_A = 30.0
PME_ACCURACY = 1e-6

# Fixed per-atom ML charge prediction for the "latent" (Mode B) leg -- a real
# checkpoint's charge head would return this per structure instead.
LATENT_CHARGE_HEAD = jnp.array([0.2, -0.1, 0.3, -0.05, 0.0])


def _fake_model_apply(params, *, atomic_numbers, positions, dst_idx, src_idx,
                       batch_segments, batch_size, batch_mask, atom_mask):
    """Analytic per-atom + pairwise energy, plus a fixed per-atom charge head.

    Respects atom_mask / batch_mask exactly like a real model, so restricting
    the masks to one monomer (as ``hybrid_forward`` does internally for the
    E(A)/E(B) legs) yields that monomer's energy alone.
    """
    import jax

    e_atom_scale = -10.0
    pair_scale = -2.0

    def energy_fn(pos):
        e_atom = e_atom_scale * jnp.cos(pos[:, 0]) * atom_mask
        e_per = jax.ops.segment_sum(e_atom, batch_segments, num_segments=batch_size)
        d = pos[dst_idx] - pos[src_idx]
        r = jnp.sqrt(jnp.maximum(jnp.sum(d * d, axis=-1), 1e-12))
        e_pair = pair_scale * jnp.exp(-r) * batch_mask
        e_per = e_per + jax.ops.segment_sum(
            e_pair, batch_segments[dst_idx], num_segments=batch_size
        )
        return jnp.sum(e_per), e_per

    (_, e_per), grad = jax.value_and_grad(energy_fn, has_aux=True)(positions)
    n = positions.shape[0]
    return {
        "energy": e_per.reshape(batch_size, 1),
        "forces": -grad,
        "charges": jnp.asarray(LATENT_CHARGE_HEAD)[:n],
    }


def _build_batch() -> dict:
    sep = MONOMER_SEPARATION_A
    pos = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [sep, 0.0, 0.0], [sep + 1.0, 0.2, 0.0], [0.0, 0.0, 0.0]]
    )
    n = int(pos.shape[0])
    atom_mask = (MOL_ID >= 0).astype(jnp.float32)
    idx = jnp.arange(n)
    dst, src = jnp.meshgrid(idx, idx, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    keep = (dst != src) & (atom_mask[dst] > 0) & (atom_mask[src] > 0)
    return {
        "R": pos,
        "Z": ATOMIC_NUMBERS,
        "mol_id": MOL_ID.reshape(1, n),
        "cgenff_type_idx": CGENFF_TYPE_IDX.reshape(1, n),
        "cgenff_charge": CGENFF_CHARGE.reshape(1, n),
        "atom_mask": atom_mask,
        "batch_mask": keep.astype(jnp.float32),
        "dst_idx": dst,
        "src_idx": src,
        "batch_segments": jnp.zeros(n, dtype=jnp.int32),
    }


def run_mode(mm_charge_mode: str) -> dict:
    from mmml.models.hybrid_energy import hybrid_forward

    batch = _build_batch()
    out = hybrid_forward(
        _fake_model_apply,
        {},
        batch,
        1,
        MASTER_SIGMAS,
        MASTER_EPSILONS,
        mm_switch_on=DEFAULT_MM_SWITCH_ON,
        mm_switch_width=DEFAULT_MM_SWITCH_WIDTH,
        ml_switch_width=DEFAULT_ML_SWITCH_WIDTH,
        mm_charge_mode=mm_charge_mode,
        short_range_wall=False,
        lr_solver="ewald",
        include_lj=False,  # native Ewald covers Coulomb only, same as MD periodic_external
        pme_box_length=PME_BOX_LENGTH_A,
        pme_accuracy=PME_ACCURACY,
    )
    return {
        "ml_scale": float(np.asarray(out["ml_scale"]).reshape(-1)[0]),
        "energy_eV": float(np.asarray(out["energy"]).reshape(-1)[0]),
        "e_mm_eV": float(np.asarray(out["e_mm"]).reshape(-1)[0]),
    }


def cross_check_e_mm_against_md_kernel(mm_charge_mode: str) -> float:
    """Recompute the same E_MM independently via the MD-side Ewald wrapper.

    Mirrors ``scripts/check_ewald_train_md_pme_parity.py``'s methodology: feed
    the *same effective per-atom charges* ``hybrid_forward`` used into the
    standalone ``compute_native_ewald_coulomb`` kernel and compare in eV.
    """
    from mmml.data.units import KCAL_MOL_TO_EV
    from mmml.interfaces.pycharmmInterface.long_range_backend import (
        compute_native_ewald_coulomb,
    )
    from mmml.models.mm_charge_mode import apply_mm_charge_mode

    batch = _build_batch()
    n = int(batch["R"].shape[0])
    q_ml = None
    if mm_charge_mode != "fixed":
        out = _fake_model_apply(
            {},
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=1,
            batch_mask=batch["batch_mask"],
            atom_mask=batch["atom_mask"],
        )
        q_ml = out["charges"].reshape(1, n)
    charges = apply_mm_charge_mode(
        mm_charge_mode,
        batch["cgenff_charge"],
        q_ml,
        batch["mol_id"],
        n_monomers=2,
    )
    real = np.asarray(batch["mol_id"]).reshape(-1) >= 0
    md = compute_native_ewald_coulomb(
        np.asarray(batch["R"])[real],
        np.asarray(charges).reshape(-1)[real],
        box_length_A=PME_BOX_LENGTH_A,
        accuracy=PME_ACCURACY,
    )
    return float(md.energy_kcalmol) * KCAL_MOL_TO_EV


def main() -> int:
    print(
        f"MM+ML(monomer) + native Ewald | separation={MONOMER_SEPARATION_A:g} A "
        f"(> handoff tail {DEFAULT_MM_SWITCH_ON + DEFAULT_MM_SWITCH_WIDTH:g} A) | "
        f"box={PME_BOX_LENGTH_A:g} A\n"
    )
    print(f"{'mode':<8} {'ml_scale':>10} {'E_total(eV)':>13} {'E_mm(eV)':>11} {'|dE_mm|(eV)':>13}")

    bad = 0
    for mode in ("fixed", "latent"):
        result = run_mode(mode)
        e_mm_md = cross_check_e_mm_against_md_kernel(mode)
        diff = abs(result["e_mm_eV"] - e_mm_md)
        ok = diff < 1e-6
        bad += not ok
        print(
            f"{mode:<8} {result['ml_scale']:>10.2e} {result['energy_eV']:>13.6f} "
            f"{result['e_mm_eV']:>11.6f} {diff:>13.3e}  {'OK' if ok else 'FAIL'}"
        )

    if bad:
        print(f"\n{bad} mode(s) failed the train<->MD Ewald cross-check", file=sys.stderr)
        return 1
    print("\nOK: fixed and latent both run MM+ML(monomer) with native Ewald.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
