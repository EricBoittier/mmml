#!/usr/bin/env python3
"""E2E gate: hybrid TRAINING energy must equal the MD calculator's energy.

The unit tests pin the pieces independently -- CGenFF parameters
(``test_cgenff_lj_parity``), the MM formula (``test_cgenff_mm_energy``), the
switching (``test_ml_switch_scale``) and the assembly (``test_hybrid_energy``).
This is the only check that the *assembled* number a model is trained on equals
the number it is deployed with.

It cannot be a CI unit test: the MD path's MM (``mm_energy_forces``) indexes
``pycharmm.param.get_atc()``, so it needs a live CHARMM session.  Run it on the
cluster after training:

    # Mode A (fixed CGenFF charges) — default, always run first
    python scripts/check_hybrid_train_md_parity.py \
        --checkpoint /path/to/ckpts/hybrid/hybrid-<uuid> \
        --data /path/to/out_combined_dedup/energies_forces_dipoles_test.npz

    # Mode B (latent) — checkpoint trained with --mm-charge-mode latent
    python scripts/check_hybrid_train_md_parity.py \
        --checkpoint /path/to/ckpts/hybrid/hybrid-<uuid> \
        --data /path/to/out_combined_dedup/energies_forces_dipoles_test.npz \
        --mm-charge-mode latent

    # Mode C (fixed+latent) — --mm-charge-mode fixed_plus_latent or --mm-charge-correction
    python scripts/check_hybrid_train_md_parity.py \
        --checkpoint /path/to/ckpts/hybrid/hybrid-<uuid> \
        --data /path/to/out_combined_dedup/energies_forces_dipoles_test.npz \
        --mm-charge-correction

IMPORTANT: point ``--checkpoint`` at a *frozen* checkpoint, not a run that is
still training.  ``_load_physnet_checkpoint`` resolves ``get_last()`` once at
startup while ``setup_calculator`` re-resolves it per structure, so against a
live run this compares two different epochs and reports a spurious ~0.1 eV
mismatch that scales with molecule size.

Known open discrepancy: ACO+ACO fails by up to 2.4e-2 eV wherever E_MM != 0
(handoff/MM-tail), while its ML-only case and *all* DCM cases pass at <=3e-4 eV.
So the ML assembly is confirmed and the gap is in the MM term for acetone only.
Atom ordering was tested and RULED OUT: the dataset's ACO order differs from the
PSF's (O first in the PSF, fourth in the dataset), but permuting into PSF order
moved E_md by only ~2e-5.  Next suspect is the ACO CGenFF LJ parameters
(dataset master tables vs the PSF's).

Compares, for real DCM+DCM and ACO+ACO dimers spanning the ML-only / handoff /
MM-tail regimes:

    training path : hybrid_forward(model.apply, params, batch)
    MD path       : setup_calculator(...)(Z, R, n_monomers=2)

See ``docs/hybrid-mm-charges.md`` for the fixed / latent / fixed+latent taxonomy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _pick_dimers(data, n_per_regime=2):
    """Real dimers spanning ML-only (<6.5), handoff (6.5-8) and MM tail (8-13)."""
    mol_id = np.asarray(data["mol_id"])
    R = np.asarray(data["R"])
    res = np.asarray(data["res_name"])
    picks = []
    for name in sorted({str(r) for r in res if "," in str(r)}):
        idx = np.where(res == name)[0]
        r_com = []
        for i in idx:
            m0, m1 = mol_id[i] == 0, mol_id[i] == 1
            r_com.append(np.linalg.norm(R[i][m0].mean(0) - R[i][m1].mean(0)))
        r_com = np.asarray(r_com)
        for lo, hi, label in ((0, 6.5, "ML-only"), (6.5, 8.0, "handoff"), (8.0, 13.0, "MM-tail")):
            sel = idx[(r_com >= lo) & (r_com < hi)][:n_per_regime]
            picks += [(int(i), name, label) for i in sel]
    return picks


def _setup_charmm_psf(resid: str, n_monomers: int) -> int:
    """Generate a CHARMM PSF for ``n_monomers`` copies of ``resid``.

    build_mm_energy_forces_fn indexes pycharmm.param.get_atc(), so the MD side
    needs a live PSF for this exact system -- there is no CHARMM-free shortcut.
    Returns the atom count per monomer.
    """
    import pycharmm

    from mmml.interfaces.pycharmmInterface import setupRes
    from mmml.interfaces.pycharmmInterface.import_pycharmm import (
        pycharmm_quiet,
        reset_block,
    )

    pycharmm_quiet()
    reset_block()
    atoms = setupRes.main(resid)
    pycharmm.read.sequence_string(" ".join([resid] * n_monomers))
    pycharmm.gen.new_segment(seg_name=resid, setup_ic=True)
    pycharmm.ic.prm_fill(replace_all=True)
    return len(atoms)


def _batch_from_structure(data, i):
    """Single-structure batch in the layout prepare_batches_jit produces."""
    import jax.numpy as jnp

    Z = np.asarray(data["Z"])[i]
    R = np.asarray(data["R"])[i]
    n = Z.shape[0]
    atom_mask = (Z > 0).astype(np.float32)
    idx = np.arange(n)
    dst, src = np.meshgrid(idx, idx, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    keep = (dst != src) & (atom_mask[dst] > 0) & (atom_mask[src] > 0)
    return {
        "R": jnp.asarray(R),
        "Z": jnp.asarray(Z),
        "atom_mask": jnp.asarray(atom_mask),
        "batch_mask": jnp.asarray(keep.astype(np.float32)),
        "dst_idx": jnp.asarray(dst),
        "src_idx": jnp.asarray(src),
        "batch_segments": jnp.zeros(n, dtype=jnp.int32),
        "mol_id": jnp.asarray(np.asarray(data["mol_id"])[i][None, :]),
        "cgenff_type_idx": jnp.asarray(np.asarray(data["cgenff_type_idx"])[i][None, :]),
        "cgenff_charge": jnp.asarray(np.asarray(data["cgenff_charge"])[i][None, :]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Trained PhysNet checkpoint dir")
    p.add_argument("--data", required=True, help="NPZ with CGenFF fields (out_combined_dedup)")
    p.add_argument("--ml-switch-width", type=float, default=1.5)
    p.add_argument("--mm-switch-on", type=float, default=8.0)
    p.add_argument("--mm-switch-width", type=float, default=5.0)
    p.add_argument(
        "--mm-charge-mode",
        choices=["fixed", "latent", "fixed_plus_latent"],
        default=None,
        help="MM Coulomb charge mode (default: fixed). Must match training.",
    )
    p.add_argument(
        "--mm-charge-correction",
        action="store_true",
        help=(
            "Alias for --mm-charge-mode fixed_plus_latent. "
            "Must match how the checkpoint was trained. Default is Mode A (fixed)."
        ),
    )
    p.add_argument("--tol", type=float, default=1e-3, help="abs energy tolerance (kcal/mol)")
    args = p.parse_args()

    import jax.numpy as jnp

    from mmml.cli.misc.physnet_evaluate import _load_physnet_checkpoint
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator
    from mmml.models.hybrid_energy import hybrid_forward
    from mmml.models.mm_charge_mode import (
        mm_charge_mode_needs_q_ml,
        resolve_hybrid_mm_charge_mode,
    )

    mode = resolve_hybrid_mm_charge_mode(
        mm_charge_mode=args.mm_charge_mode,
        charge_correction=bool(args.mm_charge_correction),
    )

    data = np.load(args.data, allow_pickle=True)
    sigmas = jnp.asarray(data["cgenff_master_sigmas"])
    epsilons = jnp.asarray(data["cgenff_master_epsilons"])
    natoms = int(np.asarray(data["Z"]).shape[1])

    _, params, model = _load_physnet_checkpoint(Path(args.checkpoint), natoms)
    if mm_charge_mode_needs_q_ml(mode) and not getattr(model, "charges", False):
        print(
            f"--mm-charge-mode {mode.value} needs a checkpoint built with charges=True",
            file=sys.stderr,
        )
        return 1

    picks = _pick_dimers(data)
    if not picks:
        print("no dimers found in the dataset", file=sys.stderr)
        return 1

    cutoff_params = CutoffParameters(
        ml_switch_width=args.ml_switch_width,
        mm_switch_on=args.mm_switch_on,
        mm_switch_width=args.mm_switch_width,
    )
    print(f"mm_charge_mode={mode.value}")

    print(f"{'idx':>6} {'species':>9} {'regime':>8} {'E_train':>14} {'E_md':>14} {'diff':>12}  ok")
    worst = 0.0
    bad = 0
    current_species = None
    for i, name, regime in sorted(picks, key=lambda t: t[1]):
        resid = name.split(",")[0].strip()
        if resid != current_species:
            _setup_charmm_psf(resid, 2)
            current_species = resid
        Z = np.asarray(data["Z"])[i]
        R = np.asarray(data["R"])[i]
        n_real = int(np.asarray(data["N"])[i])
        per_mono = n_real // 2

        # --- training path -------------------------------------------------
        batch = _batch_from_structure(data, i)
        out = hybrid_forward(
            model.apply, params, batch, 1, sigmas, epsilons,
            mm_switch_on=args.mm_switch_on,
            mm_switch_width=args.mm_switch_width,
            ml_switch_width=args.ml_switch_width,
            mm_charge_mode=mode.value,
        )
        e_train = float(np.asarray(out["energy"]).reshape(-1)[0])

        # --- MD path -------------------------------------------------------
        factory = setup_calculator(
            ATOMS_PER_MONOMER=[per_mono, per_mono],
            N_MONOMERS=2,
            ml_switch_width=args.ml_switch_width,
            mm_switch_on=args.mm_switch_on,
            mm_switch_width=args.mm_switch_width,
            complementary_handoff=True,
            doML=True,
            doMM=True,
            doML_dimer=True,
            model_restart_path=args.checkpoint,
            MAX_ATOMS_PER_SYSTEM=n_real,
            ml_energy_conversion_factor=1,
            ml_force_conversion_factor=1,
            mm_charge_mode=mode.value,
        )
        _calc, sc_fn, update_fn_factory = factory(
            atomic_numbers=Z[:n_real],
            atomic_positions=R[:n_real],
            n_monomers=2,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=True,
            doML_dimer=True,
            backprop=False,
        )
        R_jax = jnp.asarray(R[:n_real])
        Z_jax = jnp.asarray(Z[:n_real])
        mm_pair_idx = mm_pair_mask = None
        if update_fn_factory is not None:
            update_fn = update_fn_factory(R[:n_real], cutoff_params)
            if update_fn is not None:
                mm_pair_idx, mm_pair_mask = update_fn(R[:n_real])
        out_md = sc_fn(
            R_jax,
            Z_jax,
            2,
            cutoff_params,
            doML=True,
            doMM=True,
            doML_dimer=True,
            mm_pair_idx=mm_pair_idx,
            mm_pair_mask=mm_pair_mask,
        )
        e_md = float(np.asarray(out_md.energy).reshape(-1)[0])

        diff = abs(e_train - e_md)
        worst = max(worst, diff)
        ok = diff <= args.tol
        bad += (not ok)
        print(f"{i:>6} {name:>9} {regime:>8} {e_train:>14.6f} {e_md:>14.6f} {diff:>12.2e}  {'OK' if ok else 'FAIL'}")

    print(f"\nworst |E_train - E_md| = {worst:.3e} kcal/mol (tol {args.tol})")
    if bad:
        print(f"PARITY FAILED on {bad}/{len(picks)} structures", file=sys.stderr)
        return 1
    print(f"PARITY OK on {len(picks)} structures: training == MD calculator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
