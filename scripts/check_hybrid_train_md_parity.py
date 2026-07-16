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

    # Mode C (fixed+latent) — only for checkpoints trained with --mm-charge-correction
    python scripts/check_hybrid_train_md_parity.py \
        --checkpoint /path/to/ckpts/hybrid/hybrid-<uuid> \
        --data /path/to/out_combined_dedup/energies_forces_dipoles_test.npz \
        --mm-charge-correction

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
        "--mm-charge-correction",
        action="store_true",
        help=(
            "Mode C (fixed+latent): q_MM = q_CGenFF + neutralize(q_ML). "
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

    data = np.load(args.data, allow_pickle=True)
    sigmas = jnp.asarray(data["cgenff_master_sigmas"])
    epsilons = jnp.asarray(data["cgenff_master_epsilons"])
    natoms = int(np.asarray(data["Z"]).shape[1])

    _, params, model = _load_physnet_checkpoint(Path(args.checkpoint), natoms)
    if args.mm_charge_correction and not getattr(model, "charges", False):
        print(
            "--mm-charge-correction needs a checkpoint built with charges=True",
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
    mode_label = "fixed_plus_latent" if args.mm_charge_correction else "fixed"
    print(f"mm_charge_mode={mode_label}")

    print(f"{'idx':>6} {'species':>9} {'regime':>8} {'E_train':>14} {'E_md':>14} {'diff':>12}  ok")
    worst = 0.0
    bad = 0
    for i, name, regime in picks:
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
            charge_correction=bool(args.mm_charge_correction),
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
            mm_charge_correction=bool(args.mm_charge_correction),
        )
        res = factory(
            atomic_numbers=Z[:n_real],
            atomic_positions=R[:n_real],
            n_monomers=2,
            cutoff_params=cutoff_params,
            doML=True,
            doMM=True,
            doML_dimer=True,
        )
        e_md = float(res[0]) if isinstance(res, (tuple, list)) else float(res)

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
