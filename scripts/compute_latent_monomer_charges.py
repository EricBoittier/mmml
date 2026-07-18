#!/usr/bin/env python3
"""Precompute a per-monomer latent-charge template for liquid MD (Mode D).

``mm_charge_mode=latent`` (Mode B) needs a live AB-dimer forward at every MD
step, which only makes sense for a 2-monomer system: with N monomers in a
liquid box, "the AB dimer" is undefined (see
``mmml/models/mm_charge_mode.py``). This script sidesteps that by averaging
Mode B's ``neutralize_per_monomer(q_ML)`` over many *training-set* homo-dimer
forwards of one species, offline, once. The mean is a fixed per-atom charge
template for that monomer, which ``mm_charge_mode=latent_mean`` (Mode D) then
tiles across every monomer copy in a liquid box at MD setup time -- no live
q_ML, no dimer restriction, works with any ``lr_solver`` (mic/ewald/
nvalchemiops_pme).

Usage::

    python scripts/compute_latent_monomer_charges.py \\
        --checkpoint ./ckpts/mp2_nms/mp2nms_ewald \\
        --data /path/to/mp2_nms15_clean_train.npz \\
        --resid DCM \\
        --out ckpts/mp2_nms/latent_charge_template_DCM.npz

Then pass ``--mm-charge-mode latent_mean --mm-latent-charge-template
ckpts/mp2_nms/latent_charge_template_DCM.npz`` to an MD run
(``mmml/cli/run/md_system.py`` or the ``md-pbc-suite`` jaxmd/ase backends).

The dataset must be dimers (the same shape ``check_hybrid_train_md_parity.py``
consumes: ``Z``, ``R``, ``N``, ``mol_id``, ``res_name``) and the checkpoint
must have been trained with ``charges=True`` -- there is no latent charge to
average without a charge head.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _batch_from_structure(data, i):
    """Single-structure batch in the layout ``prepare_batches_jit`` produces.

    Duplicated from ``check_hybrid_train_md_parity.py`` (kept standalone
    deliberately -- these are one-off scripts, not a shared library).
    """
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
    }


def _homo_dimer_indices(data, resid: str, max_samples: int) -> list[int]:
    """Indices of ``resid,resid`` dimers (same-species pairs only)."""
    res = np.asarray(data["res_name"])
    target = f"{resid},{resid}"
    idx = [i for i, r in enumerate(res) if str(r).strip() == target]
    if not idx:
        available = sorted({str(r) for r in res})
        raise ValueError(
            f"no {target!r} homo-dimers found in dataset; available res_name "
            f"values include: {available[:20]}"
        )
    if max_samples > 0:
        idx = idx[:max_samples]
    return idx


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Trained PhysNet checkpoint dir (charges=True)")
    p.add_argument("--data", required=True, help="NPZ of dimers (e.g. mp2_nms15_clean_train.npz)")
    p.add_argument("--resid", required=True, help="Monomer species to average, e.g. DCM")
    p.add_argument("--out", required=True, help="Output template path (.npz)")
    p.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Cap on homo-dimer structures averaged over (0 = all). Default 500.",
    )
    args = p.parse_args()

    import jax.numpy as jnp

    from mmml.cli.misc.physnet_evaluate import _load_physnet_checkpoint
    from mmml.models.cgenff_mm import neutralize_per_monomer
    from mmml.models.latent_charge_template import (
        LatentChargeTemplate,
        save_latent_charge_template,
    )

    data = dict(np.load(args.data, allow_pickle=True))
    natoms = int(np.asarray(data["Z"]).shape[1])

    _, params, model = _load_physnet_checkpoint(Path(args.checkpoint), natoms)
    if not getattr(model, "charges", False):
        print(
            f"checkpoint {args.checkpoint} has no charge head (charges=False); "
            "there is no latent q_ML to average.",
            file=sys.stderr,
        )
        return 1

    idx = _homo_dimer_indices(data, args.resid, args.max_samples)
    print(f"averaging over {len(idx)} {args.resid},{args.resid} dimer(s)")

    samples: list[np.ndarray] = []
    monomer_Z: np.ndarray | None = None
    for i in idx:
        n_real = int(np.asarray(data["N"])[i])
        per_mono = n_real // 2
        mol_id_i = np.asarray(data["mol_id"])[i]
        Z_i = np.asarray(data["Z"])[i]

        z_mono = Z_i[:per_mono]
        if monomer_Z is None:
            monomer_Z = z_mono
        elif not np.array_equal(monomer_Z, z_mono):
            raise ValueError(
                f"structure {i}: monomer-A atomic numbers {z_mono.tolist()} disagree "
                f"with the first sample's {monomer_Z.tolist()} -- dataset atom "
                "ordering is not consistent for this species; cannot average a "
                "single fixed-order template."
            )

        batch = _batch_from_structure(data, i)
        out_ab = model.apply(
            params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=1,
            batch_mask=batch["batch_mask"],
            atom_mask=batch["atom_mask"],
        )
        q_ml = np.asarray(out_ab["charges"]).reshape(-1)[: batch["Z"].shape[0]]
        mol_id_j = jnp.asarray(mol_id_i)
        q_neutral = np.asarray(neutralize_per_monomer(jnp.asarray(q_ml), mol_id_j, n_monomers=2))
        samples.append(q_neutral[:per_mono])

    charges_arr = np.stack(samples, axis=0)  # (n_samples, per_mono)
    mean_charges = charges_arr.mean(axis=0)
    std_charges = charges_arr.std(axis=0)
    # Re-neutralize the mean itself: averaging n independently-neutral vectors
    # keeps the mean's sum at 0 up to fp noise, but project explicitly so the
    # saved template's net charge is exactly (not approximately) zero.
    mean_charges = mean_charges - mean_charges.mean()

    template = LatentChargeTemplate(
        atomic_numbers=monomer_Z,
        charges=mean_charges,
        charges_std=std_charges,
        n_samples=len(idx),
        resid=args.resid,
        source_checkpoint=str(args.checkpoint),
        source_data=str(args.data),
    )
    save_latent_charge_template(args.out, template)

    print(f"monomer Z:      {monomer_Z.tolist()}")
    print(f"mean charges:   {np.round(mean_charges, 4).tolist()}")
    print(f"std charges:    {np.round(std_charges, 4).tolist()}")
    print(f"net charge:     {mean_charges.sum():.6f} e (should be ~0)")
    print(f"wrote template: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
