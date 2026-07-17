#!/usr/bin/env python3
"""Validate train full-box native Ewald == MD many-to-many Ewald.

Training (``lr_solver=ewald``) and MD (``periodic_external`` + ``lr_solver
ewald``) are meant to share one operator::

    E_MM = E_ewald(all atoms, q_CGenFF)   # no exclusions, no intra subtract

Mirrors ``check_nvalchemiops_train_md_pme_parity.py`` exactly, but for the
pure-JAX ``ewald_native`` backend (no external PME library, no CUDA
requirement -- so unlike the nvalchemiops variant, this script needs nothing
beyond the base install to run).

This script checks in layers:

1. **kernel** — train helper vs ``compute_native_ewald_coulomb`` (kcal/mol)
2. **e_mm** — ``hybrid_forward`` ``e_mm`` vs MD Coulomb in eV

No CHARMM required::

    python scripts/check_ewald_train_md_pme_parity.py \\
        --data /path/to/energies_forces_dipoles_test.npz \\
        --pme-box-length 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def compare_ewald_kernel_kcalmol(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    mol_id: np.ndarray,
    *,
    box_length_A: float,
    accuracy: float = 1e-6,
    real_space_cutoff_A: float | None = None,
) -> dict[str, float]:
    """Train hybrid Ewald helper vs MD NumPy wrapper (kcal/mol)."""
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.long_range_backend import (
        compute_native_ewald_coulomb,
    )
    from mmml.models.ewald_hybrid_coulomb import hybrid_ewald_coulomb_energy

    pos = np.asarray(positions_A, dtype=np.float64)
    q = np.asarray(charges_e, dtype=np.float64).reshape(-1)
    mid = np.asarray(mol_id, dtype=np.int32).reshape(-1)
    n = int(pos.shape[0])
    if q.shape[0] != n or mid.shape[0] != n:
        raise ValueError("positions/charges/mol_id length mismatch")

    e_train = float(
        np.asarray(
            hybrid_ewald_coulomb_energy(
                jnp.asarray(pos),
                jnp.asarray(mid),
                jnp.asarray(q),
                box_length_A=float(box_length_A),
                accuracy=float(accuracy),
                real_space_cutoff_A=real_space_cutoff_A,
            )
        )
    )
    md = compute_native_ewald_coulomb(
        pos,
        q,
        box_length_A=float(box_length_A),
        accuracy=float(accuracy),
        real_space_cutoff_A=real_space_cutoff_A,
    )
    e_md = float(md.energy_kcalmol)
    return {
        "e_train_kcalmol": e_train,
        "e_md_kcalmol": e_md,
        "abs_diff_kcalmol": abs(e_train - e_md),
    }


def compare_hybrid_emm_eV(
    data: dict,
    index: int,
    *,
    box_length_A: float,
    accuracy: float = 1e-6,
    real_space_cutoff_A: float | None = None,
    checkpoint: Path | None = None,
    ml_switch_width: float = 1.5,
    mm_switch_on: float = 8.0,
    mm_switch_width: float = 5.0,
) -> dict[str, float]:
    """``hybrid_forward`` ``e_mm`` (eV) vs MD full-box Coulomb (eV)."""
    import jax.numpy as jnp

    from mmml.data.units import KCAL_MOL_TO_EV
    from mmml.interfaces.pycharmmInterface.long_range_backend import (
        compute_native_ewald_coulomb,
    )
    from mmml.models.hybrid_energy import hybrid_forward

    i = int(index)
    Z = np.asarray(data["Z"])[i]
    R = np.asarray(data["R"])[i]
    mid = np.asarray(data["mol_id"])[i]
    q = np.asarray(data["cgenff_charge"])[i]
    tidx = np.asarray(data["cgenff_type_idx"])[i]
    n = int(np.asarray(data["N"])[i]) if "N" in data else int((Z > 0).sum())
    Z, R, mid, q, tidx = Z[:n], R[:n], mid[:n], q[:n], tidx[:n]

    md = compute_native_ewald_coulomb(
        R,
        q,
        box_length_A=float(box_length_A),
        accuracy=float(accuracy),
        real_space_cutoff_A=real_space_cutoff_A,
    )
    e_md_eV = float(md.energy_kcalmol) * KCAL_MOL_TO_EV

    sigmas = jnp.asarray(data["cgenff_master_sigmas"])
    epsilons = jnp.asarray(data["cgenff_master_epsilons"])

    if checkpoint is not None:
        from mmml.cli.misc.physnet_evaluate import _load_physnet_checkpoint

        _, params, model = _load_physnet_checkpoint(Path(checkpoint), int(Z.shape[0]))
        model_apply = model.apply
    else:
        # Analytic stub: enough to exercise e_mm wiring without a checkpoint.
        def model_apply(params, *, atomic_numbers, positions, dst_idx, src_idx,
                        batch_segments, batch_size, batch_mask, atom_mask):
            e = jnp.zeros((batch_size, 1), dtype=positions.dtype)
            f = jnp.zeros_like(positions)
            return {"energy": e, "forces": f}

        params = {}

    atom_mask = (Z > 0).astype(np.float32)
    idx = np.arange(n)
    dst, src = np.meshgrid(idx, idx, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    keep = (dst != src) & (atom_mask[dst] > 0) & (atom_mask[src] > 0)
    batch = {
        "R": jnp.asarray(R),
        "Z": jnp.asarray(Z),
        "atom_mask": jnp.asarray(atom_mask),
        "batch_mask": jnp.asarray(keep.astype(np.float32)),
        "dst_idx": jnp.asarray(dst),
        "src_idx": jnp.asarray(src),
        "batch_segments": jnp.zeros(n, dtype=jnp.int32),
        "mol_id": jnp.asarray(mid[None, :]),
        "cgenff_type_idx": jnp.asarray(tidx[None, :]),
        "cgenff_charge": jnp.asarray(q[None, :]),
    }
    out = hybrid_forward(
        model_apply,
        params,
        batch,
        1,
        sigmas,
        epsilons,
        mm_switch_on=float(mm_switch_on),
        mm_switch_width=float(mm_switch_width),
        ml_switch_width=float(ml_switch_width),
        mm_charge_mode="fixed",
        short_range_wall=False,
        lr_solver="ewald",
        include_lj=False,
        pme_box_length=float(box_length_A),
        pme_accuracy=float(accuracy),
        pme_real_space_cutoff=real_space_cutoff_A,
    )
    e_mm_train = float(np.asarray(out["e_mm"]).reshape(-1)[0])
    return {
        "e_mm_train_eV": e_mm_train,
        "e_md_coulomb_eV": e_md_eV,
        "abs_diff_eV": abs(e_mm_train - e_md_eV),
    }


def _pick_indices(data: dict, n: int) -> list[int]:
    """Prefer dimers if ``res_name`` looks like A,B; else first ``n`` frames."""
    n_tot = int(np.asarray(data["Z"]).shape[0])
    if "res_name" in data:
        res = np.asarray(data["res_name"])
        dimers = [i for i in range(n_tot) if "," in str(res[i])]
        if dimers:
            return dimers[:n]
    return list(range(min(n, n_tot)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="NPZ with CGenFF fields")
    p.add_argument("--pme-box-length", type=float, required=True, help="Cubic box (Å)")
    p.add_argument("--pme-accuracy", type=float, default=1e-6)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Optional PhysNet checkpoint for hybrid_forward (layer 2)",
    )
    p.add_argument("--n-structures", type=int, default=4)
    p.add_argument("--tol-kcal", type=float, default=1e-5, help="kernel abs tol (kcal/mol)")
    p.add_argument("--tol-eV", type=float, default=1e-4, help="e_mm abs tol (eV)")
    p.add_argument("--ml-switch-width", type=float, default=1.5)
    p.add_argument("--mm-switch-on", type=float, default=8.0)
    p.add_argument("--mm-switch-width", type=float, default=5.0)
    args = p.parse_args(argv)

    data = dict(np.load(args.data, allow_pickle=True))
    for key in (
        "Z",
        "R",
        "mol_id",
        "cgenff_charge",
        "cgenff_type_idx",
        "cgenff_master_sigmas",
        "cgenff_master_epsilons",
    ):
        if key not in data:
            print(f"NPZ missing {key}", file=sys.stderr)
            return 2

    indices = _pick_indices(data, int(args.n_structures))
    if not indices:
        print("no structures to check", file=sys.stderr)
        return 2

    print(
        f"ewald train<->MD parity | box={args.pme_box_length} A | "
        f"accuracy={args.pme_accuracy} | n={len(indices)}"
    )
    print(
        f"{'idx':>5} {'|dE_kernel|':>12} {'e_mm(tr)':>12} {'E_coul(md)':>12} "
        f"{'|dE_mm|':>12}  status"
    )

    bad = 0
    worst_k = 0.0
    worst_mm = 0.0
    ckpt = Path(args.checkpoint) if args.checkpoint else None

    for i in indices:
        Z = np.asarray(data["Z"])[i]
        R = np.asarray(data["R"])[i]
        mid = np.asarray(data["mol_id"])[i]
        q = np.asarray(data["cgenff_charge"])[i]
        n = int(np.asarray(data["N"])[i]) if "N" in data else int((Z > 0).sum())
        try:
            k = compare_ewald_kernel_kcalmol(
                R[:n],
                q[:n],
                mid[:n],
                box_length_A=float(args.pme_box_length),
                accuracy=float(args.pme_accuracy),
            )
            mm = compare_hybrid_emm_eV(
                data,
                i,
                box_length_A=float(args.pme_box_length),
                accuracy=float(args.pme_accuracy),
                checkpoint=ckpt,
                ml_switch_width=float(args.ml_switch_width),
                mm_switch_on=float(args.mm_switch_on),
                mm_switch_width=float(args.mm_switch_width),
            )
        except Exception as exc:
            print(f"{i:>5}  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            bad += 1
            continue

        dk = float(k["abs_diff_kcalmol"])
        dmm = float(mm["abs_diff_eV"])
        worst_k = max(worst_k, dk)
        worst_mm = max(worst_mm, dmm)
        ok = (dk <= float(args.tol_kcal)) and (dmm <= float(args.tol_eV))
        bad += not ok
        print(
            f"{i:>5} {dk:>12.3e} {mm['e_mm_train_eV']:>12.6f} "
            f"{mm['e_md_coulomb_eV']:>12.6f} {dmm:>12.3e}  "
            f"{'OK' if ok else 'FAIL'}"
        )

    print(
        f"\nworst |E_train - E_md|_kernel = {worst_k:.3e} kcal/mol "
        f"(tol {args.tol_kcal})"
    )
    print(f"worst |e_mm - E_coul|_eV     = {worst_mm:.3e} eV (tol {args.tol_eV})")

    if bad:
        print(f"PARITY FAILED on {bad}/{len(indices)} structures", file=sys.stderr)
        return 1
    print(f"PARITY OK on {len(indices)} structures: train ewald == MD ewald")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
