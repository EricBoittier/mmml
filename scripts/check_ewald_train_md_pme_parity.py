#!/usr/bin/env python3
"""Validate train full-box native Ewald (+ optional switched LJ) == MD reference.

Training (``lr_solver=ewald``) and MD share::

    E_MM = E_ewald(all atoms, q_CGenFF)           # untapered full-box Coulomb
         + λ_MM(R) * E_LJ(σ_eff, ε_eff)          # when --include-lj

With ``--include-lj`` / ``--no-include-lj`` (default: off for Coulomb-only
backward compatibility). The MD side of layer 2 is a composed reference —
``compute_native_ewald_coulomb`` plus the same COM-switched LJ helper training
uses — so this check needs no CHARMM / no GPU::

    python scripts/check_ewald_train_md_pme_parity.py \\
        --data /path/to/energies_forces_dipoles_test.npz \\
        --pme-box-length 30

    python scripts/check_ewald_train_md_pme_parity.py \\
        --data ... --pme-box-length 30 --include-lj
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


def md_reference_emm_eV(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    mol_id: np.ndarray,
    type_idx: np.ndarray,
    master_sigmas: np.ndarray,
    master_epsilons: np.ndarray,
    *,
    box_length_A: float,
    accuracy: float = 1e-6,
    real_space_cutoff_A: float | None = None,
    include_lj: bool = False,
    sigma_scale: np.ndarray | None = None,
    epsilon_scale: np.ndarray | None = None,
    ml_switch_width: float = 1.5,
    mm_switch_on: float = 8.0,
    mm_switch_width: float = 5.0,
    complementary_handoff: bool = True,
) -> dict[str, float]:
    """Composed MD reference: native Ewald Coulomb + optional switched LJ (eV)."""
    import jax.numpy as jnp

    from mmml.data.units import KCAL_MOL_TO_EV
    from mmml.interfaces.pycharmmInterface.long_range_backend import (
        compute_native_ewald_coulomb,
    )
    from mmml.models.hybrid_energy import switched_lj_kcal

    pos = np.asarray(positions_A, dtype=np.float64)
    q = np.asarray(charges_e, dtype=np.float64).reshape(-1)
    mid = np.asarray(mol_id, dtype=np.int32).reshape(-1)
    tidx = np.asarray(type_idx, dtype=np.int32).reshape(-1)

    md = compute_native_ewald_coulomb(
        pos,
        q,
        box_length_A=float(box_length_A),
        accuracy=float(accuracy),
        real_space_cutoff_A=real_space_cutoff_A,
    )
    e_coul_kcal = float(md.energy_kcalmol)
    e_lj_kcal = 0.0
    if include_lj:
        sig_s = (
            None
            if sigma_scale is None
            else jnp.asarray(sigma_scale, dtype=jnp.float64)
        )
        eps_s = (
            None
            if epsilon_scale is None
            else jnp.asarray(epsilon_scale, dtype=jnp.float64)
        )
        e_lj_kcal = float(
            np.asarray(
                switched_lj_kcal(
                    jnp.asarray(pos),
                    jnp.asarray(tidx),
                    jnp.asarray(mid),
                    jnp.asarray(master_sigmas, dtype=jnp.float64),
                    jnp.asarray(master_epsilons, dtype=jnp.float64),
                    sigma_scale=sig_s,
                    epsilon_scale=eps_s,
                    include_lj=True,
                    mm_switch_on=float(mm_switch_on),
                    mm_switch_width=float(mm_switch_width),
                    ml_switch_width=float(ml_switch_width),
                    complementary_handoff=bool(complementary_handoff),
                )
            )
        )
    e_md_eV = (e_coul_kcal + e_lj_kcal) * KCAL_MOL_TO_EV
    return {
        "e_md_coulomb_kcalmol": e_coul_kcal,
        "e_md_lj_kcalmol": e_lj_kcal,
        "e_md_eV": e_md_eV,
    }


def compare_hybrid_emm_eV(
    data: dict,
    index: int,
    *,
    box_length_A: float,
    accuracy: float = 1e-6,
    real_space_cutoff_A: float | None = None,
    checkpoint: Path | None = None,
    include_lj: bool = False,
    sigma_scale: np.ndarray | None = None,
    epsilon_scale: np.ndarray | None = None,
    ml_switch_width: float = 1.5,
    mm_switch_on: float = 8.0,
    mm_switch_width: float = 5.0,
    complementary_handoff: bool = True,
) -> dict[str, float]:
    """``hybrid_forward`` ``e_mm`` (eV) vs composed MD Ewald(+LJ) reference (eV)."""
    import jax.numpy as jnp

    from mmml.models.hybrid_energy import hybrid_forward

    i = int(index)
    Z = np.asarray(data["Z"])[i]
    R = np.asarray(data["R"])[i]
    mid = np.asarray(data["mol_id"])[i]
    q = np.asarray(data["cgenff_charge"])[i]
    tidx = np.asarray(data["cgenff_type_idx"])[i]
    n = int(np.asarray(data["N"])[i]) if "N" in data else int((Z > 0).sum())
    Z, R, mid, q, tidx = Z[:n], R[:n], mid[:n], q[:n], tidx[:n]

    sigmas = np.asarray(data["cgenff_master_sigmas"])
    epsilons = np.asarray(data["cgenff_master_epsilons"])
    md_ref = md_reference_emm_eV(
        R,
        q,
        mid,
        tidx,
        sigmas,
        epsilons,
        box_length_A=float(box_length_A),
        accuracy=float(accuracy),
        real_space_cutoff_A=real_space_cutoff_A,
        include_lj=bool(include_lj),
        sigma_scale=sigma_scale,
        epsilon_scale=epsilon_scale,
        ml_switch_width=float(ml_switch_width),
        mm_switch_on=float(mm_switch_on),
        mm_switch_width=float(mm_switch_width),
        complementary_handoff=bool(complementary_handoff),
    )
    e_md_eV = float(md_ref["e_md_eV"])

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
    sig_j = jnp.asarray(sigmas)
    eps_j = jnp.asarray(epsilons)
    scale_kw = {}
    if sigma_scale is not None:
        scale_kw["mm_lj_sigma_scale"] = jnp.asarray(sigma_scale)
    if epsilon_scale is not None:
        scale_kw["mm_lj_epsilon_scale"] = jnp.asarray(epsilon_scale)
    out = hybrid_forward(
        model_apply,
        params,
        batch,
        1,
        sig_j,
        eps_j,
        mm_switch_on=float(mm_switch_on),
        mm_switch_width=float(mm_switch_width),
        ml_switch_width=float(ml_switch_width),
        complementary_handoff=bool(complementary_handoff),
        mm_charge_mode="fixed",
        short_range_wall=False,
        lr_solver="ewald",
        include_lj=bool(include_lj),
        pme_box_length=float(box_length_A),
        pme_accuracy=float(accuracy),
        pme_real_space_cutoff=real_space_cutoff_A,
        **scale_kw,
    )
    e_mm_train = float(np.asarray(out["e_mm"]).reshape(-1)[0])
    return {
        "e_mm_train_eV": e_mm_train,
        "e_md_coulomb_eV": e_md_eV,  # full MD ref (name kept for CLI table)
        "e_md_ref_eV": e_md_eV,
        "e_md_lj_kcalmol": float(md_ref["e_md_lj_kcalmol"]),
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
    p.add_argument(
        "--include-lj",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compare Ewald Coulomb + COM-switched LJ (default: Coulomb-only).",
    )
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

    lj_txt = "Ewald+LJ" if args.include_lj else "Ewald Coulomb-only"
    print(
        f"ewald train<->MD parity ({lj_txt}) | box={args.pme_box_length} A | "
        f"accuracy={args.pme_accuracy} | n={len(indices)}"
    )
    print(
        f"{'idx':>5} {'|dE_kernel|':>12} {'e_mm(tr)':>12} {'E_md(ref)':>12} "
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
                include_lj=bool(args.include_lj),
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
    print(f"worst |e_mm - E_md_ref|_eV   = {worst_mm:.3e} eV (tol {args.tol_eV})")

    if bad:
        print(f"PARITY FAILED on {bad}/{len(indices)} structures", file=sys.stderr)
        return 1
    print(f"PARITY OK on {len(indices)} structures: train ewald == MD ref ({lj_txt})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
