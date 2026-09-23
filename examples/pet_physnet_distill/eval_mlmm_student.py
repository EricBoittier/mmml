"""Evaluate a PhysNet ML/MM student the way MLpot uses it.

MLpot runs one network on monomers and dimers and forms
``E_int = P(AB) - P(A) - P(B)`` (then switches it by ``s(r_com)``). A small
per-sample E MAE can still hide a poor E_int, so this splits every held-out
dimer into AB, A and B, predicts all three, and reports:

* per-kind/source E and F MAE on the stored labels (mlmm mode),
* E_int MAE per r_com bin and the switched ``s(r) * E_int`` error,
* ``eval.json``, ``predictions.npz`` and ``eint_vs_rcom.png`` in ``--out``.

Usage::

    python examples/pet_physnet_distill/eval_mlmm_student.py \\
      --checkpoint ckpts/etoh_omol_l_A-<uuid> --data valid.npz --out eval_A
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KCAL = 23.060549
BINS = (0.0, 3.5, 4.5, 5.25, 6.0, 7.5)


def _predict(model, params, R, Z, N, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Energies (eV) and forces (eV/Å) in input order; pads the last batch."""
    import jax

    from mmml.models.physnetjax.physnetjax.data.batches import prepare_batches_jit

    n, pad = R.shape[0], R.shape[1]
    extra = (-n) % batch_size
    ids = np.concatenate([np.arange(n), np.zeros(extra, dtype=int)])
    data = {
        "R": R[ids],
        "Z": Z[ids],
        "N": N[ids],
        "E": np.zeros((len(ids), 1)),
        "F": np.zeros((len(ids), pad, 3)),
        "id": np.arange(len(ids)),
    }
    batches = prepare_batches_jit(
        jax.random.PRNGKey(0),
        data,
        batch_size,
        data_keys=["R", "Z", "F", "E", "N", "id"],
        num_atoms=pad,
        include_id=True,
    )
    E = np.zeros(len(ids))
    F = np.zeros((len(ids), pad, 3))

    @jax.jit
    def apply(b):
        return model.apply(
            params,
            atomic_numbers=b["Z"],
            positions=b["R"],
            dst_idx=b["dst_idx"],
            src_idx=b["src_idx"],
            batch_segments=b["batch_segments"],
            batch_size=batch_size,
            batch_mask=b["batch_mask"],
            atom_mask=b["atom_mask"],
        )

    for b in batches:
        out = apply(b)
        idx = np.asarray(b["id"]).reshape(-1)
        E[idx] = np.asarray(out["energy"]).reshape(-1)
        F[idx] = np.asarray(out["forces"]).reshape(-1, pad, 3)
    return E[:n], F[:n]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="distill NPZ (mlmm labels)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--mm-switch-on", type=float, default=6.0)
    p.add_argument("--ml-switch-width", type=float, default=1.5)
    args = p.parse_args()

    from mmml.cli.misc.physnet_evaluate import _load_physnet_checkpoint
    from mmml.interfaces.pycharmmInterface.calculator_utils import ml_switch_scale

    d = np.load(args.data, allow_pickle=True)
    R, Z, N = d["R"], d["Z"], d["N"].reshape(-1)
    E_ref, F_ref = d["E"].reshape(-1), d["F"]
    kind, src, rc, E_int = d["kind"], d["source"], d["r_com"], d["E_int"]
    pad = R.shape[1]
    _, params, model = _load_physnet_checkpoint(args.checkpoint, pad, use_ema=True)

    E_pred, F_pred = _predict(model, params, R, Z, N, args.batch_size)
    mask = Z > 0
    report: dict = {"checkpoint": str(args.checkpoint), "data": str(args.data), "by_source": {}}
    for s in sorted(set(src)):
        m = src == s
        fm = mask[m]
        report["by_source"][str(s)] = {
            "n": int(m.sum()),
            "E_MAE_kcal": float(KCAL * np.mean(np.abs(E_pred[m] - E_ref[m]))),
            "F_MAE_kcal_A": float(KCAL * np.mean(np.abs(F_pred[m] - F_ref[m])[fm])),
        }

    # Fragments of every dimer: A = first half, B = second half (n_a = N/2, homo-dimers).
    dim = np.nonzero(kind == 1)[0]
    half = N[dim] // 2
    RA = np.zeros_like(R[dim]); RB = np.zeros_like(R[dim])
    ZA = np.zeros_like(Z[dim]); ZB = np.zeros_like(Z[dim])
    for k, (i, h) in enumerate(zip(dim, half)):
        RA[k, :h], ZA[k, :h] = R[i, :h], Z[i, :h]
        RB[k, :h], ZB[k, :h] = R[i, h : 2 * h], Z[i, h : 2 * h]
    EA, _ = _predict(model, params, RA, ZA, half, args.batch_size)
    EB, _ = _predict(model, params, RB, ZB, half, args.batch_size)
    eint_pred = E_pred[dim] - EA - EB
    eint_ref = E_int[dim]
    r = rc[dim]
    s = np.asarray(
        ml_switch_scale(r, mm_switch_on=args.mm_switch_on, ml_switch_width=args.ml_switch_width)
    )
    err = KCAL * (eint_pred - eint_ref)
    report["E_int"] = {
        "n_dimers": int(len(dim)),
        "MAE_kcal": float(np.mean(np.abs(err))),
        "switched_MAE_kcal": float(np.mean(np.abs(s * err))),
        "by_r_com_bin": {},
    }
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (r >= lo) & (r < hi)
        if m.any():
            report["E_int"]["by_r_com_bin"][f"{lo}-{hi}"] = {
                "n": int(m.sum()),
                "MAE_kcal": float(np.mean(np.abs(err[m]))),
                "switched_MAE_kcal": float(np.mean(np.abs(s[m] * err[m]))),
                "ref_mean_abs_kcal": float(np.mean(np.abs(KCAL * eint_ref[m]))),
            }

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "predictions.npz",
        E_ref=E_ref, E_pred=E_pred, source=src, kind=kind, N=N,
        F_ref=F_ref, F_pred=F_pred,
        dimer_index=dim, r_com=r, E_int_ref=eint_ref, E_int_pred=eint_pred, switch=s,
    )
    (args.out / "eval.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    lim = np.percentile(np.abs(KCAL * eint_ref), 99.5)
    ax[0].scatter(KCAL * eint_ref, KCAL * eint_pred, s=1, alpha=0.3)
    ax[0].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax[0].set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="teacher E_int / kcal/mol",
              ylabel="P(AB)-P(A)-P(B) / kcal/mol")
    ax[1].scatter(r, err, s=1, alpha=0.3)
    ax[1].plot(r[np.argsort(r)], np.zeros_like(r), "k-", lw=0.5)
    ax[1].axvspan(args.mm_switch_on - args.ml_switch_width, args.mm_switch_on, color="C1", alpha=0.15,
                  label="ML→MM taper")
    ax[1].set(xlabel="r_com / Å", ylabel="E_int error / kcal/mol", ylim=(-5, 5))
    ax[1].legend(frameon=False)
    fig.savefig(args.out / "eint_vs_rcom.png", dpi=140)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
