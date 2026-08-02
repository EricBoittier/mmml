"""Acceptance gate: validation E/F accuracy of a hybrid ML/MM checkpoint, in kcal/mol.

Why this exists
---------------
The training log's ``Energy MAE`` / ``Forces MAE`` columns are **eV**, not
kcal/mol. ``train_model`` defaults to ``CONVERSION`` (eV -> kcal/mol), but the
CLI overrides it with an identity dict::

    conversion = _parse_dict_option(args.conversion) or {'energy': 1, 'forces': 1}
                                                        ^ make_training.py:1436

so the in-loop comment "convert statistics to kcal/mol for printing" does not
describe what actually happens. A reported 0.163 is 3.76 kcal/mol, not 0.163.

Training also never computes RMSE, only MAE. RMSE >= MAE always, and it is the
metric that notices a few bad structures, so a target expressed on both needs
both measured.

This script reports MAE **and** RMSE for energies and forces, in kcal/mol and
kcal/mol/A, on the same validation split the run used, scoring the same hybrid
quantity training optimised (via ``_eval_forward``).

Usage
-----
    uv run python -m mmml.cli.misc.eval_hybrid_accuracy \\
        --params  artifacts/.../params_<tag>_<stamp>.json \\
        --data    artifacts/.../des_dimers_cgenff_top50.npz \\
        --config  examples/lj_scales/train_des_warmstart.yaml \\
        --n-train 100000 --n-valid 8500 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# eV -> kcal/mol. Same factor the trainer's CONVERSION uses.
EV_TO_KCAL_MOL = 23.060548012069496

TARGET_KCAL = 1.0


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--params", required=True, type=Path, help="portable params JSON")
    p.add_argument("--data", required=True, help="training NPZ (same one the run used)")
    p.add_argument("--config", type=Path, default=None, help="training YAML, for hybrid settings")
    p.add_argument("--hybrid-mm-json", type=Path, default=None,
                   help="hybrid_mm.json holding the learned LJ scales "
                        "(default: found next to --params)")
    p.add_argument("--n-train", type=int, default=100000)
    p.add_argument("--n-valid", type=int, default=8500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--target", type=float, default=TARGET_KCAL,
                   help="pass/fail threshold in kcal/mol (and kcal/mol/A)")
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args(argv)


def _find_sidecar(params_path: Path, explicit: Path | None) -> Path | None:
    """Locate hybrid_mm.json: explicit, or unambiguously beside the params.

    Deliberately refuses to guess between several run directories. A ckpt dir
    accumulates one per run, and silently picking the wrong run's LJ scales
    would report accuracy for a parameter set that never existed -- the exact
    error this script is meant to rule out. Newest-wins and alphabetical are
    both wrong: the newest may be a *different* run still training.
    """
    if explicit is not None:
        if not Path(explicit).is_file():
            raise SystemExit(f"--hybrid-mm-json not found: {explicit}")
        return Path(explicit)

    root = Path(params_path).parent
    direct = root / "hybrid_mm.json"
    if direct.is_file():
        return direct

    hits = sorted(root.glob("*/hybrid_mm.json"))
    if len(hits) > 1:
        listed = "\n  ".join(str(h) for h in hits)
        raise SystemExit(
            f"{len(hits)} candidate hybrid_mm.json files under {root}; refusing "
            f"to guess which run produced {params_path.name}.\n"
            f"Pass --hybrid-mm-json explicitly:\n  {listed}"
        )
    return hits[0] if hits else None


def _summarize(err, n_label):
    """MAE/RMSE of a flat error array, already in kcal/mol units."""
    err = np.asarray(err, dtype=np.float64).ravel()
    return {
        "n": int(err.size),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "max_abs": float(np.abs(err).max()) if err.size else float("nan"),
        "label": n_label,
    }


def main(argv=None) -> int:
    args = _parse_args(argv)

    import jax

    from mmml.models.physnetjax.checkpoint_utils import load_physnet_checkpoint
    from mmml.models.physnetjax.physnetjax.data.data import prepare_datasets
    from mmml.models.physnetjax.physnetjax.training.evalstep import _eval_forward

    params, config = load_physnet_checkpoint(args.params)
    print(f"loaded params: {args.params}")

    # The portable checkpoint stores the bare flax tree; `model.apply` wants the
    # variables dict, i.e. {"params": tree}.
    if "params" not in params:
        params = {"params": params}

    # The learned LJ scales are NOT in the portable checkpoint -- they live in
    # the hybrid_mm.json sidecar. Without them this would score the *default*
    # CGenFF LJ and silently report the accuracy of a model that was never
    # trained, so a missing sidecar is fatal rather than a warning.
    if hybrid_sidecar := _find_sidecar(args.params, args.hybrid_mm_json):
        side = json.loads(Path(hybrid_sidecar).read_text())
        for key in ("mm_lj_sigma_scale", "mm_lj_epsilon_scale"):
            if key in side:
                params[key] = np.asarray(side[key], dtype=np.float32)
        print(f"attached learned LJ scales from {hybrid_sidecar}")
    else:
        raise SystemExit(
            "Could not find hybrid_mm.json next to the checkpoint. Pass "
            "--hybrid-mm-json; without the learned LJ scales this would score "
            "default CGenFF LJ and misreport accuracy."
        )

    # natoms must come from the checkpoint: prepare_datasets defaults to 60 and
    # reshapes coordinates to (natoms, 3), so a wrong value fails outright.
    natoms = int(config["natoms"])
    print(f"natoms (from checkpoint): {natoms}")

    # Same split the run used: data_key is the first half of split(PRNGKey(seed)).
    data_key, _train_key = jax.random.split(jax.random.PRNGKey(int(args.seed)), 2)
    _train_data, valid_data = prepare_datasets(
        data_key, args.n_train, args.n_valid, [str(args.data)], natoms=natoms
    )
    n_valid = len(valid_data["E"])
    print(f"validation frames: {n_valid}")

    # Rebuild the hybrid settings with the trainer's own builder rather than by
    # hand: the dict has ~20 keys (master LJ tables, switch widths, scale bounds,
    # trainable mask) and scoring a different hybrid quantity than training
    # optimised would silently report the wrong accuracy.
    hybrid_mm = None
    if args.config is not None:
        import yaml

        from mmml.cli.make.make_training import _build_hybrid_mm_config, build_parser

        cfg = yaml.safe_load(args.config.read_text()) or {}
        ns = build_parser().parse_args([])
        for key, value in cfg.items():
            if hasattr(ns, key):
                setattr(ns, key, value)
        # Architecture that actually shipped in the checkpoint wins over the YAML.
        for key in ("cutoff",):
            if key in config:
                setattr(ns, key, config[key])
        hybrid_mm = _build_hybrid_mm_config(ns, [str(args.data)])
        if hybrid_mm is not None:
            print(f"hybrid ML/MM scoring enabled ({hybrid_mm['hybrid_hamiltonian']})")

    from mmml.models.physnetjax.physnetjax.data.batches import _pair_indices, _prepare_batches
    from mmml.models.physnetjax.physnetjax.models.model import EF

    model = EF(**{k: v for k, v in config.items() if k in EF.__dataclass_fields__})

    # Same "fat" batching the trainer's default path uses.
    keys = ["R", "Z", "F", "N", "E", "D", "batch_segments",
            "cgenff_type_idx", "mol_id", "cgenff_charge"]
    keys = [k for k in keys if k in valid_data or k == "batch_segments"]
    batches = _prepare_batches(
        data_key,
        data=valid_data,
        batch_size=args.batch_size,
        num_atoms=natoms,
        data_keys=keys,
        pair_cache=_pair_indices(natoms, args.batch_size),
    )

    # Uncompiled, the hybrid forward retraces per batch and 8.5k frames take
    # >10 min; one compile covers all batches (they share shapes).
    _fwd = jax.jit(
        lambda p, b: _eval_forward(model.apply, p, b, args.batch_size, hybrid_mm)
    )

    e_err, f_err = [], []
    per_frame = []  # (|dE| kcal/mol, max|dF| kcal/mol/A, min inter-contact A)
    frame_de, frame_df, frame_dmin = [], [], []  # for threshold sweep
    for batch in batches:
        out = _fwd(params, batch)
        de = np.asarray(out["energy"]).ravel() - np.asarray(batch["E"]).ravel()
        e_err.append(de)
        mask = np.asarray(batch["atom_mask"]).astype(bool)
        fp = np.asarray(out["forces"])[mask]
        ft = np.asarray(batch["F"])[mask]
        f_err.append((fp - ft).ravel())

        # Per-frame worst-atom force error, to locate the RMSE tail.
        fp_b = np.asarray(out["forces"]).reshape(args.batch_size, -1, 3)
        ft_b = np.asarray(batch["F"]).reshape(args.batch_size, -1, 3)
        m_b = np.asarray(batch["atom_mask"]).reshape(args.batch_size, -1).astype(bool)
        r_b = np.asarray(batch["R"]).reshape(args.batch_size, -1, 3)
        mol_b = (np.asarray(batch["mol_id"]).reshape(args.batch_size, -1)
                 if "mol_id" in batch else None)
        for k in range(args.batch_size):
            m = m_b[k]
            if not m.any():
                continue
            dfk = np.linalg.norm((fp_b[k] - ft_b[k])[m], axis=1).max() * EV_TO_KCAL_MOL
            dmin = np.inf
            if mol_b is not None:
                r, mi = r_b[k][m], mol_b[k][m]
                a, b = np.where(mi == 0)[0], np.where(mi == 1)[0]
                if len(a) and len(b):
                    dmin = float(np.linalg.norm(
                        r[a][:, None, :] - r[b][None, :, :], axis=-1).min())
            per_frame.append((abs(de[k]) * EV_TO_KCAL_MOL, dfk, dmin))
            frame_de.append(de[k] * EV_TO_KCAL_MOL)
            frame_df.append(
                np.linalg.norm((fp_b[k] - ft_b[k])[m], axis=1) * EV_TO_KCAL_MOL
            )
            frame_dmin.append(dmin)

    energy = _summarize(np.concatenate(e_err) * EV_TO_KCAL_MOL, "kcal/mol")
    forces = _summarize(np.concatenate(f_err) * EV_TO_KCAL_MOL, "kcal/mol/A")

    target = float(args.target)
    print()
    print(f"{'metric':<22}{'MAE':>10}{'RMSE':>10}{'max|err|':>12}   unit")
    for name, s in (("energy", energy), ("forces", forces)):
        print(f"{name:<22}{s['mae']:>10.3f}{s['rmse']:>10.3f}"
              f"{s['max_abs']:>12.3f}   {s['label']}")

    print()
    failures = []
    for name, s in (("energy", energy), ("forces", forces)):
        for stat in ("mae", "rmse"):
            ok = s[stat] < target
            print(f"  {name:<8} {stat.upper():<5} {s[stat]:>9.3f} "
                  f"{'PASS' if ok else 'FAIL'} (target < {target:g})")
            if not ok:
                failures.append(f"{name}.{stat}={s[stat]:.3f}")

    # Where does the RMSE tail live? If the worst frames are close contacts,
    # the fix is the short-range handoff, not more training.
    if per_frame:
        pf = np.array(per_frame, dtype=np.float64)
        order = np.argsort(pf[:, 1])[::-1]
        print("\nworst 10 frames by max force error:")
        print(f"  {'|dE| kcal/mol':>14}{'max|dF| kcal/mol/A':>21}{'min contact A':>15}")
        for i in order[:10]:
            print(f"  {pf[i,0]:>14.3f}{pf[i,1]:>21.3f}{pf[i,2]:>15.3f}")

        tail = pf[order[: max(1, len(pf) // 100)]]
        rest = pf[order[max(1, len(pf) // 100):]]
        print(f"\n  worst 1% of frames: mean min-contact = {np.nanmean(tail[:,2]):.3f} A")
        print(f"  other 99%         : mean min-contact = {np.nanmean(rest[:,2]):.3f} A")
        f_sq = np.concatenate(f_err) ** 2
        share = np.sort(f_sq)[::-1][: max(1, f_sq.size // 1000)].sum() / f_sq.sum()
        print(f"  top 0.1% of force components own {100*share:.1f}% of the force MSE")

    # If close contacts own the tail, how much does excluding them buy? These
    # are *validation* metrics on a filtered subset -- an estimate of the
    # achievable target, not a result you can claim without retraining.
    if frame_dmin:
        dmin_a = np.asarray(frame_dmin)
        de_a = np.asarray(frame_de)
        print(f"\n{'min-contact cut':>16}{'frames kept':>13}{'E MAE':>9}"
              f"{'E RMSE':>9}{'F MAE':>9}{'F RMSE':>9}")
        for cut in (0.0, 1.0, 1.5, 2.0, 2.5):
            keep = dmin_a >= cut
            if keep.sum() < 10:
                continue
            de_k = de_a[keep]
            df_k = np.concatenate([frame_df[i] for i in np.where(keep)[0]])
            print(f"{cut:>16.1f}{keep.sum():>13d}"
                  f"{np.abs(de_k).mean():>9.3f}{np.sqrt((de_k**2).mean()):>9.3f}"
                  f"{np.abs(df_k).mean():>9.3f}{np.sqrt((df_k**2).mean()):>9.3f}")
        print(f"  ({len(dmin_a)} frames total; cut=0.0 is the unfiltered row)")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"params": str(args.params), "n_valid": n_valid, "target_kcal": target,
             "energy": energy, "forces": forces, "failures": failures}, indent=2))
        print(f"\nwrote {args.json_out}")

    if failures:
        print(f"\nBELOW TARGET: {', '.join(failures)}")
        return 1
    print(f"\nAll four metrics under {target:g} kcal/mol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
