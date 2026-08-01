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
    """Locate hybrid_mm.json: explicit, beside the params, or in a run subdir."""
    if explicit is not None:
        return explicit if Path(explicit).is_file() else None
    root = Path(params_path).parent
    direct = root / "hybrid_mm.json"
    if direct.is_file():
        return direct
    hits = sorted(root.glob("*/hybrid_mm.json"))
    return hits[-1] if hits else None


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
    import jax.numpy as jnp

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

    e_err, f_err = [], []
    for batch in batches:
        out = _eval_forward(model.apply, params, batch, args.batch_size, hybrid_mm)
        e_err.append(np.asarray(out["energy"]).ravel() - np.asarray(batch["E"]).ravel())
        mask = np.asarray(batch["atom_mask"]).astype(bool)
        fp = np.asarray(out["forces"])[mask]
        ft = np.asarray(batch["F"])[mask]
        f_err.append((fp - ft).ravel())

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
