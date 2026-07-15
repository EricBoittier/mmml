#!/usr/bin/env python3
"""Evaluate a trained SpookyPhysNet checkpoint on a prepared Orbax cache directly.

evaluate_so3lr_spooky_extxyz.py is gated on parsing an .extxyz file, but the ML/MM
splits are already Orbax caches with the same schema. This wrapper reuses that script's
model loading and evaluate_dataset() on an existing cache (e.g. a held-out test_cache),
so the top models can be scored on real held-out structures, not just the dimer grid.

Usage:
    python scripts/eval_spooky_on_cache.py \
        --checkpoint artifacts/.../step-00025000 \
        --cache /path/to/splits_des_ml_mm/test_cache \
        --output eval.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import jax
import numpy as np

_SCRIPT = Path(__file__).resolve().parent / "evaluate_so3lr_spooky_extxyz.py"
_spec = importlib.util.spec_from_file_location("evaluate_so3lr_spooky_extxyz", _SCRIPT)
_ev = importlib.util.module_from_spec(_spec)
# flax's dataclass transform resolves the defining module via sys.modules; register
# before exec so frozen dataclasses in the evaluator can be processed.
sys.modules[_spec.name] = _ev
_spec.loader.exec_module(_ev)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True, help="Prepared Orbax cache (e.g. test_cache)")
    p.add_argument("--output", help="JSON summary path")
    p.add_argument("--batch-size-per-device", type=int, default=4)
    p.add_argument("--max-pairs-per-device", type=int, default=18000)
    p.add_argument("--num-devices", type=int, default=1)
    p.add_argument("--max-eval-structures", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--plot-max-atoms", type=int, default=40)
    p.add_argument("--plots-dir", type=Path, default=None)
    args = p.parse_args()

    params, config = _ev.restore_checkpoint(Path(args.checkpoint).resolve())
    from mmml.utils.model_checkpoint import infer_trainable_zbl_config

    config = infer_trainable_zbl_config(config, params)
    args.predict_charges = config.get("predict_charges", config.get("charges", False))

    data, metadata = _ev.restore_cached_data(Path(args.cache).resolve())
    data, metadata = _ev.limit_cached_data(data, metadata, args.max_eval_structures)
    max_atoms = int(np.max(np.asarray(data["N"]).reshape(-1)))

    devices = jax.local_devices()[: args.num_devices]
    if len(devices) != args.num_devices:
        raise RuntimeError(f"Requested {args.num_devices} devices, JAX sees {len(jax.local_devices())}")
    if not any(d.platform == "gpu" for d in devices):
        raise RuntimeError(f"No GPU device found (got {devices}); refusing CPU eval.")

    model = _ev.create_model_from_config(config, max_atoms=max_atoms)
    metrics, _ = _ev.evaluate_dataset(model, params, data, metadata, args, devices)

    n = int(np.asarray(data["N"]).reshape(-1).shape[0])
    print("\n=== held-out test-set metrics ===")
    print(f"checkpoint : {args.checkpoint}")
    print(f"cache      : {args.cache}  ({n:,} structures)")
    for k, v in metrics.items():
        print(f"  {k:<24} {v:.5f}")

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"checkpoint": args.checkpoint, "cache": args.cache,
             "n_structures": n, "metrics": metrics}, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
