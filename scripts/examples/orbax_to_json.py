#!/usr/bin/env python3
"""
Convert orbax checkpoints to JSON for CPU-only and custom precision (e.g. float64).

This enables running inference on CPU without GPU, and using float64 for
numerical stability. The conversion must be run on a machine that can load
the checkpoint (GPU-saved checkpoints require GPU for conversion).

Usage:
    # Convert orbax checkpoint to JSON (includes model config from model_attributes)
    python scripts/examples/orbax_to_json.py \\
        --checkpoint mmml/physnetjax/ckpts/DESdimers/final2 \\
        --output ckpts_json/DESdimers_params.json

    # Use with run_sim: pass the JSON file or a directory containing params.json
    args = argparse.Namespace(checkpoint=Path("ckpts_json/DESdimers_params.json"), ...)
    run(args)

    # Or load directly (works on CPU, any precision)
    from mmml.utils.model_checkpoint import json_to_params, load_model_checkpoint

    ckpt = json_to_params("ckpts_json/DESdimers_params.json", dtype="float64")
    ckpt = load_model_checkpoint("ckpts_json/", dtype="float64")  # dir with params.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mmml.utils.model_checkpoint import orbax_to_json


def main():
    parser = argparse.ArgumentParser(
        description="Convert orbax checkpoint to JSON for CPU/float64 use."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to orbax checkpoint directory (e.g. mmml/physnetjax/ckpts/DESdimers/final2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path (e.g. params.json)",
    )
    parser.add_argument(
        "--params-key",
        type=str,
        default="params",
        help="Key for params in checkpoint dict (default: params)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    try:
        out = orbax_to_json(
            orbax_checkpoint_dir=checkpoint_path,
            output_path=output_path,
            params_key=args.params_key,
        )
        print(f"✓ Saved JSON checkpoint to {out}")
        print(f"  Load with: json_to_params('{out}', dtype='float64')")
    except Exception as e:
        print(f"Error converting checkpoint: {e}")
        if "cuda" in str(e).lower() or "sharding" in str(e).lower():
            print(
                "\nNote: GPU-saved checkpoints may require running this script on a "
                "machine with GPU. Alternatively, run the conversion on the same "
                "machine where the checkpoint was created."
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
