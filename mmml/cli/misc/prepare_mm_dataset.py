#!/usr/bin/env python3
"""Assign CGenFF atom types / charges to a dimer training NPZ for hybrid ML/MM.

Enriches a dense, padded training NPZ (``R (n,atoms,3)``, ``Z (n,atoms)``, ``N``,
optionally ``F``/``E``/``D``/...) with the per-atom fields the hybrid ML/MM trainer
consumes:

* ``cgenff_type_idx`` -- index into the master LJ tables (``-1`` marks padding)
* ``mol_id``          -- monomer id 0/1 (``-1`` marks padding)
* ``cgenff_charge``   -- CGenFF charge, rescaled to conserve each monomer's net charge
* ``cgenff_master_sigmas`` / ``cgenff_master_epsilons`` -- shared ``(n_types,)`` tables
* ``E_cgenff_mm`` / ``F_cgenff_mm`` -- inter-monomer CGenFF MM baseline (unless ``--no-mm-baseline``)

Frames that are not exactly two covalent components, or whose monomers have no
CGenFF template, are dropped (all per-sample arrays are filtered consistently)
unless ``--strict`` is given.

Usage:
    mmml prepare-mm-dataset -i mp2_nms15_clean_train.npz -o train_mm.npz
    mmml prepare-mm-dataset --config prepare_mm.yaml
    mmml prepare-mm-dataset --config prepare_mm.yaml -o override_out.npz

The YAML config keys mirror the long flag names (dashes or underscores), e.g.::

    data: mp2_nms15_clean_train.npz
    output: mp2_nms15_clean_train_mm.npz
    num_workers: 8
    no_mm_baseline: false
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from mmml.data.cgenff_dataset import (
    DEF_PRM_PATH,
    DEF_RTF_PATH,
    assign_frame_cgenff,
    load_reference,
)

# ─── Multiprocessing worker (module-level for pickling) ────────────────────────
_WORKER_REF = None
_WORKER_COMPUTE_MM = True


def _worker_init(prm_path: str, rtf_path: str, compute_mm: bool) -> None:
    global _WORKER_REF, _WORKER_COMPUTE_MM
    _WORKER_REF = load_reference(prm_path, rtf_path)
    _WORKER_COMPUTE_MM = compute_mm


def _worker_assign(payload):
    idx, z_i, r_i = payload
    assignment, reason = assign_frame_cgenff(
        z_i, r_i, _WORKER_REF, compute_mm=_WORKER_COMPUTE_MM
    )
    return idx, assignment, reason


# ─── Enrichment driver ─────────────────────────────────────────────────────────


def enrich_npz(
    input_path: str | Path,
    output_path: str | Path,
    *,
    prm_path: str | Path = DEF_PRM_PATH,
    rtf_path: str | Path = DEF_RTF_PATH,
    num_workers: int = 1,
    max_structures: Optional[int] = None,
    compute_mm: bool = True,
    strict: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Enrich a dense NPZ with CGenFF fields; write ``output_path``.

    Returns a small summary dict (kept/dropped counts, drop reasons).
    """
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    prm_path = str(Path(prm_path).expanduser().resolve())
    rtf_path = str(Path(rtf_path).expanduser().resolve())

    ref = load_reference(prm_path, rtf_path)

    data = dict(np.load(input_path, allow_pickle=True))
    if "R" not in data or "Z" not in data:
        raise ValueError(f"{input_path} must contain dense 'R' and 'Z' arrays")
    R = np.asarray(data["R"])
    Z = np.asarray(data["Z"])
    if R.ndim != 3 or Z.ndim != 2:
        raise ValueError(
            f"Expected dense R (n,atoms,3) and Z (n,atoms); got R{R.shape}, Z{Z.shape}"
        )
    n_samples, n_atoms = Z.shape
    limit = n_samples if not max_structures else min(n_samples, max_structures)

    if not quiet:
        print("=" * 66)
        print(" mmml prepare-mm-dataset -- CGenFF assignment for hybrid ML/MM")
        print(f"  input : {input_path}")
        print(f"  output: {output_path}")
        print(
            f"  frames: {limit:,} / {n_samples:,}  |  padded atoms: {n_atoms}  |  "
            f"types: {len(ref.sigmas)}  |  workers: {max(1, num_workers)}"
        )
        print("=" * 66)

    def frame_generator():
        for i in range(limit):
            valid = np.flatnonzero(Z[i] > 0)
            yield i, Z[i, valid], R[i, valid]

    # Dense output buffers (filled for kept frames, then filtered by keep mask).
    keep = np.zeros(limit, dtype=bool)
    out_type = np.full((limit, n_atoms), -1, dtype=np.int32)
    out_molid = np.full((limit, n_atoms), -1, dtype=np.int32)
    out_charge = np.zeros((limit, n_atoms), dtype=np.float64)
    out_e_mm = np.zeros((limit, 1), dtype=np.float64)
    out_f_mm = np.zeros((limit, n_atoms, 3), dtype=np.float64)

    drop_reasons: dict[str, int] = {}
    t0 = time.time()

    def _store(idx: int, assignment) -> None:
        valid = np.flatnonzero(Z[idx] > 0)
        keep[idx] = True
        out_type[idx, valid] = assignment.cgenff_type_idx
        out_molid[idx, valid] = assignment.mol_id
        out_charge[idx, valid] = assignment.cgenff_charge
        if compute_mm:
            out_e_mm[idx, 0] = assignment.e_cgenff_mm
            out_f_mm[idx, valid] = assignment.f_cgenff_mm

    def _drop(idx: int, reason: str) -> None:
        if strict:
            raise ValueError(f"Frame {idx}: {reason}")
        key = reason.split("(")[0].strip()[:80]
        drop_reasons[key] = drop_reasons.get(key, 0) + 1

    workers = max(1, int(num_workers))
    if workers > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        with mp.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(prm_path, rtf_path, compute_mm),
        ) as pool:
            for idx, assignment, reason in pool.imap_unordered(
                _worker_assign, frame_generator(), chunksize=256
            ):
                if assignment is None:
                    _drop(idx, reason or "unknown")
                else:
                    _store(idx, assignment)
    else:
        for idx, z_i, r_i in frame_generator():
            assignment, reason = assign_frame_cgenff(
                z_i, r_i, ref, compute_mm=compute_mm
            )
            if assignment is None:
                _drop(idx, reason or "unknown")
            else:
                _store(idx, assignment)

    n_keep = int(keep.sum())
    n_drop = limit - n_keep
    if n_keep == 0:
        raise RuntimeError(
            "No frames could be assigned CGenFF types. "
            f"Top reasons: {sorted(drop_reasons.items(), key=lambda kv: -kv[1])[:5]}"
        )

    # Filter every per-sample array (axis0 == n_samples) by the keep mask; pass
    # everything else through unchanged.
    out: dict[str, Any] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == n_samples:
            out[key] = arr[:limit][keep]
        else:
            out[key] = value

    out["cgenff_type_idx"] = out_type[keep]
    out["mol_id"] = out_molid[keep]
    out["cgenff_charge"] = out_charge[keep]
    out["cgenff_master_sigmas"] = ref.sigmas
    out["cgenff_master_epsilons"] = ref.epsilons
    if compute_mm:
        out["E_cgenff_mm"] = out_e_mm[keep]
        out["F_cgenff_mm"] = out_f_mm[keep]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)

    dt = time.time() - t0
    if not quiet:
        rate = n_keep / dt if dt > 0 else 0.0
        print(f"[+] Assigned {n_keep:,} frames in {dt:.2f}s ({rate:.0f} frames/sec)")
        if n_drop:
            print(f"[!] Dropped {n_drop:,} frames ({100 * n_drop / limit:.1f}%):")
            for reason, count in sorted(drop_reasons.items(), key=lambda kv: -kv[1])[:10]:
                print(f"      {count:>8,}  {reason}")
        print(f"[+] Wrote {output_path}")
        print("=" * 66)

    return {
        "n_input": n_samples,
        "n_processed": limit,
        "n_kept": n_keep,
        "n_dropped": n_drop,
        "drop_reasons": drop_reasons,
        "output": str(output_path),
    }


# ─── CLI plumbing ──────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml prepare-mm-dataset",
        description="Assign CGenFF atom types / charges to a dimer training NPZ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config seeding the flags below")
    parser.add_argument("-i", "--data", type=str, default=None, help="Input dense NPZ (R/Z/N/...)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output enriched NPZ")
    parser.add_argument("--prm-path", type=str, default=str(DEF_PRM_PATH), help="CGenFF parameter (.prm) file")
    parser.add_argument("--rtf-path", type=str, default=str(DEF_RTF_PATH), help="CGenFF topology (.rtf) file")
    parser.add_argument("--num-workers", type=int, default=1, help="Multiprocessing pool size (1 = serial)")
    parser.add_argument("--max-structures", type=int, default=None, help="Process only the first N frames")
    parser.add_argument(
        "--no-mm-baseline",
        action="store_true",
        help="Skip the E_cgenff_mm / F_cgenff_mm inter-monomer baseline",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Error on the first unassignable frame instead of dropping it",
    )
    parser.add_argument("--save-config", type=str, default=None, help="Write the resolved config to this YAML path")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser


def _apply_config(args: argparse.Namespace, mapping: Mapping[str, Any], parser: argparse.ArgumentParser) -> None:
    """Seed parser defaults from a YAML mapping (dash/underscore-insensitive)."""
    valid = {a.dest for a in parser._actions}
    updates: dict[str, Any] = {}
    unknown: list[str] = []
    for raw_key, value in mapping.items():
        key = str(raw_key).replace("-", "_")
        if key in valid:
            updates[key] = value
        else:
            unknown.append(str(raw_key))
    if unknown:
        raise ValueError(
            f"Unknown config key(s): {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(k for k in valid if k != 'help'))}"
        )
    parser.set_defaults(**updates)


def parse_prepare_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre.parse_known_args(argv)
    if not pre_args.config:
        return parser.parse_args(argv)

    import yaml

    cfg_path = Path(pre_args.config)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")
    # Config seeds defaults; explicit CLI flags (remaining) still win.
    _apply_config(parser, raw, parser)
    return parser.parse_args(remaining)


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_prepare_args(argv)

    if not args.data:
        raise SystemExit("prepare-mm-dataset: --data/-i (input NPZ) is required")
    if not args.output:
        raise SystemExit("prepare-mm-dataset: --output/-o (output NPZ) is required")

    if args.save_config:
        import yaml

        payload = {k: v for k, v in sorted(vars(args).items()) if k not in ("config", "save_config")}
        Path(args.save_config).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_config).write_text(yaml.safe_dump(payload, sort_keys=False))
        if not args.quiet:
            print(f"[+] Wrote resolved config to {args.save_config}")

    enrich_npz(
        args.data,
        args.output,
        prm_path=args.prm_path,
        rtf_path=args.rtf_path,
        num_workers=args.num_workers,
        max_structures=args.max_structures,
        compute_mm=not args.no_mm_baseline,
        strict=args.strict,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
