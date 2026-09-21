"""``mmml fit-liquid``: DiffTRe reweighting of the ML/MM hybrid to liquid data.

Stages::

    # pipeline check on JAX-MD NPT frames (no PhysNet / CHARMM)
    mmml fit-liquid dry-run --frames run_npt.h5 --atoms-per-molecule 10 \\
        --molecule ACO --toy --out-json fit/dry_run.json

    # decompose + cache, then fit from the cache (same toy Hamiltonian)
    mmml fit-liquid decompose --frames run_npt.h5 --atoms-per-molecule 10 \\
        --toy --cache aco_T200.npz
    mmml fit-liquid fit --cache aco_T200.npz --molecule ACO --toy \\
        --n-steps 50 --out-json fit/theta.json --sidecar fit/hybrid_mm.json

The fitted parameters are per-CGenFF-type LJ ε and Rmin log-scales and a
scale λ on the switched PhysNet dimer term. Targets are ρ(T) and ΔHvap(T);
see :mod:`mmml.fit`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mmml fit-liquid",
        description="Fit per-type CGenFF LJ scales (and a PhysNet dimer scale) "
        "to experimental rho(T) / dHvap(T) by trajectory reweighting.",
    )
    sub = p.add_subparsers(dest="stage", required=True)

    dry = sub.add_parser("dry-run", help="toy decompose + check + a few Adam steps on NPT frames")
    _add_frame_args(dry)
    _add_target_args(dry)
    dry.add_argument("--n-steps", type=int, default=2, help="Adam steps (0 = evaluate at theta_0)")
    dry.add_argument("--lr", type=float, default=0.02)
    dry.add_argument("--cache-out", type=Path, default=None, help="optional FrameCache npz")
    dry.add_argument("--out-json", type=Path, default=None)
    dry.add_argument("--overwrite", action="store_true")

    dec = sub.add_parser("decompose", help="evaluate hybrid terms per frame and write a cache")
    _add_frame_args(dec)
    dec.add_argument("--cache", type=Path, required=True)
    dec.add_argument("--overwrite", action="store_true")

    chk = sub.add_parser("check", help="same-Hamiltonian check of a cache at theta_0")
    _add_cache_args(chk)
    _add_target_args(chk)
    chk.add_argument("--out-json", type=Path, default=None)

    fit = sub.add_parser("fit", help="Adam on the reweighted liquid-observable loss")
    _add_cache_args(fit)
    _add_target_args(fit)
    fit.add_argument("--n-steps", type=int, default=50)
    fit.add_argument("--lr", type=float, default=0.02)
    fit.add_argument("--prior-weight", type=float, default=0.0)
    fit.add_argument("--no-fit-lambda", action="store_true", help="keep the dimer scale at 1")
    fit.add_argument("--out-json", type=Path, required=True)
    fit.add_argument("--sidecar", type=Path, default=None, help="hybrid_mm.json-style LJ scales")
    return p


def _add_frame_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--frames", type=Path, required=True, help="JAX-MD NPT HDF5 or ASE extxyz/traj")
    p.add_argument("--atoms-per-molecule", type=int, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument(
        "--toy",
        action="store_true",
        help="fake ML + LJ Hamiltonian (no PhysNet/CHARMM); required until checkpoint decompose lands",
    )


def _add_cache_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache", type=Path, required=True, help="FrameCache npz from decompose")
    p.add_argument(
        "--toy",
        action="store_true",
        help="rebuild the toy energy_with_lj from the cached frames",
    )


def _add_target_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--molecule", required=True, help="ACO or DCM")
    p.add_argument(
        "--e-gas",
        type=float,
        default=None,
        help="gas-phase <E> per molecule (kcal/mol); default matches the tabulated dHvap at theta_0",
    )
    p.add_argument("--reference", type=Path, default=None, help="experimental JSON (optional)")


def _require_toy(args: argparse.Namespace) -> None:
    if not args.toy:
        raise SystemExit(
            "mmml fit-liquid currently supports the --toy Hamiltonian "
            "(JAX-MD frames without PhysNet/CHARMM). Pass --toy."
        )


def _load_frames(args: argparse.Namespace):
    from mmml.fit.frames import load_frames

    path = args.frames
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return load_frames(path, args.atoms_per_molecule, start=args.start, stride=args.stride)
    index = ":" if args.start == 0 and args.stride == 1 else f"{args.start}::{args.stride}"
    return load_frames(path, args.atoms_per_molecule, index=index)


def _write_json(path: Path | None, payload: dict) -> None:
    if path is None:
        print(json.dumps(payload, indent=2))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")


def _run_dry_run(args: argparse.Namespace) -> int:
    _require_toy(args)
    from mmml.fit.driver import dry_run
    from mmml.fit.targets import load_reference

    ref = load_reference(args.reference) if args.reference is not None else None
    frames = _load_frames(args)
    result = dry_run(
        frames,
        args.molecule,
        e_gas=args.e_gas,
        n_steps=args.n_steps,
        lr=args.lr,
        cache_path=args.cache_out,
        overwrite=args.overwrite,
        ref=ref,
    )
    _write_json(args.out_json, result.as_json())
    return 0


def _run_decompose(args: argparse.Namespace) -> int:
    _require_toy(args)
    from mmml.fit.driver import decompose_frames, toy_hybrid_handles

    frames = _load_frames(args)
    cache = decompose_frames(
        frames,
        toy_hybrid_handles(frames),
        cache_path=args.cache,
        overwrite=args.overwrite,
        toy=True,
    )
    print(f"wrote {args.cache} ({cache.frames.n_frames} frames, {cache.frames.n_molecules} molecules)")
    return 0


def _result_from_cache(args: argparse.Namespace, *, n_steps: int, lr: float, prior_weight: float, fit_lambda: bool):
    from mmml.fit.driver import fit_from_cache, toy_hybrid_handles
    from mmml.fit.frames import FrameCache
    from mmml.fit.targets import load_reference

    ref = load_reference(args.reference) if args.reference is not None else None
    cache = FrameCache.load(args.cache)
    handles = toy_hybrid_handles(cache.frames)
    return fit_from_cache(
        cache,
        args.molecule,
        e_gas=args.e_gas,
        n_steps=n_steps,
        lr=lr,
        prior_weight=prior_weight,
        fit_lambda=fit_lambda,
        mm_energy_with_lj=handles.update_fn.energy_with_lj,
        ref=ref,
    )


def _run_check(args: argparse.Namespace) -> int:
    _require_toy(args)
    result = _result_from_cache(args, n_steps=0, lr=0.02, prior_weight=0.0, fit_lambda=True)
    _write_json(args.out_json, {"check": result.check, "loss": result.loss, "n_frames": result.n_frames})
    return 0


def _run_fit(args: argparse.Namespace) -> int:
    _require_toy(args)
    from mmml.fit.driver import theta_sidecar_payload

    result = _result_from_cache(
        args,
        n_steps=args.n_steps,
        lr=args.lr,
        prior_weight=args.prior_weight,
        fit_lambda=not args.no_fit_lambda,
    )
    payload = result.as_json()
    _write_json(args.out_json, payload)
    if args.sidecar is not None:
        args.sidecar.parent.mkdir(parents=True, exist_ok=True)
        args.sidecar.write_text(
            json.dumps(theta_sidecar_payload(result.theta, result.type_names, result.lam), indent=2)
        )
        print(f"wrote {args.sidecar}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "dry-run":
        return _run_dry_run(args)
    if args.stage == "decompose":
        return _run_decompose(args)
    if args.stage == "check":
        return _run_check(args)
    return _run_fit(args)


if __name__ == "__main__":
    sys.exit(main())
