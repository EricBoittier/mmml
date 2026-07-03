#!/usr/bin/env python3
"""Export aaa embedding checkpoint, validate ML monomer potential, build hybrid TRIA+water.

Example (CHARMM env required for build/hybrid steps)::

    ./scripts/mmml-charmm-mpirun.sh uv run python scripts/setup_trialanine_embedding_hybrid.py \\
      --epoch-dir artifacts/md_embedding/aaa_docs/checkpoints/aaa_long-.../epoch-49 \\
      -o artifacts/md_embedding/aaa_docs/hybrid

Steps:
  1. ``orbax-to-json`` export
  2. PhysNet E/F validation on ``valid.npz`` (monomer ML potential gate)
  3. ``md-embedding build`` if ``model.psf`` missing
  4. Register partial MLpot + ASE hybrid calculator; write ``hybrid_manifest.json``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "artifacts" / "md_embedding" / "aaa_docs"
DEFAULT_VALID = DEFAULT_OUT / "valid.npz"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--epoch-dir",
        type=Path,
        required=True,
        help="Orbax epoch directory (e.g. .../epoch-49)",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT / "hybrid",
        help="Hybrid artifact root (box + manifest; defaults under aaa_docs/hybrid)",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Exported checkpoint JSON (default: output-dir/aaa_long_epoch49_params.json)",
    )
    p.add_argument(
        "--valid-npz",
        type=Path,
        default=DEFAULT_VALID,
        help="Validation NPZ for monomer E/F gate",
    )
    p.add_argument("--target-mae", type=float, default=10.0, help="Fail if E or F MAE exceeds this (kcal/mol)")
    p.add_argument("--skip-validate", action="store_true")
    p.add_argument("--skip-hybrid", action="store_true", help="Export + validate only")
    p.add_argument("--n-waters", type=int, default=10)
    p.add_argument("--box-side-A", type=float, default=28.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--mini-probe", action="store_true", help="Run one ASE get_potential_energy() on hybrid")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    args = _parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    epoch_dir = Path(args.epoch_dir)
    json_path = Path(args.json) if args.json is not None else out / "aaa_long_epoch49_params.json"

    from mmml.interfaces.pycharmmInterface.mlpot.embedding_hybrid import (
        EmbeddingValidationResult,
        export_embedding_checkpoint,
        prepare_trialanine_hybrid_session,
        validate_embedding_monomer_potential,
    )

    print(f"Exporting {epoch_dir} -> {json_path}", flush=True)
    export_embedding_checkpoint(epoch_dir, json_path)

    validation = None
    if not args.skip_validate:
        eval_dir = out / "monomer_potential_validation"
        print(f"Validating monomer ML potential on {args.valid_npz}", flush=True)
        validation = validate_embedding_monomer_potential(
            json_path,
            args.valid_npz,
            eval_dir,
        )
        print(
            f"  energy MAE = {validation.energy_mae_kcal_mol:.4f} kcal/mol, "
            f"force MAE = {validation.force_mae_kcal_mol_A:.4f} kcal/mol/Å",
            flush=True,
        )
        if (
            validation.energy_mae_kcal_mol > args.target_mae
            or validation.force_mae_kcal_mol_A > args.target_mae
        ):
            print(
                f"WARNING: MAE above --target-mae={args.target_mae}; "
                "continuing (use --target-mae to tighten)",
                flush=True,
            )
    else:
        metrics_path = out / "monomer_potential_validation" / "metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            validation = EmbeddingValidationResult(
                checkpoint_json=json_path,
                valid_npz=Path(args.valid_npz),
                metrics_path=metrics_path,
                energy_mae_kcal_mol=float(metrics.get("energy", {}).get("mae_kcal_mol", 0.0)),
                force_mae_kcal_mol_A=float(metrics.get("forces", {}).get("mae_kcal_mol", 0.0)),
                metrics=metrics,
            )

    hybrid_info: dict[str, object] = {}
    if not args.skip_hybrid:
        print(f"Building hybrid session under {out}", flush=True)
        session = prepare_trialanine_hybrid_session(
            out,
            json_path,
            build_if_missing=True,
            n_waters=args.n_waters,
            box_side_A=args.box_side_A,
            seed=args.seed,
        )
        try:
            e_charmm = session.charmm_total_energy_kcalmol()
            hybrid_info["charmm_total_energy_kcalmol"] = e_charmm
            hybrid_info["n_atoms"] = len(session.atoms)
            hybrid_info["n_peptide_atoms"] = int(session.box_meta.get("n_peptide_atoms", 0))
            hybrid_info["training_n_atoms"] = int(session.box_meta.get("training_n_atoms", 0))
            if args.mini_probe:
                try:
                    e_ase = float(session.atoms.get_potential_energy())
                    hybrid_info["ase_hybrid_energy_ev"] = e_ase
                    print(f"  ASE hybrid E = {e_ase:.6f} eV", flush=True)
                except RuntimeError as exc:
                    hybrid_info["ase_hybrid_error"] = str(exc)
                    print(f"  WARNING: ASE hybrid probe failed: {exc}", flush=True)
            print(f"  CHARMM ENER total = {e_charmm:.4f} kcal/mol", flush=True)
        finally:
            session.close()

    manifest = {
        "epoch_dir": str(epoch_dir.resolve()),
        "checkpoint_json": str(json_path.resolve()),
        "output_dir": str(out.resolve()),
        "valid_npz": str(Path(args.valid_npz).resolve()),
        "validation": (
            {
                "energy_mae_kcal_mol": validation.energy_mae_kcal_mol,
                "force_mae_kcal_mol_A": validation.force_mae_kcal_mol_A,
                "metrics_path": str(validation.metrics_path),
            }
            if validation is not None
            else None
        ),
        "hybrid": hybrid_info or None,
        "workflow": "md-embedding partial MLpot (PEPT=ML, SOLV=CHARMM MM)",
    }
    manifest_path = out / "hybrid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
