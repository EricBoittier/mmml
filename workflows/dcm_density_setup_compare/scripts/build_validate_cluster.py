#!/usr/bin/env python3
"""Packmol cluster build + geometry validation (no MD).

Uses the bundled mmml Packmol binary (``packmol_executable()``) and CHARMM
monomer prep, then ``validate_cluster_geometry`` and optional MIC floor check.

Usage:
  python scripts/build_validate_cluster.py --config config.profile.dcm30_l30.yaml \\
    --tag minimal_dcm_30_t50_l30_ht_bussi
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from campaign_lib import (  # noqa: E402
    cell_from_tag,
    cell_workflow_cfg,
    config_for_run_tag,
    load_config,
    merge_setup_into_config,
    paths_for_run,
    run_seed,
)


def _build_args(cfg: dict, cell, *, out_dir: Path, cell_root: Path) -> Namespace:
    effective = merge_setup_into_config(cfg, cell.setup_id)
    seed = run_seed(cell, seed_base=int(cfg.get("seed_base", 4242)), cfg=cfg)
    ns = Namespace(
        composition=f"{cell.solvent}:{cell.n_monomers}",
        box_size=float(cell.box_size),
        spacing=float(effective.get("spacing", 5.0)),
        packmol_tolerance=float(effective.get("packmol_tolerance", 2.0)),
        packmol_box_padding=float(effective.get("packmol_box_padding", 2.0)),
        packmol_placement=effective.get("packmol_placement"),
        packmol_radius=effective.get("packmol_radius"),
        packmol_sphere=effective.get("packmol_sphere"),
        packmol=effective.get("packmol"),
        builder=effective.get("builder"),
        pyxtal=effective.get("pyxtal"),
        seed=seed,
        output_dir=out_dir,
        charmm_sd_steps=int(effective.get("charmm_sd_steps", 50)),
        charmm_abnr_steps=int(effective.get("charmm_abnr_steps", 100)),
        charmm_tolenr=1e-3,
        charmm_tolgrd=1e-3,
        reuse_packmol_cache=not bool(effective.get("rebuild_packmol", False)),
        packmol_cache_dir=str(cell_root / ".packmol_cache"),
        rebuild_packmol=bool(effective.get("rebuild_packmol", False)),
        quiet=False,
        mlpot_pbc=True,
        pre_mlpot_overlap_min_distance=float(
            effective.get("pre_mlpot_overlap_min_distance", 2.0)
        ),
        liquid_prep=bool(effective.get("liquid_prep", False)),
    )
    return ns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--mic-check",
        action="store_true",
        help="Abort when worst MIC contact is below pre_mlpot_overlap_min_distance",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = config_for_run_tag(cfg, args.tag)
    cell = cell_from_tag(cfg, args.tag)
    paths = paths_for_run(cfg, cell)
    cell_root = paths["out_dir"]
    out_dir = cell_root / "cluster_build"
    out_dir.mkdir(parents=True, exist_ok=True)

    from mmml.interfaces.pycharmmInterface.packmol_placement import packmol_executable

    packmol_bin = packmol_executable()
    print(f"packmol: {packmol_bin}", flush=True)

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        build_cluster_from_args_with_tag,
        use_packmol_placement,
        validate_cluster_geometry,
    )

    ns = _build_args(cfg, cell, out_dir=out_dir, cell_root=cell_root)
    if not use_packmol_placement(ns):
        print(
            "ERROR: config disables Packmol placement "
            "(set packmol: true and omit builder: liquid)",
            file=sys.stderr,
        )
        return 1

    print(
        f"Building {ns.composition} L={ns.box_size:.1f} Å "
        f"(spacing={ns.spacing}, tolerance={ns.packmol_tolerance})",
        flush=True,
    )
    z, r, n_mol, tag = build_cluster_from_args_with_tag(ns)
    stats = validate_cluster_geometry(r, n_molecules=n_mol)
    atoms_per = [len(z) // n_mol] * n_mol

    mic_worst: float | None = None
    if args.mic_check:
        from mmml.interfaces.pycharmmInterface.mlpot.density_prep_ladder import (
            assert_ml_safe_before_mlpot_registration,
        )

        mic_worst = assert_ml_safe_before_mlpot_registration(
            ns,
            positions=r,
            atoms_per_list=atoms_per,
            box_side=float(cell.box_size),
            charmm_pbc=True,
            atomic_numbers=z,
        )

    summary = {
        "tag": args.tag,
        "composition": ns.composition,
        "box_size_A": float(cell.box_size),
        "n_monomers": int(n_mol),
        "n_atoms": int(len(z)),
        "packmol": packmol_bin,
        "geometry": stats,
        "mic_worst_A": mic_worst,
    }
    summary_path = out_dir / "cluster_validation.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    npz_path = out_dir / "cluster.npz"
    import numpy as np

    np.savez(
        npz_path,
        atomic_numbers=np.asarray(z, dtype=np.int32),
        positions=np.asarray(r, dtype=np.float64),
        box_size=float(cell.box_size),
    )
    print(f"Cluster validation OK -> {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
