#!/usr/bin/env python3
"""Build DCM:3 Packmol cluster and apply rigid trimer COM placement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from cutoff_lib import (  # noqa: E402
    composition_string,
    composition_tag,
    ensure_repo_on_path,
    geometry_config,
    geometry_dir,
    load_config,
    packmol_radius_A,
    workflow_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geom_id", help="Geometry variant key from config.yaml")
    parser.add_argument(
        "--config",
        type=Path,
        default=workflow_root() / "config.yaml",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ensure_repo_on_path()
    cfg = load_config(args.config)
    geom = geometry_config(cfg, args.geom_id)
    out = geometry_dir(cfg, args.geom_id)
    out.mkdir(parents=True, exist_ok=True)
    tag = composition_tag(cfg)

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        build_cluster_from_args_with_tag,
        print_cluster_geometry_summary,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        save_cluster_topology_for_vmd,
        sync_charmm_positions,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.trimer_scan import (
        atoms_per_monomer_from_psf,
        com_distances,
        distance_report,
        place_trimer,
    )

    import pycharmm.write as write

    ns = argparse.Namespace(
        composition=composition_string(cfg),
        residue="DCM",
        n_molecules=3,
        spacing=float(cfg.get("spacing", 5.0)),
        checkpoint=None,
        packmol_sphere=True,
        packmol_radius=packmol_radius_A(cfg),
        packmol_tolerance=float(cfg.get("packmol_tolerance", 1.2)),
        seed=int(cfg.get("seed", 123)),
        output_dir=out,
        charmm_sd_steps=int(cfg.get("charmm_sd_steps", 25)),
        charmm_abnr_steps=int(cfg.get("charmm_abnr_steps", 100)),
        charmm_tolenr=1e-3,
        charmm_tolgrd=1e-3,
        reuse_packmol_cache=True,
        rebuild_packmol=False,
        packmol_cache_dir=out / ".packmol_cache",
        quiet=args.quiet,
    )

    # NBONDS must be set after the PSF has atoms (build_cluster → minimize_charmm_mm_only).
    z, ref_pos, n_mol, built_tag = build_cluster_from_args_with_tag(ns)
    if n_mol != 3 or built_tag != tag:
        raise SystemExit(f"Expected DCM:3 tag={tag}, got n_mol={n_mol} tag={built_tag}")

    atoms_per = atoms_per_monomer_from_psf()
    angle_rad = np.deg2rad(float(geom["angle_02_deg"]))
    placed = place_trimer(
        ref_pos,
        atoms_per,
        float(geom["d01"]),
        float(geom["d02"]),
        angle_rad,
    )
    sync_charmm_positions(placed)

    if not args.quiet:
        print_cluster_geometry_summary(placed, n_mol)
        dist = distance_report(placed, atoms_per)
        print(
            f"Trimer COM (Å): d01={dist['com_d01']:.3f} "
            f"d02={dist['com_d02']:.3f} d12={dist['com_d12']:.3f}",
            flush=True,
        )

    topo = save_cluster_topology_for_vmd(
        out,
        placed,
        stem=f"cluster_for_vmd_{tag}",
        title=f"DCM:3 trimer {args.geom_id}",
    )
    crd_path = out / "initial.crd"
    write.coor_card(str(crd_path), title=f"DCM:3 trimer initial {args.geom_id}")

    meta = {
        "geom_id": args.geom_id,
        "composition": composition_string(cfg),
        "d01_target_A": float(geom["d01"]),
        "d02_target_A": float(geom["d02"]),
        "angle_02_deg": float(geom["angle_02_deg"]),
        "com_distances_A": com_distances(placed, atoms_per).tolist(),
        "distance_report": distance_report(placed, atoms_per),
        "psf": str(topo["psf"]),
        "crd": str(crd_path),
    }
    meta_path = out / "geometry.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {topo['psf']}", flush=True)
    print(f"Wrote {crd_path}", flush=True)
    print(f"Wrote {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from None
