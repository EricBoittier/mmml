#!/usr/bin/env python3
"""Dump and visualize CHARMM vs MMML inter-monomer neighbor lists.

MMML pairs work from CRD/numpy alone (no PyCHARMM). CHARMM DMAT capture requires
a live PyCHARMM session loaded from PSF+CRD (``--with-charmm``).

Examples
--------
MMML only from a minimized CRD (DCM:52 @ L=38):

  uv run python scripts/dump_neighbor_lists.py \\
    --crd artifacts/dcm_density_setup_compare/resilient_dcm_52_t50_l38_ht_bussi_sw_baseline/pycharmm_mini/mini_full_mlpot_*.crd \\
    --n-monomers 52 --atoms-per-monomer 5 --box-size 38 \\
    --output-dir nl_dump_baseline

CHARMM + MMML side-by-side (PyCHARMM node):

  uv run python scripts/dump_neighbor_lists.py \\
    --artifact-dir artifacts/dcm_density_setup_compare/resilient_dcm_52_t50_l38_ht_bussi_sw_baseline/pycharmm_mini \\
    --n-monomers 52 --atoms-per-monomer 5 --box-size 38 \\
    --with-charmm --output-dir nl_dump_baseline

Auto-discover PSF/CRD from campaign leg:

  uv run python scripts/dump_neighbor_lists.py \\
    --artifact-dir artifacts/.../pycharmm_mini \\
    --n-monomers 52 --atoms-per-monomer 5 --box-size 38 \\
    --with-charmm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import read_crd_coordinates  # noqa: E402
from mmml.utils.neighbor_list_snapshot import (  # noqa: E402
    capture_charmm_inter_monomer_pairs,
    capture_mmml_inter_monomer_pairs,
    cubic_cell_matrix,
    find_artifact_geometry,
    save_neighbor_list_artifacts,
    setup_charmm_from_psf_crd,
    uniform_monomer_offsets,
)


def _load_positions(args: argparse.Namespace) -> tuple[Path | None, "np.ndarray"]:
    import numpy as np

    if args.positions_npy is not None:
        arr = np.load(args.positions_npy)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"--positions-npy must be (N,3), got {arr.shape}")
        return None, np.asarray(arr, dtype=np.float64)
    if args.crd is not None:
        pos = read_crd_coordinates(args.crd.expanduser().resolve())
        if pos is None:
            raise ValueError(f"failed to parse CRD: {args.crd}")
        return args.crd.expanduser().resolve(), pos
    if args.artifact_dir is not None and not args.with_charmm:
        _psf, crd = find_artifact_geometry(args.artifact_dir)
        pos = read_crd_coordinates(crd)
        if pos is None:
            raise ValueError(f"failed to parse CRD: {crd}")
        return crd, pos
    raise ValueError("provide --crd, --positions-npy, or --artifact-dir")


def _default_mm_r_min(mm_switch_on: float, ml_switch_width: float) -> float:
    handoff_start = float(mm_switch_on) - float(ml_switch_width)
    return handoff_start * 0.9 if handoff_start > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--crd", type=Path, help="CHARMM CRD card (EXT format)")
    src.add_argument("--positions-npy", type=Path, help="Numpy positions array (N,3)")
    src.add_argument(
        "--artifact-dir",
        type=Path,
        help="Campaign leg directory (auto-find newest PSF/CRD)",
    )
    parser.add_argument("--psf", type=Path, default=None, help="Topology PSF (with --with-charmm)")
    parser.add_argument(
        "--with-charmm",
        action="store_true",
        help="Load PSF+CRD into PyCHARMM and capture CHARMM DMAT pairs",
    )
    parser.add_argument("--n-monomers", type=int, required=True)
    parser.add_argument("--atoms-per-monomer", type=int, required=True)
    parser.add_argument("--box-size", type=float, required=True, help="Cubic box side (Å)")
    parser.add_argument("--charmm-cutoff", type=float, default=18.0, help="CHARMM cutnb (Å)")
    parser.add_argument("--mm-cutoff", type=float, default=13.0, help="MMML switched-MM cutoff (Å)")
    parser.add_argument(
        "--mm-backend",
        choices=("auto", "vesin", "cell_list", "jax_md"),
        default="vesin",
    )
    parser.add_argument("--mm-switch-on", type=float, default=8.0)
    parser.add_argument("--ml-switch-width", type=float, default=1.5)
    parser.add_argument("--top-pairs", type=int, default=30, help="Closest pairs drawn in PNG")
    parser.add_argument("--output-dir", type=Path, default=Path("neighbor_list_dump"))
    args = parser.parse_args()

    crd_path, positions = _load_positions(args)
    offsets = uniform_monomer_offsets(args.n_monomers, args.atoms_per_monomer)
    expected = int(args.n_monomers) * int(args.atoms_per_monomer)
    if positions.shape[0] != expected:
        raise ValueError(
            f"positions have {positions.shape[0]} atoms; expected "
            f"{expected} (= {args.n_monomers}×{args.atoms_per_monomer})"
        )

    cell = cubic_cell_matrix(args.box_size)
    mm_r_min = _default_mm_r_min(args.mm_switch_on, args.ml_switch_width)
    charmm_snap = None
    psf_used: Path | None = None

    if args.with_charmm:
        psf = args.psf
        crd = crd_path
        if args.artifact_dir is not None:
            psf_auto, crd_auto = find_artifact_geometry(args.artifact_dir)
            psf = psf or psf_auto
            crd = crd or crd_auto
        if psf is None or crd is None:
            raise ValueError("--with-charmm requires --psf and --crd (or --artifact-dir)")
        psf_used = psf.expanduser().resolve()
        positions, cell, eff_charmm_cut = setup_charmm_from_psf_crd(
            psf_path=psf_used,
            crd_path=crd.expanduser().resolve(),
            box_side=float(args.box_size),
            charmm_cutoff_A=float(args.charmm_cutoff),
        )
        charmm_snap = capture_charmm_inter_monomer_pairs(
            cutoff_A=eff_charmm_cut,
            monomer_offsets=offsets,
            positions=positions,
        )
        charmm_cutoff = eff_charmm_cut
    else:
        charmm_cutoff = float(args.charmm_cutoff)

    mmml_snap = capture_mmml_inter_monomer_pairs(
        positions=positions,
        cell=cell,
        cutoff_A=float(args.mm_cutoff),
        monomer_offsets=offsets,
        backend=args.mm_backend,
        mm_r_min=mm_r_min,
    )

    meta = {
        "crd": str(crd_path) if crd_path is not None else None,
        "psf": str(psf_used) if psf_used is not None else None,
        "n_monomers": int(args.n_monomers),
        "atoms_per_monomer": int(args.atoms_per_monomer),
        "box_size_A": float(args.box_size),
        "charmm_cutoff_A": charmm_cutoff,
        "mm_cutoff_A": float(args.mm_cutoff),
        "mm_r_min_A": mm_r_min,
    }
    paths = save_neighbor_list_artifacts(
        args.output_dir,
        positions=positions,
        cell=cell,
        monomer_offsets=offsets,
        charmm=charmm_snap,
        mmml=mmml_snap,
        extra_meta=meta,
        top_pairs=int(args.top_pairs),
    )

    print(json.dumps({"written": {k: str(v) for k, v in paths.items()}}, indent=2))
    if charmm_snap is not None and charmm_snap.pairs:
        w = charmm_snap.pairs[0]
        print(
            f"CHARMM closest inter-monomer pair: {w.distance_A:.4f} Å "
            f"(mon {w.monomer_i}/{w.monomer_j}, atoms {w.i}/{w.j})"
        )
    if mmml_snap.pairs:
        w = mmml_snap.pairs[0]
        print(
            f"MMML closest inter-monomer pair: {w.distance_A:.4f} Å "
            f"(mon {w.monomer_i}/{w.monomer_j}, atoms {w.i}/{w.j})"
        )
    if paths.get("comparison") is not None:
        cmp = json.loads(paths["comparison"].read_text(encoding="utf-8"))
        print(
            "comparison:",
            f"shared={cmp['n_shared']}",
            f"only_charmm={cmp['n_only_left']}",
            f"only_mmml={cmp['n_only_right']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
