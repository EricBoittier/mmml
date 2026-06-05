#!/usr/bin/env python3
"""Audit ML/MM pair coverage and cutoff zones for a cluster geometry.

Complements ``validate_mlpot_sparse_dimers.py`` (cap vs near count) with a
per-dimer table: COM distance, ML taper zone, MM inclusion (``mm_r_min``),
expected vacuum MM atom–atom pairs, and minimum inter-monomer atom distance.

No JAX or CHARMM required — numpy + CRD/DCD coordinates only.

Examples
--------
Post-mini geometry (DCM:8 NVE scaling):

  python scripts/validate_mlpot_pair_lists.py \\
    --crd workflows/dcm_nve_scaling/results/dcm_8_nve/inbfrq_50/02_mlpot_mmml_dcm_8.crd \\
    --n-monomers 8 --atoms-per-monomer 5 --free-space

NVE frame where COM QC shows drift (frame index 0-based):

  python scripts/validate_mlpot_pair_lists.py \\
    --dcd workflows/dcm_nve_scaling/results/dcm_8_nve/inbfrq_neg1/nve_dcm_8.dcd \\
    --frame 266 --n-monomers 8 --atoms-per-monomer 5 --free-space
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mmml.interfaces.pycharmmInterface.calculator_utils import dimer_permutations  # noqa: E402
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (  # noqa: E402
    build_monomer_dimer_index_arrays,
    dimer_com_distance_numpy,
    resolve_max_active_dimers,
    validate_sparse_dimer_cap,
)
from scripts.validate_mlpot_sparse_dimers import (  # noqa: E402
    _find_crd_in_output_dir,
    _load_positions_crd,
)


def _read_dcd_frame(path: Path, frame: int) -> np.ndarray:
    import importlib.util

    mod_path = _REPO / "mmml" / "utils" / "dcd_reader.py"
    spec = importlib.util.spec_from_file_location("_mmml_dcd_reader", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DCD reader from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pos, _hdr = mod.read_dcd_trajectory(path, max_frames=int(frame) + 1)
    if pos.shape[0] <= int(frame):
        raise ValueError(f"DCD {path} has {pos.shape[0]} frame(s); --frame {frame} out of range")
    return np.asarray(pos[int(frame)], dtype=np.float64)


def _default_mm_r_min(*, mm_switch_on: float, ml_switch_width: float, complementary_handoff: bool) -> float:
    if not complementary_handoff:
        return float(mm_switch_on) * 0.9
    handoff_start = float(mm_switch_on) - float(ml_switch_width)
    return handoff_start * 0.9 if handoff_start > 0 else 0.0


def _ml_taper_weight(com_dist: float, *, mm_switch_on: float, ml_switch_width: float) -> float:
    """Rough ML scale at COM distance (matches sharpstep handoff band)."""
    lo = float(mm_switch_on) - float(ml_switch_width)
    hi = float(mm_switch_on)
    if com_dist <= lo:
        return 1.0
    if com_dist >= hi:
        return 0.0
    t = (com_dist - lo) / max(hi - lo, 1e-12)
    # GAMMA_ON=6 sharpstep approximation for reporting (not bit-identical to JAX)
    t = max(0.0, min(1.0, t))
    return float(1.0 - t * t * (3.0 - 2.0 * t))


def _min_inter_monomer_atom_distance(
    positions: np.ndarray,
    offsets: np.ndarray,
    mi: int,
    mj: int,
) -> float:
    ia = slice(int(offsets[mi]), int(offsets[mi + 1]))
    ib = slice(int(offsets[mj]), int(offsets[mj + 1]))
    a = positions[ia]
    b = positions[ib]
    d = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return float(np.min(np.linalg.norm(d, axis=2)))


def _zone_label(
    com_dist: float,
    *,
    mm_switch_on: float,
    ml_switch_width: float,
    mm_r_min: float,
) -> str:
    handoff_lo = float(mm_switch_on) - float(ml_switch_width)
    if com_dist < float(mm_r_min):
        return "pure_ML"
    if com_dist < float(mm_switch_on):
        return "handoff"
    if com_dist < float(mm_switch_on) + float(ml_switch_width):
        return "MM_onset"
    return "far_MM"


def analyze_pair_lists(
    positions: np.ndarray,
    n_monomers: int,
    atoms_per_monomer: int,
    *,
    mm_switch_on: float = 5.5,
    mm_switch_width: float = 1.5,
    ml_switch_width: float = 0.1,
    mm_r_min: float | None = None,
    complementary_handoff: bool = True,
    free_space: bool = True,
    ml_max_active_dimers: int | None = None,
) -> dict[str, Any]:
    pos = np.asarray(positions, dtype=np.float64)
    n_mol = int(n_monomers)
    per = int(atoms_per_monomer)
    offsets = np.arange(0, n_mol * per + 1, per, dtype=int)

    if mm_r_min is None:
        mm_r_min = _default_mm_r_min(
            mm_switch_on=mm_switch_on,
            ml_switch_width=ml_switch_width,
            complementary_handoff=complementary_handoff,
        )

    _, dimer_idx, dimer_n_a, dimer_n_b, _ = build_monomer_dimer_index_arrays(n_mol, per)
    pairs = dimer_permutations(n_mol)
    n_dimers = len(pairs)

    cap_stats = validate_sparse_dimer_cap(
        pos,
        n_mol,
        per,
        mm_switch_on=mm_switch_on,
        max_active_dimers=ml_max_active_dimers,
        free_space=free_space,
    )
    cap = int(cap_stats["max_active_dimers_cap"])
    use_sparse = cap < n_dimers

    dimers: list[dict[str, Any]] = []
    for di, (mi, mj) in enumerate(pairs):
        com_d = dimer_com_distance_numpy(
            pos, dimer_idx[di], int(dimer_n_a[di]), int(dimer_n_b[di]), None
        )
        min_atom = _min_inter_monomer_atom_distance(pos, offsets, mi, mj)
        ml_active = com_d < float(mm_switch_on)
        mm_included = com_d >= float(mm_r_min)
        dimers.append(
            {
                "dimer_1based": [mi + 1, mj + 1],
                "com_dist_A": round(com_d, 4),
                "min_inter_atom_A": round(min_atom, 4),
                "zone": _zone_label(
                    com_d,
                    mm_switch_on=mm_switch_on,
                    ml_switch_width=ml_switch_width,
                    mm_r_min=mm_r_min,
                ),
                "ml_taper_weight": round(
                    _ml_taper_weight(
                        com_d,
                        mm_switch_on=mm_switch_on,
                        ml_switch_width=ml_switch_width,
                    ),
                    4,
                ),
                "ml_evaluated": bool(ml_active) if use_sparse else True,
                "mm_atom_pairs_included": bool(mm_included),
                "n_mm_atom_pairs": int(dimer_n_a[di]) * int(dimer_n_b[di]),
            }
        )

    n_near = int(cap_stats["n_near_mm_switch_on"])
    n_mm_pairs_expected = sum(d["n_mm_atom_pairs"] for d in dimers if d["mm_atom_pairs_included"])
    n_mm_pairs_total = sum(d["n_mm_atom_pairs"] for d in dimers)
    close_contacts = [d for d in dimers if d["min_inter_atom_A"] < 1.0]
    handoff_dimers = [d for d in dimers if d["zone"] == "handoff"]

    problems: list[str] = []
    if cap_stats["cap_saturated"]:
        problems.append(
            f"sparse cap saturated: {n_near} near dimers > cap {cap} (ML may drop pairs)"
        )
    if use_sparse:
        problems.append(
            f"sparse ML path active (cap {cap} < {n_dimers}); only COM < {mm_switch_on} Å evaluated"
        )
    else:
        problems.append(f"free-space all-pairs: all {n_dimers} ML dimer slots allocated")

    if close_contacts:
        problems.append(
            f"{len(close_contacts)} dimer(s) with inter-monomer atom distance < 1.0 Å"
        )

    ok = not cap_stats["cap_saturated"]

    return {
        "n_monomers": n_mol,
        "n_atoms": int(pos.shape[0]),
        "n_dimers_total": n_dimers,
        "mm_switch_on_A": float(mm_switch_on),
        "mm_switch_width_A": float(mm_switch_width),
        "ml_switch_width_A": float(ml_switch_width),
        "mm_r_min_A": float(mm_r_min),
        "handoff_band_A": [
            float(mm_switch_on) - float(ml_switch_width),
            float(mm_switch_on),
        ],
        "free_space": bool(free_space),
        "ml_max_active_dimers_cap": cap,
        "sparse_ml_active": use_sparse,
        "n_near_mm_switch_on": n_near,
        "n_mm_atom_pairs_total": n_mm_pairs_total,
        "n_mm_atom_pairs_included": n_mm_pairs_expected,
        "vacuum_mm_pair_policy": "all cross-monomer atom pairs per dimer (no spatial cull)",
        "dimers": dimers,
        "close_contacts_under_1A": close_contacts,
        "handoff_dimers": handoff_dimers,
        "sparse_cap": cap_stats,
        "problems": problems,
        "ok": ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--crd", type=Path)
    src.add_argument("--dcd", type=Path)
    src.add_argument("--output-dir", type=Path, help="Use mini_full_mlpot_*.crd from run dir")
    parser.add_argument("--frame", type=int, default=0, help="DCD frame index (0-based)")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--n-monomers", type=int, required=True)
    parser.add_argument("--atoms-per-monomer", type=int, default=5)
    parser.add_argument("--mm-switch-on", type=float, default=5.5)
    parser.add_argument("--mm-switch-width", type=float, default=1.5)
    parser.add_argument("--ml-switch-width", type=float, default=0.1)
    parser.add_argument("--mm-r-min", type=float, default=None)
    parser.add_argument(
        "--no-complementary-handoff",
        action="store_true",
        help="Use mm_r_min = 0.9 * mm_switch_on instead of handoff_start * 0.9",
    )
    parser.add_argument(
        "--free-space",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json", type=Path, default=None, help="Write full report JSON")
    parser.add_argument("--top", type=int, default=10, help="Show N closest dimers by COM")
    args = parser.parse_args()

    if args.crd:
        pos = _load_positions_crd(args.crd.expanduser())
        source = str(args.crd)
    elif args.dcd:
        pos = _read_dcd_frame(args.dcd.expanduser(), int(args.frame))
        source = f"{args.dcd} frame {args.frame}"
    else:
        out = args.output_dir.expanduser()
        crd = _find_crd_in_output_dir(out, args.tag)
        if crd is None:
            print(f"No mini_full_mlpot_*.crd under {out}", file=sys.stderr)
            return 2
        pos = _load_positions_crd(crd)
        source = str(crd)

    report = analyze_pair_lists(
        pos,
        int(args.n_monomers),
        int(args.atoms_per_monomer),
        mm_switch_on=args.mm_switch_on,
        mm_switch_width=args.mm_switch_width,
        ml_switch_width=args.ml_switch_width,
        mm_r_min=args.mm_r_min,
        complementary_handoff=not args.no_complementary_handoff,
        free_space=bool(args.free_space),
    )
    report["source"] = source

    print(f"Pair-list audit: {source}")
    print(f"  n_monomers={report['n_monomers']}  n_dimers={report['n_dimers_total']}")
    print(
        f"  cutoffs: mm_switch_on={report['mm_switch_on_A']:.3f} Å  "
        f"handoff=[{report['handoff_band_A'][0]:.3f}, {report['handoff_band_A'][1]:.3f}]  "
        f"mm_r_min={report['mm_r_min_A']:.3f} Å"
    )
    print(f"  ML cap={report['ml_max_active_dimers_cap']}  sparse={report['sparse_ml_active']}")
    print(f"  near dimers (COM < mm_switch_on): {report['n_near_mm_switch_on']}")
    print(
        f"  MM atom pairs: {report['n_mm_atom_pairs_included']}/"
        f"{report['n_mm_atom_pairs_total']} included (vacuum: no spatial cull)"
    )

    by_com = sorted(report["dimers"], key=lambda d: d["com_dist_A"])
    print(f"\nClosest {args.top} dimers by COM:")
    print(f"  {'pair':>8}  {'COM':>7}  {'min_atom':>8}  {'zone':>10}  ML  MM")
    for d in by_com[: max(1, int(args.top))]:
        p = d["dimer_1based"]
        print(
            f"  {p[0]:>2}-{p[1]:<2}  {d['com_dist_A']:7.3f}  "
            f"{d['min_inter_atom_A']:8.3f}  {d['zone']:>10}  "
            f"{'Y' if d['ml_evaluated'] else 'n'}  "
            f"{'Y' if d['mm_atom_pairs_included'] else 'n'}"
        )

    zones: dict[str, int] = {}
    for d in report["dimers"]:
        zones[d["zone"]] = zones.get(d["zone"], 0) + 1
    print("\nDimer zone counts:", ", ".join(f"{k}={v}" for k, v in sorted(zones.items())))

    if report["close_contacts_under_1A"]:
        print(f"\nWARN: {len(report['close_contacts_under_1A'])} dimer(s) with min inter-atom < 1.0 Å:")
        for d in report["close_contacts_under_1A"]:
            p = d["dimer_1based"]
            print(f"  {p[0]}-{p[1]}: min_atom={d['min_inter_atom_A']:.3f} Å  COM={d['com_dist_A']:.3f} Å")

    print("\nNotes:")
    for line in report["problems"]:
        print(f"  - {line}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
