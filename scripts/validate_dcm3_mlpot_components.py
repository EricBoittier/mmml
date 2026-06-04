#!/usr/bin/env python3
"""DCM:3 MLpot component scan — energies/forces vs inter-monomer separation.

Builds (or loads) a three-monomer DCM cluster, evaluates the decomposed
PhysNet + MM hybrid calculator with ``debug=True``, and records per-component
energies and force norms on:

  * a 1D scan: COM distance monomer 0–1 (monomer 2 fixed at reference separation)
  * a 2D scan: COM distances monomer 0–1 vs monomer 0–2

Also reports minimum intra-monomer and inter-monomer atom–atom distances to
check for unphysical same-monomer overlap when dimers are close.

Examples
--------
  export MMML_CKPT=/path/to/dcm_ckpt
  python scripts/validate_dcm3_mlpot_components.py \\
    --output artifacts/dcm3_component_scan/scan.npz

  python scripts/validate_dcm3_mlpot_components.py \\
    --reference-crd workflows/dcm_nve_scaling/results/dcm_3_nve/02_mlpot_mmml_dcm_3.crd \\
    --scan-1d-min 2.0 --scan-1d-max 8.0 --scan-1d-steps 25 \\
    --scan-2d-min 3.0 --scan-2d-max 10.0 --scan-2d-steps 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

EV_PER_KCAL = 1.0 / 23.0605  # kcal/mol per eV (same convention as hybrid_mlpot)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="PhysNet checkpoint (default: MMML_CKPT or repo default)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO / "artifacts" / "dcm3_component_scan" / "scan.npz",
        help="NPZ output path",
    )
    p.add_argument(
        "--reference-crd",
        type=Path,
        default=None,
        help="CHARMM CRD for DCM:3 (15 atoms); preferred when CHARMM is not on PATH",
    )
    p.add_argument(
        "--positions-npy",
        type=Path,
        default=None,
        help="Reference positions (N,3) Å; with --reference-crd or alone (needs CHARMM PSF for Z)",
    )
    p.add_argument(
        "--packmol-radius",
        type=float,
        default=6.9,
        help="Packmol sphere radius when building reference cluster",
    )
    p.add_argument("--atoms-per-monomer", type=int, default=5)
    p.add_argument("--mm-switch-on", type=float, default=5.5)
    p.add_argument("--mm-switch-width", type=float, default=1.5)
    p.add_argument("--ml-switch-width", type=float, default=0.1)
    p.add_argument("--scan-1d-min", type=float, default=2.0)
    p.add_argument("--scan-1d-max", type=float, default=10.0)
    p.add_argument("--scan-1d-steps", type=int, default=33)
    p.add_argument("--scan-2d-min", type=float, default=3.0)
    p.add_argument("--scan-2d-max", type=float, default=10.0)
    p.add_argument("--scan-2d-steps", type=int, default=17)
    p.add_argument(
        "--angle-02-deg",
        type=float,
        default=60.0,
        help="Angle (deg) of monomer-2 COM from +x axis in the 2D scan",
    )
    p.add_argument("--debug", action="store_true", help="Pass debug=True to spherical_fn")
    p.add_argument("--no-mm", action="store_true", help="ML only (doMM=False)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _load_positions_crd(path: Path) -> np.ndarray:
    """Read CHARMM EXT CRD (Å) written by PyCHARMM."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"CRD too short: {path}")
    n_atoms: int | None = None
    start = 0
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 2 and parts[-1].upper() == "EXT":
            try:
                n_atoms = int(parts[0])
                start = idx + 1
                break
            except ValueError:
                continue
    if n_atoms is None:
        raise ValueError(f"Could not parse CRD header: {path}")
    coords: list[list[float]] = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) >= 7:
            try:
                coords.append([float(parts[4]), float(parts[5]), float(parts[6])])
            except ValueError:
                continue
        elif len(parts) >= 5:
            try:
                coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
            except ValueError:
                continue
        if len(coords) >= n_atoms:
            break
    if len(coords) != n_atoms:
        raise ValueError(f"Expected {n_atoms} coords in {path}, got {len(coords)}")
    return np.asarray(coords, dtype=np.float64)


def _atomic_numbers_dcm3(atoms_per_monomer: int) -> np.ndarray:
    from mmml.cli.run.md_pbc_suite.ase import _build_cluster_psf_from_composition

    z, _names, atoms_per_list, _res = _build_cluster_psf_from_composition([("DCM", 3)])
    if list(atoms_per_list) != [atoms_per_monomer] * 3:
        raise RuntimeError(f"DCM atoms/monomer mismatch: {atoms_per_list}")
    return np.asarray(z, dtype=int)


def _load_reference_cluster(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (Z, positions Å, atoms_per_monomer list)."""
    n_mol = 3
    apm = int(args.atoms_per_monomer)
    if args.positions_npy is not None:
        pos = np.load(Path(args.positions_npy).expanduser().resolve())
        if pos.shape != (n_mol * apm, 3):
            raise ValueError(f"positions must be ({n_mol * apm}, 3), got {pos.shape}")
        z = _atomic_numbers_dcm3(apm)
        return z, np.asarray(pos, dtype=np.float64), [apm] * n_mol

    if args.reference_crd is not None:
        pos = _load_positions_crd(Path(args.reference_crd).expanduser().resolve())
        if pos.shape[0] != n_mol * apm:
            raise ValueError(
                f"CRD has {pos.shape[0]} atoms; expected {n_mol * apm} for DCM:3"
            )
        z = _atomic_numbers_dcm3(apm)
        return z, pos, [apm] * n_mol

    composition = [("DCM", n_mol)]
    from mmml.cli.run.md_pbc_suite.cluster import build_packmol_composition_cluster

    if not args.quiet:
        print("Building DCM:3 reference cluster (Packmol + CHARMM MM pre-min)...", flush=True)
    try:
        z, pos, atoms_per_list, _names = build_packmol_composition_cluster(
        composition=composition,
        center=(0.0, 0.0, 0.0),
        radius=float(args.packmol_radius),
        tolerance=1.0,
        seed=123,
        charmm_sd_steps=50,
        charmm_abnr_steps=100,
        verbose=not args.quiet,
        scratch_dir=args.output.parent / "packmol_scratch",
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "Could not build DCM:3 cluster (CHARMM/Packmol). Provide --reference-crd or "
            "--positions-npy from a minimized run, or fix CHARMMSETUP.\n"
            f"Original error: {exc}"
        ) from exc
    z = _atomic_numbers_dcm3(int(atoms_per_list[0]))
    return z, np.asarray(pos, dtype=np.float64), list(atoms_per_list)


def _monomer_offsets(atoms_per: list[int]) -> np.ndarray:
    off = np.zeros(len(atoms_per) + 1, dtype=int)
    for i, n in enumerate(atoms_per):
        off[i + 1] = off[i] + int(n)
    return off


def _monomer_com(pos: np.ndarray, off: int, n: int) -> np.ndarray:
    return np.mean(pos[off : off + n], axis=0)


def _min_intra_dist(pos: np.ndarray, off: int, n: int) -> float:
    sub = pos[off : off + n]
    if n < 2:
        return float("inf")
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
    iu = np.triu_indices(n, k=1)
    return float(d[iu].min())


def _min_inter_dist(pos: np.ndarray, off_a: int, na: int, off_b: int, nb: int) -> float:
    a = pos[off_a : off_a + na]
    b = pos[off_b : off_b + nb]
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(d.min())


def _rigid_shift_monomer(
    pos: np.ndarray,
    ref: np.ndarray,
    off: int,
    n: int,
    target_com: np.ndarray,
) -> None:
    ref_com = _monomer_com(ref, off, n)
    pos[off : off + n] = ref[off : off + n] + (target_com - ref_com)


def _place_trimer(
    ref: np.ndarray,
    atoms_per: list[int],
    d01: float,
    d02: float,
    angle_02_rad: float,
) -> np.ndarray:
    """Rigid-body move monomers 1 and 2 relative to monomer 0 COM."""
    pos = np.array(ref, dtype=np.float64, copy=True)
    off = _monomer_offsets(atoms_per)
    com0 = _monomer_com(ref, int(off[0]), int(atoms_per[0]))
    target1 = com0 + np.array([d01, 0.0, 0.0], dtype=float)
    target2 = com0 + d02 * np.array(
        [np.cos(angle_02_rad), np.sin(angle_02_rad), 0.0], dtype=float
    )
    _rigid_shift_monomer(pos, ref, int(off[1]), int(atoms_per[1]), target1)
    _rigid_shift_monomer(pos, ref, int(off[2]), int(atoms_per[2]), target2)
    return pos


def _com_distances(pos: np.ndarray, atoms_per: list[int]) -> np.ndarray:
    off = _monomer_offsets(atoms_per)
    coms = [_monomer_com(pos, int(off[i]), int(atoms_per[i])) for i in range(3)]
    d01 = float(np.linalg.norm(coms[1] - coms[0]))
    d02 = float(np.linalg.norm(coms[2] - coms[0]))
    d12 = float(np.linalg.norm(coms[2] - coms[1]))
    return np.array([d01, d02, d12], dtype=np.float64)


def _distance_report(pos: np.ndarray, atoms_per: list[int]) -> dict[str, float]:
    off = _monomer_offsets(atoms_per)
    out: dict[str, float] = {}
    for i, n in enumerate(atoms_per):
        out[f"min_intra_m{i}"] = _min_intra_dist(pos, int(off[i]), int(n))
    pairs = [(0, 1), (0, 2), (1, 2)]
    for a, b in pairs:
        out[f"min_inter_{a}{b}"] = _min_inter_dist(
            pos, int(off[a]), int(atoms_per[a]), int(off[b]), int(atoms_per[b])
        )
    com_d = _com_distances(pos, atoms_per)
    out["com_d01"] = float(com_d[0])
    out["com_d02"] = float(com_d[1])
    out["com_d12"] = float(com_d[2])
    return out


def _build_model(
    checkpoint: Path,
    z: np.ndarray,
    atoms_per: list[int],
    args: argparse.Namespace,
):
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import build_decomposed_mlpot_model

    cutoff = CutoffParameters(
        ml_switch_width=float(args.ml_switch_width),
        mm_switch_on=float(args.mm_switch_on),
        mm_switch_width=float(args.mm_switch_width),
    )
    model = build_decomposed_mlpot_model(
        checkpoint,
        z,
        atoms_per,
        n_monomers=3,
        cell=False,
        verbose=not args.quiet,
    )
    return model, cutoff


def _eval_breakdown(
    model: Any,
    cutoff: Any,
    positions: np.ndarray,
    z: np.ndarray,
    *,
    debug: bool,
    do_mm: bool,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.calculator_utils import ModelOutput

    pos_j = jnp.asarray(positions, dtype=jnp.float64)
    z_j = jnp.asarray(z, dtype=int)

    def _one(do_ml: bool, do_ml_dimer: bool, do_mm_flag: bool) -> ModelOutput:
        return model._spherical_fn(
            pos_j,
            z_j,
            model._n_monomers,
            cutoff,
            doML=do_ml,
            doMM=do_mm_flag,
            doML_dimer=do_ml_dimer,
            debug=debug,
        )

    full = _one(True, True, do_mm)
    ml_internal = _one(True, False, False)
    ml_dimer_only = _one(True, True, False)
    mm_only = _one(False, False, do_mm) if do_mm else None

    def _scalar(x) -> float:
        v = jax.device_get(x)
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)

    def _fmax(forces: jnp.ndarray) -> float:
        f = np.asarray(jax.device_get(forces), dtype=np.float64)
        return float(np.linalg.norm(f, axis=1).max())

    e_fields = {
        "energy": _scalar(full.energy),
        "hybrid_energy": _scalar(full.hybrid_energy),
        "internal_E": _scalar(full.internal_E),
        "ml_2b_E": _scalar(full.ml_2b_E),
        "dH": _scalar(full.dH),
        "mm_E": _scalar(full.mm_E) if do_mm else 0.0,
        "flat_bottom_E": _scalar(full.flat_bottom_E),
        "ml_internal_only": _scalar(ml_internal.internal_E),
        "ml_2b_contrib": _scalar(ml_dimer_only.ml_2b_E),
    }
    rec: dict[str, Any] = {
        **{f"{k}_eV": v for k, v in e_fields.items()},
        **{f"{k}_kcal": float(v) * EV_PER_KCAL for k, v in e_fields.items()},
        "force_max_eV_A": _fmax(full.forces),
        "internal_F_max_eV_A": _fmax(full.internal_F),
        "ml_2b_F_max_eV_A": _fmax(full.ml_2b_F),
        "mm_F_max_eV_A": _fmax(full.mm_F) if do_mm else 0.0,
    }
    return rec


def _print_breakdown(label: str, rec: dict[str, Any], dist: dict[str, float]) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(
        f"  COM distances (Å): d01={dist['com_d01']:.3f} d02={dist['com_d02']:.3f} "
        f"d12={dist['com_d12']:.3f}",
        flush=True,
    )
    print(
        f"  min intra-monomer (Å): m0={dist['min_intra_m0']:.3f} "
        f"m1={dist['min_intra_m1']:.3f} m2={dist['min_intra_m2']:.3f}",
        flush=True,
    )
    print(
        f"  min inter-monomer (Å): 01={dist['min_inter_01']:.3f} "
        f"02={dist['min_inter_02']:.3f} 12={dist['min_inter_12']:.3f}",
        flush=True,
    )
    print("  Energies (kcal/mol):", flush=True)
    for stem in (
        "energy",
        "hybrid_energy",
        "internal_E",
        "ml_2b_E",
        "dH",
        "mm_E",
        "ml_internal_only",
        "ml_2b_contrib",
    ):
        k = f"{stem}_kcal"
        if k in rec:
            print(f"    {stem:18s} {rec[k]:12.4f}", flush=True)
    print(
        f"  |F|_max (eV/Å): total={rec['force_max_eV_A']:.4f} "
        f"internal={rec['internal_F_max_eV_A']:.4f} "
        f"ml_2b={rec['ml_2b_F_max_eV_A']:.4f} "
        f"mm={rec['mm_F_max_eV_A']:.4f}",
        flush=True,
    )


def _run_scan_1d(
    model: Any,
    cutoff: Any,
    ref_pos: np.ndarray,
    z: np.ndarray,
    atoms_per: list[int],
    d02_fixed: float,
    d_grid: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    n = len(d_grid)
    keys = (
        "energy_kcal",
        "internal_E_kcal",
        "ml_2b_E_kcal",
        "dH_kcal",
        "mm_E_kcal",
        "ml_internal_only_kcal",
        "force_max_eV_A",
        "min_intra_m0",
        "min_intra_m1",
        "min_intra_m2",
        "min_inter_01",
        "min_inter_02",
        "min_inter_12",
        "com_d01",
        "com_d02",
        "com_d12",
    )
    store = {k: np.zeros(n, dtype=np.float64) for k in keys}
    angle = np.deg2rad(float(args.angle_02_deg))
    for i, d01 in enumerate(d_grid):
        pos = _place_trimer(ref_pos, atoms_per, float(d01), float(d02_fixed), angle)
        dist = _distance_report(pos, atoms_per)
        rec = _eval_breakdown(
            model, cutoff, pos, z, debug=args.debug, do_mm=not args.no_mm
        )
        if i == 0 or i == n // 2 or i == n - 1:
            _print_breakdown(f"1D scan i={i} d01_target={d01:.3f} Å", rec, dist)
        for k in keys:
            if k in dist:
                store[k][i] = dist[k]
            elif k in rec:
                store[k][i] = rec[k]
    return store


def _run_scan_2d(
    model: Any,
    cutoff: Any,
    ref_pos: np.ndarray,
    z: np.ndarray,
    atoms_per: list[int],
    d1_grid: np.ndarray,
    d2_grid: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    n1, n2 = len(d1_grid), len(d2_grid)
    keys = (
        "energy_kcal",
        "internal_E_kcal",
        "ml_2b_E_kcal",
        "dH_kcal",
        "mm_E_kcal",
        "min_inter_01",
        "min_inter_02",
        "min_inter_12",
    )
    store = {k: np.zeros((n1, n2), dtype=np.float64) for k in keys}
    angle = np.deg2rad(float(args.angle_02_deg))
    for i, d01 in enumerate(d1_grid):
        for j, d02 in enumerate(d2_grid):
            pos = _place_trimer(ref_pos, atoms_per, float(d01), float(d02), angle)
            dist = _distance_report(pos, atoms_per)
            rec = _eval_breakdown(
                model, cutoff, pos, z, debug=args.debug, do_mm=not args.no_mm
            )
            for k in keys:
                if k.startswith("min_") and k in dist:
                    store[k][i, j] = dist[k]
                elif k in rec:
                    store[k][i, j] = rec[k]
        if not args.quiet and (i == 0 or i == n1 - 1):
            print(f"  2D scan row {i + 1}/{n1} done", flush=True)
    return store


def main() -> int:
    args = _parse_args()
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_checkpoint

    ckpt = resolve_checkpoint(
        args.checkpoint.expanduser().resolve() if args.checkpoint else None
    )
    z, ref_pos, atoms_per = _load_reference_cluster(args)
    if len(atoms_per) != 3 or sum(atoms_per) != len(z):
        raise RuntimeError(f"Expected 3 monomers, got atoms_per={atoms_per}")

    if not args.quiet:
        print(f"Checkpoint: {ckpt}", flush=True)
        print(f"Reference cluster: {len(z)} atoms, Z={z.tolist()}", flush=True)
        span = np.ptp(ref_pos, axis=0)
        print(f"Reference span (Å): {span}", flush=True)

    model, cutoff = _build_model(ckpt, z, atoms_per, args)
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import warmup_decomposed_mlpot

    warmup_decomposed_mlpot(model, ref_pos, verbose=not args.quiet)

    ref_dist = _distance_report(ref_pos, atoms_per)
    ref_rec = _eval_breakdown(
        model, cutoff, ref_pos, z, debug=args.debug, do_mm=not args.no_mm
    )
    _print_breakdown("Reference geometry", ref_rec, ref_dist)

    d02_ref = float(ref_dist["com_d02"])
    d1 = np.linspace(float(args.scan_1d_min), float(args.scan_1d_max), int(args.scan_1d_steps))
    d2a = np.linspace(float(args.scan_2d_min), float(args.scan_2d_max), int(args.scan_2d_steps))
    d2b = np.linspace(float(args.scan_2d_min), float(args.scan_2d_max), int(args.scan_2d_steps))

    if not args.quiet:
        print("\n--- 1D scan (varying d01, fixed d02 from reference) ---", flush=True)
    scan_1d = _run_scan_1d(model, cutoff, ref_pos, z, atoms_per, d02_ref, d1, args)

    if not args.quiet:
        print("\n--- 2D scan (d01 × d02) ---", flush=True)
    scan_2d = _run_scan_2d(model, cutoff, ref_pos, z, atoms_per, d2a, d2b, args)

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save: dict[str, Any] = {
        "composition": np.array("DCM:3"),
        "checkpoint": np.array(str(ckpt)),
        "atoms_per_monomer": np.array(atoms_per, dtype=np.int32),
        "reference_positions_A": ref_pos.astype(np.float64),
        "atomic_numbers": z.astype(np.int32),
        "mm_switch_on": np.float64(args.mm_switch_on),
        "mm_switch_width": np.float64(args.mm_switch_width),
        "ml_switch_width": np.float64(args.ml_switch_width),
        "debug_mode": np.bool_(args.debug),
        "do_mm": np.bool_(not args.no_mm),
        "scan_1d_d01_A": d1.astype(np.float64),
        "scan_1d_d02_fixed_A": np.float64(d02_ref),
        "scan_2d_d01_A": d2a.astype(np.float64),
        "scan_2d_d02_A": d2b.astype(np.float64),
        "angle_02_deg": np.float64(args.angle_02_deg),
    }
    for k, v in ref_rec.items():
        if isinstance(v, (int, float, np.floating)):
            save[f"reference_{k}"] = np.float64(v)
    for k, v in ref_dist.items():
        save[f"reference_{k}"] = np.float64(v)
    for k, v in scan_1d.items():
        save[f"scan_1d_{k}"] = v
    for k, v in scan_2d.items():
        save[f"scan_2d_{k}"] = v

    np.savez_compressed(out_path, **save)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
