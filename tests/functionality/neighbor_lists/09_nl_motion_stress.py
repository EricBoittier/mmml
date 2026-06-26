#!/usr/bin/env python3
"""NL parity under motion: jitter, compression, box resize, jax-md reallocation.

Starts from a liquid-density cluster (default: dense ACO 1.5× bulk ρ), applies a
sequence of position/box perturbations, and checks every backend against the
reference oracle after each step. Also stress-tests jax-md buffer overflow and
capacity growth with a tight ``capacity_multiplier``.

Examples
--------
  uv run python tests/functionality/neighbor_lists/09_nl_motion_stress.py
  uv run python tests/functionality/neighbor_lists/09_nl_motion_stress.py \\
      --case synthetic_aco_liquid_n32_rho150 --backends vesin,jax_md,cell_list
  uv run python tests/functionality/neighbor_lists/09_nl_motion_stress.py \\
      --jax-md-capacity 1.05 --jax-md-growth 1.5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

from _common import (
    apply_motion_step,
    build_liquid_density_synthetic_case,
    effective_mass_density_g_cm3,
    liquid_density_synthetic_cases,
    motion_stress_steps,
    print_fail,
    print_header,
    print_pass,
    _composition_dict_from_liquid_case,
)
from mmml.interfaces.pycharmmInterface.jax_md_neighbor_list import (
    create_jax_md_neighbor_list,
    have_jax_md,
)
from mmml.interfaces.pycharmmInterface.nl_reference import (
    compare_pair_sets,
    extract_valid_pairs,
    filter_pairs_under_cutoff,
    reference_mic_pairs,
)

_DIR = Path(__file__).resolve().parent


def _load_compare_module():
    path = _DIR / "05_compare_nl_backends.py"
    spec = importlib.util.spec_from_file_location("nl_compare_backends", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jax_md_pairs_at(
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    monomer_offsets: np.ndarray,
    *,
    capacity_multiplier: float,
    dr_threshold: float = 0.5,
    growth_factor: float = 1.5,
    max_overflow_retries: int = 4,
    state: dict | None = None,
    update: bool = False,
) -> tuple[set[tuple[int, int]], dict, int]:
    """Build or update jax-md pairs; return pairs, mutable state, realloc_count."""
    pos = np.asarray(positions, dtype=np.float64)
    realloc_count = 0
    cap = float(state["capacity"]) if state is not None else float(capacity_multiplier)
    neighbor_fn = state["neighbor_fn"] if state is not None else None
    filter_fn = state["filter_fn"] if state is not None else None
    nbrs = state["nbrs"] if state is not None else None

    def _make_bundle(mult: float):
        bundle = create_jax_md_neighbor_list(
            cell,
            r_cutoff=cutoff,
            monomer_offsets=monomer_offsets,
            dr_threshold=dr_threshold,
            capacity_multiplier=mult,
            fractional_coordinates=False,
        )
        if bundle is None:
            raise RuntimeError("jax-md unavailable")
        return bundle

    if neighbor_fn is None:
        neighbor_fn, filter_fn, _ = _make_bundle(cap)
        nbrs = neighbor_fn.allocate(pos)
    elif update:
        nbrs = neighbor_fn.update(pos, nbrs)
        for _ in range(int(max_overflow_retries)):
            overflow = bool(np.asarray(nbrs.did_buffer_overflow))
            if not overflow:
                break
            realloc_count += 1
            cap *= float(growth_factor)
            neighbor_fn, filter_fn, _ = _make_bundle(cap)
            nbrs = neighbor_fn.allocate(pos)
            nbrs = neighbor_fn.update(pos, nbrs)
        else:
            raise RuntimeError(
                f"jax-md overflow persisted after {max_overflow_retries} realloc attempts "
                f"(capacity_multiplier={cap:.3f})"
            )
    else:
        neighbor_fn, filter_fn, _ = _make_bundle(cap)
        nbrs = neighbor_fn.allocate(pos)

    st = {
        "neighbor_fn": neighbor_fn,
        "filter_fn": filter_fn,
        "nbrs": nbrs,
        "capacity": cap,
        "growth_factor": growth_factor,
    }
    pi, pj, mask = filter_fn(nbrs.idx)
    pairs = extract_valid_pairs(pi, pj, mask)
    pairs = filter_pairs_under_cutoff(pairs, positions, cell, cutoff)
    return pairs, st, realloc_count


def _run_step_parity(
    *,
    step_name: str,
    positions: np.ndarray,
    cell: np.ndarray,
    offsets: np.ndarray,
    monomer_id: np.ndarray,
    cutoff: float,
    backends: list[str],
    jax_state: dict | None,
    jax_capacity: float,
    jax_growth: float,
) -> tuple[bool, dict | None, int]:
    nl05 = _load_compare_module()
    ref, ref_src = reference_mic_pairs(
        positions, cell, cutoff, monomer_id, monomer_offsets=offsets
    )
    print(
        f"\n  step={step_name}  reference={ref_src} ({len(ref)} pairs)  "
        f"L={float(np.diag(cell)[0]):.3f} Å"
    )
    ok = True
    total_realloc = 0
    new_jax_state = jax_state
    force_jax_allocate = step_name.startswith("box_")

    for backend in backends:
        if backend == "jax_md" and have_jax_md():
            try:
                use_update = (
                    jax_state is not None
                    and step_name != "baseline"
                    and not force_jax_allocate
                )
                pairs, new_jax_state, n_realloc = _jax_md_pairs_at(
                    positions,
                    cell,
                    cutoff,
                    offsets,
                    capacity_multiplier=jax_capacity,
                    growth_factor=jax_growth,
                    state=None if force_jax_allocate else jax_state,
                    update=use_update,
                )
                total_realloc += n_realloc
                label = "jax_md"
            except Exception as exc:
                print(f"    FAIL jax_md: {exc}")
                ok = False
                continue
        else:
            label, pairs, skip = nl05._collect_backend(
                backend,
                positions=positions,
                cell=cell,
                cutoff=cutoff,
                monomer_id=monomer_id,
                monomer_offsets=offsets,
                charmm_geometry=False,
                atomic_numbers=None,
            )
            if pairs is None:
                print(f"    SKIP {label}: {skip}")
                continue

        cmp = compare_pair_sets(ref, pairs)
        status = "PASS" if cmp.match else "FAIL"
        if not cmp.match:
            ok = False
        extra = f"  realloc={total_realloc}" if backend == "jax_md" and total_realloc else ""
        print(
            f"    {label:<12} {cmp.n_b:>6} {len(cmp.only_a):>6} {len(cmp.only_b):>6}  {status}{extra}"
        )
        if not cmp.match:
            print(cmp.summary(label_a="reference", label_b=label, max_show=4))

    return ok, new_jax_state, total_realloc


def _jax_md_trajectory_stress(
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
    offsets: np.ndarray,
    monomer_id: np.ndarray,
    capacity_multiplier: float,
    growth_factor: float,
    n_frames: int,
    max_displacement_A: float,
) -> bool:
    """Walk a random displacement trajectory; check jax-md vs reference each frame."""
    if not have_jax_md():
        print("  SKIP jax-md trajectory: unavailable")
        return True

    print(
        f"\n=== jax-md trajectory ({n_frames} frames, "
        f"cap={capacity_multiplier:.2f}, growth={growth_factor:.2f}) ==="
    )
    rng = np.random.default_rng(7)
    pos = np.asarray(positions, dtype=np.float64).copy()
    state: dict | None = None
    ok = True
    total_realloc = 0
    t0 = time.perf_counter()

    for frame in range(int(n_frames)):
        if frame > 0:
            pos = pos + (max_displacement_A / n_frames) * rng.standard_normal(pos.shape)
        ref, _ = reference_mic_pairs(pos, cell, cutoff, monomer_id, monomer_offsets=offsets)
        pairs, state, n_realloc = _jax_md_pairs_at(
            pos,
            cell,
            cutoff,
            offsets,
            capacity_multiplier=capacity_multiplier,
            growth_factor=growth_factor,
            state=state,
            update=frame > 0,
        )
        total_realloc += n_realloc
        cmp = compare_pair_sets(ref, pairs)
        if not cmp.match:
            print(f"  FAIL frame {frame}: {cmp.summary(label_a='ref', label_b='jax_md', max_show=3)}")
            ok = False
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    cap = float(state["capacity"]) if state else capacity_multiplier
    print(
        f"  trajectory: {n_frames} frames in {elapsed_ms:.1f} ms  "
        f"reallocs={total_realloc}  final_capacity_mult={cap:.3f}"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        type=str,
        default="synthetic_aco_liquid_n32_rho150",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="vesin,jax_md,ase,cell_list",
    )
    parser.add_argument(
        "--jax-md-capacity",
        type=float,
        default=1.05,
        help="Tight initial capacity multiplier to provoke overflow/realloc",
    )
    parser.add_argument(
        "--jax-md-growth",
        type=float,
        default=1.5,
        help="Capacity growth factor on jax-md overflow",
    )
    parser.add_argument("--trajectory-frames", type=int, default=12)
    parser.add_argument("--trajectory-displacement", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print_header("NL motion / reallocation stress")
    cases = {str(c["name"]): c for c in liquid_density_synthetic_cases()}
    if args.case not in cases:
        print_fail(f"unknown case {args.case!r}")
        return 1

    case = cases[args.case]
    positions, cell, offsets, monomer_id, cutoff, desc, side, rho_target = (
        build_liquid_density_synthetic_case(case)
    )
    comp = _composition_dict_from_liquid_case(case)
    rho_eff = effective_mass_density_g_cm3(comp, side)

    nl05 = _load_compare_module()
    backends = nl05._parse_backends(args.backends)

    print(
        f"case={args.case}\n"
        f"  {desc}\n"
        f"  n_atoms={positions.shape[0]}  L={side:.3f} Å  "
        f"ρ_target={rho_target:.3f} g/cm³  ρ_eff={rho_eff:.3f} g/cm³  "
        f"cutoff={cutoff:.2f} Å"
    )

    rng = np.random.default_rng(int(args.seed))
    ok = True
    jax_state: dict | None = None
    steps = motion_stress_steps()
    pos = positions.copy()
    cell_cur = cell.copy()

    print("\n=== Motion-step parity ===")
    for step in steps:
        pos, cell_cur = apply_motion_step(pos, cell_cur, step, rng=rng)
        step_ok, jax_state, _ = _run_step_parity(
            step_name=str(step["name"]),
            positions=pos,
            cell=cell_cur,
            offsets=offsets,
            monomer_id=monomer_id,
            cutoff=cutoff,
            backends=backends,
            jax_state=jax_state,
            jax_capacity=float(args.jax_md_capacity),
            jax_growth=float(args.jax_md_growth),
        )
        ok = ok and step_ok

    traj_ok = _jax_md_trajectory_stress(
        positions=positions,
        cell=cell,
        cutoff=cutoff,
        offsets=offsets,
        monomer_id=monomer_id,
        capacity_multiplier=float(args.jax_md_capacity),
        growth_factor=float(args.jax_md_growth),
        n_frames=int(args.trajectory_frames),
        max_displacement_A=float(args.trajectory_displacement),
    )
    ok = ok and traj_ok

    if ok:
        print_pass("motion / reallocation stress passed")
        return 0
    print_fail("motion / reallocation stress failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
