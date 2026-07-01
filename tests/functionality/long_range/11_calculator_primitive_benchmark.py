#!/usr/bin/env python3
"""Benchmark compile vs run time for hybrid-calculator JAX primitives.

Measures host jax-pme kernels, hybrid long-range correction sub-steps, and
(optionally) full MLpot ``spherical_cutoff`` / ``mlpot_spherical_forward`` when
a PhysNet checkpoint is available.  No PyCHARMM required for the jax-pme leg.

Examples
--------
  JAX_PLATFORMS=cpu MMML_JAX_COMPILE_TIMERS=1 \\
    uv run python tests/functionality/long_range/11_calculator_primitive_benchmark.py

  JAX_PLATFORMS=cpu uv run python tests/functionality/long_range/11_calculator_primitive_benchmark.py \\
    --checkpoint examples/ckpts_json/DESdimers_params.json --n-monomers 12

  uv run python tests/functionality/long_range/11_calculator_primitive_benchmark.py \\
    --json artifacts/calculator_primitive_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from _common import have_jax_pme_package, print_header, print_pass


@dataclass(frozen=True)
class BenchRow:
    name: str
    category: str
    compile_s: float | None
    run_s: float | None
    steady_ms: float | None
    notes: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-monomers", type=int, default=18)
    parser.add_argument("--atoms-per-monomer", type=int, default=3)
    parser.add_argument("--box-side", type=float, default=28.0)
    parser.add_argument("--method", default="ewald", choices=("ewald", "pme", "p3m"))
    parser.add_argument("--sr-cutoff", type=float, default=6.0)
    parser.add_argument("--steady-reps", type=int, default=5)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional PhysNet JSON for MLpot spherical warmup primitives",
    )
    parser.add_argument("--ml-batch-size", type=int, default=8)
    parser.add_argument("--json", type=Path, default=None, help="Write machine-readable report")
    parser.add_argument("--legacy-intra", action="store_true", help="Also bench full_minus_intra hybrid path")
    return parser.parse_args()


def _synthetic_cluster(args: argparse.Namespace):
    from jaxpme import prefactors as jpref

    from mmml.interfaces.pycharmmInterface.long_range_backend import (
        per_atom_jax_pme_c6_sqrt_for_atoms,
    )

    rng = np.random.default_rng(19)
    n_mono = int(args.n_monomers)
    size = int(args.atoms_per_monomer)
    n = n_mono * size
    pos = rng.random((n, 3)) * float(args.box_side) * 0.7
    chg = rng.normal(0.0, 0.1, n)
    offsets = np.arange(0, n + 1, size, dtype=np.int64)
    box_L = float(args.box_side)
    cell = np.diag([box_L, box_L, box_L])
    eps = np.full(n, 0.05, dtype=np.float64)
    sig = np.full(n, 2.0, dtype=np.float64)
    c6 = per_atom_jax_pme_c6_sqrt_for_atoms(eps, sig)
    ctx = {
        "pos": pos,
        "chg": chg,
        "offsets": offsets,
        "box_L": box_L,
        "cell": cell,
        "c6": c6,
        "method": str(args.method),
        "sr": float(args.sr_cutoff),
        "prefactor_coulomb": float(jpref.kcalmol_A),
        "n_atoms": n,
        "n_monomers": n_mono,
    }
    return ctx


def _bench_jax_warmup(
    name: str,
    category: str,
    fn: Callable[[], Any],
    *,
    steady_reps: int,
    notes: str = "",
) -> BenchRow:
    from mmml.utils.jax_gpu_warmup import (
        block_jax_values,
        reset_jax_compile_timers,
        run_jax_warmup_passes,
        summarize_jax_compile_timers,
    )

    reset_jax_compile_timers()
    run_jax_warmup_passes(name, 2, fn, block=lambda out: block_jax_values(out))
    records = summarize_jax_compile_timers()
    rec = next((r for r in records if r["label"] == name), None)
    compile_s = rec["compile_s"] if rec else None
    run_s = rec["run_s"] if rec else None

    steady_ms = None
    if steady_reps > 0:
        t0 = time.perf_counter()
        for _ in range(steady_reps):
            block_jax_values(fn())
        steady_ms = (time.perf_counter() - t0) * 1000.0 / steady_reps

    return BenchRow(
        name=name,
        category=category,
        compile_s=compile_s,
        run_s=run_s,
        steady_ms=steady_ms,
        notes=notes,
    )


def _bench_host_call(
    name: str,
    category: str,
    fn: Callable[[], Any],
    *,
    steady_reps: int,
    notes: str = "",
) -> BenchRow:
    """Host numpy-returning jax-pme evaluators (JIT inside evaluator)."""

    def _wrapped():
        return fn()

    return _bench_jax_warmup(
        name,
        category,
        _wrapped,
        steady_reps=steady_reps,
        notes=notes,
    )


def _bench_hybrid_components(
    ctx: dict[str, Any],
    *,
    intra_mode: str,
    steady_reps: int,
) -> list[BenchRow]:
    from mmml.interfaces.pycharmmInterface.jax_pme_cross_monomer import (
        consume_cross_monomer_profile,
    )
    from mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb import (
        consume_hybrid_jax_pme_profile,
        hybrid_jax_pme_mm_lr_correction,
    )

    os.environ["MMML_JAX_PME_INTRA_MODE"] = intra_mode
    os.environ["MMML_JAX_PME_PROFILE"] = "1"

    def _hybrid():
        return hybrid_jax_pme_mm_lr_correction(
            ctx["pos"],
            ctx["chg"],
            ctx["offsets"],
            box_length_A=ctx["box_L"],
            method=ctx["method"],
            sr_cutoff_A=ctx["sr"],
            c6_sqrt=ctx["c6"],
            pbc_cell=ctx["cell"],
            mm_switch_on=6.0,
            mm_switch_width=4.0,
        )

    total = _bench_host_call(
        f"hybrid_mm_lr_total_{intra_mode}",
        "hybrid_lr",
        _hybrid,
        steady_reps=steady_reps,
        notes=f"MMML_JAX_PME_INTRA_MODE={intra_mode}",
    )
    rows = [total]

    consume_cross_monomer_profile()
    consume_hybrid_jax_pme_profile()
    for _ in range(max(1, steady_reps)):
        _hybrid()
    for label, stats in consume_hybrid_jax_pme_profile().items():
        rows.append(
            BenchRow(
                name=f"{label} ({intra_mode})",
                category="hybrid_component",
                compile_s=None,
                run_s=None,
                steady_ms=float(stats["mean_ms"]),
                notes="MMML_JAX_PME_PROFILE steady mean",
            )
        )
    for label, stats in consume_cross_monomer_profile().items():
        rows.append(
            BenchRow(
                name=f"{label} ({intra_mode})",
                category="cross_kernel",
                compile_s=None,
                run_s=None,
                steady_ms=float(stats["mean_ms"]),
                notes="cross-monomer kernel steady mean",
            )
        )
    return rows


def _bench_jax_pme_primitives(ctx: dict[str, Any], *, steady_reps: int) -> list[BenchRow]:
    from mmml.interfaces.pycharmmInterface.jax_pme_cross_monomer import (
        compute_jax_pme_cross_monomer_power_law,
    )
    from mmml.interfaces.pycharmmInterface.long_range_backend import (
        compute_jax_pme_coulomb,
        compute_jax_pme_lj_dispersion,
        compute_jax_pme_power_law,
    )

    pos = ctx["pos"]
    chg = ctx["chg"]
    offsets = ctx["offsets"]
    box_L = ctx["box_L"]
    method = ctx["method"]
    sr = ctx["sr"]
    pref = ctx["prefactor_coulomb"]
    c6 = ctx["c6"]
    m0 = slice(int(offsets[0]), int(offsets[1]))

    rows = [
        _bench_host_call(
            "jax_pme_coulomb_full",
            "jax_pme_host",
            lambda: compute_jax_pme_coulomb(pos, chg, box_length_A=box_L, method=method, sr_cutoff_A=sr),
            steady_reps=steady_reps,
        ),
        _bench_host_call(
            "jax_pme_dispersion_full",
            "jax_pme_host",
            lambda: compute_jax_pme_lj_dispersion(pos, c6, box_length_A=box_L, method=method, sr_cutoff_A=sr),
            steady_reps=steady_reps,
        ),
        _bench_host_call(
            "jax_pme_coulomb_intra_m0",
            "jax_pme_host",
            lambda: compute_jax_pme_power_law(
                pos[m0], chg[m0], box_length_A=box_L, method=method, sr_cutoff_A=sr,
                exponent=1, prefactor=pref,
            ),
            steady_reps=steady_reps,
            notes="representative monomer slice",
        ),
        _bench_host_call(
            "jax_pme_dispersion_intra_m0",
            "jax_pme_host",
            lambda: compute_jax_pme_lj_dispersion(
                pos[m0], c6[m0], box_length_A=box_L, method=method, sr_cutoff_A=sr,
            ),
            steady_reps=steady_reps,
            notes="representative monomer slice",
        ),
        _bench_host_call(
            "cross_monomer_coulomb",
            "jax_pme_cross",
            lambda: compute_jax_pme_cross_monomer_power_law(
                pos, chg, offsets, box_length_A=box_L, method=method, sr_cutoff_A=sr,
                exponent=1, prefactor=pref,
            ),
            steady_reps=steady_reps,
        ),
        _bench_host_call(
            "cross_monomer_dispersion",
            "jax_pme_cross",
            lambda: compute_jax_pme_cross_monomer_power_law(
                pos, c6, offsets, box_length_A=box_L, method=method, sr_cutoff_A=sr,
                exponent=6, prefactor=-1.0,
            ),
            steady_reps=steady_reps,
        ),
    ]
    return rows


def _bench_mlpot_primitives(args: argparse.Namespace, *, steady_reps: int) -> list[BenchRow]:
    ckpt = args.checkpoint
    if ckpt is None:
        return []
    if not ckpt.is_file():
        print(f"  skip MLpot: checkpoint not found: {ckpt}", file=sys.stderr)
        return []

    # JAX compile timing only — avoid import-time CHARMM BLOCK/crystal on empty PSF
    # (MPI-linked libcharmm can stall for minutes with no progress output).
    os.environ.setdefault("MMML_WARMUP_MLPOT_JAX_ONLY", "1")
    print(
        "\n--- MLpot JAX warmup primitives "
        "(MMML_WARMUP_MLPOT_JAX_ONLY=1; compile may take ~30–120s on CPU) ---",
        flush=True,
    )

    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        build_decomposed_mlpot_model,
        warmup_decomposed_mlpot,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.medium_pbc_validation import (
        lattice_positions_cubic_pbc,
    )
    from mmml.utils.jax_gpu_warmup import (
        reset_hybrid_spherical_warmup_cache,
        reset_jax_compile_timers,
        summarize_jax_compile_timers,
    )

    n_mono = int(args.n_monomers)
    size = int(args.atoms_per_monomer)
    z = np.array([6, 1, 1] * n_mono, dtype=int)[: n_mono * size]
    if len(z) < n_mono * size:
        z = np.resize(z, n_mono * size)
    per = [size] * n_mono
    box = float(args.box_side)
    pos = lattice_positions_cubic_pbc(n_mono, size, box, spacing_A=3.5, seed=11)
    cell = box

    reset_hybrid_spherical_warmup_cache()
    reset_jax_compile_timers()
    t0 = time.perf_counter()
    model = build_decomposed_mlpot_model(
        ckpt,
        z,
        per,
        n_mono,
        ml_batch_size=int(args.ml_batch_size),
        cell=cell,
        defer_jax_until_mlpot_registered=False,
        defer_jax_until_after_sd=False,
    )
    warmup_decomposed_mlpot(model, pos, cell=cell, verbose=False)
    wall_s = time.perf_counter() - t0

    rows: list[BenchRow] = []
    for rec in summarize_jax_compile_timers():
        label = str(rec["label"])
        compile_s = rec["compile_s"]
        run_s = rec["run_s"]
        if isinstance(compile_s, (int, float)) and isinstance(run_s, (int, float)):
            steady_ms = float(run_s) * 1000.0
        elif isinstance(run_s, (int, float)):
            steady_ms = float(run_s) * 1000.0
        else:
            steady_ms = None
        rows.append(
            BenchRow(
                name=label,
                category="mlpot_jax",
                compile_s=float(compile_s) if compile_s is not None else None,
                run_s=float(run_s) if run_s is not None else None,
                steady_ms=steady_ms,
                notes=f"warmup_decomposed_mlpot total wall={wall_s:.2f}s",
            )
        )

    if steady_reps > 0 and model._spherical_fn is not None:
        import jax.numpy as jnp

        z_j = jnp.asarray(z)
        r_j = jnp.asarray(pos)
        side = float(box)
        box_j = jnp.asarray([[side, 0.0, 0.0], [0.0, side, 0.0], [0.0, 0.0, side]])
        mm_pair_idx = None
        mm_pair_mask = None
        if model._do_mm and model._get_update_fn is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
                _box_numpy_for_update,
            )

            update_fn = model._get_update_fn(pos, model._cutoff_params, box=box_j)
            if update_fn is not None:
                box_np = _box_numpy_for_update(box_j)
                if box_np is not None:
                    mm_pair_idx, mm_pair_mask = update_fn(pos, box=box_np)
                else:
                    mm_pair_idx, mm_pair_mask = update_fn(pos)
                mm_pair_idx = jnp.asarray(mm_pair_idx)
                mm_pair_mask = jnp.asarray(mm_pair_mask)

        def _spherical():
            return model._spherical_fn(
                positions=r_j,
                atomic_numbers=z_j,
                n_monomers=n_mono,
                cutoff_params=model._cutoff_params,
                doML=True,
                doMM=model._do_mm,
                doML_dimer=True,
                mm_pair_idx=mm_pair_idx,
                mm_pair_mask=mm_pair_mask,
                box=box_j,
            )

        row = _bench_jax_warmup(
            "spherical_cutoff_steady",
            "mlpot_jax",
            _spherical,
            steady_reps=steady_reps,
            notes="post-warmup re-bench",
        )
        rows.append(row)

    return rows


def _print_table(rows: list[BenchRow]) -> None:
    hdr = f"{'name':<36} {'category':<18} {'compile_s':>10} {'run_s':>10} {'steady_ms':>10}"
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        compile_s = f"{row.compile_s:.3f}" if row.compile_s is not None else "-"
        run_s = f"{row.run_s:.3f}" if row.run_s is not None else "-"
        steady = f"{row.steady_ms:.2f}" if row.steady_ms is not None else "-"
        print(f"{row.name:<36} {row.category:<18} {compile_s:>10} {run_s:>10} {steady:>10}")


def main() -> int:
    args = _parse_args()
    print_header("calculator primitive benchmark (compile vs run)")

    if not have_jax_pme_package():
        print("jax-pme not installed", file=sys.stderr)
        return 1

    os.environ.setdefault("MMML_JAX_COMPILE_TIMERS", "1")
    os.environ.setdefault("MMML_JAX_PME_INTRA_MODE", "cross")

    ctx = _synthetic_cluster(args)
    rows: list[BenchRow] = []

    print(f"\nSystem: {ctx['n_monomers']} monomers, {ctx['n_atoms']} atoms, L={ctx['box_L']} Å, method={ctx['method']}")
    print(f"JAX backend: {os.environ.get('JAX_PLATFORMS', 'default')}\n")

    print("--- jax-pme host primitives ---")
    rows.extend(_bench_jax_pme_primitives(ctx, steady_reps=int(args.steady_reps)))

    print("\n--- hybrid long-range (cross mode) ---")
    rows.extend(_bench_hybrid_components(ctx, intra_mode="cross", steady_reps=int(args.steady_reps)))

    if args.legacy_intra:
        print("\n--- hybrid long-range (legacy full_minus_intra) ---")
        rows.extend(_bench_hybrid_components(ctx, intra_mode="full_minus_intra", steady_reps=int(args.steady_reps)))

    mlpot_rows = _bench_mlpot_primitives(args, steady_reps=int(args.steady_reps))
    if mlpot_rows:
        rows.extend(mlpot_rows)

    print("\n=== summary ===")
    _print_table(rows)

    report = {
        "system": {
            "n_monomers": ctx["n_monomers"],
            "n_atoms": ctx["n_atoms"],
            "box_side_A": ctx["box_L"],
            "method": ctx["method"],
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "intra_mode_default": os.environ.get("MMML_JAX_PME_INTRA_MODE"),
        },
        "primitives": [asdict(r) for r in rows],
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")

    print_pass("calculator primitive benchmark complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
