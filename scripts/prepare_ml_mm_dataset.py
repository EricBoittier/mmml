#!/usr/bin/env python3
"""Multi-core CGenFF enrichment for Orbax ML/MM caches (ragged bulk datasets).

The reusable core -- CGenFF PRM/RTF parsing, monomer segmentation, template
matching, graph-isomorphism reordering and the inter-monomer MM baseline -- now
lives in :mod:`mmml.data.cgenff_dataset`.  This script is the Orbax-cache driver
that stays here for the DES-S66 bulk workflow.

For dense **NPZ** training splits (e.g. the tutorial's ``mp2_nms15_clean_*.npz``)
use the first-class CLI instead::

    mmml prepare-mm-dataset -i train.npz -o train_mm.npz

Both paths share the same assignment logic, so they produce identical
``cgenff_type_idx`` / ``cgenff_charge`` / ``mol_id`` semantics and master LJ tables.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp

# Ensure repository root is importable when run as a standalone script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mmml.data.cgenff_dataset import (  # noqa: E402
    assign_frame_cgenff,
    format_composition,
    load_reference,
)

# Loaded once per (main or spawn-worker) process via the cached loader.
_REF = load_reference()


def process_single_frame(args_tuple):
    """Worker: assign CGenFF fields to one ragged Orbax frame."""
    z_struct, r_struct, f_struct, energy_i, q_i, s_i, d_i = args_tuple
    assignment, reason = assign_frame_cgenff(z_struct, r_struct, _REF, compute_mm=True)
    if assignment is None:
        return ("SKIP", reason or "unknown")
    return (
        r_struct,
        z_struct,
        f_struct,
        assignment.f_cgenff_mm,
        energy_i,
        assignment.e_cgenff_mm,
        len(z_struct),
        q_i,
        s_i,
        d_i,
        assignment.mol_id,
        assignment.cgenff_type_idx,
        assignment.cgenff_charge,
    )


def process_orbax_cache(
    cache_dir: str | Path,
    output_cache: str | Path,
    max_structures: int | None = None,
    num_workers: int | None = None,
):
    cache_dir = Path(cache_dir).expanduser().resolve()
    output_cache = Path(output_cache).expanduser().resolve()
    workers = num_workers or min(mp.cpu_count(), 32)

    # spawn avoids JAX multithreaded os.fork() deadlocks in Python 3.13.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print("=" * 66)
    print(f" Multi-Core Orbax Cache ML/MM Pre-computer ({workers} CPU Workers)")
    print(
        f" Master Nonbonded Types: {len(_REF.nb_map):,} | "
        f"Registered RESI: {len(_REF.residues):,}"
    )
    print(" Strict Charge Conservation: sum(cgenff_charge) == target_charge (0.0 e)")
    print(f" Source Cache: {cache_dir}")
    print(f" Target Cache: {output_cache}")
    print("=" * 66)

    data = ocp.PyTreeCheckpointer().restore(cache_dir)

    Z_all = np.asarray(data["Z"]).reshape(-1)
    R_all = np.asarray(data["R"]).reshape(-1, 3)
    F_all = np.asarray(data["F"]).reshape(-1, 3)
    offsets = np.asarray(data["mol_offsets"]).reshape(-1)
    E_all = np.asarray(data["E"]).reshape(-1, 1)
    N_all = np.asarray(data["N"]).reshape(-1, 1)
    Q_all = np.asarray(data["Q"]).reshape(-1, 1)
    S_all = np.asarray(data["S"]).reshape(-1, 1)
    D_all = np.asarray(data["D"]).reshape(-1, 3)

    n_total = len(N_all)
    if max_structures:
        n_total = min(n_total, max_structures)

    print(f"[+] Total Structures to Scan: {n_total:,}")
    t0 = time.time()

    def frame_generator():
        for i in range(n_total):
            start, end = offsets[i], offsets[i + 1]
            yield (
                Z_all[start:end], R_all[start:end], F_all[start:end],
                E_all[i], Q_all[i], S_all[i], D_all[i],
            )

    kept_r, kept_z, kept_f, kept_f_cgenff = [], [], [], []
    kept_e, kept_e_cgenff, kept_n, kept_q = [], [], [], []
    kept_s, kept_d, kept_mol_id = [], [], []
    kept_cgenff_type, kept_cgenff_charge = [], []
    kept_offsets = [0]

    dimers_processed = 0
    dropped_total = 0
    dropped_reasons: dict[str, int] = {}

    with mp.Pool(processes=workers) as pool:
        for res in pool.imap(process_single_frame, frame_generator(), chunksize=5000):
            if isinstance(res, tuple) and len(res) == 2 and res[0] == "SKIP":
                dropped_total += 1
                reason = res[1][:120]
                dropped_reasons[reason] = dropped_reasons.get(reason, 0) + 1
                continue

            (r_struct, z_struct, f_struct, f_mm, energy_i, e_mm,
             n_atoms, q_i, s_i, d_i, mol_id, cgenff_type, cgenff_charge) = res

            kept_r.append(r_struct)
            kept_z.append(z_struct)
            kept_f.append(f_struct)
            kept_f_cgenff.append(f_mm)
            kept_e.append(energy_i)
            kept_e_cgenff.append(e_mm)
            kept_n.append(n_atoms)
            kept_q.append(q_i)
            kept_s.append(s_i)
            kept_d.append(d_i)
            kept_mol_id.append(mol_id)
            kept_cgenff_type.append(cgenff_type)
            kept_cgenff_charge.append(cgenff_charge)
            kept_offsets.append(kept_offsets[-1] + n_atoms)

            dimers_processed += 1
            if dimers_processed % 500000 == 0:
                dt = time.time() - t0
                print(f"  Processed {dimers_processed:,} dimers ({dimers_processed / dt:.0f} frames/sec)")

    if dropped_total > 0:
        non_dimer = sum(c for r, c in dropped_reasons.items() if r.startswith("non-dimer"))
        unmapped = sum(c for r, c in dropped_reasons.items() if r.startswith("unmapped"))
        sentinel = sum(c for r, c in dropped_reasons.items() if "DEFAULT sentinel" in r)
        other = dropped_total - non_dimer - unmapped - sentinel
        pct = 100 * dropped_total / (dropped_total + dimers_processed)
        print(f"\n[WARNING] Dropped {dropped_total:,} frames ({pct:.1f}%):")
        print(f"   {non_dimer:>10,}  non-dimer structures — expected")
        print(f"   {unmapped:>10,}  unmapped CGenFF templates")
        if sentinel:
            print(f"   {sentinel:>10,}  sentinel zero-LJ atoms")
        if other:
            print(f"   {other:>10,}  other errors")
        print("\n   Top unmapped compositions:")
        for reason, count in sorted(
            ((r, c) for r, c in dropped_reasons.items() if r.startswith("unmapped")),
            key=lambda x: -x[1],
        )[:15]:
            print(f"   {count:>8,} : {reason}")

    dt = time.time() - t0
    print(f"\n[+] Total Dimer Structures Prepared: {dimers_processed:,} in {dt:.2f}s ({dimers_processed / dt:.0f} frames/sec)")

    output_data = {
        "R": np.concatenate(kept_r, axis=0),
        "Z": np.concatenate(kept_z, axis=0),
        "F": np.concatenate(kept_f, axis=0),
        "F_cgenff_mm": np.concatenate(kept_f_cgenff, axis=0),
        "mol_offsets": np.asarray(kept_offsets, dtype=np.int64),
        "E": np.asarray(kept_e, dtype=np.float64).reshape(-1, 1),
        "E_cgenff_mm": np.asarray(kept_e_cgenff, dtype=np.float64).reshape(-1, 1),
        "N": np.asarray(kept_n, dtype=np.int32).reshape(-1, 1),
        "Q": np.asarray(kept_q, dtype=np.float64).reshape(-1, 1),
        "S": np.asarray(kept_s, dtype=np.float64).reshape(-1, 1),
        "D": np.asarray(kept_d, dtype=np.float64).reshape(-1, 3),
        "mol_id": np.concatenate(kept_mol_id, axis=0),
        "cgenff_type_idx": np.concatenate(kept_cgenff_type, axis=0),
        "cgenff_charge": np.concatenate(kept_cgenff_charge, axis=0),
        "cgenff_master_sigmas": _REF.sigmas,
        "cgenff_master_epsilons": _REF.epsilons,
    }
    output_data["metadata_n_structures"] = np.asarray(dimers_processed, dtype=np.int64)
    output_data["metadata_n_atoms_total"] = np.asarray(kept_offsets[-1], dtype=np.int64)
    output_data["metadata_max_atoms"] = np.asarray(max(kept_n), dtype=np.int32)

    output_cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving enriched Orbax data cache to: {output_cache}")
    ocp.PyTreeCheckpointer().save(output_cache, output_data, force=True)
    print("[+] Prepared dataset successfully saved!")
    print("=" * 66)


def main():
    parser = argparse.ArgumentParser(description="Multi-Core Orbax Cache ML/MM dataset preparer")
    parser.add_argument("--cache-dir", required=True, help="Input source Orbax data cache directory")
    parser.add_argument("--output-cache", default="data/orbax_cache_des_ml_mm", help="Output destination Orbax cache directory")
    parser.add_argument("--max-structures", type=int, default=None, help="Optional frame limit")
    parser.add_argument("--num-workers", type=int, default=None, help="CPU pool size (default: auto-detect)")
    args = parser.parse_args()
    process_orbax_cache(args.cache_dir, args.output_cache, max_structures=args.max_structures, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
