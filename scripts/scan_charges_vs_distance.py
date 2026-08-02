#!/usr/bin/env python3
"""Per-atom predicted charges (and energy terms) across a dimer distance scan.

run_dimer_scan_campaign.py's scan_results.csv has per-term ENERGY
decomposition but never saved per-atom CHARGES -- this script fills that
gap: for each requested pair, reuses the exact same oriented scan geometries
(mmml.analysis.dimer_molecules.make_oriented_scan_geometries) at the exact
distances already present in an existing scan_results.csv (for direct
overlay with the energy plots), runs the model once per distance point (no
monomer subtraction needed -- we want the raw per-atom charges the model
assigns in the AB context, not an interaction-referenced quantity), and
records every atom's charge plus the predicted total (sum_charges).

Usage:
    python scripts/scan_charges_vs_distance.py \\
        --checkpoint artifacts/spooky_so3lr_muon3/epoch-0010 \\
        --reference-csv results/dimer_scan_campaign_muon3_ep10/scan_results.csv \\
        --pairs TIP3:TIP3 MEOH:MEOH TIP3:MEOH DCM:DCM ACE:ACE \\
        --out-csv eval_out/charges_vs_distance_ep10.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _distances_for_pair(reference_csv: Path, label_a: str, label_b: str) -> list[float]:
    """Distinct offset=0 distances already scanned for this pair in an existing CSV."""
    with reference_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    seen = set()
    out = []
    for r in rows:
        if r["molecule_a"] != label_a or r["molecule_b"] != label_b:
            continue
        try:
            off = float(r["offset_angstrom"])
        except ValueError:
            continue
        if off != 0.0:
            continue
        d = float(r["distance_angstrom"])
        if d not in seen:
            seen.add(d)
            out.append(d)
    return sorted(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Orbax epoch-N checkpoint dir")
    p.add_argument("--reference-csv", required=True, type=Path, help="Existing scan_results.csv to reuse distances from")
    p.add_argument("--pairs", nargs="+", required=True, help="Pairs as LABEL_A:LABEL_B, e.g. TIP3:TIP3")
    p.add_argument("--charge-key", default="charge")
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()

    import e3x
    import jax
    import jax.numpy as jnp

    import evaluate_so3lr_spooky_extxyz as ev
    from mmml.analysis.dimer_molecules import make_oriented_scan_geometries
    from mmml.utils.model_checkpoint import infer_trainable_zbl_config

    checkpoint_path = Path(args.checkpoint).resolve()
    params, config = ev.restore_checkpoint(checkpoint_path)
    config = infer_trainable_zbl_config(config, params)

    pairs = []
    for spec in args.pairs:
        a, b = spec.split(":")
        pairs.append((a, b))

    # Gather all (pair, atoms, distance) samples first, so we build ONE model
    # padded to the global max atom count (matches decompose_so3lr_terms_vs_natoms.py).
    samples: list[tuple[str, str, float, object]] = []
    for a, b in pairs:
        distances = _distances_for_pair(args.reference_csv, a, b)
        if not distances:
            print(f"WARNING: no offset=0 distances found for {a}+{b} in {args.reference_csv}; skipping")
            continue
        for geom in make_oriented_scan_geometries(a, b, distances, offsets_angstrom=[0.0]):
            samples.append((a, b, geom.distance_angstrom, geom.atoms))
    if not samples:
        raise ValueError("No samples generated")

    max_atoms = max(len(atoms) for _, _, _, atoms in samples)
    print(f"{len(samples)} total (pair, distance) samples; model padded to max_atoms={max_atoms}")

    model = ev.create_model_from_config(config, max_atoms=max_atoms)
    dst_idx_np, src_idx_np = e3x.ops.sparse_pairwise_indices(max_atoms)
    dst_idx = jnp.asarray(dst_idx_np)
    src_idx = jnp.asarray(src_idx_np)
    batch_segments = jnp.zeros((max_atoms,), dtype=jnp.int32)

    def _apply(z, pos, q, s, atom_mask, batch_mask):
        return model.apply(
            params,
            atomic_numbers=z,
            charges=q,
            spins=s,
            positions=pos,
            dst_idx=dst_idx,
            src_idx=src_idx,
            batch_segments=batch_segments,
            batch_size=1,
            batch_mask=batch_mask,
            atom_mask=atom_mask,
            compute_forces=False,
        )

    apply_fn = jax.jit(_apply)

    rows: list[dict] = []
    for pair_a, pair_b, distance, atoms in samples:
        n_real = len(atoms)
        z_real = atoms.get_atomic_numbers()
        z = np.zeros(max_atoms, dtype=np.int32)
        z[:n_real] = z_real
        pos = np.zeros((max_atoms, 3), dtype=np.float32)
        pos[:n_real] = atoms.get_positions()
        pad = max_atoms - n_real
        if pad:
            far = 1.0e4 + 100.0 * np.arange(pad, dtype=np.float32)
            pos[n_real:] = np.stack([far, np.zeros(pad, dtype=np.float32), np.zeros(pad, dtype=np.float32)], axis=1)

        atom_mask = (z > 0).astype(np.float32)
        valid_pairs = (atom_mask[dst_idx_np] > 0) & (atom_mask[src_idx_np] > 0)
        batch_mask = valid_pairs.astype(np.float32)

        charge_target = float(atoms.info.get(args.charge_key, 0.0)) if hasattr(atoms, "info") else 0.0
        spin = ev._infer_spin_multiplicity(atoms, charge_target)
        q = jnp.full((max_atoms, 1), charge_target, dtype=jnp.float32)
        s = jnp.full((max_atoms, 1), spin, dtype=jnp.float32)

        out = apply_fn(jnp.asarray(z), jnp.asarray(pos), q, s, jnp.asarray(atom_mask), jnp.asarray(batch_mask))
        charges_arr = np.asarray(out["charges"]).reshape(max_atoms, -1)[:n_real, 0]
        elec = float(np.asarray(out["electrostatics"]).reshape(-1)[0])
        sum_charges_pred = float(np.asarray(out["sum_charges"]).reshape(-1)[0])

        # mol_id: which fragment (A=0 / B=1) each real atom belongs to, from the
        # geometry helper's own "mol_id" array info (set by make_oriented_scan_geometries).
        mol_id = np.asarray(atoms.arrays.get("mol_id", np.zeros(n_real, dtype=int)))[:n_real]

        for i in range(n_real):
            rows.append(
                {
                    "pair": f"{pair_a}+{pair_b}",
                    "distance": distance,
                    "atom_index": i,
                    "fragment": "A" if mol_id[i] == 0 else "B",
                    "element_Z": int(z_real[i]),
                    "charge": float(charges_arr[i]),
                    "sum_charges_total": sum_charges_pred,
                    "electrostatics_total": elec,
                    "target_charge": charge_target,
                }
            )
        print(
            f"{pair_a}+{pair_b:6s} d={distance:6.2f}  sum_q={sum_charges_pred:8.4f} "
            f"(target={charge_target:g})  q_range=[{charges_arr.min():.4f}, {charges_arr.max():.4f}]"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
