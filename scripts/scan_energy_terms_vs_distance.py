#!/usr/bin/env python3
"""Training-consistent per-term interaction-energy decomposition across a dimer scan.

Unlike scripts/run_dimer_scan_campaign.py, this NEVER constructs or passes
CGenFF metadata (mol_id / cgenff_type_idx / master sigma-epsilon tables) --
so3lr_train.extxyz carries no such fields, so the CGenFF-vdW-scale head was
never trained and injecting that metadata at eval time activates an
untrained pathway that contaminates the reported total energy (energy =
atomic_energies + electrostatics + repulsion + atomic_vdw is additive, so a
spurious atomic_vdw term changes "energy" itself, not just a separate
line). This script only ever calls the model the same way training did:
atomic_numbers/positions/charges/spins, nothing else.

For each requested pair, reuses the exact same oriented scan geometries as
scan_charges_vs_distance.py, computes E_dimer, E_fragA, E_fragB (and their
electrostatics/repulsion/cgenff_vdw/atomic-energy sub-terms) with ONE model
call each, and reports E_int = E_dimer - E_fragA - E_fragB per term.

Usage:
    python scripts/scan_energy_terms_vs_distance.py \\
        --checkpoint artifacts/spooky_so3lr_muon3/epoch-0010 \\
        --reference-csv results/dimer_scan_campaign_muon3_ep10/scan_results.csv \\
        --pairs TIP3:TIP3 MEOH:MEOH TIP3:MEOH DCM:DCM ACE:ACE \\
        --out-csv eval_out/energy_terms_clean_ep10.csv
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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference-csv", required=True, type=Path)
    p.add_argument("--pairs", nargs="+", required=True, help="LABEL_A:LABEL_B, e.g. TIP3:TIP3")
    p.add_argument("--charge-key", default="charge")
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()

    import e3x
    import jax
    import jax.numpy as jnp

    import evaluate_so3lr_spooky_extxyz as ev
    from mmml.analysis.dimer_molecules import make_oriented_scan_geometries
    from mmml.utils.model_checkpoint import infer_trainable_zbl_config

    EV_TO_KCAL_MOL = 23.060548867

    checkpoint_path = Path(args.checkpoint).resolve()
    params, config = ev.restore_checkpoint(checkpoint_path)
    config = infer_trainable_zbl_config(config, params)

    pairs = [tuple(spec.split(":")) for spec in args.pairs]

    # Gather (pair, distance, dimer_atoms, fragA_atoms, fragB_atoms) samples.
    samples = []
    max_atoms = 0
    for a, b in pairs:
        distances = _distances_for_pair(args.reference_csv, a, b)
        if not distances:
            print(f"WARNING: no offset=0 distances for {a}+{b}; skipping")
            continue
        for geom in make_oriented_scan_geometries(a, b, distances, offsets_angstrom=[0.0]):
            mol_id = np.asarray(geom.atoms.arrays["mol_id"])
            idx_a = np.flatnonzero(mol_id == 0)
            idx_b = np.flatnonzero(mol_id == 1)
            frag_a = geom.atoms[idx_a]
            frag_b = geom.atoms[idx_b]
            samples.append((f"{a}+{b}", geom.distance_angstrom, geom.atoms, frag_a, frag_b))
            max_atoms = max(max_atoms, len(geom.atoms), len(frag_a), len(frag_b))

    if not samples:
        raise ValueError("No samples generated")
    print(f"{len(samples)} (pair, distance) samples; model padded to max_atoms={max_atoms}")

    model = ev.create_model_from_config(config, max_atoms=max_atoms)
    dst_idx_np, src_idx_np = e3x.ops.sparse_pairwise_indices(max_atoms)
    dst_idx = jnp.asarray(dst_idx_np)
    src_idx = jnp.asarray(src_idx_np)
    batch_segments = jnp.zeros((max_atoms,), dtype=jnp.int32)

    def _apply(z, pos, q, s, atom_mask, batch_mask):
        return model.apply(
            params, atomic_numbers=z, charges=q, spins=s, positions=pos,
            dst_idx=dst_idx, src_idx=src_idx, batch_segments=batch_segments,
            batch_size=1, batch_mask=batch_mask, atom_mask=atom_mask,
            compute_forces=False,
        )

    apply_fn = jax.jit(_apply)

    def _terms(atoms):
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

        def _batch_scalar(key):
            val = out.get(key)
            return 0.0 if val is None else float(np.asarray(val).reshape(-1)[0])

        def _sum_per_atom(key):
            val = out.get(key)
            if val is None:
                return 0.0
            return float(np.sum(np.asarray(val).reshape(max_atoms, -1)[:n_real]))

        e_total = _batch_scalar("energy")
        elec = _batch_scalar("electrostatics")
        rep = _sum_per_atom("repulsion")
        cgenff = _batch_scalar("cgenff_vdw")  # will be 0.0/None: never supplied -> never active
        neural = e_total - elec - rep - cgenff
        return {"total": e_total, "neural": neural, "electrostatics": elec, "zbl": rep, "cgenff_vdw": cgenff}

    rows = []
    for pair, distance, dimer_atoms, frag_a, frag_b in samples:
        t_dimer = _terms(dimer_atoms)
        t_a = _terms(frag_a)
        t_b = _terms(frag_b)
        row = {"pair": pair, "distance": distance}
        for key in ("total", "neural", "electrostatics", "zbl", "cgenff_vdw"):
            e_int = t_dimer[key] - t_a[key] - t_b[key]
            row[f"Eint_{key}_ev"] = e_int
            row[f"Eint_{key}_kcal_mol"] = e_int * EV_TO_KCAL_MOL
        rows.append(row)
        print(
            f"{pair:12s} d={distance:6.2f}  "
            f"total={row['Eint_total_kcal_mol']:9.3f}  neural={row['Eint_neural_kcal_mol']:9.3f}  "
            f"elec={row['Eint_electrostatics_kcal_mol']:8.3f}  zbl={row['Eint_zbl_kcal_mol']:8.3f}  "
            f"cgenff={row['Eint_cgenff_vdw_kcal_mol']:8.3f} (should be ~0, never supplied)"
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
