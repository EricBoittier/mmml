#!/usr/bin/env python3
"""Convert an SO3LR-format dimer HDF5 (``qcell_dimers.h5``) to a dense padded NPZ.

This is the missing bridge between the DES dimer set and the hybrid ML/MM
ladder in ``examples/lj_scales``: that ladder wants ``R (n,atoms,3)`` /
``Z (n,atoms)`` / ``N`` / ``E`` / ``F`` / ``D``, and the DES data is per-structure
HDF5 groups.

**Units are already right and are not converted.** FHI-aims writes eV and eV/Ang,
which is what the hybrid MM baseline (``E_cgenff_mm``) and the existing lj_scales
datasets use. ``formation_energy`` is already referenced against
``metadata/free_atom_energy``, so it is the direct analogue of ``E`` in
``examples/dcm_mp2_psf_order.npz``.

Downstream::

    scripts/des_h5_to_npz.py qcell_dimers.h5 -o des_dimers.npz --pad 34
    mmml prepare-mm-dataset --data des_dimers.npz --output des_dimers_cgenff.npz
    scripts/filter_mm_dataset_by_residue.py des_dimers_cgenff.npz \\
        --top 40 -o des_top40.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def convert(
    h5_path: Path,
    out_path: Path,
    *,
    pad: int,
    charge_filter: float | None,
    max_structures: int | None,
    energy_key: str,
    force_key: str,
) -> dict:
    import h5py

    keys_seen: set[str] = set()
    R_l, Z_l, N_l, E_l, F_l, D_l = [], [], [], [], [], []
    n_seen = n_skipped_charge = n_skipped_size = 0

    with h5py.File(h5_path, "r") as fh:
        groups = [k for k in fh.keys() if k != "metadata"]
        groups.sort()
        for gi, name in enumerate(groups):
            if max_structures and len(R_l) >= max_structures:
                break
            g = fh[name]
            if not keys_seen:
                keys_seen = set(g.keys())
                missing = {energy_key, force_key, "positions", "atomic_numbers"} - keys_seen
                if missing:
                    raise KeyError(
                        f"{h5_path} group '{name}' is missing {sorted(missing)}; "
                        f"available: {sorted(keys_seen)}"
                    )
            n_seen += 1

            if charge_filter is not None:
                q = float(g["charge"][()]) if "charge" in g else 0.0
                if abs(q - charge_filter) > 1e-6:
                    n_skipped_charge += 1
                    continue

            z = np.asarray(g["atomic_numbers"][()], dtype=np.int32).reshape(-1)
            n_at = z.size
            if n_at > pad:
                n_skipped_size += 1
                continue

            r = np.asarray(g["positions"][()], dtype=np.float64).reshape(n_at, 3)
            f = np.asarray(g[force_key][()], dtype=np.float64).reshape(n_at, 3)

            # Zero-padded to `pad`; Z == 0 is the padding sentinel the rest of
            # the pipeline (prepare-mm-dataset, physnet-train) keys off.
            zp = np.zeros(pad, dtype=np.int32)
            rp = np.zeros((pad, 3), dtype=np.float64)
            fp = np.zeros((pad, 3), dtype=np.float64)
            zp[:n_at] = z
            rp[:n_at] = r
            fp[:n_at] = f

            Z_l.append(zp)
            R_l.append(rp)
            F_l.append(fp)
            N_l.append(n_at)
            E_l.append(float(g[energy_key][()]))
            D_l.append(np.asarray(g["dipole"][()], dtype=np.float64).reshape(3)
                       if "dipole" in g else np.zeros(3))

            if (gi + 1) % 25000 == 0:
                print(f"  {gi + 1}/{len(groups)} groups, {len(R_l)} kept",
                      file=sys.stderr, flush=True)

    if not R_l:
        raise RuntimeError(f"No structures survived conversion from {h5_path}")

    out = {
        "R": np.stack(R_l),
        "Z": np.stack(Z_l),
        "F": np.stack(F_l),
        "N": np.asarray(N_l, dtype=np.int64),
        "E": np.asarray(E_l, dtype=np.float64),
        "D": np.stack(D_l),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    return {
        "n_groups": n_seen,
        "n_kept": len(R_l),
        "n_skipped_charge": n_skipped_charge,
        "n_skipped_size": n_skipped_size,
        "pad": pad,
        "output": str(out_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5", type=Path, help="e.g. ~/qcell/qcell_dimers.h5")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--pad", type=int, default=34,
                    help="padded atom count; 34 is the DES dimer maximum")
    ap.add_argument("--charge", type=float, default=0.0,
                    help="keep only structures with this net charge "
                         "(default 0.0 = neutral only, matching trainDES)")
    ap.add_argument("--all-charges", action="store_true",
                    help="disable the charge filter")
    ap.add_argument("--max-structures", type=int, default=None)
    ap.add_argument("--energy-key", default="formation_energy",
                    help="free-atom-referenced energy (eV)")
    ap.add_argument("--force-key", default="total_forces", help="forces (eV/Ang)")
    a = ap.parse_args(argv)

    info = convert(
        a.h5.expanduser(), a.output.expanduser(),
        pad=a.pad,
        charge_filter=None if a.all_charges else a.charge,
        max_structures=a.max_structures,
        energy_key=a.energy_key, force_key=a.force_key,
    )
    print(f"{info['n_kept']:,} / {info['n_groups']:,} structures kept "
          f"(pad {info['pad']}); dropped {info['n_skipped_charge']:,} on charge, "
          f"{info['n_skipped_size']:,} oversized")
    print(f"wrote {info['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
