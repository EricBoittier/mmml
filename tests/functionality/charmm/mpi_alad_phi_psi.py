#!/usr/bin/env python3
"""Workshop-style MPI smoke: alanine dipeptide phi/psi grid (Phase 0).

Port of Brooks pyCHARMM-Workshop ``3SimpleMPIExample`` — embarrassingly parallel
CHARMM minimizations sharded across mpi4py ranks, gathered on rank 0.

User-run on a CHARMM + OpenMPI node (not CI):

  MMML_MPI_NP=4 ./scripts/mmml-charmm-mpirun.sh python \\
    tests/functionality/charmm/mpi_alad_phi_psi.py --n-phi 12 --n-psi 12 \\
    -o /tmp/alad_phi_psi_mpi.json

Serial reference:

  ./scripts/mmml-charmm-mpirun.sh python \\
    tests/functionality/charmm/mpi_alad_phi_psi.py --n-phi 12 --n-psi 12 \\
    -o /tmp/alad_phi_psi_serial.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPI alanine dipeptide phi/psi smoke")
    parser.add_argument("--n-phi", type=int, default=12, help="Phi grid points (-180..180)")
    parser.add_argument("--n-psi", type=int, default=12, help="Psi grid points (-180..180)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("phi_psi_energies.json"),
        help="Output JSON (rank 0 only)",
    )
    parser.add_argument("--dihedral-k", type=float, default=500.0, help="Dihedral restraint k")
    parser.add_argument("--mini-steps", type=int, default=200, help="ABNR steps per grid point")
    return parser.parse_args()


def _protein_toppar_paths() -> tuple[str, str]:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CHARMM_HOME

    base = Path(CHARMM_HOME) / "toppar"
    rtf = base / "top_all36_prot.rtf"
    prm = base / "par_all36m_prot.prm"
    if not rtf.is_file() or not prm.is_file():
        raise FileNotFoundError(
            f"Protein toppar not found under {base}. "
            "Set CHARMM_HOME to a full CHARMM installation (workshop smoke)."
        )
    return str(rtf), str(prm)


def _build_alad_and_minimize_base() -> None:
    from pycharmm import generate, ic, minimize, read, settings
    from pycharmm.scripts import NonBondedScript

    rtf, prm = _protein_toppar_paths()
    settings.set_verbosity(5)
    settings.set_warn_level(-5)
    read.rtf(rtf)
    read.prm(prm)
    read.sequence_string("ALA")
    generate.new_segment(
        seg_name="ALAD",
        first_patch="ACE",
        last_patch="CT3",
        setup_ic=True,
    )
    ic.prm_fill(replace_all=True)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()
    NonBondedScript(
        cutnb=16,
        ctofnb=14,
        ctonnb=12,
        atom=True,
        vatom=True,
        eps=1,
        switch=True,
        vswitch=True,
        cdie=True,
    ).run()
    minimize.run_abnr(nstep=500, tolenr=1e-3, tolgrd=1e-3)


def _minimize_at_dihedrals(phi: float, psi: float, *, k: float, nstep: int) -> float:
    from pycharmm import energy, minimize, restraints, settings

    f_cons = "1 CY 1 N 1 CA 1 C"
    y_cons = "1 N 1 CA 1 C 1 NT"
    restraints.angles.dihedral(selection=f_cons, force=k, minimum=f"{phi:4.2f}")
    restraints.angles.dihedral(selection=y_cons, force=k, minimum=f"{psi:4.2f}")
    settings.set_verbosity(5)
    minimize.run_abnr(nstep=nstep, tolenr=1e-3, tolgrd=1e-3)
    restraints.angles.dihedral_clear()
    return float(energy.get_total())


def main() -> int:
    args = _parse_args()
    try:
        from mpi4py import MPI
    except ImportError as exc:
        print(f"mpi_alad_phi_psi: mpi4py required: {exc}", file=sys.stderr)
        return 2

    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    nproc = int(comm.Get_size())

    if rank == 0:
        print(f"mpi_alad_phi_psi: {nproc} ranks, grid {args.n_phi}x{args.n_psi}", flush=True)

    _build_alad_and_minimize_base()
    comm.barrier()

    phi_vals = np.linspace(-180.0, 180.0, int(args.n_phi), endpoint=False)
    psi_vals = np.linspace(-180.0, 180.0, int(args.n_psi), endpoint=False)

    local_map: dict[str, list[float]] = {"phi": [], "psi": [], "energy": []}
    for iphi, phi in enumerate(phi_vals):
        if iphi % nproc != rank:
            continue
        for psi in psi_vals:
            e = _minimize_at_dihedrals(
                float(phi),
                float(psi),
                k=float(args.dihedral_k),
                nstep=int(args.mini_steps),
            )
            local_map["phi"].append(float(phi))
            local_map["psi"].append(float(psi))
            local_map["energy"].append(float(e))

    comm.barrier()

    merged: dict[str, list[float]] = {"phi": [], "psi": [], "energy": []}
    if rank == 0:
        for key in merged:
            merged[key].extend(local_map[key])
    for src in range(1, nproc):
        if rank == src:
            comm.send(local_map, dest=0, tag=10 + src)
        elif rank == 0:
            remote = comm.recv(source=src, tag=10 + src)
            for key in merged:
                merged[key].extend(remote[key])

    comm.barrier()

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_ranks": nproc,
            "n_points": len(merged["phi"]),
            "phi": merged["phi"],
            "psi": merged["psi"],
            "energy_kcal_mol": merged["energy"],
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"mpi_alad_phi_psi: wrote {args.output} ({payload['n_points']} points)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
