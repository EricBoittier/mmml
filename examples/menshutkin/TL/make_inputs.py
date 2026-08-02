"""Write one ORCA directory per frame of tl_subset.npz.

    python make_inputs.py                 # writes frames/frame_0001 ... frame_1477

Each directory gets `engrad.inp` and `run.sh`. The name `engrad.inp` is fixed:
`run.sh` invokes ORCA on exactly that filename, so renaming the input silently
produces a job that runs, finds nothing, and exits successfully.

Level of theory
---------------
    ! RI-MP2 aug-cc-pVTZ aug-cc-pVTZ/C TightSCF EnGrad
    %pal nprocs 32 end
    %maxcore 4000

RI-MP2 has **analytic gradients** in ORCA 6.1, so `EnGrad` gives forces as well
as energies. That is the reason to prefer it here over DLPNO-CCSD(T), which has
no analytic (T) gradient -- the earlier CCSD(T) set could only ever have
delivered energies, and a force-free correction constrains what the transfer
model can learn.

`%pal nprocs 32` matches `--ntasks=32` in run.sh, and `%maxcore 4000` matches
`--mem-per-cpu=4000`. Change one and change the other: ORCA sizing its buffers
above the cgroup limit is an OOM kill partway through, not an error at startup.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 17: "Cl", 35: "Br", 53: "I"}

HEADER = """! RI-MP2 aug-cc-pVTZ aug-cc-pVTZ/C TightSCF EnGrad
%pal nprocs {nprocs} end
%maxcore {maxcore}

* xyz {charge} {mult}
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=HERE / "tl_subset.npz")
    p.add_argument("--outdir", type=Path, default=HERE / "frames")
    p.add_argument("--nprocs", type=int, default=32)
    p.add_argument("--maxcore", type=int, default=4000)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    runsh = HERE / "run.sh"
    if not runsh.exists():
        raise SystemExit(f"{runsh} missing -- it is the SLURM template")

    d = np.load(a.npz)
    R, Z, N, Q = d["R"], d["Z"], d["N"], d["Q"]
    xi = d["xi"] if "xi" in d.files else np.full(len(R), np.nan)
    src = d["source_index"] if "source_index" in d.files else np.arange(len(R))

    if a.outdir.exists() and any(a.outdir.iterdir()) and not a.overwrite:
        raise SystemExit(
            f"{a.outdir} already has contents. Pass --overwrite to replace, or "
            f"move it aside -- regenerating over a submitted set would rewrite "
            f"inputs underneath running jobs.")
    a.outdir.mkdir(parents=True, exist_ok=True)

    for i in range(len(R)):
        n = int(N[i])
        z, r = Z[i][:n], R[i][:n]
        charge = int(round(float(Q[i])))
        mult = 1                       # every frame here is closed shell, Q = 0

        dd = a.outdir / f"frame_{i + 1:04d}"
        dd.mkdir(exist_ok=True)

        lines = [HEADER.format(nprocs=a.nprocs, maxcore=a.maxcore,
                               charge=charge, mult=mult)]
        for zi, ri in zip(z, r):
            lines.append(f"  {SYMBOL[int(zi)]:2s} {ri[0]:16.10f} "
                         f"{ri[1]:16.10f} {ri[2]:16.10f}\n")
        lines.append("*\n")
        (dd / "engrad.inp").write_text("".join(lines))

        shutil.copy2(runsh, dd / "run.sh")
        (dd / "run.sh").chmod(0o755)
        (dd / "provenance.txt").write_text(
            f"source_row={int(src[i])} n_atoms={n} charge={charge} "
            f"xi={xi[i]:+.4f}\n" if np.isfinite(xi[i]) else
            f"source_row={int(src[i])} n_atoms={n} charge={charge} "
            f"xi=fragment\n")

    print(f"wrote {len(R)} directories under {a.outdir}")
    print(f"  level  RI-MP2 aug-cc-pVTZ aug-cc-pVTZ/C TightSCF EnGrad")
    print(f"  pal    {a.nprocs}   maxcore {a.maxcore} MB")
    print(f"  next   bash submit_tl.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
