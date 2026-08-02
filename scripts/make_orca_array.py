#!/usr/bin/env python3
"""Generate an ORCA SLURM array job for the dimer-scan geometries.

Reference upgrade for the dataset: GFN2-xTB -> PBE0-D4/def2-TZVP. The GEOMETRIES
are identical (read straight from the GFN2 npz), so the two sets are directly
comparable and every diagnostic built against the GFN2 run transfers unchanged.

Why an array here rather than the GPU: measured, PBE0-D4/def2-TZVP/RIJCOSX
TightSCF + EnGrad on an acetone dimer is 29 s on 16 cores. With 128 cores/node
that is 8 concurrent jobs per node, so ~128 concurrent across 16 idle nodes:
22322 geometries in ~1.4 h, against ~23 GPU-hours for gpu4pyscf on one card.

DISK: ORCA writes a ~1.7 MB .gbw plus densities per job -- ~40 GB at this scale,
on a filesystem already at 87%. Each task therefore extracts E/gradient/dipole
into a small .dat and deletes the scratch immediately. Do not remove that.

    python scripts/make_orca_array.py --data gfn2_nms15_train.npz \\
        gfn2_nms15_valid.npz gfn2_nms15_test.npz --out orca_run
    # then, on the cluster:
    sbatch orca_run/run_array.sh
    python scripts/collect_orca_array.py --run-dir orca_run --out pbe0_nms15.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}

INP = """! {keywords}

%pal
  nprocs {nprocs}
end

%maxcore {maxcore}

%scf
  MaxIter 300
  ConvForced true
end

* xyz {charge} {mult}
{geom}
*
"""

RUNNER = """#!/bin/bash
#SBATCH -p {partition}
# ORCA's MPI ranks share scratch files, so every calculation must stay on one
# physical node. A bare ``-n`` lets Slurm spread ranks over several nodes.
#SBATCH --nodes=1
#SBATCH --ntasks={nprocs}
#SBATCH --ntasks-per-node={nprocs}
#SBATCH --cpus-per-task=1
#SBATCH -t {walltime}
#SBATCH --array=0-{last}%{throttle}
#SBATCH -o {run_dir}/logs/task_%a.out
#SBATCH -J orca_dimer

# One task handles a contiguous chunk of geometries, sequentially: job startup
# (~seconds) would otherwise be a large fraction of a 29 s calculation.
#
# sbatch inherits the submitting shell's environment; the `module` function is
# only defined for interactive login shells (sourced from /etc/profile.d), so
# a job submitted from a non-interactive shell (e.g. a plain `ssh host sbatch
# ...`) silently lacks it. Source it explicitly so this works either way.
source /etc/profile.d/00-module.sh 2>/dev/null || true
module load {module}
ORCA=$(which orca)
if [ -z "$ORCA" ]; then echo "orca not on PATH after module load" >&2; exit 1; fi

RUN={run_dir}
CHUNK={chunk}
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK - 1 ))
NTOT={ntot}
[ $END -ge $NTOT ] && END=$(( NTOT - 1 ))

# Node-local scratch: ORCA is IO-heavy and $HOME is shared and 87% full.
WORK=${{TMPDIR:-/tmp}}/orca_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}_$$
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

for i in $(seq $START $END); do
  IDX=$(printf "%06d" $i)
  OUT="$RUN/dat/$IDX.dat"
  [ -s "$OUT" ] && continue            # resumable: skip what is already done

  cp "$RUN/inp/$IDX.inp" "$WORK/j.inp"
  cd "$WORK"
  $ORCA j.inp > j.out 2>&1

  if [ -s j.engrad ]; then
    {{
      echo "# idx $i"
      grep -A2 "current total energy" j.engrad | tail -1
      echo "# gradient Eh/bohr"
      awk '/current gradient/{{f=1;next}} /^#/{{if(f&&n>0)f=0}} f&&NF==1{{print;n++}}' j.engrad
      echo "# dipole au"
      grep "Total Dipole Moment" j.out | tail -1 | awk '{{print $5, $6, $7}}'
    }} > "$OUT"
  else
    echo "# idx $i FAILED" > "$RUN/failed/$IDX.txt"
    grep -iE "error|aborting" j.out | head -3 >> "$RUN/failed/$IDX.txt" 2>/dev/null
  fi
  # Delete the scratch NOW: .gbw + densities are ~3 MB per geometry.
  rm -f j.gbw j.densities* j.prop* j.bibtex j.tmp* j_property.txt
done
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="npz file(s) of geometries")
    p.add_argument("--out", default="orca_run")
    p.add_argument("--keywords",
                   default="PBE0 D4 def2-TZVP def2/J RIJCOSX TightSCF EnGrad",
                   help="EnGrad is required: forces are half the training signal")
    p.add_argument("--nprocs", type=int, default=16)
    p.add_argument("--maxcore", type=int, default=3000)
    p.add_argument("--chunk", type=int, default=140,
                   help="geometries per array task")
    p.add_argument("--throttle", type=int, default=128,
                   help="max concurrent tasks (128 cores/node / nprocs * nodes)")
    p.add_argument("--walltime", default="12:00:00")
    p.add_argument("--partition", default="long")
    p.add_argument("--module", default="orca/orca-openmpi-6.1.0")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--mult", type=int, default=1)
    args = p.parse_args()

    R, Z, N, names = [], [], [], []
    for f in args.data:
        d = np.load(f, allow_pickle=True)
        R.append(np.asarray(d["R"])); Z.append(np.asarray(d["Z"]))
        N.append(np.asarray(d["N"]))
        names.append(np.array([str(x) for x in d["res_name"]]))
    R = np.concatenate(R); Z = np.concatenate(Z)
    N = np.concatenate(N); names = np.concatenate(names)
    n_tot = len(R)

    run = Path(args.out)
    for sub in ("inp", "dat", "logs", "failed"):
        (run / sub).mkdir(parents=True, exist_ok=True)

    for i in range(n_tot):
        n = int(N[i])
        lines = [f"{SYM[int(z)]:2s} {r[0]:14.8f} {r[1]:14.8f} {r[2]:14.8f}"
                 for z, r in zip(Z[i][:n], R[i][:n])]
        (run / "inp" / f"{i:06d}.inp").write_text(
            INP.format(keywords=args.keywords, nprocs=args.nprocs,
                       maxcore=args.maxcore, charge=args.charge, mult=args.mult,
                       geom="\n".join(lines)))

    # index -> geometry metadata, so the collector never has to re-derive it
    np.savez(run / "index.npz", Z=Z, N=N, res_name=names,
             source=np.array(args.data))

    n_tasks = -(-n_tot // args.chunk)
    (run / "run_array.sh").write_text(RUNNER.format(
        partition=args.partition, nprocs=args.nprocs, walltime=args.walltime,
        last=n_tasks - 1, throttle=args.throttle, run_dir=str(run.resolve()),
        module=args.module, chunk=args.chunk, ntot=n_tot))
    (run / "run_array.sh").chmod(0o755)

    per = 29.0  # measured: ACO dimer, PBE0-D4/def2-TZVP/RIJCOSX, 16 cores
    wall = n_tot * per / args.throttle
    print(f"{n_tot} geometries -> {n_tasks} array tasks x {args.chunk} each")
    print(f"  {run}/inp/  ({n_tot} .inp)")
    print(f"  sbatch {run}/run_array.sh")
    print(f"\nESTIMATE at {per:.0f} s/geom, {args.throttle} concurrent: "
          f"{wall/3600:.1f} h wall ({n_tot*per/3600:.0f} core-node-hours)")
    print(f"  per task: {args.chunk} x {per:.0f}s = {args.chunk*per/3600:.1f} h "
          f"(walltime {args.walltime})")
    if args.chunk * per > 0.8 * _secs(args.walltime):
        print("  WARNING: chunk may exceed walltime -- lower --chunk")
    print("\nResumable: tasks skip geometries whose dat/ file already exists.")
    print("Then: python scripts/collect_orca_array.py --run-dir "
          f"{run} --out pbe0_nms15.npz")
    return 0


def _secs(hms: str) -> float:
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


if __name__ == "__main__":
    raise SystemExit(main())
