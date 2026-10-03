#!/usr/bin/env python3
"""Run every restraint through every energy engine and report what happened.

CHARMM's engines -- the CPU code, DOMDEC, DOMDEC_GPU, OpenMM and BLaDE -- do
not implement the same restraints.  A restraint one engine honours may be
refused by another and left out entirely by a third, which means a script can
be running different physics depending on how the energy was evaluated.

This drives one small system through every (restraint, engine) pair and
classifies the outcome:

  AGREE    the restraint energy matches the CPU reference
  DIFFER   the engine evaluated it and got a different answer
  ZERO     the engine ran, reported no energy for the term, and said nothing
  REFUSED  the engine stopped the run with a message
  CRASH    the run died without a message that named a restraint
  SKIP     that engine is not in this build

ZERO is the interesting one: the run finished, the number is wrong, and
nothing told the user.

Usage:
    run_matrix.py --charmm PATH [--engines cpu,domdec,openmm] [--np 4]

BLaDE and DOMDEC_GPU need a CUDA build, so they are only reachable where one
exists; the probe self-skips when the engine is absent from the build.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "probe.inp")

# restraint id -> the CHARMM energy term it reports through
RESTRAINTS = [
    ("consharm", "HARM"),
    ("consdihe", "CDIH"),
    ("hmcm", "HMCM"),
    ("noe", "NOE"),
    ("resd", "RESD"),
    ("geo", "GEO"),
    ("geocom", "GEO"),
    ("mdip", "MDIP"),
    ("pull", "PULL"),
    ("rgy", "RGY"),
    ("cdro", "CDRO"),
]

ALL_ENGINES = ["cpu", "domdec", "domdecnosplit", "domdecgpu", "openmm", "blade"]

# A term an engine really computes on the GPU comes back in single precision,
# so it cannot be held to the tolerance used for one CHARMM computed itself.
ENGINE_REL_TOL = {"openmm": 1e-4, "blade": 1e-4}

# engines that must run under MPI to mean anything
NEEDS_MPI = {"domdec", "domdecnosplit"}

# GPU engines want one rank per GPU; more OpenMP threads than GPUs stops BLaDE
SERIAL_ONLY = {"blade", "domdecgpu", "openmm"}

PROBE_RE = re.compile(
    r"^PROBE\s+(\S+)\s+(\S+)\s+TERM\s+(\S+)\s+TOTAL\s+(\S+)", re.M)
SKIP_RE = re.compile(r"^PROBE\s+(\S+)\s+(\S+)\s+SKIP\s+(\S+)", re.M)


def parse_number(tok):
    """CHARMM prints **** when a value overflows its format."""
    try:
        return float(tok)
    except ValueError:
        return None


def run_one(charmm, datadir, rtype, eterm, engine, np, timeout, mpi_cmd):
    """Run one (restraint, engine) pair in its own directory."""
    workdir = tempfile.mkdtemp(prefix="rmx_")
    try:
        os.symlink(os.path.join(datadir, "data"), os.path.join(workdir, "data"))
        shutil.copy(os.path.join(datadir, "datadir.def"), workdir)
        os.mkdir(os.path.join(workdir, "scratch"))

        cmd = []
        if engine in NEEDS_MPI and np > 1:
            # On a Slurm cluster this has to be srun: mpirun launched inside an
            # allocation does not deadlock loudly, it just hangs, and the pair
            # then burns the whole per-run timeout before anyone notices.
            cmd += mpi_cmd.split() + [str(np)]
        cmd += [charmm, "-prevclcg",
                "rtype=" + rtype, "eterm=" + eterm, "engine=" + engine,
                "-i", PROBE]

        try:
            p = subprocess.run(cmd, cwd=workdir, timeout=timeout,
                               capture_output=True, text=True)
            out = p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            return {"status": "CRASH", "note": "timed out"}

        skip = SKIP_RE.search(out)
        if skip:
            return {"status": "SKIP", "note": skip.group(3)}

        m = PROBE_RE.search(out)
        if not m:
            # died before reporting -- find out whether it said why
            why = ""
            for line in out.splitlines():
                if "WRNDIE" in line.upper() or "NOT READY" in line.upper() \
                        or "NOT SUPPORTED" in line.upper() \
                        or "ONLY SUPPORTS" in line.upper():
                    why = line.strip()
                    break
            return {"status": "REFUSED" if why else "CRASH",
                    "note": why[:90] or "no message naming a restraint"}

        return {"status": None,
                "term": parse_number(m.group(3)),
                "total": parse_number(m.group(4)),
                "note": ""}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def classify(ref, got, rel_tol):
    """Compare one engine's result against the CPU reference."""
    if got.get("status"):
        return got["status"], got.get("note", "")
    term = got.get("term")
    if term is None:
        return "DIFFER", "value overflowed its print format"
    if ref is None:
        return "?", "no reference"
    if abs(term) < 1e-10 and abs(ref) > 1e-10:
        return "ZERO", "term reported as zero, no message"
    denom = max(abs(ref), 1e-10)
    if abs(term - ref) / denom <= rel_tol:
        return "AGREE", ""
    return "DIFFER", "%.6f vs %.6f" % (term, ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--charmm", required=True, help="path to the charmm binary")
    ap.add_argument("--datadir", default=None,
                    help="directory holding data/ and datadir.def "
                         "(default: the test/ directory of this checkout)")
    ap.add_argument("--engines", default=",".join(ALL_ENGINES))
    ap.add_argument("--restraints", default=None,
                    help="comma-separated subset, default all")
    ap.add_argument("--np", type=int, default=4, help="MPI ranks for DOMDEC")
    ap.add_argument("--rel-tol", type=float, default=1e-6)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--mpi-cmd", default="mpirun -np",
                    help="launcher for the MPI engines, rank count appended. "
                         "Use 'srun -n' under Slurm.")
    args = ap.parse_args()

    # the probe runs in a scratch directory, so every path handed to it,
    # including the binary, has to be absolute
    args.charmm = os.path.abspath(args.charmm)
    if not os.access(args.charmm, os.X_OK):
        sys.exit("not executable: %s" % args.charmm)

    datadir = args.datadir or os.path.abspath(
        os.path.join(HERE, os.pardir, os.pardir, "test"))
    if not os.path.isdir(os.path.join(datadir, "data")):
        sys.exit("no data/ under %s -- pass --datadir" % datadir)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if "cpu" not in engines:
        engines.insert(0, "cpu")          # the reference has to run first
    wanted = None
    if args.restraints:
        wanted = {r.strip() for r in args.restraints.split(",")}
    restraints = [r for r in RESTRAINTS if wanted is None or r[0] in wanted]

    rows = {}
    for rtype, eterm in restraints:
        rows[rtype] = {}
        ref = None
        for engine in engines:
            got = run_one(args.charmm, datadir, rtype, eterm, engine,
                          args.np, args.timeout, args.mpi_cmd)
            if engine == "cpu" and not got.get("status"):
                ref = got.get("term")
            tol = ENGINE_REL_TOL.get(engine, args.rel_tol)
            status, note = classify(ref, got, tol)
            rows[rtype][engine] = (status, note)
            print("  %-10s %-14s %-8s %s" % (rtype, engine, status, note),
                  flush=True)

    width = max(len(r) for r, _ in restraints) + 2
    print("\n" + "restraint".ljust(width) +
          "".join(e.ljust(16) for e in engines))
    print("-" * (width + 16 * len(engines)))
    for rtype, _ in restraints:
        line = rtype.ljust(width)
        for engine in engines:
            line += rows[rtype][engine][0].ljust(16)
        print(line)

    bad = [(r, e) for r in rows for e in rows[r]
           if rows[r][e][0] in ("DIFFER", "ZERO", "CRASH")]
    print("\n%d pair(s) need attention" % len(bad))
    for r, e in bad:
        print("  %s / %s: %s %s" % (r, e, rows[r][e][0], rows[r][e][1]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
