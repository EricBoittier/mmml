#!/bin/bash
# Does each restraint actually act during dynamics through a given engine?
#
# An energy call cannot answer this: CHARMM evaluates a restraint the engine
# does not implement and reports it in the total anyway, so the number looks
# right while the trajectory runs without it.
#
# The same 200 steps are run three times from identical coordinates:
#
#   no  (a)   unrestrained
#   no  (b)   unrestrained again -- establishes the run-to-run noise floor
#   yes       restrained
#
# On a CPU the two unrestrained runs are bit-identical and the noise floor is
# zero.  On a GPU they are not: BLaDE's reductions are not reproducible run to
# run, so a bare equality test calls every restraint "acting", including ones
# the engine has no code for.  Comparing the restrained run against that
# measured floor is what makes the answer mean anything.
#
#   run_dyn_probe.sh <charmm> <datadir> <engine> <restraint>...

set -e
CHM=$1; DATADIR=$2; ENGINE=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)

for r in "$@"; do
    work=$(mktemp -d)
    ln -s "$DATADIR/data" "$work/data"
    cp "$DATADIR/datadir.def" "$work/"
    mkdir -p "$work/scratch"

    for tag in noa nob yes; do
        rest=yes; [ "$tag" = yes ] || rest=no
        ( cd "$work" && "$CHM" -prevclcg rtype="$r" engine="$ENGINE" rest="$rest" \
             -i "$HERE/dynamics_probe.inp" > "$tag.out" 2>&1 ) || true
    done

    python3 - "$work" "$r" "$ENGINE" <<'PY'
import re, sys
work, r, engine = sys.argv[1:4]

def vals(tag):
    try:
        txt = open("%s/%s.out" % (work, tag)).read()
    except OSError:
        return None
    m = re.search(r"^DYNEFFECT .*?CX (\S+) CY (\S+) CZ (\S+) RMS (\S+)", txt, re.M)
    if not m:
        return None
    try:
        return [float(x) for x in m.groups()]
    except ValueError:
        return None

a, b, y = vals("noa"), vals("nob"), vals("yes")
if a is None or b is None or y is None:
    print("%-10s %-8s CRASH      no result; output kept in %s" % (r, engine, work))
    sys.exit()

def dist(p, q):
    return max(abs(i - j) for i, j in zip(p, q))

noise  = dist(a, b)          # engine's own run-to-run spread
signal = dist(y, a)          # what the restraint did on top of it

# Require the restraint to move things well clear of the noise floor, and by
# something physically visible rather than the last bits of a float.
threshold = max(10.0 * noise, 1.0e-5)
verdict = "ACTS     " if signal > threshold else "NO-EFFECT"
print("%-10s %-8s %s  signal=%.3e noise=%.3e" % (r, engine, verdict, signal, noise))
PY
    rm -rf "$work"
done
