#!/bin/bash
# Submit the TL frames, never letting more than MAX_PENDING sit in PD.
#
#   bash submit_tl.sh                 # submit everything not yet done
#   MAX_PENDING=5 bash submit_tl.sh
#   DRY_RUN=1 bash submit_tl.sh       # print what would be submitted
#
# Throttles on PENDING jobs, not on total jobs. A total-jobs cap stalls as soon
# as the queue fills with your own RUNNING work, so the submitter sleeps while
# the cluster is busy doing exactly what you asked. Capping PD keeps a short
# ready queue and lets RUNNING grow to whatever the scheduler will give you.
#
# Resumable: a frame whose engrad.out already reports a finished ORCA run is
# skipped, so re-running this after an interruption picks up where it stopped
# rather than double-submitting 1477 jobs.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMES="${FRAMES:-${HERE}/frames}"
MAX_PENDING="${MAX_PENDING:-10}"
SLEEP_TIME="${SLEEP_TIME:-30}"
DRY_RUN="${DRY_RUN:-0}"

[[ -d "${FRAMES}" ]] || { echo "no ${FRAMES} -- run make_inputs.py first"; exit 1; }

pending() { squeue -u "${USER}" -h -t PD 2>/dev/null | wc -l; }

# A frame is done when ORCA said so. Checking for the file alone would skip
# frames that died halfway and left a truncated output.
finished() {
  [[ -f "$1/engrad.out" ]] && \
    grep -q "ORCA TERMINATED NORMALLY" "$1/engrad.out" 2>/dev/null
}

total=0; done_already=0; submitted=0
for dir in "${FRAMES}"/frame_*; do
  [[ -d "${dir}" ]] || continue
  total=$((total + 1))
done

echo "frames        ${total} under ${FRAMES}"
echo "max pending   ${MAX_PENDING}   poll ${SLEEP_TIME}s"
[[ "${DRY_RUN}" == "1" ]] && echo "DRY RUN -- nothing will be submitted"
echo

for dir in "${FRAMES}"/frame_*; do
  [[ -d "${dir}" ]] || continue
  name="$(basename "${dir}")"

  if finished "${dir}"; then
    done_already=$((done_already + 1))
    continue
  fi

  while (( $(pending) >= MAX_PENDING )); do
    echo "  [$(date +%H:%M:%S)] $(pending) pending, waiting ${SLEEP_TIME}s..."
    sleep "${SLEEP_TIME}"
  done

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  would submit ${name}"
  else
    ( cd "${dir}" && sbatch --job-name="tl_${name}" run.sh >/dev/null ) \
      && echo "  submitted ${name}  (pending now $(pending))" \
      || echo "  FAILED to submit ${name}"
    sleep 1                     # let slurmdbd register it before recounting
  fi
  submitted=$((submitted + 1))
done

echo
echo "already finished ${done_already}"
echo "submitted        ${submitted}"
echo
echo "watch:   squeue -u ${USER} -h | wc -l"
echo "collect: python collect_results.py"
