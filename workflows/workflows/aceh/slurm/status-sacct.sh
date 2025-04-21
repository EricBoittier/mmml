#!/bin/bash

# Check if a job ID was provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <job_id>"
    exit 1
fi

jobid=$1

# Get the job status from sacct
status=$(sacct -j "$jobid" --format=State --noheader | head -n 1 | awk '{print $1}')

# Check the status and return appropriate code
case $status in
    COMPLETED)
        echo "success"
        exit 0
        ;;
    RUNNING|PENDING|COMPLETING)
        echo "running"
        exit 0
        ;;
    FAILED|TIMEOUT|CANCELLED|PREEMPTED)
        echo "failed"
        exit 0
        ;;
    *)
        echo "failed"
        exit 0
        ;;
esac