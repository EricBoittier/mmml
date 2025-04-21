#!/usr/bin/env python
import sys
import os
import subprocess

# Submit job using sbatch
jobid = subprocess.check_output(sys.argv[1:]).decode().strip()

# Print jobid for snakemake
print(jobid)