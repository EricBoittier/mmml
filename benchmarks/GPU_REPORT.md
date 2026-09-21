# GPU benchmark with a private browser report

`run_gpu_report.py` wraps the existing ASV suite. It does not change branches,
install packages, stop GPU jobs, or upload data. Use a dedicated **clean,
committed checkout** containing the existing ASV suite and gate tests. The runner
can also live outside it via `--repo`; its own hash is recorded separately. The interpreter must
already have MMML, the benchmark dependencies, ASV and pytest installed.

## Run on gpu09 GPU 1

From the repository root in the GPU environment:

```bash
python benchmarks/run_gpu_report.py --dry-run
python benchmarks/run_gpu_report.py --gpu 1 --x64 1 --serve --port 8765
```

To use the runner before its changes are committed, copy just the runner outside
the target checkout. From your laptop:

```bash
scp benchmarks/run_gpu_report.py boittier@gpu09:~/run_gpu_report.py
```

Then on gpu09, using the environment that has MMML and its GPU dependencies:

```bash
python ~/run_gpu_report.py --repo ~/mmml-main-runs --gpu 1 --serve
```

The target checkout must still be clean. No changes are made to it, except ignored
report artifacts under `.asv/`. The runtime check rejects an MMML import from a
different worktree; set up the interpreter for the intended checkout first.

If the activated interpreter is wrong, use `.venv/bin/python` explicitly.
Use `--checkpoint /absolute/path/to/checkpoint` for a different model. Both the
checkpoint content hash and precision are recorded. The default is the bundled
DESdimers checkpoint, not Student A. `--threads` sets OMP/BLAS thread counts;
it is **not CPU affinity or an XLA thread-pool limit**.

The runner checks the selected physical GPU's utilization and allocated memory
before starting, then verifies that JAX sees exactly one GPU. It records the
other GPUs' initial load too. This is a snapshot, not a scheduler reservation:
reserve the GPU with your normal job coordination and do not launch another job
on it during the run. Other GPU jobs can compete for shared CPU/memory resources.

It runs the callback, vectorized-pair, PBC-transform and radius unit tests before
benchmarking. Zero tests, skipped tests, errors or failures stop the run. These
are regression gates, not a replacement for production-frame parity, a matched
teacher/student benchmark, or a long NVE/restart conservation test.

The default benchmark selection covers host pair construction and JAX ML
calculator calls. ASV's fixtures warm the functions and synchronize the timed
results. Raw ASV samples are retained; the report does not sum estimates into
claimed speedups. Missing/skipped/failed benchmark cases prevent a successful
report. See `benchmark.log` for dependency or runtime failures.

For a broader **synthetic MM** MD benchmark:

```bash
python benchmarks/run_gpu_report.py --gpu 1 --serve \
  --bench 'bench_md_driver.MDSystemSize|bench_neighbors.PairListBackends|bench_calculator.JaxMLEnergy'
```

The MDSystemSize series uses synthetic water and the MM driver. Its ns/day is
**not** the ETOH:181 PhysNet/PyCHARMM production throughput. Changing the ASV
regex does not create a production trajectory replay or a pair-capacity sweep.

## Open the private endpoint

While `--serve` is running, open an SSH tunnel **from your laptop**:

```bash
ssh -N -L 8765:127.0.0.1:8765 boittier@gpu09
```

Use your usual SSH alias/jump-host options if needed. Then open
<http://localhost:8765/> in your laptop browser. The server binds only to the
remote loopback interface. It serves only that run directory, not your checkout
or home directory. The page shows current status, provenance and logs; after
publishing it links to ASV charts. Refresh for updates. The command remains in
the foreground after the benchmark finishes so the report stays available;
Ctrl-C stops the server. Use your usual terminal multiplexer for a long run.

The default output is `.asv/gpu-reports/<UTC timestamp>-<commit>/`:

- `index.html`, `status.json`: status and provenance, including failure state.
- `gate.xml`, `gate.log`: correctness gate results.
- `preflight.log`, `packages.log`: runtime identity and package versions.
- `benchmark.log`, `results/`: ASV output and raw samples.
- `asv/`: ASV's published static report.

Each invocation gets a separate directory, preserving same-commit repeats rather
than overwriting them. This report contains the tested commit only, including
feature branches. To resume serving an old report, use the exact command printed
at the end of the run, or:

```bash
python -m http.server 8765 --bind 127.0.0.1 \
  --directory .asv/gpu-reports/REPLACE_WITH_RUN_DIRECTORY
```

For an external web server, copy the **contents of one run directory** into a
chosen static-web directory. No external upload or public deployment is enabled
by this script. Logs and provenance can include machine names and local paths;
keep the endpoint private unless those are intended to be public.

## Comparing builds

Run the same command on clean baseline and candidate checkouts, sequentially on
the same GPU and matched machine load. Keep checkpoint, precision, benchmark
selection and environment fixed. Each run has its own report and raw samples;
the script does not switch commits or silently combine incompatible series.
ASV's existing `benchmarks/run_bench.sh` remains available for shared historical
results under `benchmarks/results/`.

Do not interpret lower dispatch time by itself as a gain. Compare synchronized
wall-clock measurements, and retain separate production-frame tests for energy,
forces, pair completeness and trajectory drift before deploying a capacity or
neighbor-list change.
