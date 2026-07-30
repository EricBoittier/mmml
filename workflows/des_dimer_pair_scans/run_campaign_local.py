#!/usr/bin/env python3
"""Run the des_dimer_pair_scans campaign locally without Snakemake."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add repository root and scripts directory to path
_WORKFLOW_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _WORKFLOW_ROOT.parents[1]
_SCRIPTS = _WORKFLOW_ROOT / "scripts"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from scan_lib import load_config, iter_pairs, output_dir  # noqa: E402


def _truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "yes", "true")


def _pair_scan_cmd(pair_tag: str, cfg_path: Path, repo_root: Path) -> list[str]:
    """Build argv for one pair scan (optional CHARMM MPI wrapper)."""
    scan_py = str(_SCRIPTS / "run_pair_scan.py")
    args = ["--config", str(cfg_path), "--pair", pair_tag]
    wrapper = Path(
        os.environ.get(
            "MMML_MPIRUN_WRAPPER",
            str(repo_root / "scripts" / "mmml-charmm-mpirun.sh"),
        )
    )
    # Opt-in only: serial (macOS --no-mpi) CHARMM must keep bare python.
    # run_campaign_gpu.sh exports MMML_USE_CHARMM_MPIRUN=1 on scicore.
    if (
        _truthy("MMML_USE_CHARMM_MPIRUN")
        and not _truthy("MMML_DES_SCAN_NO_MPIRUN")
        and wrapper.is_file()
        and os.access(wrapper, os.X_OK)
    ):
        return [str(wrapper), "python", scan_py, *args]
    return [sys.executable, scan_py, *args]


def _tail_text(path: Path, n: int = 40) -> str:
    if not path.is_file():
        return f"(no log at {path})"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"(could not read {path}: {exc})"
    if not lines:
        return f"(empty log at {path})"
    return "\n".join(lines[-n:])


def run_pair(pair_tag, cfg_path, repo_root, workflow_root):
    cfg = load_config(cfg_path)
    from scan_lib import pair_from_tag

    pair = pair_from_tag(cfg, pair_tag)
    out_dir = output_dir(cfg, pair)
    done_file = out_dir / "done.txt"
    npz_file = out_dir / "scan_2d.npz"
    stdout_file = out_dir / "stdout.log"

    if done_file.exists() and npz_file.exists():
        print(f"Pair {pair_tag} already completed. Skipping.")
        return pair_tag, True

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = _pair_scan_cmd(pair_tag, Path(cfg_path), Path(repo_root))

    print(f"Running scan for {pair_tag}...")
    try:
        with open(stdout_file, "w", encoding="utf-8") as f:
            proc = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                # Per-pair cwd avoids parallel CHARMM scratch collisions in repo root.
                cwd=str(out_dir),
                check=False,
                env={
                    **os.environ,
                    "PYTHONPATH": f"{repo_root}:{os.environ.get('PYTHONPATH', '')}",
                },
            )
        # OpenMPI/PRRTE often returns 1 after a successful CHARMM run; trust NPZ.
        if npz_file.is_file():
            done_file.write_text("ok\n", encoding="utf-8")
            if proc.returncode != 0:
                print(
                    f"Completed {pair_tag} (npz present; process exit={proc.returncode})."
                )
            else:
                print(f"Completed {pair_tag}.")
            return pair_tag, True

        print(f"Error running {pair_tag}: exit={proc.returncode}", file=sys.stderr)
        print(f"----- tail {stdout_file} -----", file=sys.stderr)
        print(_tail_text(stdout_file), file=sys.stderr)
        print("----- end log -----", file=sys.stderr)
        return pair_tag, False
    except Exception as e:
        print(f"Error running {pair_tag}: {e}", file=sys.stderr)
        print(f"----- tail {stdout_file} -----", file=sys.stderr)
        print(_tail_text(stdout_file), file=sys.stderr)
        print("----- end log -----", file=sys.stderr)
        return pair_tag, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("-j", "--jobs", type=int, default=4)
    args = parser.parse_args()

    workflow_root = Path(__file__).resolve().parent
    repo_root = workflow_root.parents[1]
    cfg_arg = Path(args.config)
    cfg_path = cfg_arg if cfg_arg.is_absolute() else (workflow_root / cfg_arg).resolve()

    cfg = load_config(cfg_path)
    pairs = list(iter_pairs(cfg))

    # Prevent JAX preallocation issue in parallel processes
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    print(f"Loaded config from {cfg_path}")
    print(f"Found {len(pairs)} pairs to scan.")
    if _truthy("MMML_USE_CHARMM_MPIRUN"):
        print(
            f"CHARMM launcher: {os.environ.get('MMML_MPIRUN_WRAPPER', 'mmml-charmm-mpirun.sh')}"
        )

    # Create results folder
    (workflow_root / "results").mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_pairs = []

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_pair, pair.tag, cfg_path, repo_root, workflow_root): pair.tag
            for pair in pairs
        }

        for future in as_completed(futures):
            tag = futures[future]
            try:
                tag, success = future.result()
                if success:
                    success_count += 1
                else:
                    failed_pairs.append(tag)
            except Exception as e:
                print(f"Unhandled exception in future for {tag}: {e}", file=sys.stderr)
                failed_pairs.append(tag)

    print(f"Scan phase complete. Successfully completed {success_count}/{len(pairs)} pairs.")
    if failed_pairs:
        print(f"Failed pairs: {', '.join(failed_pairs)}", file=sys.stderr)

    # Run collect and report scripts
    print("Collecting scan results...")
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS / "collect_scans.py"),
            "--config",
            str(cfg_path),
            "--output-csv",
            "results/summary.csv",
            "--output-md",
            "results/summary.md",
        ],
        check=True,
        cwd=str(workflow_root),
    )

    print("Generating HTML report...")
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS / "build_report.py"),
            "--config",
            str(cfg_path),
            "--output-html",
            "results/report.html",
        ],
        check=True,
        cwd=str(workflow_root),
    )

    print("Report generated at results/report.html")
    return 0 if success_count == len(pairs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
