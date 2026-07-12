#!/usr/bin/env python3
"""Run the des_dimer_pair_scans campaign locally without Snakemake."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add scripts directory to path
_SCRIPTS = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from scan_lib import load_config, iter_pairs, output_dir


def run_pair(pair_tag, cfg_path, repo_root, workflow_root):
    cfg = load_config(cfg_path)
    from scan_lib import pair_from_tag
    pair = pair_from_tag(cfg, pair_tag)
    out_dir = output_dir(cfg, pair)
    done_file = out_dir / "done.txt"
    stdout_file = out_dir / "stdout.log"

    if done_file.exists() and (out_dir / "scan_2d.npz").exists():
        print(f"Pair {pair_tag} already completed. Skipping.")
        return pair_tag, True

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(_SCRIPTS / "run_pair_scan.py"),
        "--config",
        str(cfg_path),
        "--pair",
        pair_tag,
    ]

    print(f"Running scan for {pair_tag}...")
    try:
        with open(stdout_file, "w") as f:
            subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,  # Capture stderr to log too
                cwd=str(repo_root),
                check=True,
            )
        done_file.touch()
        print(f"Completed {pair_tag}.")
        return pair_tag, True
    except Exception as e:
        print(f"Error running {pair_tag}: {e}", file=sys.stderr)
        return pair_tag, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("-j", "--jobs", type=int, default=4)
    args = parser.parse_args()

    workflow_root = Path(__file__).resolve().parent
    repo_root = workflow_root.parents[1]
    cfg_path = (workflow_root / args.config).resolve()

    cfg = load_config(cfg_path)
    pairs = list(iter_pairs(cfg))

    # Prevent JAX preallocation issue in parallel processes
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    print(f"Loaded config from {cfg_path}")
    print(f"Found {len(pairs)} pairs to scan.")

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


if __name__ == "__main__":
    main()
