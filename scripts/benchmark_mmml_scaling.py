#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def _parse_counts(spec: str) -> list[int]:
    # Supports: "10,20,30" or "10:60:10"
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        if len(parts) != 3:
            raise ValueError("Range format must be start:stop:step")
        start, stop, step = parts
        return list(range(start, stop + (1 if step > 0 else -1), step))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _run_case(
    *,
    python_exe: str,
    script_path: Path,
    backend: str,
    n_mol: int,
    ps: float,
    dt_fs: float,
    base_out_dir: Path,
    timeout_s: int,
    extra_args: list[str],
) -> dict:
    case_dir = base_out_dir / f"{backend}_n{n_mol}"
    case_dir.mkdir(parents=True, exist_ok=True)
    if backend == "ase":
        cmd = [
            python_exe,
            str(script_path / "md_10mer_mmml_pbc_suite.py"),
            "--only",
            "pbc_nve",
            "--n-molecules",
            str(n_mol),
            "--ps",
            str(ps),
            "--dt-fs",
            str(dt_fs),
            "--output-dir",
            str(case_dir),
            "--fd-check-atoms",
            "0",
            "--skip-jit-warmup",
        ]
    elif backend == "jaxmd":
        cmd = [
            python_exe,
            str(script_path / "md_10mer_mmml_pbc_suite_jaxmd.py"),
            "--ensemble",
            "nvt",
            "--n-molecules",
            str(n_mol),
            "--ps",
            str(ps),
            "--dt-fs",
            str(dt_fs),
            "--output-dir",
            str(case_dir),
            "--steps-per-recording",
            "25",
        ]
    else:
        raise ValueError(f"Unknown backend: {backend}")
    cmd.extend(extra_args)

    t0 = time.perf_counter()
    result = {
        "backend": backend,
        "n_molecules": n_mol,
        "ps": ps,
        "dt_fs": dt_fs,
        "status": "unknown",
        "wall_s": None,
        "md_per_step_ms": None,
        "md_integrator_s": None,
        "steps_completed": None,
        "error": None,
        "output_dir": str(case_dir),
    }
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        result["wall_s"] = time.perf_counter() - t0
        (case_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (case_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            result["status"] = "failed"
            result["error"] = f"exit_code={proc.returncode}"
            return result
        result["status"] = "ok"
        if backend == "ase":
            summary_path = case_dir / "suite_summary.json"
            if summary_path.exists():
                s = json.loads(summary_path.read_text(encoding="utf-8"))
                run = s.get("runs", {}).get("pbc_nve", {})
                t = run.get("timings_s", {})
                result["md_per_step_ms"] = t.get("md_per_step_mean_ms")
                result["md_integrator_s"] = t.get("md_integrator_loop_s")
                result["steps_completed"] = s.get("md", {}).get("nsteps")
        else:
            summary_path = case_dir / "suite_summary_jaxmd.json"
            if summary_path.exists():
                s = json.loads(summary_path.read_text(encoding="utf-8"))
                result["steps_completed"] = s.get("nsteps_completed")
        return result
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["wall_s"] = time.perf_counter() - t0
        result["error"] = f"timeout>{timeout_s}s"
        return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ase", "jaxmd"], default="ase")
    p.add_argument("--counts", type=str, default="10,20,30,40,50")
    p.add_argument("--ps", type=float, default=0.2)
    p.add_argument("--dt-fs", type=float, default=0.25)
    p.add_argument("--timeout-s", type=int, default=1200)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/mmml_scaling"))
    p.add_argument("--stop-on-fail", action="store_true")
    p.add_argument("--extra-arg", action="append", default=[], help="Extra arg forwarded to target script.")
    args = p.parse_args()

    counts = _parse_counts(args.counts)
    out_dir = (Path.cwd() / args.output_dir.expanduser()).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path(__file__).resolve().parent

    rows: list[dict] = []
    for n_mol in counts:
        row = _run_case(
            python_exe=sys.executable,
            script_path=scripts_dir,
            backend=args.backend,
            n_mol=n_mol,
            ps=args.ps,
            dt_fs=args.dt_fs,
            base_out_dir=out_dir,
            timeout_s=args.timeout_s,
            extra_args=list(args.extra_arg),
        )
        rows.append(row)
        print(json.dumps(row, indent=2))
        if args.stop_on_fail and row["status"] != "ok":
            break

    json_path = out_dir / f"scaling_{args.backend}.json"
    csv_path = out_dir / f"scaling_{args.backend}.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = [
        "backend",
        "n_molecules",
        "ps",
        "dt_fs",
        "status",
        "wall_s",
        "md_per_step_ms",
        "md_integrator_s",
        "steps_completed",
        "error",
        "output_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

