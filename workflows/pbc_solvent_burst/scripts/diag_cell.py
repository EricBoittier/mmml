#!/usr/bin/env python3
"""Post-mortem for one pbc_solvent_burst matrix cell."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from campaign_lib import cell_from_tag, load_config, paths_for_run  # noqa: E402
from monitor_lib import (  # noqa: E402
    classify_failure,
    extract_campaign_markers,
    grep_errors,
    inspect_run,
    read_text_tail,
)

_SLURM_LOG = re.compile(r"^\d+\.log$")


def _read_text_lines(path: Path, *, max_lines: int = 400) -> list[str]:
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    if b"\x00" in raw[:4096]:
        return []
    try:
        text = raw.decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    if len(lines) > max_lines:
        return lines[-max_lines:]
    return lines


def _slurm_logs(workflow_root: Path, run_tag: str) -> list[Path]:
    base = workflow_root / ".snakemake" / "slurm_logs" / "rule_run_burst"
    if not base.is_dir():
        return []
    found: list[Path] = []
    for sub in base.iterdir():
        if not sub.is_dir() or run_tag not in sub.name:
            continue
        for path in sub.iterdir():
            if path.is_file() and _SLURM_LOG.match(path.name):
                found.append(path)
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_tag", help="e.g. dcm_50_t100_l36")
    parser.add_argument(
        "--config",
        type=Path,
        default=_SCRIPTS.parent / "config.yaml",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="One-line matrix triage (status, buckets, markers; no Slurm tail dump)",
    )
    args = parser.parse_args()

    workflow = _SCRIPTS.parent
    cfg = load_config(args.config)
    cell = cell_from_tag(cfg, args.run_tag)
    paths = paths_for_run(cfg, cell)
    out = paths["out_dir"]
    init = out / "pycharmm_init"

    print(f"=== {args.run_tag} ===")
    print(
        f"cell: {cell.solvent} N={cell.n_monomers} "
        f"T={cell.temperature}K L={cell.box_size}Å"
    )
    print(f"out_dir: {out}")
    done = paths["done"]
    print(f"done.txt: {done.is_file()} ({done.stat().st_size if done.is_file() else 0} B)")

    monitor = inspect_run(cfg, cell)

    stdout = out / "stdout.log"
    log_text = read_text_tail(stdout) if stdout.is_file() else ""
    buckets = classify_failure(log_text) if log_text else []
    markers = extract_campaign_markers(log_text) if log_text else {}

    if args.brief:
        parts = [
            args.run_tag,
            f"status={monitor.status}",
            f"health={monitor.health}",
        ]
        if monitor.log_stage:
            parts.append(f"log_stage={monitor.log_stage}")
        if buckets:
            parts.append(f"buckets={','.join(buckets)}")
        if markers:
            parts.append(" ".join(f"{k}={v}" for k, v in markers.items()))
        parts.append(monitor.progress_note)
        print("  ".join(parts))
        return 0

    print(f"monitor: status={monitor.status} health={monitor.health} {monitor.progress_note}")

    if buckets:
        print(f"failure_buckets: {', '.join(buckets)}")
    if markers:
        print("campaign_markers: " + " ".join(f"{k}={v}" for k, v in markers.items()))
    if monitor.dyna.get("n_frames"):
        print(
            f"  DYNA: {monitor.dyna['n_frames']} frames, "
            f"T_last={monitor.dyna.get('temperature_last_K', 0):.1f} K, "
            f"E_drift={monitor.dyna.get('total_energy_drift_kcal', 0):.2f} kcal/mol"
        )
    if monitor.last_dyna_lines:
        print("  last DYNA>:")
        for ln in monitor.last_dyna_lines[-3:]:
            print(f"    {ln.rstrip()}")

    summary_path = paths["campaign_summary"]
    if summary_path.is_file():
        jobs = json.loads(summary_path.read_text(encoding="utf-8")).get("jobs", [])
        if jobs:
            last = jobs[-1]
            print(f"campaign_summary: {len(jobs)} job(s)")
            print(f"  last job: {last.get('job_id')} exit={last.get('exit_code')}")
            for st in last.get("stages") or []:
                if st.get("status") == "error" or st.get("truncated"):
                    print(
                        f"  stage {st.get('stage')}: status={st.get('status')} "
                        f"frames={st.get('frames_written')} "
                        f"ps={st.get('ps_completed')}/{st.get('ps_requested')}"
                    )
        else:
            print("campaign_summary: empty jobs list")
    else:
        print("campaign_summary: missing")

    stage_summary = init / "stage_summary.json"
    if stage_summary.is_file():
        payload = json.loads(stage_summary.read_text(encoding="utf-8"))
        print(f"stage_summary: exit={payload.get('exit_code')}")
        for st in payload.get("stages") or []:
            if st.get("status") != "complete":
                print(
                    f"  {st.get('stage')}: {st.get('status')} "
                    f"frames={st.get('frames_written')}"
                )

    handoff = init / "handoff" / "state.npz"
    print(f"pycharmm_init handoff: {handoff.is_file()}")

    heat_segs = sorted(init.glob("heat_*.res"))
    if heat_segs:
        print("heat restarts:")
        for p in heat_segs[:12]:
            step_note = ""
            try:
                from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
                    read_restart_last_step,
                )

                st = read_restart_last_step(p)
                if st is not None:
                    step_note = f" step={st}"
            except Exception:
                pass
            print(f"  {p.name}: {p.stat().st_size} B{step_note}")
        if len(heat_segs) > 12:
            print(f"  ... {len(heat_segs) - 12} more")
    heat_dcds = sorted(init.glob("heat_*.dcd"))
    if heat_dcds:
        print("heat trajectories:")
        for p in heat_dcds[:6]:
            print(f"  {p.name}: {p.stat().st_size} B")

    pretreat = init / "pretreat"
    if pretreat.is_dir():
        print(f"pretreat/: {sum(1 for _ in pretreat.rglob('*') if _.is_file())} files")

    if stage_summary.is_file():
        ec = json.loads(stage_summary.read_text(encoding="utf-8")).get("exit_code")
        if ec == 2:
            print(
                "\nNote: exit=2 means pycharmm_mlpot raised RuntimeError/ValueError "
                "(overlap, intra-monomer, handoff, etc.) — not a Slurm issue."
            )

    text_sources: list[Path] = []
    stdout = out / "stdout.log"
    if stdout.is_file():
        text_sources.append(stdout)
    if stage_summary.is_file():
        text_sources.append(stage_summary)
    text_sources.extend(_slurm_logs(workflow, args.run_tag)[:3])

    printed_tail = False
    for path in text_sources:
        hits = grep_errors(read_text_tail(path))
        if hits:
            print(f"\n--- errors in {path} ---")
            for line in hits:
                print(line)
            printed_tail = True
        elif path.suffix == ".log" and path in _slurm_logs(workflow, args.run_tag)[:1]:
            lines = _read_text_lines(path, max_lines=30)
            if lines:
                print(f"\n--- tail {path} ---")
                for line in lines:
                    print(line)
                printed_tail = True

    slurm_logs = _slurm_logs(workflow, args.run_tag)
    if not slurm_logs:
        print(
            f"\nSlurm logs: none under "
            f"{workflow / '.snakemake' / 'slurm_logs' / 'rule_run_burst'}"
            f" matching *{args.run_tag}*/*.log"
        )
    elif not printed_tail:
        latest = slurm_logs[0]
        print(f"\n--- tail {latest} ---")
        for line in _read_text_lines(latest, max_lines=30):
            print(line)

    fort = out / "fort.51"
    if fort.is_file():
        print(f"\nWARN: CHARMM binary fort.51 in {out} (ignore; not a log file)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
