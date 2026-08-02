#!/usr/bin/env python3
"""Rank candidate ML checkpoints by running a cheap-to-expensive triage ladder.

Why this exists
---------------
A checkpoint's validation MAE is necessary but not sufficient. MD explores
regions the test split never sampled, and a gradient descent needs no thermal
barrier to find an artifact -- so a model can win on MAE and still be unusable.
This driver runs the tiers that catch that, cheapest first, and records the
outcome per checkpoint so a choice can be justified rather than asserted.

Tiers
-----
0 ``load``      checkpoint parses; config present; params non-degenerate.
1 ``dataset``   held-out error via ``scripts/evaluate_so3lr_spooky_extxyz.py``.
2 ``physics``   ``mmml mode-check`` -- finite-difference forces, X-H stretch,
                vibrations, kick stability.
3 ``smoothness`` rigid dimer scan (``scripts/scan_hybrid_dimer.py``) screened by
                ``scripts/check_spurious_minima.py`` for extra minima or a
                non-monotonic repulsive wall.

Honesty contract
----------------
A tier that cannot run (missing dataset, no GPU, absent scan input) is recorded
as ``skipped`` with a reason -- never as ``passed``. ``proof.json`` only asserts
a check when the tier actually ran and returned a verdict, so an unrunnable
tier leaves the task INCOMPLETE rather than silently green.

Example::

    python workflows/validation_campaign/scripts/checkpoint_triage.py \\
        --output-dir artifacts/validation_campaign/<run>/checkpoint.triage \\
        --checkpoint-glob 'examples/sppoky-epoch-*_params.json' \\
        --tiers load,physics
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from glob import glob
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import campaign_lib as lib  # noqa: E402

REPO = Path(__file__).resolve().parents[3]

TIERS = ("load", "dataset", "physics", "smoothness")

#: Tier -> the acceptance check it asserts in ``proof.json``.
TIER_CHECK = {
    "load": "all_checkpoints_load",
    "dataset": "dataset_error_ranked",
    "physics": "fd_forces_pass",
    "smoothness": "no_spurious_minima",
}


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _result(status: str, **kw: Any) -> dict[str, Any]:
    """One tier's outcome for one checkpoint. ``status`` in pass/fail/skipped."""
    return {"status": status, **kw}


def _run(cmd: list[str], log: Path, timeout_s: int) -> tuple[int, str]:
    """Run a subprocess, tee output to ``log``, return (rc, short reason)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, cwd=REPO
        )
    except subprocess.TimeoutExpired:
        log.write_text(f"TIMEOUT after {timeout_s}s\ncmd: {' '.join(cmd)}\n")
        return 124, f"timed out after {timeout_s}s"
    except FileNotFoundError as exc:
        log.write_text(f"NOT FOUND: {exc}\ncmd: {' '.join(cmd)}\n")
        return 127, str(exc)
    log.write_text(
        f"cmd: {' '.join(cmd)}\nrc: {proc.returncode}\n\n--- stdout ---\n"
        f"{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
    )
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return proc.returncode, (tail[-1][:200] if tail else "")


# ---------------------------------------------------------------------------
# Tier 0 -- load
# ---------------------------------------------------------------------------


def tier_load(ckpt: Path) -> dict[str, Any]:
    """Parse the checkpoint and sanity-check its contents. Cheap, CPU, no model."""
    try:
        blob = json.loads(ckpt.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return _result("fail", reason=f"unreadable: {exc}")

    if not isinstance(blob, dict) or "params" not in blob:
        return _result("fail", reason="no 'params' key -- not a checkpoint dump")

    cfg = blob.get("config")
    if not isinstance(cfg, dict):
        return _result("fail", reason="no 'config' -- provenance missing")

    n_leaves, n_finite, n_values = 0, 0, 0

    def walk(node: Any) -> None:
        nonlocal n_leaves, n_finite, n_values
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            n_leaves += 1
            flat = _flatten(node)
            n_values += len(flat)
            n_finite += sum(1 for x in flat if isinstance(x, (int, float)) and x == x)

    walk(blob["params"])
    if n_values == 0:
        return _result("fail", reason="params contain no numeric leaves")
    if n_finite != n_values:
        return _result(
            "fail", reason=f"{n_values - n_finite}/{n_values} non-finite parameters"
        )

    return _result(
        "pass",
        n_param_arrays=n_leaves,
        n_parameters=n_values,
        model_type=cfg.get("model_type"),
        workdir=cfg.get("workdir"),
        cutoff=cfg.get("cutoff"),
        train_extxyz=cfg.get("extxyz"),
        size_mb=round(ckpt.stat().st_size / 1e6, 2),
    )


def _flatten(node: Any) -> list[Any]:
    if isinstance(node, list):
        out: list[Any] = []
        for v in node:
            out.extend(_flatten(v))
        return out
    return [node]


# ---------------------------------------------------------------------------
# Tier 1 -- dataset error
# ---------------------------------------------------------------------------


def tier_dataset(
    ckpt: Path, out: Path, *, test_extxyz: str | None, cache_dir: str | None, timeout_s: int
) -> dict[str, Any]:
    """Evaluate against each named test set separately.

    Deliberately one evaluator call per file rather than pointing ``--extxyz``
    at the directory: the directory holds ten sets covering very different
    chemistry, and we only want the ones asked for. Metrics are kept **per
    dataset** and never averaged -- an energy MAE over small molecules and one
    over torsion scans are not commensurable, so a mean of the two would be a
    number with no meaning.
    """
    if not test_extxyz:
        return _result("skipped", reason="--test-extxyz not given")
    if not cache_dir:
        return _result("skipped", reason="--cache-dir required by the evaluator")

    cache = Path(cache_dir).expanduser()
    paths = [Path(p).expanduser() for p in test_extxyz.split(",") if p.strip()]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        return _result("skipped", reason=f"test set(s) not found: {', '.join(missing)}")

    per_dataset: dict[str, Any] = {}
    for path in paths:
        summary = out / "eval" / f"{ckpt.stem}.{path.stem}.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        rc, why = _run(
            [
                sys.executable,
                "scripts/evaluate_so3lr_spooky_extxyz.py",
                "--checkpoint", str(ckpt),
                "--extxyz", str(path),
                "--cache-dir", str(cache),
                "--output", str(summary),
            ],
            out / "logs" / f"{ckpt.stem}.{path.stem}.dataset.log",
            timeout_s,
        )
        if rc != 0:
            return _result("fail", reason=f"evaluator exited {rc} on {path.name}: {why}")
        if not summary.is_file():
            return _result("fail", reason=f"evaluator wrote no summary for {path.name}")

        data = json.loads(summary.read_text())
        # {dataset_name: {energy_mae, forces_mae, ...}} -- usually one entry here.
        for dname, metrics in data.items():
            if not isinstance(metrics, dict):
                continue
            per_dataset[dname] = {
                k: metrics.get(k)
                for k in ("energy_mae", "energy_rmse", "forces_mae", "forces_rmse")
                if metrics.get(k) is not None
            }

    if not per_dataset:
        return _result("fail", reason="evaluator produced no per-dataset metrics")
    return _result("pass", datasets=per_dataset, n_datasets=len(per_dataset))


# ---------------------------------------------------------------------------
# Tier 2 -- physics
# ---------------------------------------------------------------------------


def tier_physics(
    ckpt: Path, out: Path, *, composition: str, checks: str, timeout_s: int
) -> dict[str, Any]:
    dest = out / "mode_check" / ckpt.stem
    rc, why = _run(
        [
            sys.executable, "-m", "mmml.cli.__main__", "mode-check",
            "--checkpoint", str(ckpt),
            "--composition", composition,
            "--checks", checks,
            "--output-dir", str(dest),
        ],
        out / "logs" / f"{ckpt.stem}.physics.log",
        timeout_s,
    )
    if rc != 0:
        return _result("fail", reason=f"mode-check exited {rc}: {why}")

    # mode-check writes its own JSON; surface the force error if we can find it.
    payload = None
    for cand in sorted(dest.rglob("*.json")):
        try:
            payload = json.loads(cand.read_text())
            break
        except (OSError, json.JSONDecodeError):
            continue
    if payload is None:
        return _result("pass", note="mode-check exited 0 but wrote no parseable JSON")
    return _result("pass", fd_max_error=_dig(payload, "fd_max_error", "max_error"),
                   report=str(dest))


def _dig(payload: Any, *keys: str) -> Any:
    """First value found under any of ``keys``, at any depth."""
    if isinstance(payload, dict):
        for k in keys:
            if k in payload:
                return payload[k]
        for v in payload.values():
            found = _dig(v, *keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for v in payload:
            found = _dig(v, *keys)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# Tier 3 -- smoothness
# ---------------------------------------------------------------------------


def tier_smoothness(
    ckpt: Path, out: Path, *, dimer_npz: str | None, resids: str, timeout_s: int
) -> dict[str, Any]:
    if not dimer_npz:
        return _result("skipped", reason="--dimer-npz not given")
    if not Path(dimer_npz).exists():
        return _result("skipped", reason=f"dimer NPZ not found: {dimer_npz}")

    scan_dir = out / "scans" / ckpt.stem
    rc, why = _run(
        [
            sys.executable, "scripts/scan_hybrid_dimer.py",
            "--checkpoint", str(ckpt),
            "--data", dimer_npz,
            "--resids", resids,
            "--out", str(scan_dir),
        ],
        out / "logs" / f"{ckpt.stem}.scan.log",
        timeout_s,
    )
    if rc != 0:
        return _result("fail", reason=f"dimer scan exited {rc}: {why}")

    verdict = out / "minima" / f"{ckpt.stem}.json"
    verdict.parent.mkdir(parents=True, exist_ok=True)
    rc, why = _run(
        [sys.executable, "scripts/check_spurious_minima.py", "--scan-dir", str(scan_dir)],
        out / "logs" / f"{ckpt.stem}.minima.log",
        timeout_s,
    )
    # A non-zero exit here means artifacts were found -- a real verdict, not an error.
    log = (out / "logs" / f"{ckpt.stem}.minima.log").read_text()
    return _result(
        "pass" if rc == 0 else "fail",
        reason="clean" if rc == 0 else f"spurious features detected (rc={rc}): {why}",
        scan_dir=str(scan_dir),
        log_excerpt=log[-800:],
    )


# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--checkpoint-glob",
        default="examples/sppoky-epoch-*_params.json",
        help="glob of candidate checkpoints, relative to the repo root",
    )
    p.add_argument("--tiers", default=",".join(TIERS), help=f"subset of {','.join(TIERS)}")
    p.add_argument(
        "--test-extxyz",
        default=os.environ.get("MMML_TEST_EXTXYZ"),
        help="comma-separated .extxyz test sets; each is evaluated separately "
             "and reported separately (never averaged together)",
    )
    p.add_argument("--cache-dir", default=os.environ.get("MMML_EVAL_CACHE_DIR"))
    p.add_argument("--dimer-npz", default=os.environ.get("MMML_DIMER_NPZ"))
    p.add_argument("--composition", default="TIP3:2", help="system for mode-check")
    p.add_argument("--mode-checks", default="minimize,fd,bond-scan,vibrations,kick")
    p.add_argument("--resids", default="DCM,ACO")
    p.add_argument("--timeout-s", type=int, default=3600)
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    unknown = [t for t in tiers if t not in TIERS]
    if unknown:
        raise SystemExit(f"unknown tier(s): {unknown}; choose from {list(TIERS)}")

    candidates = sorted(Path(p_) for p_ in glob(str(REPO / args.checkpoint_glob)))
    if not candidates:
        lib.write_json(
            out / "proof.json",
            {
                "checks": [_check("all_checkpoints_load", False,
                                  f"no checkpoints matched {args.checkpoint_glob!r}")],
                "task": "checkpoint.triage",
            },
        )
        print(f"checkpoint_triage: no checkpoints matched {args.checkpoint_glob!r}", file=sys.stderr)
        return 1

    print(f"checkpoint_triage: {len(candidates)} candidate(s), tiers={tiers}")
    results: dict[str, dict[str, Any]] = {}
    for ckpt in candidates:
        started = time.time()
        per_tier: dict[str, Any] = {}
        for tier in tiers:
            if tier == "load":
                r = tier_load(ckpt)
            elif tier == "dataset":
                r = tier_dataset(ckpt, out, test_extxyz=args.test_extxyz,
                                 cache_dir=args.cache_dir, timeout_s=args.timeout_s)
            elif tier == "physics":
                r = tier_physics(ckpt, out, composition=args.composition,
                                 checks=args.mode_checks, timeout_s=args.timeout_s)
            else:
                r = tier_smoothness(ckpt, out, dimer_npz=args.dimer_npz,
                                    resids=args.resids, timeout_s=args.timeout_s)
            per_tier[tier] = r
            print(f"  {ckpt.name:44s} {tier:11s} {r['status']}"
                  f"{' -- ' + r['reason'] if r.get('reason') else ''}")
            # A checkpoint that will not load cannot be meaningfully tested further.
            if tier == "load" and r["status"] == "fail":
                break
        per_tier["_wall_s"] = round(time.time() - started, 1)
        results[ckpt.name] = per_tier

    lib.write_json(out / "metrics.json", results)

    # Assert a check only for tiers that actually produced a verdict somewhere.
    proof_checks = []
    for tier in tiers:
        name = TIER_CHECK[tier]
        verdicts = [r.get(tier, {}).get("status") for r in results.values()]
        ran = [v for v in verdicts if v in {"pass", "fail"}]
        if not ran:
            continue  # every candidate skipped -> assert nothing, stay INCOMPLETE
        passed = all(v == "pass" for v in ran)
        proof_checks.append(
            _check(name, passed,
                   f"{sum(v == 'pass' for v in ran)}/{len(ran)} checkpoints passed tier {tier!r}"
                   + (f"; {len(verdicts) - len(ran)} skipped" if len(ran) != len(verdicts) else ""))
        )

    lib.write_json(
        out / "proof.json",
        {"task": "checkpoint.triage", "checks": proof_checks,
         "checkpoint_glob": args.checkpoint_glob, "tiers": tiers},
    )
    print(f"checkpoint_triage: wrote metrics.json + proof.json to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
