#!/usr/bin/env python
"""Report per-window state of a hybrid umbrella campaign and clear failed ones.

Snakemake's ``window`` rule treats ``windows/wXXX.npz`` as its output, but a
window that aborts non-finite still *writes* that file (``status=failed``, all
NaN) and exits 0. So a rerun sees the output present, skips the window, and
``assemble`` -- which runs ``--no-resume-failed`` -- bakes the NaN row into the
PMF. Failed windows have to be deleted before they will be redone.

    uv run python workflows/hybrid_umbrella_windows/scripts/window_status.py
    uv run python workflows/hybrid_umbrella_windows/scripts/window_status.py --reset-failed

``--reset-failed`` also removes ``umbrella_snapshots.npz`` and friends, because
``bootstrap_windows_from_snapshots`` recreates any *missing* window from the
aggregate as a failed placeholder -- which would silently undo the deletions.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from campaign_lib import load_config, resolve_path, window_ids  # noqa: E402

_RELAX_RE = re.compile(r"relax_steps=(\d+)")


@dataclass(frozen=True)
class WindowReport:
    wid: int
    status: str  # "ok" | "failed" | "missing"
    xi0: float
    cv_mean: float
    finite: float
    relax_steps: int | None  # None when the log predates seed relaxation
    fail_reason: str

    @property
    def needs_rerun(self) -> bool:
        return self.status != "ok"


def window_log_relax_steps(output_dir: Path, wid: int) -> int | None:
    """FIRE steps this window actually ran, read back from its log."""
    log = Path(output_dir) / "logs" / f"window_w{int(wid):03d}.log"
    if not log.is_file():
        return None
    hits = _RELAX_RE.findall(log.read_text(encoding="utf-8", errors="replace"))
    return int(hits[-1]) if hits else None


def scan_windows(output_dir: Path, wids: list[int]) -> list[WindowReport]:
    from mmml.umbrella.hybrid_windows import load_window_checkpoint

    reports: list[WindowReport] = []
    for wid in wids:
        relax = window_log_relax_steps(output_dir, wid)
        chk = load_window_checkpoint(Path(output_dir), wid)
        if chk is None:
            reports.append(
                WindowReport(wid, "missing", float("nan"), float("nan"), 0.0, relax, "")
            )
            continue
        cv = np.asarray(chk["cv"], dtype=np.float64).reshape(-1)
        finite = float(np.isfinite(cv).mean()) if cv.size else 0.0
        with np.errstate(invalid="ignore"):
            cv_mean = float(np.nanmean(cv)) if np.any(np.isfinite(cv)) else float("nan")
        status = str(chk.get("status") or "failed")
        if status == "ok" and finite < 1.0:
            status = "failed"
        reports.append(
            WindowReport(
                wid=wid,
                status=status,
                xi0=float(chk.get("xi0", float("nan"))),
                cv_mean=cv_mean,
                finite=finite,
                relax_steps=relax,
                fail_reason=str(chk.get("fail_reason") or ""),
            )
        )
    return reports


def files_to_reset(
    reports: list[WindowReport],
    output_dir: Path,
    *,
    reset_failed: bool = True,
    reset_unrelaxed: bool = False,
) -> list[Path]:
    """Window checkpoints to delete, plus the aggregates that would restore them."""
    from mmml.umbrella.hybrid_windows import window_npz_path

    out = Path(output_dir)
    doomed: list[Path] = []
    for r in reports:
        drop = (reset_failed and r.status == "failed") or (
            reset_unrelaxed and r.status == "ok" and not r.relax_steps
        )
        if drop:
            path = window_npz_path(out, r.wid)
            if path.is_file():
                doomed.append(path)
    if not doomed:
        return []
    # Any deletion is undone by a whole-campaign --resume unless the aggregate
    # goes too: bootstrap_windows_from_snapshots refills missing windows from it.
    for name in ("umbrella_snapshots.npz", "umbrella_summary.json"):
        if (out / name).is_file():
            doomed.append(out / name)
    if (out / "mbar" / "status.json").is_file():
        doomed.append(out / "mbar" / "status.json")
    return doomed


def format_table(reports: list[WindowReport]) -> str:
    lines = [
        f"{'wid':>4} {'status':>7} {'xi0':>7} {'cv_mean':>8} {'finite':>6}"
        f" {'relax':>5}  reason"
    ]
    for r in reports:
        relax = "-" if r.relax_steps is None else str(r.relax_steps)
        reason = r.fail_reason if r.status != "ok" else ""
        lines.append(
            f"{r.wid:4d} {r.status:>7} {r.xi0:+7.3f} {r.cv_mean:+8.3f}"
            f" {r.finite:6.2f} {relax:>5}  {reason[:70]}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="workflow config (default: config.yaml)")
    ap.add_argument(
        "--reset-failed",
        action="store_true",
        help="delete failed window checkpoints so Snakemake redoes them",
    )
    ap.add_argument(
        "--reset-unrelaxed",
        action="store_true",
        help=(
            "also redo windows that finished without seed relaxation, so the PMF "
            "is not a mix of relaxed and unrelaxed seeds"
        ),
    )
    ap.add_argument("--dry-run", action="store_true", help="list removals, delete nothing")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    repo = _SCRIPTS.parents[2]
    out = resolve_path(repo, cfg.get("output_dir", "artifacts/nh3_ch3cl/umbrella_nc_acn_prod"))
    wids = window_ids(cfg)

    reports = scan_windows(out, wids)
    print(f"campaign: {out}")
    print(format_table(reports))

    n_ok = sum(r.status == "ok" for r in reports)
    n_failed = sum(r.status == "failed" for r in reports)
    n_missing = sum(r.status == "missing" for r in reports)
    n_unrelaxed = sum(r.status == "ok" and not r.relax_steps for r in reports)
    print(
        f"\n{n_ok} ok / {n_failed} failed / {n_missing} missing"
        f"  ({n_unrelaxed} ok window(s) ran without seed relaxation)"
    )

    if not (args.reset_failed or args.reset_unrelaxed):
        print("\nnothing removed (pass --reset-failed to clear them for a rerun)")
        return 0

    doomed = files_to_reset(
        reports,
        out,
        reset_failed=args.reset_failed,
        reset_unrelaxed=args.reset_unrelaxed,
    )
    if not doomed:
        print("\nnothing to reset")
        return 0
    print(f"\n{'would remove' if args.dry_run else 'removing'} {len(doomed)} file(s):")
    for path in doomed:
        print(f"  {path}")
        if not args.dry_run:
            path.unlink()
    if not args.dry_run:
        print("\nnow relaunch: nohup bash scripts/snakemake_slurm.sh 8 > snakemake_gpu.log 2>&1 &")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
