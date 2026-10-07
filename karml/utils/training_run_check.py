"""Postconditions for a training run, so a job's exit code means what it says.

SLURM state is not evidence about the thing under test. In the Q⁰ campaign both
directions were observed in one afternoon:

* job 206099 reported ``TIMEOUT`` after the warm-start fix it existed to verify
  had already succeeded — it then spent its remaining budget on auto-batch
  probing;
* job 206089 reported ``COMPLETED`` while training a partly-random model whose
  forces were off by three orders of magnitude.

The remedy is for the run to state its own verdict against explicit
postconditions. These helpers are pure so they can be unit-tested; the CLI in
``scripts/check_training_run.py`` wires them to a workdir and an exit code.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Check",
    "Verdict",
    "check_run",
    "parse_epoch_summaries",
    "parse_step_metrics",
    "parse_warm_start",
]

_WARM_START_RE = re.compile(
    r"Warm-started from (?P<path>\S+?):\s*"
    r"loaded (?P<loaded>\d+) parameter leaves,\s*"
    r"initialized (?P<initialized>\d+) new leaves,\s*"
    r"skipped (?P<skipped>\d+) incompatible leaves"
)
_STEP_RE = re.compile(r"^epoch\s+(?P<epoch>\d+)\s+step\s+(?P<step>\d+)\s")
_EPOCH_DONE_RE = re.compile(r"^epoch\s+(?P<epoch>\d+)\s+done in\s+(?P<seconds>[\d.]+)s\s")
_KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z_|][\w|]*)=(?P<value>-?[\d.eE+-]+|nan|inf|-inf)")


def _floats(line: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for match in _KEY_VALUE_RE.finditer(line):
        try:
            out[match.group("key")] = float(match.group("value"))
        except ValueError:
            continue
    return out


def parse_warm_start(lines: Iterable[str]) -> dict[str, Any] | None:
    """The last ``Warm-started from ...`` line, or None if the run had none."""
    found = None
    for line in lines:
        match = _WARM_START_RE.search(line)
        if match:
            found = {
                "path": match.group("path"),
                "loaded": int(match.group("loaded")),
                "initialized": int(match.group("initialized")),
                "skipped": int(match.group("skipped")),
            }
    return found


def parse_step_metrics(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Per-step training metric lines, in order."""
    out = []
    for line in lines:
        match = _STEP_RE.match(line.strip())
        if not match:
            continue
        entry = {"epoch": int(match.group("epoch")), "step": int(match.group("step"))}
        entry.update(_floats(line))
        out.append(entry)
    return out


def parse_epoch_summaries(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Per-epoch ``done in`` summary lines, in order."""
    out = []
    for line in lines:
        match = _EPOCH_DONE_RE.match(line.strip())
        if not match:
            continue
        entry = {
            "epoch": int(match.group("epoch")),
            "wall_time_s": float(match.group("seconds")),
        }
        entry.update(_floats(line))
        out.append(entry)
    return out


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Verdict:
    status: str
    checks: list[Check] = field(default_factory=list)
    observed: dict[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checks": [asdict(c) for c in self.checks],
            "observed": self.observed,
        }

    def render(self) -> str:
        lines = [f"{'PASS' if c.ok else 'FAIL'}  {c.name}: {c.detail}" for c in self.checks]
        lines.append(f"verdict: {self.status}")
        return "\n".join(lines)


def _checkpoint_dirs(workdir: Path) -> list[Path]:
    return sorted(
        p
        for p in workdir.glob("*")
        if p.is_dir()
        and (p.name.startswith("epoch-") or p.name.startswith("step-"))
        and (p / "_CHECKPOINT_METADATA").exists()
    )


def check_run(
    workdir: Path,
    log_lines: Iterable[str],
    *,
    require_steps: int = 1,
    require_checkpoint: bool = True,
    require_full_warm_start: bool = True,
    require_distillation: bool = False,
    max_force_mae: float | None = None,
) -> Verdict:
    """Judge a finished (or killed) training run against explicit postconditions."""
    workdir = Path(workdir)
    log_lines = list(log_lines)
    checks: list[Check] = []

    warm = parse_warm_start(log_lines)
    steps = parse_step_metrics(log_lines)
    epochs = parse_epoch_summaries(log_lines)

    checks.append(
        Check(
            "run_config",
            (workdir / "run_config.json").exists(),
            f"{workdir / 'run_config.json'}",
        )
    )

    # The failure that made a COMPLETED job worthless: a partial warm-start
    # trains a partly-random model while logging only counts.
    if warm is None:
        checks.append(
            Check(
                "warm_start",
                not require_full_warm_start,
                "no warm-start line in the log"
                + ("" if not require_full_warm_start else " (expected one)"),
            )
        )
    else:
        clean = warm["initialized"] == 0 and warm["skipped"] == 0
        checks.append(
            Check(
                "warm_start",
                clean or not require_full_warm_start,
                f"loaded {warm['loaded']}, initialized {warm['initialized']}, "
                f"skipped {warm['skipped']}",
            )
        )

    # The failure that made a TIMEOUT job look like a regression: the job died
    # before running any training step, so its state said nothing about the code.
    last_step = steps[-1]["step"] if steps else 0
    checks.append(
        Check(
            "training_steps",
            # require_steps <= 0 waives the requirement, for verdicts scoped to
            # something that happens before training (e.g. judging a warm-start
            # on a run the wall clock killed during batch-size probing).
            require_steps <= 0 or (bool(steps) and last_step >= require_steps),
            f"{len(steps)} logged, last step {last_step} (need >= {require_steps})",
        )
    )

    if require_checkpoint:
        found = _checkpoint_dirs(workdir)
        checks.append(
            Check(
                "checkpoint",
                bool(found),
                ", ".join(p.name for p in found) if found else "none written",
            )
        )

    finite_bad = [
        f"{key}={value}"
        for entry in (epochs[-1:] or steps[-1:])
        for key, value in entry.items()
        if isinstance(value, float) and not math.isfinite(value)
    ]
    checks.append(
        Check(
            "metrics_finite",
            not finite_bad,
            ", ".join(finite_bad) if finite_bad else "all final metrics finite",
        )
    )

    if max_force_mae is not None:
        observed_f = None
        for entry in reversed(epochs):
            if "valid_F_MAE" in entry:
                observed_f = entry["valid_F_MAE"]
                break
        if observed_f is None:
            for entry in reversed(steps):
                if "F_MAE" in entry:
                    observed_f = entry["F_MAE"]
                    break
        checks.append(
            Check(
                "force_mae",
                observed_f is not None and observed_f <= max_force_mae,
                f"{observed_f} (limit {max_force_mae})"
                if observed_f is not None
                else "no force MAE found in the log",
            )
        )

    if require_distillation:
        path = workdir / "distillation.json"
        detail = "missing"
        ok = False
        if path.exists():
            try:
                meta = json.loads(path.read_text())
                teacher = meta.get("teacher", {}) if isinstance(meta, Mapping) else {}
                sha = teacher.get("sha256")
                ok = bool(sha) and isinstance(meta.get("targets"), Mapping)
                detail = f"teacher {str(sha)[:12]}, targets {meta.get('targets')}"
            except (OSError, ValueError) as exc:
                detail = f"unreadable: {exc}"
        checks.append(Check("distillation_provenance", ok, detail))

    verdict = Verdict(
        status="PASS" if all(c.ok for c in checks) else "FAIL",
        checks=checks,
        observed={
            "warm_start": warm,
            "n_steps_logged": len(steps),
            "last_step": steps[-1] if steps else None,
            "last_epoch": epochs[-1] if epochs else None,
            "checkpoints": [p.name for p in _checkpoint_dirs(workdir)],
        },
    )
    return verdict
