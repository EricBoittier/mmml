"""Capability-aware, receipt-producing scientific smoke matrices."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SmokeCase:
    id: str
    category: str
    description: str
    command: tuple[str, ...]
    requires_env: tuple[str, ...] = ()
    requires_commands: tuple[str, ...] = ()
    requires_modules: tuple[str, ...] = ()
    requires_any_modules: tuple[str, ...] = ()
    expected_artifacts: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SmokeManifest:
    schema_version: int
    cases: tuple[SmokeCase, ...]


def load_smoke_manifest(path: Path | str) -> SmokeManifest:
    """Load and validate a smoke manifest without probing the runtime."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or int(data.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"smoke manifest must use schema_version {SCHEMA_VERSION}")
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("smoke manifest requires a non-empty cases list")
    cases: list[SmokeCase] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, Mapping):
            raise ValueError("each smoke case must be a mapping")
        case_id = str(raw.get("id", "")).strip()
        command = raw.get("command")
        if not case_id or case_id in seen:
            raise ValueError(f"smoke case id is empty or duplicated: {case_id!r}")
        if not isinstance(command, list) or not command:
            raise ValueError(f"smoke case {case_id!r} requires a command list")
        seen.add(case_id)
        cases.append(
            SmokeCase(
                id=case_id,
                category=str(raw.get("category", "other")),
                description=str(raw.get("description", "")),
                command=tuple(str(item) for item in command),
                requires_env=tuple(map(str, raw.get("requires_env", ()))),
                requires_commands=tuple(map(str, raw.get("requires_commands", ()))),
                requires_modules=tuple(map(str, raw.get("requires_modules", ()))),
                requires_any_modules=tuple(map(str, raw.get("requires_any_modules", ()))),
                expected_artifacts=tuple(map(str, raw.get("expected_artifacts", ()))),
                tags=tuple(map(str, raw.get("tags", ()))),
            )
        )
    return SmokeManifest(schema_version=SCHEMA_VERSION, cases=tuple(cases))


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _blocked_reasons(case: SmokeCase, env: Mapping[str, str]) -> list[str]:
    reasons = [f"environment variable {name} is unset" for name in case.requires_env if not env.get(name)]
    reasons.extend(
        f"path from {name} does not exist: {env[name]}"
        for name in case.requires_env
        if env.get(name) and not Path(env[name]).expanduser().exists()
    )
    reasons.extend(
        f"command {name!r} is unavailable" for name in case.requires_commands if shutil.which(name) is None
    )
    reasons.extend(
        f"Python module {name!r} is unavailable"
        for name in case.requires_modules
        if importlib.util.find_spec(name) is None
    )
    if case.requires_any_modules and not any(
        importlib.util.find_spec(name) is not None for name in case.requires_any_modules
    ):
        reasons.append("none of the Python modules are available: " + ", ".join(case.requires_any_modules))
    return reasons


def _resolve_command(case: SmokeCase, *, repo: Path, output_dir: Path) -> list[str]:
    values = {"python": sys.executable, "repo": str(repo), "output_dir": str(output_dir)}
    command: list[str] = []
    for token in case.command:
        expanded = token
        for name, value in values.items():
            expanded = expanded.replace("{" + name + "}", value)
        expanded = os.path.expandvars(expanded)
        if "$" in expanded:
            raise ValueError(f"unresolved environment variable in command token {token!r}")
        command.append(expanded)
    return command


def run_smoke_matrix(
    manifest: SmokeManifest,
    *,
    output_root: Path,
    repo: Path,
    selected: Sequence[str] = (),
    tags: Sequence[str] = (),
) -> dict[str, Any]:
    """Run selected cases in isolated subprocesses and write complete receipts."""
    output_root.mkdir(parents=True, exist_ok=True)
    selected_set, tag_set = set(selected), set(tags)
    cases = [
        case
        for case in manifest.cases
        if (not selected_set or case.id in selected_set)
        and (not tag_set or tag_set.intersection(case.tags))
    ]
    unknown = selected_set.difference(case.id for case in manifest.cases)
    if unknown:
        raise ValueError(f"unknown smoke cases: {', '.join(sorted(unknown))}")

    results: list[dict[str, Any]] = []
    for case in cases:
        case_dir = output_root / case.id
        case_dir.mkdir(parents=True, exist_ok=True)
        reasons = _blocked_reasons(case, os.environ)
        command = _resolve_command(case, repo=repo, output_dir=case_dir) if not reasons else []
        request = {
            "schema_version": SCHEMA_VERSION,
            "id": case.id,
            "category": case.category,
            "description": case.description,
            "command": command,
            "requirements": {
                "environment": list(case.requires_env),
                "commands": list(case.requires_commands),
                "modules": list(case.requires_modules),
                "any_modules": list(case.requires_any_modules),
            },
        }
        (case_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        started = time.monotonic()
        if reasons:
            status = "BLOCKED"
            returncode = None
            (case_dir / "stdout.log").write_text("")
            (case_dir / "stderr.log").write_text("\n".join(reasons) + "\n")
        else:
            proc = subprocess.run(command, cwd=repo, env=os.environ.copy(), capture_output=True, text=True)
            (case_dir / "stdout.log").write_text(proc.stdout or "")
            (case_dir / "stderr.log").write_text(proc.stderr or "")
            returncode = proc.returncode
            missing = [name for name in case.expected_artifacts if not (case_dir / name).exists()]
            if proc.returncode == 0 and not missing:
                status = "PASS"
            else:
                status = "FAIL"
                reasons.extend(f"missing expected artifact: {name}" for name in missing)
        result = {
            "schema_version": SCHEMA_VERSION,
            "id": case.id,
            "category": case.category,
            "status": status,
            "returncode": returncode,
            "reasons": reasons,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        (case_dir / "status.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
        ).stdout.strip(),
        "input_hashes": {
            name: _sha256(Path(value).expanduser())
            for name, value in os.environ.items()
            if name.startswith("MMML_SMOKE_") and value
        },
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "counts": {state: sum(item["status"] == state for item in results) for state in ("PASS", "FAIL", "BLOCKED")},
        "results": results,
        "provenance": provenance,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--strict-blocked", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_smoke_manifest(args.manifest)
    if args.list:
        for case in manifest.cases:
            print(f"{case.id}\t{case.category}\t{case.description}")
        return 0
    summary = run_smoke_matrix(
        manifest,
        output_root=args.output_root.resolve(),
        repo=Path.cwd().resolve(),
        selected=args.case,
        tags=args.tag,
    )
    counts = summary["counts"]
    print(f"smoke matrix: {counts['PASS']} pass, {counts['FAIL']} fail, {counts['BLOCKED']} blocked")
    return 1 if counts["FAIL"] or (args.strict_blocked and counts["BLOCKED"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
