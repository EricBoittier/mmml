"""YAML configuration helpers for ``mmml md-system``."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

_CONFIG_ALIASES: dict[str, str] = {
    "output": "output_dir",
    "checkpoint_path": "checkpoint",
    "job": "job_id",
    "job-id": "job_id",
    "run-all": "run_all",
    "resume-campaign": "resume_campaign",  # alias; prefer ``resume``
    "continue-from": "continue_from",
    "continue-from-frame": "continue_from_frame",
    "handoff-write-res": "handoff_write_res",
    "handoff-template-res": "handoff_template_res",
    "no-stage-summary": "no_stage_summary",
    "campaign_output": "campaign_output_dir",
    "campaign-output": "campaign_output_dir",
}


def _normalize_config_key(key: str) -> str:
    k = str(key).strip()
    if k in _CONFIG_ALIASES:
        return _CONFIG_ALIASES[k]
    return k.replace("-", "_")


def _deep_merge_mappings(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into a copy of ``base`` (``defaults`` deep-merge)."""
    result: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if key == "include":
            continue
        if (
            key == "defaults"
            and isinstance(value, Mapping)
            and isinstance(result.get(key), Mapping)
        ):
            result[key] = _deep_merge_mappings(result[key], value)
        elif (
            key == "runs"
            and isinstance(value, Mapping)
            and isinstance(result.get(key), Mapping)
        ):
            merged_runs = dict(result[key])
            merged_runs.update(value)
            result[key] = merged_runs
        elif (
            key == "jobs"
            and isinstance(value, Mapping)
            and isinstance(result.get(key), Mapping)
        ):
            merged_jobs = dict(result[key])
            merged_jobs.update(value)
            result[key] = merged_jobs
        else:
            result[key] = value
    return result


def _resolve_include_path(config_path: Path, include_ref: str) -> Path:
    ref = Path(include_ref)
    if ref.is_absolute():
        return ref
    # Prefer relative to the including file, then repo-root relative (mmml/cli/run/...).
    candidate = (config_path.parent / ref).resolve()
    if candidate.is_file():
        return candidate
    repo_root = config_path.resolve()
    for _ in range(8):
        if (repo_root / "mmml" / "cli" / "run").is_dir():
            break
        if repo_root.parent == repo_root:
            break
        repo_root = repo_root.parent
    alt = (repo_root / ref).resolve()
    if alt.is_file():
        return alt
    return candidate


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")

    includes = raw.get("include") or []
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list):
        raise ValueError(f"'include' must be a string or list of paths, got {type(includes).__name__}")

    merged: dict[str, Any] = {}
    for include_ref in includes:
        include_path = _resolve_include_path(config_path, str(include_ref))
        included = load_yaml_config(include_path)
        merged = _deep_merge_mappings(merged, included)

    overlay = {k: v for k, v in raw.items() if k != "include"}
    return _deep_merge_mappings(merged, overlay)


def resume_requested(
    args: argparse.Namespace | None = None,
    *,
    mapping: Mapping[str, Any] | None = None,
) -> bool:
    """True when resume was requested via ``--resume``, ``--resume-campaign``, or YAML."""
    if args is not None:
        if bool(getattr(args, "resume", False)) or bool(getattr(args, "resume_campaign", False)):
            return True
    if mapping is not None:
        if mapping.get("resume") is True or mapping.get("resume_campaign") is True:
            return True
    return False


def normalize_resume_flags(args: argparse.Namespace) -> None:
    """Keep ``resume`` and ``resume_campaign`` in sync after CLI/YAML parsing."""
    if resume_requested(args):
        args.resume = True
        args.resume_campaign = True


def campaign_resume_enabled(
    args: argparse.Namespace,
    campaign: Mapping[str, Any],
) -> bool:
    """Resume mode for ``--run-all`` (CLI flags or ``defaults.resume`` in YAML)."""
    if resume_requested(args):
        return True
    defaults = campaign.get("defaults")
    if isinstance(defaults, Mapping):
        return resume_requested(mapping=defaults)
    return False


def collect_explicit_cli_dests(
    argv: list[str],
    parser: argparse.ArgumentParser,
) -> set[str]:
    """Return argparse ``dest`` names set explicitly on ``argv`` (for campaign overrides)."""
    option_dests: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        for opt in action.option_strings:
            if opt.startswith("--"):
                option_dests[opt.split("=", 1)[0]] = action.dest
    explicit: set[str] = set()
    i = 0
    while i < len(argv):
        tok = str(argv[i])
        if tok.startswith("--"):
            flag = tok.split("=", 1)[0]
            dest = option_dests.get(flag)
            if dest is not None:
                explicit.add(dest)
            if "=" not in tok and i + 1 < len(argv) and not str(argv[i + 1]).startswith("-"):
                i += 1
        i += 1
    return explicit


def apply_mapping_to_namespace(
    args: argparse.Namespace,
    mapping: Mapping[str, Any],
    *,
    source: str,
    allowed_prefixes: tuple[str, ...] = (),
) -> None:
    unknown: list[str] = []
    for raw_key, value in mapping.items():
        if raw_key in {"defaults", "runs", "jobs"}:
            continue
        key = _normalize_config_key(str(raw_key))
        if not hasattr(args, key):
            if allowed_prefixes and key.startswith(allowed_prefixes):
                setattr(args, key, value)
                continue
            unknown.append(str(raw_key))
            continue
        setattr(args, key, value)
    if unknown:
        valid = sorted(k for k in vars(args) if not k.startswith("_"))
        raise ValueError(
            f"Unknown {source} key(s): {', '.join(sorted(unknown))}. "
            f"Valid keys include: {', '.join(valid[:40])}..."
        )


def namespace_from_yaml(path: str | Path, parse_args_fn) -> argparse.Namespace:
    args = parse_args_fn([])
    apply_mapping_to_namespace(args, load_yaml_config(path), source=f"config '{path}'")
    return args


def merge_campaign_job_config(
    campaign: dict[str, Any],
    job_id: str,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    runs = campaign.get("runs") or campaign.get("jobs") or {}
    if job_id not in runs:
        raise KeyError(f"Unknown job_id {job_id!r}")
    merged: dict[str, Any] = {}
    defaults = campaign.get("defaults") or {}
    if isinstance(defaults, dict):
        merged.update(defaults)
    job = runs[job_id]
    if isinstance(job, dict):
        merged.update(job)
    merged["job_id"] = job_id
    merged["job_name"] = job_id
    if output_dir is not None:
        merged["output_dir"] = str(output_dir)
    elif "output_dir" not in merged and "output_root" in campaign:
        merged["output_dir"] = str(Path(str(campaign["output_root"])) / job_id)
    return merged


def campaign_job_ids(campaign: dict[str, Any]) -> list[str]:
    runs = campaign.get("runs") or campaign.get("jobs") or {}
    return list(runs.keys())


def config_is_campaign(cfg: Mapping[str, Any]) -> bool:
    """True when YAML defines a ``runs`` / ``jobs`` table (campaign mode)."""
    runs = cfg.get("runs") or cfg.get("jobs")
    return isinstance(runs, Mapping) and bool(runs)


def campaign_has_job(cfg: Mapping[str, Any], job_id: str) -> bool:
    runs = cfg.get("runs") or cfg.get("jobs") or {}
    return job_id in runs


def topological_job_order(campaign: dict[str, Any]) -> list[str]:
    runs = campaign.get("runs") or campaign.get("jobs") or {}
    ids = list(runs.keys())
    deps: dict[str, set[str]] = {}
    for jid in ids:
        job = runs[jid] or {}
        dep = job.get("depends_on")
        deps[jid] = {dep} if isinstance(dep, str) and dep else set()
    ordered: list[str] = []
    seen: set[str] = set()

    def visit(node: str) -> None:
        if node in seen:
            return
        for d in deps.get(node, set()):
            if d not in runs:
                raise ValueError(f"Job {node!r} depends_on unknown job {d!r}")
            visit(d)
        seen.add(node)
        ordered.append(node)

    for jid in ids:
        visit(jid)
    return ordered


def expand_repeated_jobs(campaign: dict[str, Any], job_order: list[str]) -> list[tuple[str, str, int]]:
    """Return ``(base_id, run_id, replicate_index)`` tuples."""
    runs = campaign.get("runs") or campaign.get("jobs") or {}
    expanded: list[tuple[str, str, int]] = []
    for jid in job_order:
        job = runs[jid] or {}
        repeat = int(job.get("repeat", 1))
        for rep in range(repeat):
            run_id = jid if repeat == 1 else f"{jid}.{rep}"
            expanded.append((jid, run_id, rep))
    return expanded


def save_md_system_config(args: argparse.Namespace, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in sorted(vars(args).items())
        if k not in {"extra_args"} and not k.startswith("_")
    }
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
