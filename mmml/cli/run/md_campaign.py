"""In-process campaign runner for ``mmml md-system``."""

from __future__ import annotations

import sys
import time
import uuid
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping

from mmml.cli.run.md_config import (
    campaign_job_ids,
    expand_repeated_jobs,
    load_yaml_config,
    merge_campaign_job_config,
    topological_job_order,
)
from mmml.cli.run.md_handoff import (
    clear_handoff_context,
    enrich_handoff_from_restart_files,
    find_latest_charmm_restart_in_dir,
    get_handoff_out,
    handoff_is_valid,
    load_dependency_handoff,
    load_handoff,
    save_handoff,
    set_handoff_in,
    set_handoff_out,
)
from mmml.cli.run.md_stage_summary import (
    MdJobSummary,
    MdStageSummary,
    build_pycharmm_plan_rows,
    build_single_leg_plan_row,
    print_campaign_plan,
    print_campaign_report,
    print_stage_summary,
    write_campaign_plan,
    write_campaign_summary,
    write_stage_summary_json,
)


def _campaign_needs_pycharmm(campaign: dict[str, Any]) -> bool:
    runs = campaign.get("runs") or campaign.get("jobs") or {}
    return any(str((runs[j] or {}).get("backend", "")) == "pycharmm" for j in runs)


def _pycharmm_bonded_mm_mini_enabled(cfg: dict[str, Any], job: dict[str, Any]) -> bool:
    """PyCHARMM campaign jobs strain-check after heat by default."""
    if job.get("bonded_mm_mini") is False or cfg.get("bonded_mm_mini") is False:
        return False
    return True


def _resolve_output_dir(merged: dict[str, Any], run_id: str, *, rep: int = 0) -> Path:
    repeat = int(merged.get("repeat", 1))
    if merged.get("output_dir"):
        base = Path(str(merged["output_dir"])).expanduser().resolve()
        if repeat > 1:
            return (base / f"rep{int(rep):02d}").resolve()
        return base
    root = merged.get("output_root")
    if root is None and merged.get("campaign_output_dir"):
        root = merged["campaign_output_dir"]
    if root is None:
        root = "results"
    return (Path(str(root)) / run_id).resolve()


def _unique_output_dir_if_exists(path: Path, *, resume: bool) -> Path:
    """Return ``path`` when missing or resuming; else ``path_<uuid>`` if ``path`` exists."""
    resolved = path.expanduser().resolve()
    if resume or not resolved.exists():
        return resolved
    while True:
        candidate = resolved.parent / f"{resolved.name}_{uuid.uuid4().hex[:8]}"
        if not candidate.exists():
            return candidate


def _lookup_resolved_output_dir(
    resolved: dict[str, Path],
    campaign: dict[str, Any],
    job_id: str,
    *,
    rep: int = 0,
) -> Path:
    """Prefer in-run output paths (after uniquification) over static YAML layout."""
    if job_id in resolved:
        return resolved[job_id]
    for run_id, path in resolved.items():
        if run_id == job_id or run_id.startswith(f"{job_id}."):
            return path
    return _resolve_output_dir(merge_campaign_job_config(campaign, job_id), job_id, rep=rep)


_CAMPAIGN_ONLY_KEYS = frozenset(
    {
        "description",
        "depends_on",
        "repeat",
        "optional",
        "integrator",
        "pbc",
        "handoff_write_res",
        "config",
        "run_all",
        "resume",
        "resume_campaign",
        "campaign_output_dir",
        "output_root",
        "cleanup_strategy_name",
        "packmol_cache_dir",
        "job_id",
        "job_name",
        "continue_from",
        "continue_from_frame",
        "no_stage_summary",
        "handoff_template_res",
        "handoff_pre_minimize",
    }
)


def strip_campaign_metadata_keys(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Drop workflow-only keys before ``parse_md_system_args`` applies campaign defaults."""
    return {k: v for k, v in mapping.items() if k not in _CAMPAIGN_ONLY_KEYS}


# Parent ``md-system`` CLI flags that override per-job YAML when explicitly set.
_CAMPAIGN_CLI_OVERRIDE_KEYS: tuple[str, ...] = (
    "ml_batch_size",
    "ml_gpu_count",
    "ml_max_active_dimers",
    "charmm_omp_threads",
    "skip_jit_warmup",
    "handoff_pre_minimize",
)

# Long-range / MM-stack flags: only override job YAML when present on the parent CLI.
_CAMPAIGN_CLI_EXPLICIT_OVERRIDE_KEYS: tuple[str, ...] = (
    "lr_solver",
    "jax_pme_method",
    "jax_pme_sr_cutoff",
    "jax_pme_dispersion",
    "scafacos_method",
    "mm_nonbond_mode",
    "periodic_charmm_vdw",
    "include_mm",
)


def apply_campaign_cli_overrides(merged: dict[str, Any], parent: Namespace) -> None:
    """Merge top-level CLI flags into each campaign job before ``namespace_from_merged``."""
    explicit = getattr(parent, "_cli_explicit", None) or set()
    for key in _CAMPAIGN_CLI_OVERRIDE_KEYS:
        val = getattr(parent, key, None)
        if val is None:
            continue
        if isinstance(val, bool):
            if val or key == "handoff_pre_minimize":
                merged[key] = val
        else:
            merged[key] = val
    for key in _CAMPAIGN_CLI_EXPLICIT_OVERRIDE_KEYS:
        if key not in explicit:
            continue
        merged[key] = getattr(parent, key)
    if getattr(parent, "ml_spatial_mpi", False):
        merged["ml_spatial_mpi"] = True


def namespace_from_merged(merged: dict[str, Any]) -> Namespace:
    from mmml.cli.run import md_system

    parser_keys = set(vars(md_system.parse_args([])))
    argv: list[str] = []
    extra_tail: list[str] = []
    passthrough: dict[str, Any] = {}
    raw_extra = merged.get("extra_args")
    if raw_extra:
        extra_tail.extend(["--extra-args", *[str(x) for x in raw_extra]])
    for key, value in merged.items():
        if key in _CAMPAIGN_ONLY_KEYS or value is None or key == "extra_args":
            continue
        if key not in parser_keys:
            passthrough[key] = value
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            elif key == "bonded_mm_mini":
                argv.append("--no-bonded-mm-mini")
            elif key.startswith("no_") or key in {
                "handoff_write_res",
                "continue_velocities",
                "traj_export_molecular_wrap",
            }:
                argv.append(f"--no-{key[3:].replace('_', '-')}" if key.startswith("no_") else f"--no-{key.replace('_', '-')}")
        elif isinstance(value, list):
            if key == "extra_args":
                argv.extend(["--extra-args", *value])
            else:
                argv.append(flag)
                argv.extend(str(v) for v in value)
        else:
            argv.extend([flag, str(value)])
    argv.extend(extra_tail)
    old = sys.argv[:]
    sys.argv = ["md-system", *argv]
    try:
        args = md_system.parse_md_system_args()
        md_system._apply_backend_setup_defaults(args)
        for key, value in merged.items():
            if key in _CAMPAIGN_ONLY_KEYS or key == "extra_args" or value is None:
                continue
            if key in parser_keys:
                setattr(args, key, value)
        for key, value in passthrough.items():
            setattr(args, key, value)
        return args
    finally:
        sys.argv = old


def build_plan_rows(campaign: dict[str, Any], job_order: list[str]) -> list[MdStageSummary]:
    rows: list[MdStageSummary] = []
    for jid in job_order:
        merged = merge_campaign_job_config(campaign, jid)
        args = namespace_from_merged(merged)
        desc = merged.get("description")
        backend = str(merged.get("backend", args.backend))
        if backend == "pycharmm":
            rows.extend(build_pycharmm_plan_rows(jid, args, description=desc))
        else:
            rows.append(build_single_leg_plan_row(jid, args, backend, description=desc))
    return rows


def run_single_backend(
    args: Namespace,
    *,
    handoff_in=None,
) -> tuple[int, Any, list[MdStageSummary]]:
    from mmml.cli.run import md_system

    clear_handoff_context()
    set_handoff_in(handoff_in)
    set_handoff_out(None)
    backend, argv = md_system.build_command(args)
    exit_code = md_system.run_backend(backend, argv, args)
    handoff_out = get_handoff_out()
    stages: list[MdStageSummary] = getattr(md_system, "_last_job_stages", []) or []
    return exit_code, handoff_out, stages


def run_campaign(args: Namespace) -> int:
    from mmml.cli.run import md_system

    campaign = load_yaml_config(args.config)
    from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy import (
        sync_spatial_mpi_env_from_campaign,
    )

    sync_spatial_mpi_env_from_campaign(campaign, args)
    if getattr(args, "job_id", None):
        order = [str(args.job_id)]
    elif getattr(args, "run_all", False):
        order = topological_job_order(campaign)
    else:
        raise ValueError("Campaign config requires --job-id or --run-all")

    if _campaign_needs_pycharmm(campaign):
        md_system._apply_charmm_omp_threads_env(args)
        from mmml.interfaces.pycharmmInterface.charmm_mpi import (
            maybe_rerun_md_system_under_mpirun,
            prepare_serial_charmm_mpi_env,
        )

        prepare_serial_charmm_mpi_env()
        rerun = maybe_rerun_md_system_under_mpirun(sys.argv[1:])
        if rerun is not None:
            return int(rerun)

    plan_rows = build_plan_rows(campaign, order)
    campaign_root = Path(getattr(args, "campaign_output_dir", None) or campaign.get("campaign_output", "artifacts/md_campaign"))
    campaign_root = campaign_root.expanduser().resolve()
    from mmml.cli.run.md_config import campaign_resume_enabled

    resume = campaign_resume_enabled(args, campaign)
    run_all = bool(getattr(args, "run_all", False))
    requested_campaign_root = campaign_root
    if run_all and not resume:
        campaign_root = _unique_output_dir_if_exists(campaign_root, resume=False)
        if campaign_root != requested_campaign_root and not getattr(args, "quiet", False):
            print(
                f"mmml md-system: campaign output dir {requested_campaign_root} exists; "
                f"using {campaign_root}",
                flush=True,
            )
    args.campaign_output_dir = campaign_root
    campaign_root.mkdir(parents=True, exist_ok=True)
    write_campaign_plan(campaign_root / "campaign_plan.json", plan_rows)
    if not getattr(args, "quiet", False):
        print_campaign_plan(plan_rows)

    expanded = expand_repeated_jobs(campaign, order)
    handoff_by_run: dict[str, Any] = {}
    resolved_output_dirs: dict[str, Path] = {}
    job_summaries: list[MdJobSummary] = []
    campaign_rc = 0

    for base_id, run_id, rep in expanded:
        merged = merge_campaign_job_config(campaign, base_id)
        if rep:
            merged["seed"] = int(merged.get("seed", 123)) + rep
        merged["campaign_output_dir"] = str(campaign_root)
        requested_out_dir = _resolve_output_dir(merged, run_id, rep=rep)
        out_dir = requested_out_dir
        if run_all and not resume:
            out_dir = _unique_output_dir_if_exists(out_dir, resume=False)
            if out_dir != requested_out_dir and not getattr(args, "quiet", False):
                print(
                    f"mmml md-system: job {run_id!r} output dir exists; using {out_dir}",
                    flush=True,
                )
        resolved_output_dirs[run_id] = out_dir
        merged["output_dir"] = str(out_dir)
        merged["job_name"] = run_id
        dep = merged.get("depends_on")
        handoff_in = None
        if dep:
            dep_key = str(dep)
            for _, rid, _ in expanded:
                if rid == dep_key or rid.startswith(dep_key + "."):
                    handoff_in = handoff_by_run.get(rid)
                    break
            if handoff_in is None:
                dep_dir = _lookup_resolved_output_dir(
                    resolved_output_dirs, campaign, dep_key
                )
                dep_merged = merge_campaign_job_config(campaign, dep_key)
                handoff_in = load_dependency_handoff(
                    dep_dir,
                    quiet=bool(getattr(args, "quiet", False)),
                    fallback_box_side_A=dep_merged.get("box_size"),
                )
            elif dep:
                dep_dir = _lookup_resolved_output_dir(
                    resolved_output_dirs, campaign, dep_key
                )
                dep_merged = merge_campaign_job_config(campaign, dep_key)
                handoff_in = enrich_handoff_from_restart_files(
                    handoff_in,
                    dep_dir,
                    fallback_box_side_A=dep_merged.get("box_size"),
                )
            if handoff_in is not None and not merged.get("continue_from"):
                dep_dir = _lookup_resolved_output_dir(
                    resolved_output_dirs, campaign, dep_key
                )
                npz = dep_dir / "handoff" / "state.npz"
                if npz.is_file():
                    merged["continue_from"] = str(npz)
                else:
                    res = find_latest_charmm_restart_in_dir(dep_dir)
                    if res is not None:
                        merged["continue_from"] = str(res)
        if handoff_in is None and merged.get("continue_from"):
            handoff_in = load_handoff(Path(str(merged["continue_from"])))

        if resume and handoff_is_valid(out_dir):
            print(f"mmml md-system: resume skip complete job {run_id!r}", flush=True)
            if (out_dir / "handoff" / "state.npz").is_file():
                handoff_by_run[run_id] = load_handoff(out_dir / "handoff" / "state.npz")
            continue

        apply_campaign_cli_overrides(merged, args)
        ns = namespace_from_merged(merged)
        ns.output_dir = out_dir
        ns.job_name = run_id
        t0 = time.perf_counter()
        rc, handoff_out, stages = run_single_backend(ns, handoff_in=handoff_in)
        wall = time.perf_counter() - t0
        if handoff_out is not None:
            template = None
            if dep and handoff_by_run.get(str(dep)):
                prev_dir = _lookup_resolved_output_dir(
                    resolved_output_dirs, campaign, str(dep)
                )
                cand = prev_dir / "handoff" / "final.res"
                if cand.is_file():
                    template = cand
            paths = save_handoff(
                handoff_out,
                out_dir,
                template_res=template,
                write_res=bool(getattr(ns, "handoff_write_res", True)),
            )
            handoff_by_run[run_id] = handoff_out
        else:
            paths = {}

        if not getattr(args, "no_stage_summary", False):
            for st in stages:
                st.wall_time_s = wall / max(1, len(stages))
                if not getattr(args, "quiet", False):
                    print_stage_summary(st)
            job_summary = MdJobSummary(
                job_id=run_id,
                backend=str(ns.backend),
                setup=str(ns.setup),
                stages=stages,
                handoff={k: str(v) for k, v in paths.items()},
                wall_time_s=wall,
                exit_code=rc,
            )
            write_stage_summary_json(job_summary, out_dir)
            job_summaries.append(job_summary)

        if rc != 0:
            campaign_rc = rc
            if not merged.get("optional"):
                break

    write_campaign_summary(campaign_root / "campaign_summary.json", job_summaries)
    if not getattr(args, "quiet", False):
        print_campaign_report(job_summaries)
    return campaign_rc


def build_md_system_argv_from_campaign(
    campaign: dict[str, Any],
    job_id: str,
    *,
    output_dir: Path | None = None,
) -> list[str]:
    """Build flat argv for one campaign job (benchmark workflows)."""
    merged = merge_campaign_job_config(campaign, job_id, output_dir=output_dir)
    args = namespace_from_merged(merged)
    from mmml.cli.run import md_system

    _backend, backend_argv = md_system.build_command(args)
    return backend_argv


def build_benchmark_md_system_argv(
    cfg: dict[str, Any],
    job_id: str,
    *,
    output_dir: Path | None,
    resolve_checkpoint,
    job_output_dir,
) -> list[str]:
    """Build flat ``mmml md-system`` CLI argv for legacy benchmark ``config.yaml`` jobs."""
    job = cfg["jobs"][job_id]
    out = output_dir or job_output_dir(cfg, job_id)

    argv: list[str] = [
        "--setup",
        str(job["setup"]),
        "--backend",
        str(job["backend"]),
        "--composition",
        str(cfg["composition"]),
        "--checkpoint",
        str(resolve_checkpoint(str(cfg["checkpoint"]))),
        "--output-dir",
        str(out),
        "--job-name",
        job_id,
        "--seed",
        str(cfg["seed"]),
        "--dt-fs",
        str(cfg["dt_fs"]),
        "--ps",
        str(job.get("ps", cfg["ps"])),
        "--temperature",
        str(cfg["temperature"]),
        "--spacing",
        str(cfg["spacing"]),
        "--mm-switch-on",
        str(cfg["mm_switch_on"]),
        "--mm-switch-width",
        str(cfg["mm_switch_width"]),
        "--ml-switch-width",
        str(cfg["ml_switch_width"]),
    ]
    if cfg.get("min_com_restraint_distance") is not None:
        argv.extend(
            [
                "--min-com-restraint-distance",
                str(cfg["min_com_restraint_distance"]),
                "--min-com-restraint-k",
                str(cfg.get("min_com_restraint_k", 1.0)),
            ]
        )

    if job.get("pbc"):
        box_size = job.get("box_size", cfg["box_size"])
        argv.extend(["--box-size", str(box_size)])
    else:
        box_size = cfg.get("box_size")
        if box_size is not None:
            argv.extend(["--box-size", str(box_size)])
        argv.append("--free-space")

    argv.extend(
        [
            "--packmol-placement",
            "cube",
            "--packmol-tolerance",
            str(cfg["packmol_tolerance"]),
        ]
    )
    if cfg.get("packmol_radius") is not None:
        argv.extend(["--packmol-radius", str(cfg["packmol_radius"])])

    if job.get("nvt_integrator"):
        argv.extend(["--nvt-integrator", str(job["nvt_integrator"])])

    if str(job["backend"]) == "pycharmm":
        argv.extend(
            [
                "--mini-nstep",
                str(cfg["mini_nstep"]),
                "--dyn-nprint",
                str(cfg["dyn_nprint"]),
                "--dcd-nsavc",
                str(cfg["dcd_nsavc"]),
                "--dynamics-overlap-action",
                "rescue",
                "--dynamics-overlap-check-interval",
                str(cfg.get("dynamics_overlap_check_interval", 8000)),
                "--echeck",
                str(cfg.get("echeck", 500.0)),
                "--charmm-sd-steps",
                "25",
                "--charmm-abnr-steps",
                "100",
                "--skip-energy-show",
                "--ml-gpu-count",
                "1",
            ]
        )
        if job.get("md_stages"):
            argv.extend(["--md-stages", str(job["md_stages"])])
        if job.get("ps_nve") is not None:
            argv.extend(["--ps-nve", str(job["ps_nve"])])
        if job.get("ps_heat") is not None:
            argv.extend(["--ps-heat", str(job["ps_heat"])])
        if job.get("heat_thermostat"):
            argv.extend(["--heat-thermostat", str(job["heat_thermostat"])])
        if job.get("no_echeck_heat") or cfg.get("no_echeck_heat"):
            argv.append("--no-echeck-heat")
        if job.get("dynamics_overlap_memory_handoff", cfg.get("dynamics_overlap_memory_handoff")):
            argv.append("--dynamics-overlap-memory-handoff")
        if job.get("no_echeck") or cfg.get("no_echeck"):
            argv.append("--no-echeck")
        heat_ihtfrq = job.get("heat_ihtfrq")
        if heat_ihtfrq is not None:
            argv.extend(["--heat-ihtfrq", str(heat_ihtfrq)])
        argv.extend(["--heat-firstt", str(job.get("heat_firstt", cfg["heat_firstt"]))])
        argv.extend(["--heat-finalt", str(job.get("heat_finalt", cfg["heat_finalt"]))])
        if str(job["backend"]) == "pycharmm":
            if _pycharmm_bonded_mm_mini_enabled(cfg, job):
                argv.append("--bonded-mm-mini")
                argv.extend(
                    [
                        "--bonded-mm-mini-after",
                        str(
                            job.get(
                                "bonded_mm_mini_after",
                                cfg.get("bonded_mm_mini_after", "mini,heat"),
                            )
                        ),
                    ]
                )
                argv.extend(
                    [
                        "--bonded-mm-mini-steps",
                        str(int(cfg.get("bonded_mm_mini_steps", 50))),
                    ]
                )
                if cfg.get("bonded_mm_mini_always") or job.get("bonded_mm_mini_always"):
                    argv.append("--bonded-mm-mini-always")
            else:
                argv.append("--no-bonded-mm-mini")

    if str(job["setup"]) == "pbc_npt":
        argv.extend(["--pressure", str(job.get("pressure", cfg["pressure"]))])

    extra = list(job.get("extra_args") or [])
    if extra:
        argv.append("--extra-args")
        argv.extend(extra)

    return argv
