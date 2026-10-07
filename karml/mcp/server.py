"""MMML MCP server — FastMCP tools over the mmml CLI."""

from __future__ import annotations

import json
from typing import Any

from mmml.cli.configure_presets import PRESETS, list_presets_text
from mmml.cli.registry import COMMAND_REGISTRY
from mmml.cli.run.health_check import run_health_check
from mmml.mcp.allowlist import ALLOWED_COMMANDS, ALLOWED_CONSOLE_SCRIPTS
from mmml.mcp.env import default_checkpoint, repo_root, runs_root
from mmml.mcp.recipes import configure_run, list_recipe_names, load_recipe, run_recipe_stage
from mmml.mcp.runner import run_console_script, run_mmml
from mmml.mcp.status import get_run_status, list_runs, tail_log


try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:
    _FASTMCP_IMPORT_ERROR = exc

    class FastMCP:  # type: ignore[no-redef]
        """Minimal decorator shim so non-server helpers remain importable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def tool(self, *_args: Any, **_kwargs: Any) -> Any:
            def decorator(func: Any) -> Any:
                return func

            return decorator

        def run(self) -> None:
            raise RuntimeError(
                "The MMML MCP server requires the optional 'mcp' dependency. "
                "Install with `pip install 'mmml[mcp]'` or `pip install 'mmml[dev]'`."
            ) from _FASTMCP_IMPORT_ERROR


mcp = FastMCP(
    "mmml",
    instructions=(
        "Orchestrate MMML molecular simulations: residue/box builds (make-res, liquid-box), "
        "hybrid MD (ASE / JAX-MD / PyCHARMM via setup_calculator), QM labeling, "
        "PhysNet training, MD campaigns, and IR analysis. Prefer configure_run + "
        "run_recipe_stage for pipelines; use submit_mmml_command for individual steps. "
        "Recipes: dimer_smoke (MD→IR), build_smoke (geometry + hybrid backends). "
        "All run artifacts live under artifacts/mcp_runs/<run_id>/."
    ),
)


@mcp.tool()
def list_capabilities() -> str:
    """List MCP recipes, configure presets, and allowlisted mmml commands."""
    recipes = list_recipe_names()
    preset_lines = [f"  - {p.key}: {p.title}" for p in PRESETS]
    allowed = sorted(ALLOWED_COMMANDS)
    console = sorted(ALLOWED_CONSOLE_SCRIPTS)
    active_cmds = [
        f"  - {s.name}: {s.summary}"
        for s in COMMAND_REGISTRY
        if s.status == "active" and s.name in ALLOWED_COMMANDS
    ]
    body = {
        "repo_root": str(repo_root()),
        "runs_root": str(runs_root()),
        "default_checkpoint": str(default_checkpoint()),
        "recipes": recipes,
        "presets": preset_lines,
        "allowed_mmml_commands": allowed,
        "allowed_console_scripts": console,
        "active_command_summaries": active_cmds,
        "configure_presets_help": list_presets_text(),
    }
    return json.dumps(body, indent=2)


@mcp.tool()
def health_check(
    skip_live: bool = True,
    checkpoint: str | None = None,
) -> str:
    """Run mmml health-check (core, jax, charmm, mlpot, packmol, checkpoint, mpi)."""
    ckpt = None
    if checkpoint:
        from pathlib import Path

        ckpt = Path(checkpoint)
    report = run_health_check(
        skip=["live"] if skip_live else None,
        checkpoint=ckpt or default_checkpoint(),
        strict=False,
    )
    return json.dumps(report.to_dict(), indent=2)


@mcp.tool()
def configure_run_tool(
    run_id: str,
    recipe: str = "dimer_smoke",
    mode: str = "smoke",
    preset: str | None = None,
    cluster: str = "pc-bach",
) -> str:
    """Create a new MCP run directory, manifest, and config templates."""
    result = configure_run(
        run_id,
        recipe=recipe,
        mode=mode,
        preset=preset,
        cluster=cluster,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
def run_recipe_stage_tool(
    run_id: str,
    stage: str,
    mode: str = "smoke",
    dry_run: bool = False,
    background: bool = False,
) -> str:
    """Execute one pipeline stage (configure, make_res, box_build, hybrid_md_*, md, ir, …)."""
    result = run_recipe_stage(
        run_id,
        stage,
        mode=mode,
        dry_run=dry_run,
        background=background,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
def submit_mmml_command(
    command: str,
    args: list[str] | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
    background: bool = False,
) -> str:
    """Run an allowlisted mmml subcommand (e.g. md-system, physnet-train)."""
    from mmml.mcp.env import ensure_run_dir

    run_dir = ensure_run_dir(run_id) if run_id else None
    result = run_mmml(
        command,
        args=args,
        run_dir=run_dir,
        dry_run=dry_run,
        background=background,
    )
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def submit_console_script(
    script: str,
    args: list[str] | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
    background: bool = False,
) -> str:
    """Run an allowlisted console script (currently mmml-spectra-md)."""
    from mmml.mcp.env import ensure_run_dir

    run_dir = ensure_run_dir(run_id) if run_id else None
    result = run_console_script(
        script,
        args=args,
        run_dir=run_dir,
        dry_run=dry_run,
        background=background,
    )
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def get_run_status_tool(run_id: str) -> str:
    """Read manifest, stage states, artifacts, and current Slurm queue."""
    return json.dumps(get_run_status(run_id), indent=2)


@mcp.tool()
def list_runs_tool() -> str:
    """List MCP runs under artifacts/mcp_runs/."""
    return json.dumps(list_runs(), indent=2)


@mcp.tool()
def describe_recipe(name: str = "dimer_smoke") -> str:
    """Return the full YAML recipe definition."""
    return json.dumps(load_recipe(name), indent=2)


@mcp.tool()
def tail_run_log(path: str, lines: int = 40) -> str:
    """Tail a log file under the repo or artifacts/mcp_runs (max 200 lines)."""
    n = max(1, min(int(lines), 200))
    return json.dumps(tail_log(path, lines=n), indent=2)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
