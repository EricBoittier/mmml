# MMML MCP server

Natural-language orchestration over the `mmml` CLI for Cursor and other MCP clients.

## Install

```bash
cd /path/to/mmml
uv sync --extra mcp
# or: uv pip install --python .venv/bin/python 'mcp>=1.0.0'
```

## Run (stdio)

```bash
mmml-mcp
# or: python -m mmml.mcp
```

## Cursor

The repo includes `.cursor/mcp.json` pointing at the local `.venv`. Reload MCP servers
in Cursor after checkout.

## Tools (MVP)

| Tool | Purpose |
|------|---------|
| `list_capabilities` | Recipes, presets, allowlisted commands |
| `health_check` | Wraps `mmml health-check` |
| `configure_run_tool` | Create `artifacts/mcp_runs/<run_id>/` + manifest |
| `run_recipe_stage_tool` | Run one stage of a recipe |
| `submit_mmml_command` | Allowlisted `mmml` subcommand |
| `submit_console_script` | e.g. `mmml-spectra-md` |
| `get_run_status_tool` | Manifest + artifacts + `squeue` |
| `list_runs_tool` | All MCP runs |
| `describe_recipe` | Recipe YAML |
| `tail_run_log` | Tail logs under `artifacts/mcp_runs/` |

## Smoke test workflow

In Cursor (with MCP connected), ask the agent:

1. Call `health_check`
2. Call `configure_run_tool` with `run_id="smoke001"`, `recipe="dimer_smoke"`, `mode="smoke"`
3. Call `run_recipe_stage_tool` with `stage="md"`, `background=true`
4. Poll `get_run_status_tool` until MD finishes
5. Call `run_recipe_stage_tool` with `stage="ir"`

Smoke mode skips QM/training and uses `examples/ckpts_json/DESdimers_params.json`.

## Layout

```
artifacts/mcp_runs/<run_id>/
  manifest.json
  configs/
    md_smoke.yaml
    qm_pipeline/          # full mode
  md/
  spectra/
  logs/
```

## Security

- Only allowlisted `mmml` subcommands may run
- CLI args reject shell metacharacters
- Writes restricted to `artifacts/mcp_runs/` (except read-only status tools)

## Environment

| Variable | Default |
|----------|---------|
| `MMML_REPO_ROOT` | auto-detected from package path |
| `MMML_MCP_RUNS_ROOT` | `$REPO/artifacts/mcp_runs` |
| `MMML_BIN` / `MMML_PYTHON` | `.venv/bin/mmml` / `.venv/bin/python` |
| `MMML_CKPT` | `examples/ckpts_json/DESdimers_params.json` |
