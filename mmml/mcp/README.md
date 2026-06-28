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

### `dimer_smoke` (MD → IR)

In Cursor (with MCP connected), ask the agent:

1. Call `health_check`
2. Call `configure_run_tool` with `run_id="smoke001"`, `recipe="dimer_smoke"`, `mode="smoke"`
3. Call `run_recipe_stage_tool` with `stage="md"`, `background=true`
4. Poll `get_run_status_tool` until MD finishes
5. Call `run_recipe_stage_tool` with `stage="ir"`

Smoke mode skips QM/training and uses `examples/ckpts_json/DESdimers_params.json`
(PhysNet, `charges=False`). IR uses **classical CGENFF dipole** autocorrelation — not
`mmml-spectra-md` / EFieldPhysNet (incompatible with this checkpoint).

Default smoke MD: **20.0 ps** at **0.1 fs** (200k steps), **record every step**
(frame spacing **0.1 fs**). IR uses centered dipole fluctuations and a non-negative
ω-weighted periodogram (not EField).

### `build_smoke` (geometry + hybrid backends)

Tests `make-res`, `liquid-box`, and short hybrid MD with **ASE**, **JAX-MD**, and
**PyCHARMM** (`setup_calculator` ML/MM). Requires `CHARMM_HOME`, `CHARMM_LIB_DIR`,
and Packmol.

```bash
source examples/md_cpu/_env.sh
bash examples/mcp/run_build_smoke.sh build001          # full smoke
DRY_RUN=1 bash examples/mcp/run_build_smoke.sh build001 # print commands only
MODE=minimal bash examples/mcp/run_build_smoke.sh build001  # skip box + pycharmm
```

Or via MCP tools: `configure_run_tool` with `recipe="build_smoke"`, then stages
`make_res` → `box_build` → `hybrid_md_ase` / `hybrid_md_jaxmd` / `hybrid_md_pycharmm`.

Hybrid vacuum configs use `packmol_radius` for cluster placement (required for
`free_nve` + JAX-MD, which omits `--box-size` from the subprocess argv). See
[`mmml/mcp/examples/README.md`](examples/README.md) for direct `mmml md-system` usage
and programmatic `setup_calculator` examples.

| Stage | Command | Outputs |
|-------|---------|---------|
| `make_res` | `mmml make-res` | `residue/` |
| `box_build` | `mmml liquid-box` | `boxes/liquid/` |
| `hybrid_md_ase` | `mmml md-system` (ASE) | `results/hybrid_ase/` |
| `hybrid_md_jaxmd` | `mmml md-system` (JAX-MD) | `results/hybrid_jaxmd/` |
| `hybrid_md_pycharmm` | `mmml md-system` (PyCHARMM) | `results/hybrid_pycharmm/` |

## Layout

```
artifacts/mcp_runs/<run_id>/
  manifest.json
  configs/
    md_smoke.yaml           # dimer_smoke
    build_box.yaml          # build_smoke
    hybrid_ase.yaml         # build_smoke
    hybrid_jaxmd.yaml
    hybrid_pycharmm.yaml
    qm_pipeline/            # full mode
  residue/                  # build_smoke make_res
  boxes/liquid/             # build_smoke box_build
  results/hybrid_*/         # build_smoke hybrid MD
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
