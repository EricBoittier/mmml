# Getting Started

## Install

### Using `uv`

```bash
git clone https://github.com/EricBoittier/mmml.git
cd mmml
uv sync
```

Optional extras:

```bash
uv sync --extra dev    # tests + MkDocs
uv sync --extra cli    # shell tab completion (argcomplete)
uv sync --extra gpu    # JAX CUDA 13 + CuPy (GPU nodes)
```

### Using `pip`

```bash
pip install -e ".[dev]"
pip install -e ".[cli]"   # tab completion
```

## CLI quick start

```bash
mmml -h                 # compact top-level help
mmml commands           # all subcommands by category
mmml examples           # copy-paste invocations
mmml configure          # interactive YAML / Snakemake wizard
mmml env                # checkpoints + CHARMM paths
mmml md-system --help   # flags for one command
```

Enable tab completion (bash/zsh):

```bash
uv sync --extra cli
eval "$(register-python-argcomplete mmml)"
```

See the [CLI overview](cli/index.md) and [tab completion](cli/completion.md) pages for details.

## Serve docs locally

```bash
uv sync --extra dev
make docs-serve
```

Then open <http://127.0.0.1:8000>.

Per-command reference pages are auto-generated before each `make docs-build` from
`scripts/generate_cli_docs.py`.
