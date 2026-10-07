# Getting Started

## Install

### Using `uv`

```bash
git clone https://github.com/EricBoittier/karml.git
cd karml
uv sync
```

Optional extras:

```bash
uv sync --extra dev    # tests + MkDocs
uv sync --extra cli    # shell tab completion (argcomplete)
uv sync --extra gpu    # JAX CUDA 13 + CuPy (GPU nodes)
uv sync --extra metatomic  # TorchScript AtomisticModel ASE + CHARMM MLpot
```

### Using `pip`

```bash
pip install -e ".[dev]"
pip install -e ".[cli]"   # tab completion
```

## CLI quick start

```bash
karml -h                 # compact top-level help
karml commands           # all subcommands by category
karml examples           # copy-paste invocations
karml configure          # interactive YAML / Snakemake wizard
karml env                # checkpoints + CHARMM paths
karml md-system --help   # flags for one command
```

Enable tab completion (bash/zsh):

```bash
uv sync --extra cli
eval "$(register-python-argcomplete karml)"
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
