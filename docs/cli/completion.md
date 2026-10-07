# Tab completion

KARML uses [argcomplete](https://github.com/kislyuk/argcomplete) for bash, zsh,
and fish tab completion on subcommand names and flags.

## Install

```bash
uv sync --extra cli
# or: pip install 'karml[cli]'
```

The `cli` extra adds `argcomplete>=3.5.0`. Without it, `karml completion` still
prints a **top-level-only** script (subcommand names, no flags).

## Enable in your shell

### Bash / zsh (recommended)

```bash
eval "$(register-python-argcomplete karml)"
```

Add that line to `~/.bashrc` or `~/.zshrc` after activating the KARML virtualenv
(or use the full path to the `karml` executable in your env).

### Via `karml completion`

```bash
eval "$(karml completion bash)"
eval "$(karml completion zsh)"
karml completion fish | source   # fish
```

Use `--executable` if the CLI entry point is not named `karml`:

```bash
karml completion bash --executable /path/to/.venv/bin/karml
```

`--install-hint` prints setup reminders on stderr after the script.

## What gets completed

1. **Top level** — all `karml` subcommands (`md-system`, `physnet-train`, …).
2. **Per command** — flags and choices when the subcommand defines
   `build_parser()` (see `karml commands --audit` for coverage).

Completion hooks run before heavy imports when possible; `_ARGCOMPLETE` is set
by the shell integration and handled in `karml.cli.completion.try_autocomplete()`.

## Audit completion coverage

```bash
karml commands --audit
```

Active commands with `✓ flags` have full flag completion. Others complete the
subcommand name only until `build_parser()` is added.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Only subcommand names complete | Install `karml[cli]` / `argcomplete` |
| Wrong `karml` binary completed | Set `KARML_PYTHON` / activate venv before `eval` |
| Completion runs wrong parser | Ensure `karml` on `PATH` matches the env you use for MD |

For cluster jobs, completion is optional — production launches use YAML configs
and `karml-charmm-mpirun.sh` wrappers documented under
[PyCHARMM MPI](../pycharmm-mpi.md).
