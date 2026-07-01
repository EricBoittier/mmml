# Tab completion

MMML uses [argcomplete](https://github.com/kislyuk/argcomplete) for bash, zsh,
and fish tab completion on subcommand names and flags.

## Install

```bash
uv sync --extra cli
# or: pip install 'mmml[cli]'
```

The `cli` extra adds `argcomplete>=3.5.0`. Without it, `mmml completion` still
prints a **top-level-only** script (subcommand names, no flags).

## Enable in your shell

### Bash / zsh (recommended)

```bash
eval "$(register-python-argcomplete mmml)"
```

Add that line to `~/.bashrc` or `~/.zshrc` after activating the MMML virtualenv
(or use the full path to the `mmml` executable in your env).

### Via `mmml completion`

```bash
eval "$(mmml completion bash)"
eval "$(mmml completion zsh)"
mmml completion fish | source   # fish
```

Use `--executable` if the CLI entry point is not named `mmml`:

```bash
mmml completion bash --executable /path/to/.venv/bin/mmml
```

`--install-hint` prints setup reminders on stderr after the script.

## What gets completed

1. **Top level** — all `mmml` subcommands (`md-system`, `physnet-train`, …).
2. **Per command** — flags and choices when the subcommand defines
   `build_parser()` (see `mmml commands --audit` for coverage).

Completion hooks run before heavy imports when possible; `_ARGCOMPLETE` is set
by the shell integration and handled in `mmml.cli.completion.try_autocomplete()`.

## Audit completion coverage

```bash
mmml commands --audit
```

Active commands with `✓ flags` have full flag completion. Others complete the
subcommand name only until `build_parser()` is added.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Only subcommand names complete | Install `mmml[cli]` / `argcomplete` |
| Wrong `mmml` binary completed | Set `MMML_PYTHON` / activate venv before `eval` |
| Completion runs wrong parser | Ensure `mmml` on `PATH` matches the env you use for MD |

For cluster jobs, completion is optional — production launches use YAML configs
and `mmml-charmm-mpirun.sh` wrappers documented under
[PyCHARMM MPI](../pycharmm-mpi.md).
