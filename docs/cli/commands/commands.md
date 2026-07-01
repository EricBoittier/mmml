# `mmml commands`

Browse subcommands (grouped).


`mmml commands` lists every subcommand grouped by task area — a browsable
alternative to the compact top-level `mmml -h`.

```bash
mmml commands
mmml commands --audit    # deprecated/legacy + tab-completion coverage
```

The grouped list is defined in `mmml/cli/help_text.py` and kept in sync with
`mmml/cli/registry.py`.

## Usage

```bash
mmml commands --help
```

!!! note
    No `build_parser()` hook — see module docstring or run the command without arguments for usage.

Implementation: `mmml.cli.commands_help`

## Related docs

- [CLI overview](../index.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
