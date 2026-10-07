# `karml commands`

Browse subcommands (grouped).


`karml commands` lists every subcommand grouped by task area — a browsable
alternative to the compact top-level `karml -h`.

```bash
karml commands
karml commands --audit    # deprecated/legacy + tab-completion coverage
```

The grouped list is defined in `karml/cli/help_text.py` and kept in sync with
`karml/cli/registry.py`.

## Usage

```bash
karml commands --help
```

!!! note
    No `build_parser()` hook — see module docstring or run the command without arguments for usage.

Implementation: `karml.cli.commands_help`


## Related docs

- [CLI overview](../index.md)

---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
