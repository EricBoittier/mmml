# `mmml orbax-to-json`

Export Orbax checkpoint to JSON.


## Usage

```bash
mmml orbax-to-json --help
```

## Options

```text
usage: mmml orbax-to-json [-h] -o OUTPUT [--params-key PARAMS_KEY] checkpoint

Convert an Orbax checkpoint to a portable JSON file.

positional arguments:
  checkpoint            Orbax checkpoint directory (epoch-* dir or experiment root)

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output JSON path (e.g. DESdimers_params.json)
  --params-key PARAMS_KEY
                        Key to extract from restored checkpoint dict (default: "params")

CLI: export an Orbax checkpoint to a portable JSON file.

Usage:
    mmml orbax-to-json path/to/epoch-1985 -o DESdimers_params.json
```



---

[← CLI overview](../index.md) · [All commands](../index.md#command-index)
